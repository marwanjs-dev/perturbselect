"""Fold-safe prediction of gene-level functional phenotypes."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .qc import finite_vector, response_magnitude, spearman_association, validate_cell_counts
from .signatures import mask_target_expression


@dataclass
class BenchmarkResult:
    """Metrics and aligned out-of-fold predictions."""
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def make_cv(n_splits: int = 5, random_state: int = 42) -> KFold:
    """Create shuffled perturbation-level folds."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_mean_model() -> DummyRegressor:
    """Training-fold phenotype mean baseline."""
    return DummyRegressor(strategy="mean")


def make_scalar_ridge_model() -> Pipeline:
    """Scale and select ridge regularization inside each training fold."""
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 17)))])


def make_pca_ridge_model(n_components: int = 100, random_state: int = 42) -> Pipeline:
    """Build the validated scaler/randomized-PCA/RidgeCV estimator."""
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=n_components, svd_solver="randomized",
                                 random_state=random_state)),
                     ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 17)))])


def cross_validated_predictions(model: BaseEstimator, X: ArrayLike, y: ArrayLike,
                                 cv: KFold, n_jobs: int = 1) -> NDArray:
    """Predict every held-out perturbation; reject infeasible PCA dimensions."""
    X, y = np.asarray(X), finite_vector(y, "y")
    response_magnitude(X)  # shared finite/2D validation, without changing precision
    if len(y) != X.shape[0] or len(y) < 2:
        raise ValueError("X and y must have equal row counts and at least two rows.")
    splits = list(cv.split(X, y))
    if isinstance(model, Pipeline) and "pca" in model.named_steps:
        components = model.named_steps["pca"].n_components
        limit = min(X.shape[1], min(len(train) for train, _ in splits))
        if isinstance(components, int) and components > limit:
            raise ValueError(f"PCA n_components={components} exceeds training-fold limit {limit}.")
    return cross_val_predict(model, X, y, cv=splits, n_jobs=n_jobs)


def evaluate_predictions(name: str, y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Score predictions; constant vectors have undefined (NaN) correlations."""
    y, pred = finite_vector(y_true, "y_true"), finite_vector(y_pred, "y_pred")
    association = spearman_association(y, pred)
    if len(y) < 2:
        raise ValueError("Evaluation requires at least two observations.")
    pearson = float("nan")
    if np.ptp(y) != 0 and np.ptp(pred) != 0:
        pearson = float(pearsonr(y, pred).statistic)
    return {"model": name, "spearman_rho": association["rho"], "pearson_r": pearson,
            "r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def run_benchmarks(X: ArrayLike, y: ArrayLike, n_cells: ArrayLike, *,
                   n_components: int = 100, n_splits: int = 5,
                   random_state: int = 42) -> BenchmarkResult:
    """Run mean, log-cell-count, magnitude and full-response baselines."""
    magnitude = response_magnitude(X)
    counts = validate_cell_counts(n_cells, len(magnitude))
    cv = make_cv(n_splits, random_state)
    specifications = [
        ("Mean baseline", "pred_mean", make_mean_model(), np.zeros((len(magnitude), 1))),
        ("Cell count only", "pred_cell_count", make_scalar_ridge_model(), np.log1p(counts)[:, None]),
        ("RNA magnitude only", "pred_rna_magnitude", make_scalar_ridge_model(), magnitude[:, None]),
        ("PCA + Ridge", "pred_pca_ridge", make_pca_ridge_model(n_components, random_state), X),
    ]
    predictions, metrics = {}, []
    for name, column, model, features in specifications:
        predictions[column] = cross_validated_predictions(model, features, y, cv)
        metrics.append(evaluate_predictions(name, y, predictions[column]))
    return BenchmarkResult(pd.DataFrame(metrics), pd.DataFrame(predictions))


def evaluate_target_masked(X: NDArray, y: ArrayLike, targets: Sequence[str],
                           genes: Sequence[str], model: BaseEstimator,
                           cv: KFold) -> BenchmarkResult:
    """Evaluate a copy with the per-row target feature masked."""
    masked, _ = mask_target_expression(X, targets, genes)
    pred = cross_validated_predictions(model, masked, y, cv)
    return BenchmarkResult(pd.DataFrame([evaluate_predictions("PCA + Ridge (target masked)", y, pred)]),
                           pd.DataFrame({"pred_target_masked": pred}))


def cv_stability(X: ArrayLike, y: ArrayLike, seeds: Sequence[int],
                  model: BaseEstimator | Callable[[], BaseEstimator], *,
                  n_splits: int = 5) -> pd.DataFrame:
    """Vary only fold seeds; keep estimator/PCA settings fixed."""
    rows = []
    for seed in seeds:
        estimator = model() if callable(model) else model
        pred = cross_validated_predictions(estimator, X, y, make_cv(n_splits, seed))
        rows.append({"seed": seed, **evaluate_predictions("PCA + Ridge (target masked)", y, pred)})
    return pd.DataFrame(rows, columns=["seed", "model", "spearman_rho", "pearson_r", "r2", "mae", "rmse"])


def permutation_control(model: BaseEstimator, X: ArrayLike, y: ArrayLike,
                         cv: KFold, seed: int = 42) -> BenchmarkResult:
    """Fit and score the same estimator/folds against Generator-permuted labels."""
    shuffled = np.random.default_rng(seed).permutation(finite_vector(y, "y"))
    pred = cross_validated_predictions(model, X, shuffled, cv)
    return BenchmarkResult(pd.DataFrame([evaluate_predictions("PCA + Ridge (shuffled labels)", shuffled, pred)]),
                           pd.DataFrame({"shuffled_fitness": shuffled, "pred_shuffled": pred}))
