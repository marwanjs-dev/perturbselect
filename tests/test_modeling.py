import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from perturbselect.modeling import (cross_validated_predictions, cv_stability,
    evaluate_predictions, evaluate_target_masked, make_cv, make_mean_model,
    make_pca_ridge_model, make_scalar_ridge_model, permutation_control, run_benchmarks)


@pytest.fixture
def signal():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(120, 4))
    return X, 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=.05, size=120)


def test_benchmark_signal_and_all_predictions(signal):
    X, y = signal
    result = run_benchmarks(X, y, np.full(len(y), 30), n_components=4)
    assert result.predictions.shape == (120, 4)
    assert np.isfinite(result.predictions).all().all()
    scores = result.metrics.set_index("model")
    assert scores.loc["PCA + Ridge", "r2"] > .95
    assert scores.loc["PCA + Ridge", "r2"] > scores.loc["Mean baseline", "r2"]


def test_permutation_breaks_signal_and_is_reproducible(signal):
    X, y = signal
    model = make_pca_ridge_model(4)
    a = permutation_control(model, X, y, make_cv())
    b = permutation_control(model, X, y, make_cv())
    np.testing.assert_array_equal(a.predictions, b.predictions)
    np.testing.assert_array_equal(a.predictions.shuffled_fitness, np.random.default_rng(42).permutation(y))
    assert a.metrics.iloc[0].r2 < .2


def test_metric_values_and_constant_predictions():
    metrics = evaluate_predictions("example", [1, 2, 3], [1, 2, 4])
    assert metrics["r2"] == pytest.approx(.5)
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["spearman_rho"] == pytest.approx(1)
    constant = evaluate_predictions("constant", [1, 2, 3], [2, 2, 2])
    assert np.isnan(constant["spearman_rho"])
    assert np.isnan(constant["pearson_r"])
    assert constant["r2"] == 0


def test_pipeline_structure_and_fold_reproducibility(signal):
    model = make_pca_ridge_model(3)
    assert isinstance(model.named_steps["scale"], StandardScaler)
    assert isinstance(model.named_steps["pca"], PCA)
    assert isinstance(model.named_steps["ridge"], RidgeCV)
    assert model.named_steps["pca"].svd_solver == "randomized"
    X, y = signal
    a = cross_validated_predictions(model, X, y, make_cv())
    b = cross_validated_predictions(model, X, y, make_cv())
    np.testing.assert_array_equal(a, b)
    # cross_val_predict clones the whole pipeline, leaving this estimator unfitted.
    assert not hasattr(model.named_steps["scale"], "mean_")
    stability = cv_stability(X, y, [11, 22], lambda: make_pca_ridge_model(3))
    assert stability.seed.tolist() == [11, 22]


class NoSeenRows(RegressorMixin, BaseEstimator):
    """Fail if any held-out row was present during fitting."""
    def fit(self, X, y):
        self.seen_ = set(X[:, 0])
        return self

    def predict(self, X):
        assert not self.seen_.intersection(X[:, 0])
        return np.ones(len(X))


def test_predictions_are_held_out():
    pred = cross_validated_predictions(NoSeenRows(), np.arange(20)[:, None], np.arange(20), make_cv())
    np.testing.assert_array_equal(pred, np.ones(20))


def test_masked_evaluation_preserves_input(signal):
    X, y = signal
    original = X.copy()
    result = evaluate_target_masked(X, y, [f"g{i}" for i in range(120)],
                                    [f"g{i}" for i in range(4)], make_pca_ridge_model(3), make_cv())
    assert len(result.predictions) == len(y)
    np.testing.assert_array_equal(X, original)


def test_invalid_dimensions_and_pca_limit(signal):
    X, y = signal
    with pytest.raises(ValueError, match="equal row"):
        cross_validated_predictions(make_mean_model(), X, y[:-1], make_cv())
    with pytest.raises(ValueError, match="2D"):
        cross_validated_predictions(make_scalar_ridge_model(), X[:, 0], y, make_cv())
    with pytest.raises(ValueError, match="training-fold limit"):
        cross_validated_predictions(make_pca_ridge_model(100), X, y, make_cv())
    with pytest.raises(ValueError, match="finite"):
        evaluate_predictions("bad", [1, 2], [1, np.nan])
