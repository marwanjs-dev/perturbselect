"""Small extensible registry wrapping standard numerical-library metrics."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metric:
    """A numeric scorer and its improvement direction."""
    name: str
    score: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool


METRICS: dict[str, Metric] = {}


def register_metric(metric: Metric, *, replace: bool = False) -> None:
    """Register a metric explicitly; accidental replacement is rejected."""
    if metric.name in METRICS and not replace:
        raise ValueError(f"Metric already registered: {metric.name}")
    METRICS[metric.name] = metric


def _correlation(y: np.ndarray, pred: np.ndarray, scorer: Callable) -> float:
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return float("nan")
    return float(scorer(y, pred).statistic)


for _metric in (
    Metric("spearman_rho", lambda y, p: _correlation(y, p, spearmanr), True),
    Metric("pearson_r", lambda y, p: _correlation(y, p, pearsonr), True),
    Metric("r2", lambda y, p: float(r2_score(y, p)) if len(y) >= 2 else float("nan"), True),
    Metric("mae", lambda y, p: float(mean_absolute_error(y, p)), False),
    Metric("rmse", lambda y, p: float(np.sqrt(mean_squared_error(y, p))), False),
):
    register_metric(_metric)


def score_metric(name: str, y: np.ndarray, pred: np.ndarray, *, bootstrap: bool = False) -> float:
    """Score finite aligned vectors; constant outcomes invalidate bootstrap R2.

    Ordinary R2 retains sklearn's force_finite convention. For resampling, an
    undefined constant-outcome denominator is not treated as a valid replicate.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Metric vectors must have identical nonempty 1D shapes.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must be finite.")
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}")
    if bootstrap and name == "r2" and np.ptp(y) == 0:
        return float("nan")
    return float(METRICS[name].score(y, pred))
