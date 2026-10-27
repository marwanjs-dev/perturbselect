"""Statistics on matched independent sample identities and fixed predictions."""
import numpy as np
import pandas as pd

from .metrics import METRICS, score_metric
from .schema import AXES, validate_predictions, validate_results


def percentile_interval(values: np.ndarray, confidence: float = .95) -> tuple[float, float]:
    """Percentile bounds of finite replicates; all-invalid input gives NaN bounds."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan"), float("nan")
    return tuple(np.percentile(valid, [50 * (1 - confidence), 50 * (1 + confidence)]))


def match_predictions(predictions: pd.DataFrame, method: str, baseline: str) -> pd.DataFrame:
    """Strictly pair two methods on sample identity, truth, and fold assignment.

    Select a single study/task/dataset/context/split/seed before calling. Multiple
    variants/representations of one method require caller filtering. Fold is
    matched per sample; repeated observations are never independently resampled.
    """
    data = validate_predictions(predictions)
    data = data[data.method.isin([method, baseline])]
    for axis in ("study", "task", "dataset", "context", "split", "seed"):
        if data[axis].nunique(dropna=False) != 1:
            raise ValueError(f"Select one {axis} before paired comparison.")
    a, b = (data[data.method == name].set_index("sample_id") for name in (method, baseline))
    if a.empty or b.empty or a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Each compared method needs one prediction per independent sample.")
    for method_data in (a, b):
        for axis in ("method_variant", "representation", "baseline"):
            if method_data[axis].nunique(dropna=False) != 1:
                raise ValueError(f"Select one {axis} for each compared method.")
    if set(a.index) != set(b.index):
        raise ValueError("Compared sample universes must match exactly.")
    a, b = a.sort_index(), b.loc[sorted(a.index)]
    if not np.array_equal(a.y_true, b.y_true):
        raise ValueError("Paired ground truths disagree.")
    if not a.fold.fillna("__missing__").equals(b.fold.fillna("__missing__")):
        raise ValueError("Paired fold assignments disagree.")
    return pd.DataFrame({"sample_id": a.index, "y_true": a.y_true.to_numpy(),
                         "candidate": a.y_pred.to_numpy(), "baseline": b.y_pred.to_numpy()})


def paired_bootstrap(predictions: pd.DataFrame, method: str, baseline: str, metric: str, *,
                     n_bootstrap: int = 5000, seed: int = 42, confidence: float = .95) -> dict:
    """Paired fixed-prediction bootstrap; positive improvement always means better.

    Raw deltas are candidate minus baseline. Direction-adjusted effects reverse
    lower-is-better metrics. Proportion_better is not a p-value. Intervals are
    conditional on these fits, not retraining uncertainty.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Require positive integer n_bootstrap and nonnegative integer seed.")
    paired = match_predictions(predictions, method, baseline)
    y, a, b = (paired[key].to_numpy() for key in ("y_true", "candidate", "baseline"))
    if len(y) < 2:
        raise ValueError("Bootstrap requires at least two independent samples.")
    observed = score_metric(metric, y, a, bootstrap=True) - score_metric(metric, y, b, bootstrap=True)
    direction = 1 if METRICS[metric].higher_is_better else -1
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.integers(0, len(y), size=len(y))
        deltas[i] = score_metric(metric, y[sample], a[sample], bootstrap=True) - score_metric(metric, y[sample], b[sample], bootstrap=True)
    valid = deltas[np.isfinite(deltas)]
    low, high = percentile_interval(valid, confidence)
    better_low, better_high = percentile_interval(direction * valid, confidence)
    return {"method": method, "baseline": baseline, "metric": metric, "observed_delta": observed,
            "ci_low": low, "ci_high": high, "observed_improvement": direction * observed,
            "improvement_ci_low": better_low, "improvement_ci_high": better_high,
            "bootstrap_mean": float(valid.mean()) if len(valid) else np.nan,
            "bootstrap_median": float(np.median(valid)) if len(valid) else np.nan,
            "proportion_better": float(np.mean(direction * valid > 0)) if len(valid) else np.nan,
            "n_requested": n_bootstrap, "n_valid": len(valid), "n_invalid": n_bootstrap - len(valid),
            "n_samples": len(y), "seed": seed, "confidence": confidence}


def summarize_values(values: np.ndarray) -> dict:
    """Describe finite values using sample SD; counts expose omitted undefined values."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return {"n_valid": len(valid), "n_invalid": len(values) - len(valid),
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "median": float(np.median(valid)) if len(valid) else np.nan,
            "sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            "iqr": float(np.diff(np.percentile(valid, [25, 75]))[0]) if len(valid) else np.nan}


def summarize_repeats(results: pd.DataFrame, repeat: str = "seed") -> pd.DataFrame:
    """Summarize seeds or folds without pooling different tasks/representations."""
    if repeat not in ("seed", "fold"):
        raise ValueError("repeat must be seed or fold.")
    data = validate_results(results)
    if data[repeat].isna().any():
        raise ValueError(f"Repeated summaries require explicit {repeat} values.")
    groups = [c for c in AXES if c != repeat] + ["metric"]
    return pd.DataFrame([{**dict(zip(groups, keys)), **summarize_values(group.value.to_numpy())}
                         for keys, group in data.groupby(groups, dropna=False, sort=False)])


def adjust_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment of supplied p-values; NaNs remain unknown.

    This never converts bootstrap support proportions into p-values. The caller
    defines the testing family; only finite supplied p-values enter its size.
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be a 1D array in [0, 1], optionally containing NaN.")
    result = np.full(values.shape, np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    return result
