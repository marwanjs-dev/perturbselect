"""Explicit baseline-relative paired comparisons; no automatic significance claims."""
import pandas as pd

from .statistics import paired_bootstrap


def compare_to_baseline(predictions: pd.DataFrame, baseline: str, metrics: list[str], *,
                         methods: list[str] | None = None, **bootstrap_options) -> pd.DataFrame:
    """Return raw deltas and direction-aware improvements for each requested metric."""
    methods = methods if methods is not None else sorted(set(predictions.method) - {baseline})
    if baseline in methods or len(set(methods)) != len(methods):
        raise ValueError("Comparison methods must be unique and exclude the baseline.")
    return pd.DataFrame([paired_bootstrap(predictions, method, baseline, metric, **bootstrap_options)
                         for method in methods for metric in metrics])
