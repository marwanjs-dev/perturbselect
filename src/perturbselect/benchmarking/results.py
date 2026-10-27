"""Evaluation and serialization of canonical tables; no estimators are fitted."""
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, score_metric
from .schema import AXES, validate_predictions, validate_results


def evaluate_predictions(predictions: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Evaluate each complete method/split/fold/seed identity independently."""
    predictions = validate_predictions(predictions)
    rows = []
    for keys, group in predictions.groupby(AXES, dropna=False, sort=False):
        for metric in (list(METRICS) if metrics is None else metrics):
            value = score_metric(metric, group.y_true.to_numpy(), group.y_pred.to_numpy())
            rows.append({**dict(zip(AXES, keys)), "metric": metric, "value": value,
                         "value_status": "undefined" if np.isnan(value) else "defined", "n_samples": len(group)})
    if not rows:
        raise ValueError("No predictions/metrics to evaluate.")
    return validate_results(pd.DataFrame(rows))


def write_results(results: pd.DataFrame, path: str | Path) -> None:
    """Write validated numeric values at full precision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_results(results).to_csv(path, index=False)


def read_results(path: str | Path) -> pd.DataFrame:
    """Read with round-trip precision, preserving stored floating-point ties."""
    return validate_results(pd.read_csv(path, float_precision="round_trip"))
