"""Canonical tidy schemas; optional identity fields are explicit, never inferred."""
import numpy as np
import pandas as pd

AXES = ["study", "task", "dataset", "context", "method", "method_variant",
        "representation", "baseline", "split", "fold", "seed"]
RESULT_ID = AXES + ["metric"]
PREDICTION_ID = AXES + ["sample_id"]


def require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    """Raise a readable error instead of implicitly manufacturing required values."""
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")


def finite_numeric(values: pd.Series, name: str, *, allow_nan: bool = False) -> None:
    """Require real numbers; strings, booleans and infinities are not observations."""
    if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)):
        raise ValueError(f"{name} must contain real numeric values.")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        raise ValueError(f"{name} must contain finite values.")


def _base(table: pd.DataFrame, required: list[str], identity: list[str]) -> pd.DataFrame:
    require_columns(table, required)
    result = table.copy(deep=True)
    for name in ("task", "dataset", "method", "split"):
        if result[name].isna().any() or any(not isinstance(v, str) or not v.strip() for v in result[name]):
            raise ValueError(f"{name} must contain nonempty strings.")
    for name in AXES:
        if name not in result:
            result[name] = None
    if result.duplicated(identity).any():
        raise ValueError("Duplicate result/prediction identities; aggregate or select explicitly.")
    return result


def validate_results(table: pd.DataFrame) -> pd.DataFrame:
    """Validate metric rows and preserve extra metadata without rounding.

    NaN is allowed only with value_status='undefined'; infinities are rejected.
    Optional axes participate in identity when present. Counts are nonnegative
    integers; n_features may be unknown, while n_samples is required.
    """
    result = _base(table, ["task", "dataset", "method", "split", "metric", "value", "n_samples"], RESULT_ID)
    if result.metric.isna().any() or any(not isinstance(v, str) or not v.strip() for v in result.metric):
        raise ValueError("metric must contain nonempty strings.")
    if "value_status" not in result:
        result["value_status"] = "defined"
    if not result.value_status.isin(["defined", "undefined"]).all():
        raise ValueError("value_status must be defined or undefined.")
    finite_numeric(result.value, "value", allow_nan=True)
    if not np.array_equal(result.value.isna(), result.value_status.eq("undefined")):
        raise ValueError("Undefined values must be NaN and explicitly marked undefined.")
    for name in ("n_samples", "n_features"):
        if name not in result:
            result[name] = None
        present = result[name].dropna()
        if name == "n_samples" and len(present) != len(result):
            raise ValueError("n_samples is required for every result.")
        if len(present):
            finite_numeric(present, name)
            if ((present < 0) | (present % 1 != 0)).any():
                raise ValueError(f"{name} must contain nonnegative integers.")
    return result


def validate_predictions(table: pd.DataFrame) -> pd.DataFrame:
    """Validate one finite prediction per named sample and method identity."""
    result = _base(table, ["sample_id", "task", "dataset", "method", "split", "y_true", "y_pred"], PREDICTION_ID)
    if result.sample_id.isna().any() or any(not isinstance(v, str) or not v.strip() for v in result.sample_id):
        raise ValueError("sample_id must contain nonempty strings.")
    for name in ("y_true", "y_pred"):
        finite_numeric(result[name], name)
    return result
