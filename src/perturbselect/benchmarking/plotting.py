"""Publication-oriented views of precomputed tables; never evaluate or fit models."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .metrics import METRICS
from .schema import finite_numeric, require_columns, validate_results


def _axes(n: int = 5) -> tuple[Figure, Axes]:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        return plt.subplots(figsize=(7, max(3, .38 * n + 1.4)), layout="constrained")


def _order(table: pd.DataFrame, order: list[str] | None) -> pd.DataFrame:
    if table.method.duplicated().any():
        raise ValueError("Select one observation per method before plotting; no implicit aggregation.")
    names = sorted(table.method) if order is None else order
    if len(set(names)) != len(names) or set(names) != set(table.method):
        raise ValueError("method_order must contain every plotted method exactly once.")
    return table.set_index("method").loc[names].reset_index()


def _metric_rows(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = validate_results(results)
    data = data[data.metric == metric].copy()
    if data.empty:
        raise ValueError(f"No results for metric {metric}.")
    return data


def _points(data: pd.DataFrame, value: str, low: str, high: str,
            labels: dict[str, str] | None = None) -> tuple[Figure, Axes]:
    fig, ax = _axes(len(data))
    locations = np.arange(len(data))
    if low in data or high in data:
        require_columns(data, [low, high])
        for column in (low, high):
            finite_numeric(data[column], column, allow_nan=True)
        if (data[low] > data[high]).any():
            raise ValueError("Interval lower bound exceeds upper bound.")
        # Segment rather than asymmetric errorbar: percentile intervals need not contain the point estimate.
        ax.hlines(locations, data[low], data[high], color="C0", linewidth=1.5)
    ax.scatter(data[value], locations, color="C0", s=32, zorder=3)
    ax.set_yticks(locations, [(labels or {}).get(m, m) for m in data.method])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    return fig, ax


def plot_model_comparison(results: pd.DataFrame, metric: str, *, method_order: list[str] | None = None,
                           display_labels: dict[str, str] | None = None) -> tuple[Figure, Axes]:
    """One method point with optional ci_low/ci_high supplied by the caller."""
    data = _order(_metric_rows(results, metric), method_order)
    fig, ax = _points(data, "value", "ci_low", "ci_high", display_labels)
    ax.set_xlabel(metric)
    return fig, ax


def plot_forest(comparisons: pd.DataFrame, metric: str, *, method_order: list[str] | None = None,
                display_labels: dict[str, str] | None = None) -> tuple[Figure, Axes]:
    """Plot direction-adjusted improvement; positive always favors the candidate."""
    require_columns(comparisons, ["method", "baseline", "metric", "observed_improvement",
                                  "improvement_ci_low", "improvement_ci_high"])
    data = _order(comparisons[comparisons.metric == metric], method_order)
    if data.empty or data.baseline.nunique() != 1:
        raise ValueError("Select a nonempty comparison against one baseline.")
    finite_numeric(data.observed_improvement, "observed_improvement", allow_nan=True)
    fig, ax = _points(data, "observed_improvement", "improvement_ci_low", "improvement_ci_high", display_labels)
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{metric}: improvement over {data.baseline.iloc[0]}")
    return fig, ax


def plot_stability(results: pd.DataFrame, metric: str, *, repeat: str = "seed",
                    method_order: list[str] | None = None,
                    display_labels: dict[str, str] | None = None) -> tuple[Figure, Axes]:
    """Display all supplied fold/seed observations without computing confidence intervals."""
    if repeat not in ("seed", "fold"):
        raise ValueError("repeat must be seed or fold.")
    data = _metric_rows(results, metric)
    if data[repeat].isna().any() or data.duplicated(["method", repeat]).any():
        raise ValueError("Select one observation per method and explicit repeat identity.")
    methods = _order(data[["method"]].drop_duplicates(), method_order).method.tolist()
    fig, ax = _axes(len(methods))
    for i, method in enumerate(methods):
        group = data[data.method == method].sort_values(repeat)
        offsets = np.linspace(-.15, .15, len(group)) if len(group) > 1 else np.array([0.])
        ax.scatter(group.value, i + offsets, color="C0", alpha=.7, s=25)
    ax.set_yticks(range(len(methods)), [(display_labels or {}).get(m, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} across {repeat} values")
    return fig, ax


def validate_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    """Coverage view schema: method, n_supported, n_total, coverage_fraction."""
    require_columns(coverage, ["method", "n_supported", "n_total", "coverage_fraction"])
    data = coverage.copy()
    for column in ("n_supported", "n_total", "coverage_fraction"):
        finite_numeric(data[column], column)
    if ((data.n_total <= 0) | (data.n_supported < 0) | (data.n_supported > data.n_total)).any():
        raise ValueError("Coverage counts must satisfy 0 <= n_supported <= positive n_total.")
    if ((data.n_total % 1 != 0) | (data.n_supported % 1 != 0)).any():
        raise ValueError("Coverage counts must be integers.")
    if not np.allclose(data.coverage_fraction, data.n_supported / data.n_total):
        raise ValueError("Coverage fractions disagree with counts.")
    return data


def plot_coverage(coverage: pd.DataFrame, *, method_order: list[str] | None = None,
                  display_labels: dict[str, str] | None = None) -> tuple[Figure, Axes]:
    """Plot supplied coverage fractions without inventing unknown coverage."""
    data = _order(validate_coverage(coverage), method_order)
    fig, ax = _points(data, "coverage_fraction", "ci_low", "ci_high", display_labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coverage fraction")
    return fig, ax


def plot_performance_vs_coverage(results: pd.DataFrame, coverage: pd.DataFrame, metric: str) -> tuple[Figure, Axes]:
    """Join explicitly selected single-slice metrics and coverage by method name."""
    performance = _order(_metric_rows(results, metric), None)
    coverage = _order(validate_coverage(coverage), None)
    if set(performance.method) != set(coverage.method):
        raise ValueError("Performance and coverage must contain identical methods.")
    data = performance.merge(coverage, on="method", validate="one_to_one")
    fig, ax = _axes()
    ax.scatter(data.coverage_fraction, data.value, color="C0")
    for row in data.itertuples():
        ax.annotate(row.method, (row.coverage_fraction, row.value), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Coverage fraction", ylabel=metric)
    return fig, ax


def plot_metric_correlation(results: pd.DataFrame, x_metric: str, y_metric: str) -> tuple[Figure, Axes]:
    """Compare paired precomputed metric values by method; no scoring is performed."""
    x, y = (_order(_metric_rows(results, metric), None) for metric in (x_metric, y_metric))
    if set(x.method) != set(y.method):
        raise ValueError("Both metrics must contain identical methods.")
    data = x[["method", "value"]].merge(y[["method", "value"]], on="method", suffixes=("_x", "_y"), validate="one_to_one")
    fig, ax = _axes()
    ax.scatter(data.value_x, data.value_y, color="C0")
    for row in data.itertuples():
        ax.annotate(row.method, (row.value_x, row.value_y), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel=x_metric, ylabel=y_metric)
    return fig, ax


def plot_ranked_methods(results: pd.DataFrame, metric: str) -> tuple[Figure, Axes]:
    """Order methods by the registered metric direction, using names to break ties."""
    data = _order(_metric_rows(results, metric), None)
    order = data.sort_values(["value", "method"], ascending=[not METRICS[metric].higher_is_better, True], na_position="last").method.tolist()
    return plot_model_comparison(data, metric, method_order=order)


def save_figure(fig: Figure, stem: str | Path, formats: tuple[str, ...] | list[str] = ("svg", "png", "pdf"), dpi: int = 300) -> list[Path]:
    """Save vector/raster exports; caller controls figure lifetime and customization."""
    if not formats or any(f not in ("svg", "png", "pdf") for f in formats):
        raise ValueError("Supported figure formats: svg, png, pdf.")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in formats:
            path = stem.parent / f"{stem.name}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", format=extension)
            paths.append(path)
    return paths
