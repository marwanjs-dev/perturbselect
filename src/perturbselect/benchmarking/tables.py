"""Numeric publication tables and independent display serialization."""
from pathlib import Path

import pandas as pd

from .schema import require_columns, validate_results
from .statistics import summarize_repeats


def performance_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot a caller-selected result slice; never silently average duplicates."""
    data = validate_results(results)
    if data.duplicated(["method", "metric"]).any():
        raise ValueError("Select one result identity per method/metric before tabulation.")
    return data.pivot(index="method", columns="metric", values="value").reset_index()


def comparison_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Retain numeric raw deltas, directional improvements and uncertainty."""
    require_columns(comparisons, ["method", "baseline", "metric", "observed_delta", "ci_low", "ci_high"])
    if comparisons.duplicated(["method", "baseline", "metric"]).any():
        raise ValueError("Select unique comparison identities before tabulation.")
    return comparisons.copy(deep=True)


def stability_table(results: pd.DataFrame, repeat: str = "seed") -> pd.DataFrame:
    """Numeric descriptive stability summaries; no confidence intervals inferred."""
    return summarize_repeats(results, repeat)


def coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    """Preserve supplied coverage metadata and values."""
    require_columns(coverage, ["method", "n_supported", "n_total", "coverage_fraction"])
    return coverage.copy(deep=True)


def dataset_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Expose observed dataset/task counts rather than guessing cohort-wide totals."""
    data = validate_results(results)
    return data[["study", "task", "dataset", "context", "method", "representation", "split", "n_samples", "n_features"]].drop_duplicates().reset_index(drop=True)


def format_table(table: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """Format a separate string-valued display copy; numeric input is untouched."""
    if type(precision) is not int or precision < 0:
        raise ValueError("precision must be a nonnegative integer.")
    def display(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)
    return table.apply(lambda column: column.map(display))


def render_table(table: pd.DataFrame, format: str, precision: int = 3) -> str:
    """CSV preserves numeric precision; Markdown/LaTeX use display rounding only."""
    if format == "csv":
        return table.to_csv(index=False)
    display = format_table(table, precision)
    rows = [list(map(str, display.columns)), *display.values.tolist()]
    if format == "markdown":
        escape = lambda v: v.replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(escape(v) for v in row) + " |" for row in rows]
        lines.insert(1, "| " + " | ".join(["---"] * len(display.columns)) + " |")
        return "\n".join(lines)
    if format == "latex":
        replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        escape = lambda v: "".join(replacements.get(c, c) for c in v)
        lines = [" & ".join(escape(v) for v in row) + r" \\" for row in rows]
        lines.insert(1, r"\hline")
        return "\\begin{tabular}{" + "l" * len(display.columns) + "}\n" + "\n".join(lines) + "\n\\end{tabular}\n"
    raise ValueError("Table format must be csv, markdown or latex.")


def save_table(table: pd.DataFrame, path: str | Path, format: str, precision: int = 3) -> Path:
    """Save a numeric CSV or separately rounded presentation artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_table(table, format, precision), encoding="utf-8")
    return path
