"""Lightweight Markdown reports and configuration-driven presentation exports."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import pandas as pd
import yaml

from .schema import validate_results
from .tables import dataset_summary_table, performance_table, save_table, stability_table


@dataclass(frozen=True)
class ArtifactLayout:
    """New artifacts are separated from existing input paths; nothing is migrated."""
    root: Path
    study: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.study):
            raise ValueError("study directory must be a simple alphanumeric/hyphen/underscore name.")
        object.__setattr__(self, "root", Path(self.root))

    def results(self, section: str) -> Path:
        if section not in ("raw", "summaries", "statistics"):
            raise ValueError("Unknown result section.")
        return self.root / "results" / self.study / section

    def figures(self, section: str = "manuscript") -> Path:
        if section not in ("manuscript", "supplementary"):
            raise ValueError("Unknown figure section.")
        return self.root / "figures" / self.study / section

    def tables(self, section: str = "manuscript") -> Path:
        if section not in ("manuscript", "supplementary"):
            raise ValueError("Unknown table section.")
        return self.root / "tables" / self.study / section


def load_presentation_config(path: str | Path) -> dict:
    """Display-only settings; unknown keys (including statistical choices) are errors."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = {"metrics", "method_order", "baseline", "display_labels", "output_formats", "table_precision"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Presentation config contains unsupported keys; statistical decisions belong outside it.")
    config.setdefault("metrics", [])
    config.setdefault("method_order", None)
    config.setdefault("display_labels", {})
    config.setdefault("output_formats", ["svg", "png", "pdf"])
    config.setdefault("table_precision", 3)
    for key in ("metrics", "output_formats"):
        if not isinstance(config[key], list) or any(not isinstance(v, str) for v in config[key]):
            raise ValueError(f"{key} must be a list of strings.")
    if any(f not in ("svg", "png", "pdf") for f in config["output_formats"]):
        raise ValueError("Unsupported figure format.")
    if type(config["table_precision"]) is not int or config["table_precision"] < 0:
        raise ValueError("table_precision must be a nonnegative integer.")
    if not isinstance(config["display_labels"], dict):
        raise ValueError("display_labels must be a mapping.")
    if config["method_order"] is not None and (not isinstance(config["method_order"], list)
            or any(not isinstance(v, str) for v in config["method_order"])):
        raise ValueError("method_order must be a list of method names or null.")
    if config.get("baseline") is not None and not isinstance(config["baseline"], str):
        raise ValueError("baseline must be a method name or null.")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in config["display_labels"].items()):
        raise ValueError("display_labels must map method names to strings.")
    return config


def write_report(path: str | Path, *, title: str, metadata: dict,
                  artifacts: dict[str, list[Path]], provenance: dict) -> Path:
    """Assemble metadata and links for metrics/coverage/comparisons/robustness."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "```json", json.dumps(metadata, indent=2), "```", ""]
    for section, files in artifacts.items():
        lines += [f"## {section}", ""]
        for file in files:
            link = quote(Path(os.path.relpath(file, path.parent)).as_posix(), safe="/._-")
            lines.append(f"- [{Path(file).name}]({link})")
        lines.append("")
    lines += ["## Provenance", "", "```json", json.dumps(provenance, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_presentation(results: pd.DataFrame, layout: ArtifactLayout, config: dict, *,
                            metadata: dict | None = None, provenance: dict | None = None,
                            comparisons: pd.DataFrame | None = None,
                            coverage: pd.DataFrame | None = None,
                            robustness: pd.DataFrame | None = None) -> Path:
    """Present supplied observations; never run bootstrap, scoring or model code."""
    import matplotlib.pyplot as plt
    from .plotting import plot_model_comparison, plot_forest, plot_coverage, plot_stability, save_figure

    results = validate_results(results)
    requested = config["metrics"]
    if requested and (len(set(requested)) != len(requested) or set(requested) - set(results.metric)):
        raise ValueError("Requested metrics must be unique and present in the result table.")
    selected = results[results.metric.isin(requested)] if requested else results
    if selected.empty:
        raise ValueError("Presentation selection contains no results.")
    artifacts: dict[str, list[Path]] = {"Primary metrics": [], "Figures": []}
    numeric = performance_table(selected)
    if requested:
        numeric = numeric[["method", *requested]]
    order = config.get("method_order")
    if order is not None:
        if len(set(order)) != len(order) or set(order) != set(numeric.method):
            raise ValueError("method_order must contain every presented method once.")
        numeric = numeric.set_index("method").loc[order].reset_index()
    for extension, format in (("csv", "csv"), ("md", "markdown"), ("tex", "latex")):
        display = numeric if format == "csv" else numeric.assign(method=numeric.method.replace(config.get("display_labels", {})))
        artifacts["Primary metrics"].append(save_table(display, layout.tables() / f"performance.{extension}", format, config["table_precision"]))
    artifacts["Dataset summary"] = [save_table(dataset_summary_table(selected),
        layout.tables("supplementary") / "dataset_summary.csv", "csv")]
    for i, metric in enumerate(requested or selected.metric.unique()):
        fig, _ = plot_model_comparison(selected, metric, method_order=order, display_labels=config.get("display_labels"))
        try:
            artifacts["Figures"] += save_figure(fig, layout.figures() / f"performance_{i}", config["output_formats"])
        finally:
            plt.close(fig)
    for label, table in (("Comparisons", comparisons), ("Coverage", coverage), ("Robustness", robustness)):
        if table is None:
            continue
        artifacts[label] = [save_table(table, layout.tables("supplementary") / f"{label.lower()}.csv", "csv")]
        if label == "Comparisons":
            subset = table if config.get("baseline") is None else table[table.baseline == config["baseline"]]
            figures = [(f"comparison_{i}", plot_forest(subset, metric)) for i, metric in enumerate(subset.metric.unique())]
        elif label == "Coverage":
            figures = [("coverage", plot_coverage(table))]
        else:
            artifacts[label].append(save_table(stability_table(table),
                layout.tables("supplementary") / "stability_summary.csv", "csv"))
            figures = [(f"stability_{i}", plot_stability(table, metric)) for i, metric in enumerate(table.metric.unique())]
        for name, (fig, _) in figures:
            try:
                artifacts["Figures"] += save_figure(fig, layout.figures("supplementary") / name, config["output_formats"])
            finally:
                plt.close(fig)
    raw = [path for path in (layout.results("raw") / "metrics.csv",
           layout.results("raw") / "predictions.csv", layout.results("raw") / "provenance.json")
           if path.is_file()]
    if raw:
        artifacts["Standardized source tables"] = raw
    return write_report(layout.results("summaries") / "report.md", title=layout.study,
                        metadata=metadata or {}, artifacts=artifacts, provenance=provenance or {})
