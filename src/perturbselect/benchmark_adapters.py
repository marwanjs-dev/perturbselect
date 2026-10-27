"""PerturbSelect-only adapters; the generic benchmarking package never imports this."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarking.metrics import METRICS
from .benchmarking.schema import validate_predictions, validate_results


SELECTIVITY_PREDICTIONS = {
    "pred_mean": "Mean baseline", "pred_cell_counts": "Cell count only (both contexts)",
    "pred_magnitude_difference": "RNA magnitude difference", "pred_k562_only": "K562 RNA only",
    "pred_rpe1_only": "RPE1 RNA only", "pred_differential": "Differential RNA (K562 - RPE1)",
    "pred_joint": "Joint RNA (K562 + RPE1)",
}


def adapt_metrics(table: pd.DataFrame, *, task: str, dataset: str, n_samples: int,
                   study: str = "perturbselect", seed: int | None = None) -> pd.DataFrame:
    """Melt stored metrics without recalculating or rounding any observation."""
    if "model" not in table:
        raise ValueError("Source metrics need a model column.")
    metric_columns = [name for name in METRICS if name in table]
    if not metric_columns:
        raise ValueError("No recognized saved metrics.")
    data = table.rename(columns={"model": "method"}).melt(
        id_vars=[column for column in table.rename(columns={"model": "method"}) if column not in metric_columns],
        value_vars=metric_columns, var_name="metric", value_name="value")
    data["study"], data["task"], data["dataset"] = study, data.get("endpoint", task), dataset
    data["split"], data["n_samples"] = "out_of_fold", n_samples
    if "seed" not in data:
        data["seed"] = seed
    data["value_status"] = np.where(data.value.isna(), "undefined", "defined")
    return validate_results(data)


def adapt_predictions(table: pd.DataFrame, *, target_column: str, prediction_columns: dict[str, str],
                       task: str, dataset: str, study: str = "perturbselect",
                       seed: int | None = None) -> pd.DataFrame:
    """Map each saved wide prediction to a named independent gene sample.

    No fold IDs are reconstructed. A caller must provide separately saved fold
    assignments if needed; unknown folds remain null.
    """
    required = {"gene", target_column, *prediction_columns}
    if not required <= set(table):
        raise ValueError(f"Missing saved prediction columns: {sorted(required - set(table))}")
    pieces = []
    for column, method in prediction_columns.items():
        piece = table[["gene", target_column, column]].copy()
        piece.columns = ["sample_id", "y_true", "y_pred"]
        piece = piece.assign(study=study, task=task, dataset=dataset, method=method,
                             split="out_of_fold", seed=seed)
        pieces.append(piece)
    return validate_predictions(pd.concat(pieces, ignore_index=True))


def adapt_incremental_bootstrap(table: pd.DataFrame) -> pd.DataFrame:
    """Rename the saved diagnostic fields and add metric-aware improvement views."""
    required = {"comparison", "metric", "observed_delta", "ci_2_5", "ci_97_5",
                "proportion_gt_zero", "n_requested_bootstrap", "n_valid_bootstrap",
                "n_invalid_bootstrap"}
    if not required <= set(table):
        raise ValueError(f"Missing saved comparison columns: {sorted(required - set(table))}")
    result = table.rename(columns={"ci_2_5": "ci_low", "ci_97_5": "ci_high",
        "n_requested_bootstrap": "n_requested", "n_valid_bootstrap": "n_valid",
        "n_invalid_bootstrap": "n_invalid"}).copy()
    pair = result.comparison.str.split(" vs ", n=1, expand=True)
    if pair.shape[1] != 2 or pair.isna().any().any():
        raise ValueError("Saved comparisons must name candidate vs baseline.")
    aliases = {"Joint RNA": "Joint RNA (K562 + RPE1)",
               "Differential RNA": "Differential RNA (K562 - RPE1)"}
    result["method"], result["baseline"] = pair[0].replace(aliases), pair[1].replace(aliases)
    if not result.metric.isin(METRICS).all():
        raise ValueError("Saved comparison uses an unregistered metric.")
    direction = result.metric.map(lambda name: 1 if METRICS[name].higher_is_better else -1)
    result["observed_improvement"] = direction * result.observed_delta
    result["improvement_ci_low"] = np.where(direction == 1, result.ci_low, -result.ci_high)
    result["improvement_ci_high"] = np.where(direction == 1, result.ci_high, -result.ci_low)
    # The saved fraction is P(raw delta > 0). For lower-is-better metrics,
    # P(raw delta < 0) cannot be recovered without the number of exact ties.
    result["proportion_better"] = np.where(direction == 1, result.proportion_gt_zero, np.nan)
    if result.duplicated(["method", "baseline", "metric"]).any():
        raise ValueError("Duplicate saved comparison identities.")
    return result


def export_selectivity(input_directory: str | Path, root: str | Path,
                        presentation_config: str | Path | None = None) -> Path:
    """Write new canonical artifacts and optionally present them; original files untouched."""
    from .benchmarking.provenance import capture_provenance
    from .benchmarking.reporting import ArtifactLayout, generate_presentation, load_presentation_config
    from .benchmarking.results import write_results

    source = Path(input_directory)
    layout = ArtifactLayout(Path(root), "perturbselect_standardized")
    paths = [source / "selectivity_model_metrics.csv", source / "selectivity_predictions.csv",
             source / "selectivity_summary.json", source / "resolved_config.json"]
    metrics, predictions = [pd.read_csv(path, float_precision="round_trip") for path in paths[:2]]
    summary = json.loads(paths[2].read_text(encoding="utf-8"))
    config = json.loads(paths[3].read_text(encoding="utf-8"))
    seed = config.get("analysis", {}).get("random_seed")
    if predictions.gene.nunique() != len(predictions) or len(predictions) != summary["n_common_targets"]:
        raise ValueError("Saved prediction sample identities/count disagree with summary.")
    canonical = adapt_metrics(metrics, task="functional_selectivity", dataset="K562_RPE1_common",
                              n_samples=len(predictions), seed=seed)
    canonical_predictions = adapt_predictions(predictions, target_column="functional_selectivity",
        prediction_columns=SELECTIVITY_PREDICTIONS, task="functional_selectivity", dataset="K562_RPE1_common", seed=seed)
    layout.results("raw").mkdir(parents=True, exist_ok=True)
    write_results(canonical, layout.results("raw") / "metrics.csv")
    canonical_predictions.to_csv(layout.results("raw") / "predictions.csv", index=False)
    comparison_path = source / "selectivity_incremental_bootstrap.csv"
    stability_path = source / "selectivity_cv_stability.csv"
    comparisons = robustness = None
    if comparison_path.is_file():
        comparisons = adapt_incremental_bootstrap(pd.read_csv(comparison_path, float_precision="round_trip"))
        layout.results("statistics").mkdir(parents=True, exist_ok=True)
        comparisons.to_csv(layout.results("statistics") / "comparisons.csv", index=False)
        paths.append(comparison_path)
    if stability_path.is_file():
        robustness = adapt_metrics(pd.read_csv(stability_path, float_precision="round_trip"),
            task="functional_selectivity", dataset="K562_RPE1_common", n_samples=len(predictions))
        layout.results("statistics").mkdir(parents=True, exist_ok=True)
        write_results(robustness, layout.results("statistics") / "stability.csv")
        paths.append(stability_path)
    provenance = capture_provenance(sources=paths, config=presentation_config,
                                    seeds=[] if seed is None else [seed],
                                    packages=["numpy", "pandas", "scipy", "scikit-learn"], git_directory=root)
    (layout.results("raw") / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if presentation_config is not None:
        return generate_presentation(canonical, layout, load_presentation_config(presentation_config),
                                     metadata=summary, provenance=provenance,
                                     comparisons=comparisons, robustness=robustness)
    return layout.results("raw")


def main() -> None:
    """Optional export/presentation command; no scientific analysis is rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/selectivity")
    parser.add_argument("--root", default=".")
    parser.add_argument("--presentation")
    args = parser.parse_args()
    print(export_selectivity(args.input, args.root, args.presentation))


if __name__ == "__main__":
    main()
