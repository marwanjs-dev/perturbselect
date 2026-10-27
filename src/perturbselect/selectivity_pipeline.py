"""Selectivity analysis from saved context outputs; never loads single-cell H5AD."""
import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import load_table, require_columns
from .modeling import (cross_validated_predictions, cv_stability, evaluate_predictions,
                       make_cv, make_mean_model, make_pca_ridge_model,
                       make_scalar_ridge_model, permutation_control)
from .qc import spearman_association
from .selectivity import (AlignedContexts, ContextData, align_contexts,
                          build_selectivity_endpoints, cell_count_features,
                          differential_responses, joint_responses, unique_identifiers)


DIFFERENTIAL = "Differential RNA (K562 - RPE1)"
JOINT = "Joint RNA (K562 + RPE1)"
K562_ONLY = "K562 RNA only"
RPE1_ONLY = "RPE1 RNA only"
CONTEXT_ABLATIONS = (K562_ONLY, RPE1_ONLY, DIFFERENTIAL, JOINT)
MASKED = "Differential RNA (K562 - RPE1; target masked)"


@dataclass
class PreparedSelectivity:
    """Validated common-universe data before any predictive fitting."""
    config: dict[str, Any]
    aligned: AlignedContexts
    endpoints: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class SelectivityResult:
    """Aligned inputs, saved tables and output location."""
    prepared: PreparedSelectivity
    tables: dict[str, pd.DataFrame]
    output_directory: Path


def load_selectivity_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a small saved-output configuration; paths follow its explicit root."""
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        base = path.parent
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("Selectivity configuration must be a mapping.")
    result = deepcopy(dict(config))
    root = result.get("project_root", base)
    if not isinstance(root, (str, Path)):
        raise ValueError("project_root must be a path.")
    root = (base / root).resolve()
    result["project_root"] = root
    for name in ("k562", "rpe1"):
        context = result.get(name)
        if not isinstance(context, dict):
            raise ValueError(f"Required context '{name}' must be a mapping.")
        for key in ("responses", "metadata", "fitness", "gene_column", "fitness_column", "assay"):
            if not isinstance(context.get(key), (str, Path)) or not str(context[key]).strip():
                raise ValueError(f"Missing or invalid {name}.{key}")
        for key in ("responses", "metadata", "fitness"):
            context[key] = (root / context[key]).resolve()
        context.setdefault("metadata_gene_column", "gene")
        context.setdefault("cell_count_column", "n_cells")
        for key in ("metadata_gene_column", "cell_count_column"):
            if not isinstance(context[key], str) or not context[key]:
                raise ValueError(f"{name}.{key} must be a nonempty column name.")
        provenance = context.setdefault("provenance_files", [])
        if not isinstance(provenance, list) or any(not isinstance(p, (str, Path)) for p in provenance):
            raise ValueError(f"{name}.provenance_files must be a list of paths.")
        context["provenance_files"] = [(root / p).resolve() for p in provenance]
    analysis = result.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a mapping.")
    defaults = {"pca_components": 100, "cv_folds": 5, "random_seed": 42,
                "stability_seeds": [11, 22, 33, 44, 55], "expected_features": 7226}
    for key, value in defaults.items():
        analysis.setdefault(key, value)
    for key in ("pca_components", "cv_folds", "random_seed", "expected_features"):
        minimum = 0 if key == "random_seed" else (2 if key == "cv_folds" else 1)
        if type(analysis[key]) is not int or analysis[key] < minimum:
            raise ValueError(f"analysis.{key} must be an integer >= {minimum}.")
    seeds = analysis["stability_seeds"]
    if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("analysis.stability_seeds must be a list of nonnegative integers.")
    output = result.get("output", {})
    if (not isinstance(output, dict)
            or not isinstance(output.get("directory"), (str, Path))
            or not str(output["directory"]).strip()):
        raise ValueError("output.directory must be a path.")
    output["directory"] = (root / output["directory"]).resolve()
    result["output"] = output
    return result


def load_context_outputs(config: Mapping[str, Any]) -> ContextData:
    """Read NPZ and CSV schemas from context pipelines or the original notebooks.

    Cell counts are joined from metadata by target identity. When NPZ counts
    exist, they must agree with that join. Pickled arrays are never accepted.
    """
    with np.load(config["responses"], allow_pickle=False) as archive:
        missing = {"X", "targets", "genes"} - set(archive.files)
        if missing:
            raise ValueError(f"Response NPZ is missing keys: {sorted(missing)}")
        X, targets, genes = archive["X"], archive["targets"], archive["genes"]
        saved_counts = archive["n_cells"] if "n_cells" in archive.files else None
    targets = unique_identifiers(targets, "NPZ targets")
    metadata = load_table(config["metadata"])
    gene, count = config["metadata_gene_column"], config["cell_count_column"]
    require_columns(metadata, [gene, count], table_name="response metadata")
    unique_identifiers(metadata[gene].to_numpy(), "metadata genes")
    metadata = metadata.set_index(gene)
    missing = set(targets) - set(metadata.index)
    if missing:
        raise ValueError(f"Response metadata missing target genes: {sorted(missing)[:10]}")
    counts = metadata.loc[targets, count].to_numpy()
    if saved_counts is not None and not np.array_equal(saved_counts, counts):
        raise ValueError("NPZ and gene-aligned metadata cell counts disagree.")
    fitness = load_table(config["fitness"])
    gene, value = config["gene_column"], config["fitness_column"]
    require_columns(fitness, [gene, value], table_name="gene fitness")
    return ContextData(X, targets, genes, counts, fitness.set_index(gene)[value])


def prepare_selectivity(config: str | Path | Mapping[str, Any]) -> PreparedSelectivity:
    """Align saved outputs and build endpoints without running any model."""
    config = load_selectivity_config(config)
    k562, rpe1 = (load_context_outputs(config[name]) for name in ("k562", "rpe1"))
    aligned = align_contexts(k562, rpe1)
    n_features = len(aligned.k562.genes)
    if n_features != config["analysis"]["expected_features"]:
        raise ValueError(f"Aligned expression feature count is {n_features}, expected "
                         f"{config['analysis']['expected_features']}.")
    endpoints = build_selectivity_endpoints(aligned)
    summary = {"n_common_targets": len(endpoints), "n_features": n_features,
               "n_positive_selectivity": int((endpoints.functional_selectivity > 0).sum()),
               "n_negative_selectivity": int((endpoints.functional_selectivity < 0).sum()),
               "n_zero_selectivity": int((endpoints.functional_selectivity == 0).sum()),
               "input_k562_response_targets": len(k562.targets),
               "input_rpe1_response_targets": len(rpe1.targets),
               "input_k562_fitness_genes": len(k562.fitness),
               "input_rpe1_fitness_genes": len(rpe1.fitness),
               "n_masked_targets": len(set(aligned.k562.targets) & set(aligned.k562.genes)),
               "target_order": "lexicographically sorted four-way intersection",
               "feature_order": "K562 NPZ order; RPE1 reordered by gene identity",
               "primary_endpoint": "rank_pct(-k562_raw_fitness) - rank_pct(rpe1_raw_fitness)",
               "rank_method": "average", "rank_universe": "final common targets only",
               "secondary_endpoint": "z(-k562_raw_fitness) - z(rpe1_raw_fitness)",
               "zscore_ddof": 0,
               "positive_selectivity": "stronger relative dependency in K562",
               "assay_caveat": "CRISPRi gamma and CRISPR-KO BAGEL are distinct assays; rank contrast is not an absolute effect-size difference.",
               "response_definition": "context perturbation mean minus context control mean",
               "stability_input": "target-masked differential RNA; primary rank endpoint",
               "permutation_inputs": "unmasked and target-masked differential RNA; identical shuffled primary rank endpoint"}
    return PreparedSelectivity(config, aligned, endpoints, summary)


def _evaluate_models(prepared: PreparedSelectivity, delta: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark all RNA inputs on identical aligned targets, endpoint and folds."""
    options, endpoints = prepared.config["analysis"], prepared.endpoints
    y = endpoints.functional_selectivity.to_numpy()
    cv = make_cv(options["cv_folds"], options["random_seed"])
    specifications = [
        ("Mean baseline", "pred_mean", "baseline", make_mean_model(), np.zeros((len(y), 1))),
        ("Cell count only (both contexts)", "pred_cell_counts", "baseline", make_scalar_ridge_model(), cell_count_features(prepared.aligned)),
        ("RNA magnitude difference", "pred_magnitude_difference", "baseline", make_scalar_ridge_model(), endpoints.magnitude_difference.to_numpy()[:, None]),
        (K562_ONLY, "pred_k562_only", "ablation", make_pca_ridge_model(options["pca_components"], options["random_seed"]), prepared.aligned.k562.X),
        (RPE1_ONLY, "pred_rpe1_only", "ablation", make_pca_ridge_model(options["pca_components"], options["random_seed"]), prepared.aligned.rpe1.X),
        (DIFFERENTIAL, "pred_differential", "primary", make_pca_ridge_model(options["pca_components"], options["random_seed"]), delta),
        (JOINT, "pred_joint", "secondary", make_pca_ridge_model(options["pca_components"], options["random_seed"]), joint_responses(prepared.aligned)),
    ]
    rows, predictions = [], endpoints[["gene", "functional_selectivity", "z_selectivity"]].copy()
    for name, column, role, model, X in specifications:
        prediction = cross_validated_predictions(model, X, y, cv, n_jobs=1)
        predictions[column] = prediction
        rows.append({"endpoint": "functional_selectivity", "role": role,
                     **evaluate_predictions(name, y, prediction)})
    return pd.DataFrame(rows), predictions


def _robustness(prepared: PreparedSelectivity, delta: np.ndarray,
                predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Masked-rank stability, both differential permutations, and z-endpoint fit."""
    options, endpoints = prepared.config["analysis"], prepared.endpoints
    model = make_pca_ridge_model(options["pca_components"], options["random_seed"])
    cv = make_cv(options["cv_folds"], options["random_seed"])
    y, z = endpoints.functional_selectivity.to_numpy(), endpoints.z_selectivity.to_numpy()
    masked = differential_responses(prepared.aligned, masked=True)
    predictions["pred_differential_masked"] = cross_validated_predictions(model, masked, y, cv)
    predictions["pred_differential_z"] = cross_validated_predictions(model, delta, z, cv)
    rows = [
        {"endpoint": "functional_selectivity", **evaluate_predictions(DIFFERENTIAL, y, predictions.pred_differential)},
        {"endpoint": "functional_selectivity", **evaluate_predictions(MASKED, y, predictions.pred_differential_masked)},
        {"endpoint": "z_selectivity", **evaluate_predictions(DIFFERENTIAL, z, predictions.pred_differential_z)},
    ]
    stability = cv_stability(masked, y, options["stability_seeds"], model, n_splits=options["cv_folds"])
    stability["model"] = MASKED
    stability["endpoint"] = "functional_selectivity"
    permutation = permutation_control(model, masked, y, cv, seed=options["random_seed"])
    predictions["shuffled_functional_selectivity"] = permutation.predictions.shuffled_fitness.to_numpy()
    predictions["pred_shuffled"] = permutation.predictions.pred_shuffled.to_numpy()
    permutation.metrics["model"] = MASKED + " (shuffled labels)"
    permutation.metrics["endpoint"] = "functional_selectivity"
    unmasked_permutation = permutation_control(model, delta, y, cv, seed=options["random_seed"])
    if not np.array_equal(permutation.predictions.shuffled_fitness,
                          unmasked_permutation.predictions.shuffled_fitness):
        raise RuntimeError("Differential permutation controls must use identical shuffled labels.")
    predictions["pred_shuffled_differential"] = unmasked_permutation.predictions.pred_shuffled.to_numpy()
    unmasked_permutation.metrics["model"] = "Differential RNA (shuffled selectivity)"
    unmasked_permutation.metrics["endpoint"] = "functional_selectivity"
    associations = []
    for column in ("magnitude_difference", "cell_count_difference", "log_cell_count_difference", "z_selectivity"):
        associations.append({"association": f"{column}_vs_functional_selectivity",
                             **spearman_association(endpoints[column], y)})
    return {"selectivity_robustness_metrics": pd.DataFrame(rows),
            "selectivity_cv_stability": stability,
            "selectivity_permutation_metrics": pd.concat(
                [permutation.metrics, unmasked_permutation.metrics], ignore_index=True),
            "selectivity_associations": pd.DataFrame(associations)}


def _source_provenance(config: dict) -> dict:
    """Fingerprint the exact saved artifacts and preserve optional source metadata."""
    provenance = {}
    for name in ("k562", "rpe1"):
        context = config[name]
        paths = [context[key] for key in ("responses", "metadata", "fitness")]
        paths += context["provenance_files"]
        entries = []
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            entry = {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
            if path.suffix == ".json":
                entry["source_metadata"] = json.loads(path.read_text(encoding="utf-8"))
            entries.append(entry)
        provenance[name] = {"assay": context["assay"], "files": entries}
    return provenance


def run_selectivity_pipeline(config: str | Path | Mapping[str, Any]) -> SelectivityResult:
    """Run OOF selectivity analysis and save outputs without modifying context results."""
    prepared = prepare_selectivity(config)
    provenance = _source_provenance(prepared.config)
    delta = differential_responses(prepared.aligned)
    metrics, predictions = _evaluate_models(prepared, delta)
    tables = _robustness(prepared, delta, predictions)
    tables.update({"common_targets": prepared.endpoints[["gene"]].copy(),
                   "functional_selectivity": prepared.endpoints,
                   "selectivity_model_metrics": metrics, "selectivity_predictions": predictions})
    tables["selectivity_context_ablation_metrics"] = (
        metrics.set_index("model").loc[list(CONTEXT_ABLATIONS)].reset_index())
    output = prepared.config["output"]["directory"]
    # Never allow a configured output filename to overwrite a source artifact.
    destinations = {output / f"{name}.csv" for name in tables}
    destinations |= {output / name for name in ("differential_responses.npz", "selectivity_summary.json", "resolved_config.json")}
    sources = {Path(entry["path"]) for source in provenance.values() for entry in source["files"]}
    if sources & destinations:
        raise ValueError("Selectivity outputs must not overwrite context input artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "differential_responses.npz", X_delta=delta,
                        targets=prepared.aligned.k562.targets, genes=prepared.aligned.k562.genes)
    summary = {**prepared.summary, "created_utc": datetime.now(timezone.utc).isoformat(),
               "context_ablation": {"models": list(CONTEXT_ABLATIONS),
                                    "endpoint": "functional_selectivity",
                                    "same_targets_and_folds": True,
                                    "preprocessing_and_hyperparameters": "identical configured PCA + Ridge pipeline"},
               "response_dtype": str(delta.dtype), "provenance": provenance,
               "library_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}}
    for filename, content in (("selectivity_summary.json", summary), ("resolved_config.json", prepared.config)):
        (output / filename).write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    return SelectivityResult(prepared, tables, output)


def main() -> None:
    """Thin CLI used by Make; audit prepares endpoints without fitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit", "check-config"))
    parser.add_argument("--config", default="configs/selectivity.yaml")
    args = parser.parse_args()
    if args.action == "check-config":
        print(json.dumps(load_selectivity_config(args.config), indent=2, default=str))
    elif args.action == "audit":
        print(json.dumps(prepare_selectivity(args.config).summary, indent=2))
    else:
        result = run_selectivity_pipeline(args.config)
        print("Common targets:", result.prepared.summary["n_common_targets"])
        print(result.tables["selectivity_model_metrics"].to_string(index=False))
        print("Outputs:", result.output_directory)


if __name__ == "__main__":
    main()
