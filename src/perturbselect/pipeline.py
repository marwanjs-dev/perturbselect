"""Config-driven measured-expression to independent functional-fitness analysis."""
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .fitness import aggregate_replicates, collapse_constructs_to_genes, filter_fitness_to_targets
from .io import (close_expression_data, load_expression_data, load_table,
                 require_columns, validate_expression_schema)
from .modeling import (cv_stability, evaluate_target_masked, make_cv,
                       make_pca_ridge_model, permutation_control, run_benchmarks)
from .qc import (adjust_response_magnitude_for_cell_count, cell_count_qc,
                 cell_count_sensitivity, response_magnitude, spearman_association)
from .signatures import ResponseMatrix, build_response_matrix, mask_target_expression


@dataclass
class PipelineResult:
    """Numerical outputs and the directory to which they were saved."""
    summary: dict[str, Any]
    responses: ResponseMatrix
    tables: dict[str, pd.DataFrame]
    output_directory: Path


def _identifiers(path: Path, column: str) -> list[str]:
    """Read an ordered unique identifier column without silently dropping rows."""
    table = load_table(path)
    require_columns(table, [column], table_name=str(path))
    if table[column].isna().any():
        raise ValueError(f"Missing identifiers in {path}: {column}")
    values = table[column].astype(str).tolist()
    if len(set(values)) != len(values) or any(not v.strip() for v in values):
        raise ValueError(f"Identifiers must be nonempty and unique in {path}: {column}")
    return values


def _features(adata: Any, config: dict) -> list[str]:
    """Resolve a transparent feature selection, retaining specified feature order."""
    if config["mode"] == "all":
        return adata.var_names.astype(str).tolist()
    if config["mode"] == "list":
        return _identifiers(config["path"], config.get("column", "gene"))
    other = load_expression_data(config["path"], backed="r")
    try:
        if other.var_names.has_duplicates:
            raise ValueError("Secondary expression feature names contain duplicates.")
        available = set(other.var_names)
        return [gene for gene in adata.var_names if gene in available]
    finally:
        close_expression_data(other)


def run_pipeline(config: str | Path | Mapping[str, Any]) -> PipelineResult:
    """Run and save the complete analysis without changing phenotype signs.

    Input expression must already be normalized. Only perturbation means are
    modeled. QC uses float64 accumulated responses; modeling uses float32, as in
    the validated notebooks. Existing files in the configured output are replaced.
    """
    config = load_config(config)
    expression, fitness_cfg = config["expression"], config["fitness"]
    analysis = config["analysis"]
    adata = load_expression_data(expression["path"], backed="r")
    try:
        summary = validate_expression_schema(
            adata, perturbation_column=expression["perturbation_column"],
            control_label=expression["control_label"])
        raw = load_table(fitness_cfg["path"], sheet_name=fitness_cfg.get("sheet_name", 0))
        gene_column, value_column = fitness_cfg["gene_column"], fitness_cfg["output_column"]
        require_columns(raw, [gene_column], table_name="raw fitness")
        constructs = aggregate_replicates(
            raw, fitness_cfg["replicate_columns"], output_column=value_column,
            aggregation=fitness_cfg["replicate_aggregation"],
            require_complete=fitness_cfg["require_complete_replicates"])
        fitness = collapse_constructs_to_genes(
            constructs, gene_column=gene_column, value_column=value_column,
            construct_column=fitness_cfg.get("construct_column"),
            aggregation=fitness_cfg["construct_aggregation"],
            excluded_gene_labels=fitness_cfg["excluded_gene_labels"])
        available = set(adata.obs[expression["perturbation_column"]].astype(str))
        available.discard(expression["control_label"])
        available &= set(fitness[gene_column])
        target_cfg = config["targets"]
        if "path" in target_cfg:
            targets = _identifiers(target_cfg["path"], target_cfg.get("column", "gene"))
            missing = set(targets) - available
            if missing:
                raise ValueError(f"Requested targets lack expression or fitness: {sorted(missing)[:10]}")
        else:
            targets = sorted(available)
        fitness = filter_fitness_to_targets(fitness, targets, gene_column=gene_column)
        responses = build_response_matrix(
            adata, targets, _features(adata, config["features"]),
            perturbation_column=expression["perturbation_column"],
            control_label=expression["control_label"], chunk_size=analysis["chunk_size"])
        if fitness[gene_column].tolist() != responses.targets.tolist():
            raise RuntimeError("Fitness and response target identities/order do not match.")
        y = fitness[value_column].to_numpy()
        summary.update({"dataset": config["dataset"]["name"], "raw_fitness_rows": len(raw),
                        "retained_construct_rows": len(constructs),
                        "raw_unique_genes": int(raw[gene_column].nunique()),
                        # After replicate filtering, before excluded labels or target intersection.
                        "retained_construct_unique_genes": int(constructs[gene_column].nunique()),
                        "n_targets": len(targets), "n_features": len(responses.genes)})
    finally:
        close_expression_data(adata)

    magnitude = response_magnitude(responses.X)
    residual = adjust_response_magnitude_for_cell_count(responses.n_cells, magnitude)
    associations = cell_count_qc(responses.n_cells, magnitude, y)
    associations["adjusted_rna_magnitude_vs_fitness"] = spearman_association(residual, y)
    X = responses.X.astype(np.float32)
    benchmark = run_benchmarks(X, y, responses.n_cells,
                               n_components=analysis["pca_components"],
                               n_splits=analysis["cv_folds"], random_state=analysis["random_seed"])
    model = make_pca_ridge_model(analysis["pca_components"], analysis["random_seed"])
    cv = make_cv(analysis["cv_folds"], analysis["random_seed"])
    masked_result = evaluate_target_masked(X, y, responses.targets, responses.genes, model, cv)
    masked, n_masked = mask_target_expression(X, responses.targets, responses.genes)
    stability = cv_stability(masked, y, analysis["stability_seeds"], model,
                             n_splits=analysis["cv_folds"])
    permutation = permutation_control(model, masked, y, cv, seed=analysis["random_seed"])
    predictions = pd.DataFrame({"gene": targets, "observed_fitness": y,
                                "n_cells": responses.n_cells})
    predictions = pd.concat([predictions, benchmark.predictions,
                             masked_result.predictions, permutation.predictions], axis=1)
    tables = {
        "gene_fitness": fitness,
        "response_metadata": pd.DataFrame({"gene": targets, "n_cells": responses.n_cells,
                                            "response_magnitude": magnitude,
                                            "adjusted_response_magnitude": residual}),
        "qc_associations": pd.DataFrame([{"association": key, **value}
                                         for key, value in associations.items()]),
        "qc_cell_count_sensitivity": cell_count_sensitivity(
            responses.n_cells, magnitude, y, analysis["cell_count_thresholds"]),
        "model_metrics": benchmark.metrics, "predictions": predictions,
        "target_masked_metrics": masked_result.metrics, "cv_stability": stability,
        "permutation_metrics": permutation.metrics,
    }
    summary.update({"n_masked_targets": n_masked,
                    "mean_absolute_response": float(magnitude.mean()),
                    "min_perturbation_cells": int(responses.n_cells.min()),
                    "median_perturbation_cells": float(np.median(responses.n_cells)),
                    "max_perturbation_cells": int(responses.n_cells.max()),
                    "response_dtype": "float64", "modeling_dtype": "float32",
                    "library_versions": {name: version(name) for name in
                                         ("numpy", "pandas", "scipy", "scikit-learn", "anndata")},
                    "phenotype_semantics": fitness_cfg.get("semantics", "Raw values; no sign or scale transformation."),
                    "feature_selection": config["features"], "target_selection": config["targets"]})
    output = config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "measured_responses.npz", X=X,
                        targets=responses.targets, genes=responses.genes,
                        n_cells=responses.n_cells, n_controls=responses.n_controls)
    for name, content in (("dataset_summary", summary), ("resolved_config", config)):
        (output / f"{name}.json").write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    return PipelineResult(summary, responses, tables, output)
