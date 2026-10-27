"""Small YAML configuration loader; relative paths use an explicit project root."""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a configuration, resolving paths without requiring data yet.

    YAML project_root is relative to its file; dictionary project_root is relative
    to the current directory. Other paths are relative to project_root. With no
    project_root, use the YAML's directory (or the dictionary caller's cwd).
    """
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        base = path.parent
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a YAML mapping.")
    result = deepcopy(dict(config))
    required = {
        "dataset": ("name",),
        "expression": ("path", "perturbation_column", "control_label"),
        "fitness": ("path", "gene_column", "replicate_columns"),
        "features": ("mode",), "output": ("directory",),
    }
    for section, keys in required.items():
        if not isinstance(result.get(section), dict):
            raise ValueError(f"Required section '{section}' must be a mapping.")
        for key in keys:
            if key not in result[section] or result[section][key] in (None, "", []):
                raise ValueError(f"Missing required configuration key: {section}.{key}")
    root_value = result.get("project_root", base)
    if not isinstance(root_value, (str, Path)):
        raise ValueError("project_root must be a path.")
    root = (base / root_value).resolve()
    result["project_root"] = root
    defaults = {"chunk_size": 5000, "pca_components": 100, "cv_folds": 5,
                "random_seed": 42, "stability_seeds": [11, 22, 33, 44, 55],
                "cell_count_thresholds": [30, 50, 75, 100, 150]}
    analysis = result.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a mapping.")
    for key, value in defaults.items():
        analysis.setdefault(key, value)
    for key in ("chunk_size", "pca_components", "cv_folds", "random_seed"):
        minimum = 2 if key == "cv_folds" else (0 if key == "random_seed" else 1)
        if type(analysis[key]) is not int or analysis[key] < minimum:
            raise ValueError(f"analysis.{key} must be an integer >= {minimum}.")
    for key in ("stability_seeds", "cell_count_thresholds"):
        if not isinstance(analysis[key], list) or any(type(v) is not int or v < 0 for v in analysis[key]):
            raise ValueError(f"analysis.{key} must be a list of nonnegative integers.")
    fitness = result["fitness"]
    columns = fitness["replicate_columns"]
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("fitness.replicate_columns must be a nonempty list of column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("fitness.replicate_columns must be unique.")
    for key, default in (("replicate_aggregation", "mean"), ("construct_aggregation", "median")):
        fitness.setdefault(key, default)
        if fitness[key] not in ("mean", "median"):
            raise ValueError(f"fitness.{key} must be mean or median.")
    fitness.setdefault("require_complete_replicates", True)
    if type(fitness["require_complete_replicates"]) is not bool:
        raise ValueError("fitness.require_complete_replicates must be a boolean.")
    fitness.setdefault("excluded_gene_labels", [])
    if not isinstance(fitness["excluded_gene_labels"], list):
        raise ValueError("fitness.excluded_gene_labels must be a list.")
    fitness.setdefault("output_column", "fitness")
    for section in ("features", "targets"):
        if isinstance(result.get(section), dict) and "column" in result[section]:
            value = result[section]["column"]
            if not isinstance(value, str) or not value:
                raise ValueError(f"{section}.column must be a nonempty column name.")
    if fitness.get("construct_column") is not None:
        if not isinstance(fitness["construct_column"], str) or not fitness["construct_column"]:
            raise ValueError("fitness.construct_column must be a nonempty column name or null.")
    if fitness["output_column"] in (fitness["gene_column"], fitness.get("construct_column")):
        raise ValueError("fitness.output_column must not overwrite gene or construct identifiers.")
    features = result["features"]
    if features["mode"] not in ("all", "list", "intersection"):
        raise ValueError("features.mode must be all, list or intersection.")
    if features["mode"] != "all" and not features.get("path"):
        raise ValueError("features.path is required for list or intersection mode.")
    targets = result.setdefault("targets", {})
    if not isinstance(targets, dict):
        raise ValueError("targets must be a mapping.")
    for section, key in (("dataset", "name"), ("expression", "perturbation_column"),
                         ("expression", "control_label"), ("fitness", "gene_column"),
                         ("fitness", "output_column")):
        if not isinstance(result[section][key], str) or not result[section][key]:
            raise ValueError(f"{section}.{key} must be a nonempty string.")
    for section, key in (("expression", "path"), ("fitness", "path"), ("output", "directory"),
                         ("features", "path"), ("targets", "path")):
        if key in result[section]:
            value = result[section][key]
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"{section}.{key} must be a nonempty path.")
            result[section][key] = (root / value).resolve()
    return result
