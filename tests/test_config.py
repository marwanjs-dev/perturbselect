from copy import deepcopy

import pytest
import yaml

from perturbselect.config import load_config


@pytest.fixture
def minimal_config(tmp_path):
    return {"project_root": str(tmp_path), "dataset": {"name": "synthetic"},
            "expression": {"path": "expression.h5ad", "perturbation_column": "label", "control_label": "ctrl"},
            "fitness": {"path": "fitness.csv", "gene_column": "target", "replicate_columns": ["r1", "r2"]},
            "features": {"mode": "all"}, "output": {"directory": "results"}}


def test_yaml_paths_relative_to_config_and_explicit_root(tmp_path, minimal_config):
    directory = tmp_path / "configs"
    directory.mkdir()
    minimal_config["project_root"] = ".."
    path = directory / "test.yaml"
    path.write_text(yaml.safe_dump(minimal_config), encoding="utf-8")
    result = load_config(path)
    assert result["expression"]["path"] == tmp_path / "expression.h5ad"
    assert result["analysis"]["pca_components"] == 100
    del minimal_config["project_root"]
    path.write_text(yaml.safe_dump(minimal_config), encoding="utf-8")
    assert load_config(path)["expression"]["path"] == directory / "expression.h5ad"


def test_dictionary_is_copied_and_resolution_is_idempotent(minimal_config):
    before = deepcopy(minimal_config)
    result = load_config(minimal_config)
    assert minimal_config == before
    assert load_config(result) == result


@pytest.mark.parametrize("section,key,value,match", [
    ("features", "mode", "implicit", "features.mode"),
    ("fitness", "replicate_columns", "r1", "replicate_columns"),
    ("fitness", "require_complete_replicates", "false", "boolean"),
    ("fitness", "construct_aggregation", "sum", "construct_aggregation"),
    ("analysis", "cv_folds", 1, "cv_folds"),
])
def test_validation(minimal_config, section, key, value, match):
    minimal_config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=match):
        load_config(minimal_config)


def test_missing_sections_and_feature_source(tmp_path, minimal_config):
    path = tmp_path / "empty.yaml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
    minimal_config["features"] = {"mode": "intersection"}
    with pytest.raises(ValueError, match="features.path"):
        load_config(minimal_config)
    del minimal_config["fitness"]
    with pytest.raises(ValueError, match="fitness"):
        load_config(minimal_config)
