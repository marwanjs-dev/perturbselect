"""Small, real backed-h5ad end-to-end checks; no project data required."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbselect.pipeline import run_pipeline


@pytest.fixture
def synthetic_config(tmp_path):
    targets = [f"g{i}" for i in range(12)]
    rng = np.random.default_rng(18)
    profiles = rng.normal(size=(12, 4))
    matrix = np.vstack([np.zeros((3, 4)), np.repeat(profiles, 2, axis=0)])
    data = ad.AnnData(sparse.csr_matrix(matrix),
        obs=pd.DataFrame({"label": ["ctrl"] * 3 + list(np.repeat(targets, 2))},
                         index=[f"cell{i}" for i in range(27)]),
        var=pd.DataFrame(index=["g2", "g0", "other", "g1"]))
    data.write_h5ad(tmp_path / "expression.h5ad")
    # Raw row order deliberately differs from requested targets.
    pd.DataFrame({"target": targets, "r1": profiles[:, 0], "r2": profiles[:, 0]}).iloc[::-1].to_csv(
        tmp_path / "fitness.csv", index=False)
    pd.DataFrame({"gene": targets[::2] + targets[1::2]}).to_csv(tmp_path / "targets.csv", index=False)
    return {"project_root": str(tmp_path), "dataset": {"name": "synthetic"},
        "expression": {"path": "expression.h5ad", "perturbation_column": "label", "control_label": "ctrl"},
        "fitness": {"path": "fitness.csv", "gene_column": "target", "replicate_columns": ["r1", "r2"],
                    "output_column": "effect"},
        "targets": {"path": "targets.csv", "column": "gene"}, "features": {"mode": "all"},
        "analysis": {"pca_components": 2, "cv_folds": 3, "chunk_size": 5,
                     "stability_seeds": [11], "cell_count_thresholds": [2, 3]},
        "output": {"directory": "output"}}


def test_end_to_end_alignment_and_saved_outputs(synthetic_config, tmp_path):
    # A gene with an incomplete replicate pair counts in raw input only.
    raw = pd.read_csv(tmp_path / "fitness.csv")
    incomplete = pd.DataFrame({"target": ["incomplete"], "r1": [1.0], "r2": [np.nan]})
    pd.concat([raw, incomplete], ignore_index=True).to_csv(tmp_path / "fitness.csv", index=False)
    result = run_pipeline(synthetic_config)
    expected = pd.read_csv(tmp_path / "targets.csv").gene.tolist()
    assert result.responses.targets.tolist() == expected
    assert result.tables["gene_fitness"].target.tolist() == expected
    assert result.tables["predictions"].gene.tolist() == expected
    # phenotype is the mean response of g2 for this synthetic assay.
    np.testing.assert_allclose(result.tables["predictions"].observed_fitness, result.responses.X[:, 0])
    assert result.summary["n_controls"] == 3
    assert result.summary["raw_fitness_rows"] == 13
    assert result.summary["raw_unique_genes"] == 13
    assert result.summary["retained_construct_rows"] == 12
    assert result.summary["retained_construct_unique_genes"] == 12
    assert result.summary["n_masked_targets"] == 3
    for name, table in result.tables.items():
        saved = pd.read_csv(result.output_directory / f"{name}.csv")
        assert saved.shape == table.shape
    with np.load(result.output_directory / "measured_responses.npz", allow_pickle=False) as archive:
        assert archive["targets"].tolist() == expected
        assert archive["X"].dtype == np.float32
    summary = json.loads((result.output_directory / "dataset_summary.json").read_text())
    assert summary["n_features"] == 4


@pytest.mark.parametrize("mode", ["list", "intersection"])
def test_feature_selection_and_order(synthetic_config, tmp_path, mode):
    if mode == "list":
        pd.DataFrame({"gene": ["g1", "g2"]}).to_csv(tmp_path / "features.csv", index=False)
        synthetic_config["features"] = {"mode": mode, "path": "features.csv"}
        expected = ["g1", "g2"]
    else:
        other = ad.AnnData(np.zeros((1, 3)), var=pd.DataFrame(index=["g1", "g2", "absent"]))
        other.write_h5ad(tmp_path / "other.h5ad")
        synthetic_config["features"] = {"mode": mode, "path": "other.h5ad"}
        expected = ["g2", "g1"]
    assert run_pipeline(synthetic_config).responses.genes.tolist() == expected


def test_handles_closed_on_failure(synthetic_config, tmp_path, monkeypatch):
    import perturbselect.pipeline as pipeline
    opened = []
    original = pipeline.load_expression_data

    def capture(*args, **kwargs):
        data = original(*args, **kwargs)
        opened.append(data)
        return data

    monkeypatch.setattr(pipeline, "load_expression_data", capture)
    pd.DataFrame({"gene": ["missing"]}).to_csv(tmp_path / "targets.csv", index=False)
    with pytest.raises(ValueError, match="lack expression or fitness"):
        run_pipeline(synthetic_config)
    assert opened and all(not data.file.is_open for data in opened)
    assert not (tmp_path / "output").exists()


def test_default_target_intersection(synthetic_config):
    del synthetic_config["targets"]
    result = run_pipeline(synthetic_config)
    assert result.responses.targets.tolist() == sorted(result.responses.targets)
