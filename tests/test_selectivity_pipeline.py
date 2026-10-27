"""Saved-output orchestration using small NPZ/CSV fixtures, never H5AD."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from perturbselect.selectivity_pipeline import (load_selectivity_config,
    prepare_selectivity, run_selectivity_pipeline)


@pytest.fixture
def saved_config(tmp_path):
    rng = np.random.default_rng(92)
    targets = np.array([f"g{i:02}" for i in range(24)])
    genes = targets[:4]
    config = {"project_root": str(tmp_path),
              "analysis": {"pca_components": 3, "cv_folds": 3, "expected_features": 4,
                           "stability_seeds": [11, 22]}, "output": {"directory": "out"}}
    for name in ("k562", "rpe1"):
        folder = tmp_path / name
        folder.mkdir()
        matrix = rng.normal(size=(24, 4)).astype(np.float32)
        order, columns = rng.permutation(24), rng.permutation(4)
        counts = np.arange(24) + 30
        np.savez_compressed(folder / "measured_responses.npz", X=matrix[np.ix_(order, columns)],
                            targets=targets[order], genes=genes[columns], n_cells=counts[order])
        pd.DataFrame({"gene": targets[::-1], "n_cells": counts[::-1]}).to_csv(folder / "response_metadata.csv", index=False)
        pd.DataFrame({"symbol": targets, "score": matrix[:, 0]}).to_csv(folder / "gene_fitness.csv", index=False)
        config[name] = {"responses": f"{name}/measured_responses.npz",
                        "metadata": f"{name}/response_metadata.csv", "fitness": f"{name}/gene_fitness.csv",
                        "gene_column": "symbol", "fitness_column": "score", "assay": f"synthetic {name}"}
    return config


def test_config_paths_and_read_only_preparation(saved_config, tmp_path):
    original = deepcopy(saved_config)
    path = tmp_path / "selectivity.yaml"
    path.write_text(yaml.safe_dump(saved_config), encoding="utf-8")
    loaded = load_selectivity_config(path)
    assert load_selectivity_config(loaded) == loaded
    prepared = prepare_selectivity(path)
    assert prepared.summary["n_common_targets"] == 24
    assert prepared.endpoints.gene.tolist() == prepared.aligned.k562.targets.tolist()
    assert saved_config == original
    assert not (tmp_path / "out").exists()


def test_end_to_end_saved_outputs_and_provenance(saved_config, tmp_path):
    result = run_selectivity_pipeline(saved_config)
    predictions = result.tables["selectivity_predictions"]
    assert len(predictions) == 24
    assert np.isfinite(predictions.drop(columns="gene").to_numpy()).all()
    assert len(result.tables["selectivity_model_metrics"]) == 7
    assert result.tables["selectivity_context_ablation_metrics"].model.tolist() == [
        "K562 RNA only", "RPE1 RNA only", "Differential RNA (K562 - RPE1)", "Joint RNA (K562 + RPE1)"]
    assert {"pred_k562_only", "pred_rpe1_only", "pred_shuffled_differential"} <= set(predictions.columns)
    assert len(result.tables["selectivity_permutation_metrics"]) == 2
    assert result.tables["selectivity_cv_stability"].seed.tolist() == [11, 22]
    assert set(result.tables["selectivity_robustness_metrics"].endpoint) == {"functional_selectivity", "z_selectivity"}
    for name, table in result.tables.items():
        assert pd.read_csv(result.output_directory / f"{name}.csv").shape == table.shape
    summary = json.loads((result.output_directory / "selectivity_summary.json").read_text())
    assert summary["n_common_targets"] == 24
    assert summary["zscore_ddof"] == 0
    assert len(summary["provenance"]["k562"]["files"][0]["sha256"]) == 64
    with np.load(result.output_directory / "differential_responses.npz", allow_pickle=False) as archive:
        assert archive["targets"].tolist() == predictions.gene.tolist()
        np.testing.assert_array_equal(archive["X_delta"], result.prepared.aligned.k562.X - result.prepared.aligned.rpe1.X)


def test_legacy_npz_without_counts_uses_named_metadata(saved_config, tmp_path):
    path = tmp_path / "k562/measured_responses.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("X", "targets", "genes")}
    np.savez_compressed(path, **values)
    prepared = prepare_selectivity(saved_config)
    np.testing.assert_array_equal(prepared.aligned.k562.n_cells, np.arange(24) + 30)


def test_inconsistent_counts_fail(saved_config, tmp_path):
    path = tmp_path / "rpe1/response_metadata.csv"
    table = pd.read_csv(path)
    table.loc[0, "n_cells"] += 1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cell counts disagree"):
        prepare_selectivity(saved_config)


def test_missing_metadata_genes_fail(saved_config, tmp_path):
    path = tmp_path / "rpe1/response_metadata.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="metadata missing target genes"):
        prepare_selectivity(saved_config)


def test_partial_fitness_intersects_before_ranking(saved_config, tmp_path):
    path = tmp_path / "rpe1/gene_fitness.csv"
    pd.read_csv(path).iloc[:10].to_csv(path, index=False)
    prepared = prepare_selectivity(saved_config)
    assert prepared.summary["n_common_targets"] == 10
    expected = pd.Series(prepared.endpoints.k562_dependency).rank(method="average", pct=True)
    np.testing.assert_array_equal(prepared.endpoints.k562_dependency_percentile, expected)


def test_config_and_feature_count_errors(saved_config):
    saved_config["analysis"]["expected_features"] = 7226
    with pytest.raises(ValueError, match="feature count"):
        prepare_selectivity(saved_config)
    del saved_config["k562"]["fitness_column"]
    with pytest.raises(ValueError, match="k562.fitness_column"):
        load_selectivity_config(saved_config)


def test_context_ablations_share_endpoint_folds_and_estimator_settings(saved_config, monkeypatch):
    """Inspect actual model inputs and folds, including both single-context ablations."""
    import perturbselect.selectivity_pipeline as module
    prepared = prepare_selectivity(saved_config)
    delta = module.differential_responses(prepared.aligned)
    calls = []

    def capture(model, X, y, cv, n_jobs=1):
        if hasattr(model, "named_steps") and "pca" in model.named_steps:
            calls.append((model, np.asarray(X).copy(), y.copy(), list(cv.split(X, y)), n_jobs))
        return np.linspace(-.2, .2, len(y))

    monkeypatch.setattr(module, "cross_validated_predictions", capture)
    _, predictions = module._evaluate_models(prepared, delta)
    assert len(calls) == 4
    expected_inputs = [prepared.aligned.k562.X, prepared.aligned.rpe1.X,
                       delta, module.joint_responses(prepared.aligned)]
    reference_params = calls[0][0].get_params()
    for (model, X, y, splits, n_jobs), expected in zip(calls, expected_inputs):
        np.testing.assert_array_equal(X, expected)
        np.testing.assert_array_equal(y, prepared.endpoints.functional_selectivity)
        assert n_jobs == 1
        params = model.get_params()
        for key in ("pca__n_components", "pca__svd_solver", "pca__random_state",
                    "scale__with_mean", "scale__with_std"):
            assert params[key] == reference_params[key]
        np.testing.assert_array_equal(params["ridge__alphas"], reference_params["ridge__alphas"])
        for (train, test), (expected_train, expected_test) in zip(splits, calls[0][3]):
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(test, expected_test)
    assert predictions.gene.tolist() == prepared.aligned.k562.targets.tolist()
