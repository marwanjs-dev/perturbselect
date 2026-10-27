"""Opt-in full K562 regression: expensive, uses local data and temporary outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perturbselect.config import load_config
from perturbselect.pipeline import run_pipeline

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def k562_result(tmp_path_factory):
    config = load_config(ROOT / "configs/k562.yaml")
    for section in ("expression", "fitness", "targets", "features"):
        if not config[section]["path"].is_file():
            pytest.skip(f"Local K562 regression input missing: {config[section]['path']}")
    config["output"]["directory"] = tmp_path_factory.mktemp("k562_pipeline")
    return run_pipeline(config)


def test_k562_dimensions_and_response_reference(k562_result):
    result = k562_result
    for key, value in {"n_cells": 308646, "n_genes": 8563, "n_controls": 10691,
                       "n_perturbations": 1971, "n_targets": 1641,
                       "raw_fitness_rows": 21378, "retained_construct_rows": 21168,
                       "raw_unique_genes": 18902,
                       "retained_construct_unique_genes": 18728,
                       "n_masked_targets": 1415}.items():
        assert result.summary[key] == value, f"Unexpected {key}"
    assert result.responses.X.shape == (1641, 7226)
    assert np.isfinite(result.responses.X).all()
    counts = result.responses.n_cells
    assert (counts.min(), np.median(counts), counts.max()) == (30, 130, 992)
    assert np.abs(result.responses.X).mean() == pytest.approx(.03653514, abs=1e-7)
    reference = pd.read_csv(ROOT / "results/phase0/k562_gene_fitness.csv")
    assert result.responses.targets.tolist() == reference.gene.tolist()
    np.testing.assert_allclose(result.tables["gene_fitness"].fitness, reference.k562_gamma,
                               rtol=1e-10, atol=1e-12)
    phenotype = result.tables["gene_fitness"].fitness
    np.testing.assert_allclose(
        [phenotype.mean(), phenotype.std(), phenotype.min(), phenotype.median(), phenotype.max()],
        [-.251596, .210301, -.913132, -.232802, .454688], atol=1e-6, rtol=0)


def test_k562_qc_reference(k562_result):
    associations = k562_result.tables["qc_associations"].set_index("association")
    assert associations.loc["cell_count_vs_rna_magnitude", "rho"] == pytest.approx(-.381310, abs=1e-5)
    assert associations.loc["rna_magnitude_vs_fitness", "rho"] == pytest.approx(-.543233, abs=1e-5)
    assert associations.loc["adjusted_rna_magnitude_vs_fitness", "rho"] == pytest.approx(-.623, abs=.001)
    sensitivity = k562_result.tables["qc_cell_count_sensitivity"]
    assert sensitivity.n_perturbations.tolist() == [1641, 1545, 1367, 1145, 669]
    np.testing.assert_allclose(sensitivity.rna_magnitude_vs_fitness_rho,
        [-.543233, -.575644, -.610841, -.630919, -.668968], atol=1e-5)


def test_k562_predictive_reference(k562_result):
    columns = ["spearman_rho", "pearson_r", "r2", "mae", "rmse"]
    reference = pd.read_csv(ROOT / "results/phase0/k562_fitness_model_metrics.csv").set_index("model")
    observed = k562_result.tables["model_metrics"].set_index("model")
    # Allow small BLAS/sklearn differences, not changes in the scientific result.
    np.testing.assert_allclose(observed.loc[reference.index, columns], reference[columns], atol=.003, rtol=0)
    masked = k562_result.tables["target_masked_metrics"].iloc[0]
    assert masked.spearman_rho == pytest.approx(.699898, abs=.003)
    assert masked.r2 == pytest.approx(.407594, abs=.003)
    shuffled = k562_result.tables["permutation_metrics"].iloc[0]
    assert shuffled.spearman_rho == pytest.approx(.043326, abs=.005)
    assert shuffled.r2 == pytest.approx(-.010791, abs=.005)
    stability = k562_result.tables["cv_stability"]
    np.testing.assert_allclose(stability.spearman_rho,
                               [.707211, .705562, .704298, .700096, .702103], atol=.003, rtol=0)
