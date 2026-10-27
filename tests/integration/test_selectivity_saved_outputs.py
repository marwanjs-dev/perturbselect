"""Opt-in experiment on saved context outputs, with no single-cell aggregation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perturbselect.selectivity_pipeline import (MASKED, load_selectivity_config,
                                                run_selectivity_pipeline)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def selectivity_result(tmp_path_factory):
    config = load_selectivity_config(ROOT / "configs/selectivity.yaml")
    for name in ("k562", "rpe1"):
        paths = [config[name][key] for key in ("responses", "metadata", "fitness")]
        for path in paths + config[name]["provenance_files"]:
            if not path.is_file():
                pytest.skip(f"Saved context artifact unavailable: {path}")
    config["output"]["directory"] = tmp_path_factory.mktemp("selectivity")
    return run_selectivity_pipeline(config)


def test_saved_context_alignment_and_universe(selectivity_result):
    prepared = selectivity_result.prepared
    k, r = prepared.aligned.k562, prepared.aligned.rpe1
    # Independently intersect the four source identity lists, without a fixed count.
    target_sets = []
    for name in ("k562", "rpe1"):
        source = prepared.config[name]
        with np.load(source["responses"], allow_pickle=False) as archive:
            target_sets.append(set(archive["targets"].tolist()))
        target_sets.append(set(pd.read_csv(source["fitness"])[source["gene_column"]]))
    expected = sorted(set.intersection(*target_sets))
    assert len(expected) > 100
    assert prepared.summary["n_common_targets"] == len(expected)
    assert k.targets.tolist() == r.targets.tolist() == expected
    assert k.genes.tolist() == r.genes.tolist()
    assert k.X.shape == r.X.shape == (len(expected), 7226)
    assert prepared.endpoints.gene.tolist() == expected
    assert k.fitness.index.tolist() == r.fitness.index.tolist() == expected
    assert np.isfinite(k.X).all() and np.isfinite(r.X).all()
    assert np.isfinite(prepared.endpoints.drop(columns="gene").to_numpy()).all()
    endpoint = prepared.endpoints.functional_selectivity
    assert endpoint.min() < 0 < endpoint.max()


def test_real_signal_exceeds_permutation(selectivity_result):
    # Scientific acceptance criterion, not a claim of already established success.
    # Investigate a failure; do not weaken this criterion to manufacture a result.
    real = selectivity_result.tables["selectivity_robustness_metrics"].set_index("model").loc[MASKED]
    shuffled = selectivity_result.tables["selectivity_permutation_metrics"].iloc[0]
    assert abs(shuffled.spearman_rho) < .15, "Permutation correlation is not near zero."
    assert real.spearman_rho > abs(shuffled.spearman_rho) + .10, "Selectivity signal has not clearly exceeded permutation."
    assert real.r2 > shuffled.r2, "Real-label R2 did not exceed the permutation control."
    for row in selectivity_result.tables["selectivity_permutation_metrics"].itertuples():
        assert abs(row.spearman_rho) < .15, f"{row.model}: permutation correlation is not near zero."
        assert row.r2 <= 0, f"{row.model}: shuffled selectivity has positive R2."
