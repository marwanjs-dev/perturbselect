from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perturbselect.fitness import (
    aggregate_replicates,
    build_gene_fitness,
    collapse_constructs_to_genes,
    filter_fitness_to_targets,
)
from perturbselect.io import load_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]

K562_FITNESS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "k562_crispri_growth.csv"
)

K562_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "phase0"
    / "k562_gene_fitness.csv"
)


def test_aggregate_replicates_mean():
    df = pd.DataFrame({
        "gene": ["A", "B", "C"],
        "rep1": [-0.2, -0.4, np.nan],
        "rep2": [-0.4, -0.6, -0.8],
    })

    result = aggregate_replicates(
        df,
        ["rep1", "rep2"],
        output_column="fitness",
        aggregation="mean",
        require_complete=True,
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result["fitness"].to_numpy(),
        [-0.3, -0.5],
    )


def test_collapse_constructs_uses_median():
    df = pd.DataFrame({
        "gene": [
            "A",
            "A",
            "A",
            "B",
        ],
        "sgID": [
            "A1",
            "A2",
            "A3",
            "B1",
        ],
        "fitness": [
            -0.2,
            -0.4,
            -2.0,
            -0.5,
        ],
    })

    result = collapse_constructs_to_genes(
        df,
        gene_column="gene",
        value_column="fitness",
        construct_column="sgID",
        aggregation="median",
    )

    gene_a = result.loc[
        result["gene"] == "A"
    ].iloc[0]

    assert gene_a["fitness"] == -0.4
    assert gene_a["n_constructs"] == 3


def test_excluded_control_gene_is_removed():
    df = pd.DataFrame({
        "gene": [
            "A",
            "non-targeting",
        ],
        "sgID": [
            "A1",
            "control1",
        ],
        "fitness": [
            -0.4,
            0.0,
        ],
    })

    result = collapse_constructs_to_genes(
        df,
        gene_column="gene",
        value_column="fitness",
        construct_column="sgID",
        excluded_gene_labels=[
            "non-targeting",
        ],
    )

    assert (
        "non-targeting"
        not in result["gene"].tolist()
    )


def test_filter_fitness_preserves_target_order():
    fitness = pd.DataFrame({
        "gene": [
            "A",
            "B",
            "C",
        ],
        "fitness": [
            -0.1,
            -0.2,
            -0.3,
        ],
    })

    targets = [
        "C",
        "A",
        "NOT_PRESENT",
    ]

    result = filter_fitness_to_targets(
        fitness,
        targets,
    )

    assert result["gene"].tolist() == [
        "C",
        "A",
    ]


@pytest.mark.integration
def test_k562_complete_dual_guide_count():
    if not K562_FITNESS.is_file():
        pytest.skip("Local K562 fitness data are unavailable.")
    raw = load_table(
        K562_FITNESS
    )

    construct_level = aggregate_replicates(
        raw,
        [
            "dual_rep1_gamma",
            "dual_rep2_gamma",
        ],
        output_column="fitness",
        aggregation="mean",
        require_complete=True,
    )

    assert len(construct_level) == 21168


@pytest.mark.integration
def test_k562_gene_fitness_matches_reference():
    if not K562_FITNESS.is_file() or not K562_REFERENCE.is_file():
        pytest.skip("Local K562 fitness/reference data are unavailable.")
    raw = load_table(
        K562_FITNESS
    )

    gene_level = build_gene_fitness(
        raw,
        gene_column="gene",
        replicate_columns=[
            "dual_rep1_gamma",
            "dual_rep2_gamma",
        ],
        construct_column="sgID",
        replicate_aggregation="mean",
        construct_aggregation="median",
        output_column="fitness",
        excluded_gene_labels=[
            "non-targeting",
        ],
        require_complete_replicates=True,
    )

    reference = pd.read_csv(
        K562_REFERENCE
    )

    expected = (
        reference
        .set_index("gene")
        ["k562_gamma"]
    )

    observed = (
        gene_level
        .set_index("gene")
        .loc[expected.index, "fitness"]
    )

    assert len(observed) == 1641

    np.testing.assert_allclose(
        observed.to_numpy(),
        expected.to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
