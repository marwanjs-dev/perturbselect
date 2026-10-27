from pathlib import Path

import pytest
import pandas as pd

from perturbselect.io import (
    close_expression_data,
    load_expression_data,
    load_table,
    require_columns,
    validate_expression_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

K562_EXPRESSION = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "replogle22k562.h5ad"
)

K562_FITNESS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "k562_crispri_growth.csv"
)


@pytest.fixture(scope="module")
def k562():
    """Load the K562 AnnData once for this test module."""
    if not K562_EXPRESSION.is_file():
        pytest.skip("Local K562 expression data are unavailable.")
    adata = load_expression_data(
        K562_EXPRESSION,
        backed="r",
    )

    try:
        yield adata
    finally:
        close_expression_data(adata)


@pytest.mark.integration
def test_load_expression_data(k562):
    assert k562.n_obs == 308646
    assert k562.n_vars == 8563
    assert k562.isbacked


@pytest.mark.integration
def test_validate_expression_schema(k562):
    summary = validate_expression_schema(
        k562,
        perturbation_column="perturbation",
        control_label="control",
    )

    assert summary["n_cells"] == 308646
    assert summary["n_genes"] == 8563
    assert summary["n_controls"] == 10691
    assert summary["n_perturbations"] == 1971


@pytest.mark.integration
def test_load_k562_fitness():
    if not K562_FITNESS.is_file():
        pytest.skip("Local K562 fitness data are unavailable.")
    fitness = load_table(
        K562_FITNESS
    )

    assert fitness.shape == (21378, 14)


@pytest.mark.integration
def test_k562_fitness_required_columns():
    if not K562_FITNESS.is_file():
        pytest.skip("Local K562 fitness data are unavailable.")
    fitness = load_table(
        K562_FITNESS
    )

    require_columns(
        fitness,
        [
            "gene",
            "dual_rep1_gamma",
            "dual_rep2_gamma",
        ],
        table_name="K562 fitness",
    )


def test_missing_file_raises_error():
    fake_path = (
        PROJECT_ROOT
        / "data"
        / "this_file_does_not_exist.csv"
    )

    with pytest.raises(FileNotFoundError):
        load_table(fake_path)


def test_missing_required_column_raises_error():
    fitness = pd.DataFrame({"gene": ["A"]})

    with pytest.raises(KeyError):
        require_columns(
            fitness,
            ["definitely_not_a_real_column"],
            table_name="K562 fitness",
        )
