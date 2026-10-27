"""Functional phenotype processing for PerturbSelect.

This module converts perturbation-screen measurements into a standardized
gene-level fitness table.

It does not perform statistical quality control or predictive modeling.
Those belong in qc.py and modeling.py.
"""

from collections.abc import Iterable, Sequence
from typing import Literal

import pandas as pd

from .io import require_columns


Aggregation = Literal["mean", "median"]


def aggregate_replicates(
    df: pd.DataFrame,
    replicate_columns: Sequence[str],
    *,
    output_column: str = "fitness",
    aggregation: Aggregation = "mean",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Combine replicate phenotype measurements for each construct.

    Parameters
    ----------
    df
        Raw functional-screen table.

    replicate_columns
        Columns containing replicate phenotype measurements.

    output_column
        Name of the new aggregated phenotype column.

    aggregation
        How replicate measurements are combined.

        ``"mean"`` averages the replicate values.
        ``"median"`` takes their median.

    require_complete
        If True, retain only rows where every requested replicate
        has a non-missing value.

        If False, retain rows with at least one available replicate.

    Returns
    -------
    pandas.DataFrame
        Copy of the input containing the aggregated phenotype column.
    """
    replicate_columns = list(replicate_columns)

    if len(replicate_columns) == 0:
        raise ValueError(
            "At least one replicate column is required."
        )

    require_columns(
        df,
        replicate_columns,
        table_name="fitness table",
    )

    if aggregation not in {"mean", "median"}:
        raise ValueError(
            "aggregation must be 'mean' or 'median'."
        )

    result = df.copy()

    replicate_values = result[
        replicate_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if require_complete:
        keep = replicate_values.notna().all(axis=1)
    else:
        keep = replicate_values.notna().any(axis=1)

    result = result.loc[keep].copy()
    replicate_values = replicate_values.loc[keep]

    if aggregation == "mean":
        result[output_column] = (
            replicate_values.mean(axis=1)
        )
    else:
        result[output_column] = (
            replicate_values.median(axis=1)
        )

    return result


def collapse_constructs_to_genes(
    df: pd.DataFrame,
    *,
    gene_column: str,
    value_column: str = "fitness",
    construct_column: str | None = None,
    aggregation: Aggregation = "median",
    excluded_gene_labels: Iterable[str] = (),
) -> pd.DataFrame:
    """Collapse construct-level phenotypes into one value per gene.

    Parameters
    ----------
    df
        Construct-level phenotype table.

    gene_column
        Column containing target-gene identifiers.

    value_column
        Construct-level phenotype to aggregate.

    construct_column
        Optional column identifying individual perturbation constructs.

        When supplied, the returned table also contains ``n_constructs``.

    aggregation
        How multiple constructs targeting the same gene are combined.

    excluded_gene_labels
        Gene labels to remove before aggregation, for example
        ``"non-targeting"``.

    Returns
    -------
    pandas.DataFrame
        Gene-level phenotype table containing one row per gene.
    """
    required = [
        gene_column,
        value_column,
    ]

    if construct_column is not None:
        required.append(construct_column)

    require_columns(
        df,
        required,
        table_name="construct-level fitness table",
    )

    if aggregation not in {"mean", "median"}:
        raise ValueError(
            "aggregation must be 'mean' or 'median'."
        )

    result = df.dropna(
        subset=[
            gene_column,
            value_column,
        ]
    ).copy()

    result[gene_column] = (
        result[gene_column]
        .astype(str)
        .str.strip()
    )

    excluded_gene_labels = {
        str(label)
        for label in excluded_gene_labels
    }

    if excluded_gene_labels:
        result = result[
            ~result[gene_column].isin(
                excluded_gene_labels
            )
        ].copy()

    grouped = result.groupby(
        gene_column,
        sort=True,
    )

    if aggregation == "median":
        gene_values = grouped[
            value_column
        ].median()
    else:
        gene_values = grouped[
            value_column
        ].mean()

    gene_level = (
        gene_values
        .rename(value_column)
        .reset_index()
    )

    if construct_column is not None:
        construct_counts = (
            grouped[construct_column]
            .nunique()
            .rename("n_constructs")
            .reset_index()
        )

        gene_level = gene_level.merge(
            construct_counts,
            on=gene_column,
            how="left",
            validate="one_to_one",
        )

    if gene_level[gene_column].duplicated().any():
        raise RuntimeError(
            "Gene-level fitness table unexpectedly "
            "contains duplicate genes."
        )

    return gene_level


def build_gene_fitness(
    df: pd.DataFrame,
    *,
    gene_column: str,
    replicate_columns: Sequence[str],
    construct_column: str | None = None,
    replicate_aggregation: Aggregation = "mean",
    construct_aggregation: Aggregation = "median",
    output_column: str = "fitness",
    excluded_gene_labels: Iterable[str] = (),
    require_complete_replicates: bool = True,
) -> pd.DataFrame:
    """Convert a raw perturbation screen into gene-level fitness values.

    This is a convenience function combining:

    1. replicate aggregation;
    2. optional removal of controls;
    3. construct-to-gene aggregation.

    Returns
    -------
    pandas.DataFrame
        Standardized table containing one phenotype value per gene.
    """
    construct_level = aggregate_replicates(
        df,
        replicate_columns,
        output_column=output_column,
        aggregation=replicate_aggregation,
        require_complete=require_complete_replicates,
    )

    gene_level = collapse_constructs_to_genes(
        construct_level,
        gene_column=gene_column,
        value_column=output_column,
        construct_column=construct_column,
        aggregation=construct_aggregation,
        excluded_gene_labels=excluded_gene_labels,
    )

    return gene_level


def filter_fitness_to_targets(
    fitness: pd.DataFrame,
    targets: Iterable[str],
    *,
    gene_column: str = "gene",
) -> pd.DataFrame:
    """Restrict gene-level fitness values to a target set.

    The ordering of ``targets`` is preserved.

    Parameters
    ----------
    fitness
        Gene-level phenotype table.

    targets
        Target genes to retain.

    gene_column
        Gene identifier column.

    Returns
    -------
    pandas.DataFrame
        Fitness rows matching the supplied targets.
    """
    require_columns(
        fitness,
        [gene_column],
        table_name="gene-level fitness table",
    )

    targets = [
        str(target)
        for target in targets
    ]

    if fitness[gene_column].duplicated().any():
        raise ValueError(
            "Fitness table must contain one row per gene "
            "before target filtering."
        )

    indexed = fitness.set_index(
        gene_column
    )

    available_targets = [
        target
        for target in targets
        if target in indexed.index
    ]

    result = (
        indexed
        .loc[available_targets]
        .reset_index()
    )

    return result