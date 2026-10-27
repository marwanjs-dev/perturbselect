"""Input/output utilities for PerturbSelect.

This module handles loading expression datasets and tabular phenotype data.
Dataset-specific biological processing should live in other modules.
"""

from pathlib import Path
from typing import Iterable, Literal

import anndata as ad
import pandas as pd
import scanpy as sc


BackedMode = Literal["r", "r+"] | None


def _require_file(path: str | Path) -> Path:
    """Return a Path after verifying that the file exists.

    Parameters
    ----------
    path
        File path to validate.

    Returns
    -------
    pathlib.Path
        Validated path.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )

    return path


def load_expression_data(
    path: str | Path,
    *,
    backed: BackedMode = "r",
) -> ad.AnnData:
    """Load an AnnData expression dataset.

    Parameters
    ----------
    path
        Path to an .h5ad file.

    backed
        AnnData backed mode.

        ``"r"`` keeps the expression matrix on disk and reads data
        when needed. ``None`` loads the entire dataset into memory.

    Returns
    -------
    anndata.AnnData
        Loaded expression dataset.
    """
    path = _require_file(path)

    if path.suffix.lower() != ".h5ad":
        raise ValueError(
            f"Expression dataset must be an .h5ad file: {path}"
        )

    adata = sc.read_h5ad(
        path,
        backed=backed,
    )

    if adata.n_obs == 0:
        close_expression_data(adata)
        raise ValueError(
            f"Expression dataset contains no cells: {path}"
        )

    if adata.n_vars == 0:
        close_expression_data(adata)
        raise ValueError(
            f"Expression dataset contains no genes: {path}"
        )

    return adata


def validate_expression_schema(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
) -> dict[str, int]:
    """Validate the minimum expression-data structure PerturbSelect needs.

    Parameters
    ----------
    adata
        Expression dataset.

    perturbation_column
        Column in ``adata.obs`` containing perturbation identities.

    control_label
        Value identifying control cells.

    Returns
    -------
    dict
        Basic dataset statistics.
    """
    if perturbation_column not in adata.obs.columns:
        raise KeyError(
            f"Missing perturbation column "
            f"'{perturbation_column}'. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )

    if adata.var_names.has_duplicates:
        raise ValueError(
            "Expression feature names contain duplicates."
        )

    labels = (
        adata.obs[perturbation_column]
        .astype(str)
    )

    unique_labels = set(labels.unique())

    if control_label not in unique_labels:
        raise ValueError(
            f"Control label '{control_label}' was not found in "
            f"column '{perturbation_column}'."
        )

    n_controls = int(
        (labels == control_label).sum()
    )

    n_perturbations = (
        len(unique_labels) - 1
    )

    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_controls": n_controls,
        "n_perturbations": n_perturbations,
    }


def load_table(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load a CSV, TSV or Excel table.

    Parameters
    ----------
    path
        Input table.

    sheet_name
        Excel worksheet name/index. Ignored for CSV/TSV files.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    path = _require_file(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)

    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(
            path,
            sep="\t",
        )

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

    else:
        raise ValueError(
            "Unsupported table format "
            f"'{suffix}'. Expected CSV, TSV, TXT, XLSX or XLS."
        )

    if df.empty:
        raise ValueError(
            f"Loaded table is empty: {path}"
        )

    return df


def require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    table_name: str = "table",
) -> None:
    """Check that a DataFrame contains all required columns.

    Parameters
    ----------
    df
        Table to inspect.

    required_columns
        Columns required by the next analysis step.

    table_name
        Human-readable name used in error messages.
    """
    required_columns = list(required_columns)

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{table_name} is missing required columns: "
            f"{missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def close_expression_data(
    adata: ad.AnnData,
) -> None:
    """Close the backing HDF5 file if AnnData is disk-backed."""
    if adata.isbacked:
        adata.file.close()
