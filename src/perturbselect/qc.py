"""Perturbation-level QC; undefined correlations are reported as NaN."""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def finite_vector(values: ArrayLike, name: str) -> NDArray:
    """Validate a finite one-dimensional numerical vector without changing dtype."""
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional numeric vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite observations.")
    return array


def response_magnitude(X: ArrayLike) -> NDArray:
    """Return mean absolute response per perturbation, preserving input precision."""
    X = np.asarray(X)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must be a nonempty 2D response matrix.")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("X must contain only finite numeric responses.")
    return np.mean(np.abs(X), axis=1)


def spearman_association(x: ArrayLike, y: ArrayLike) -> dict[str, float | int]:
    """Return rho, p_value and n; fewer than two or constant values give NaN."""
    x, y = finite_vector(x, "x"), finite_vector(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have equal lengths.")
    rho = p = float("nan")
    if len(x) >= 2 and np.ptp(x) != 0 and np.ptp(y) != 0:
        rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p), "n": len(x)}


def validate_cell_counts(n_cells: ArrayLike, n: int) -> NDArray:
    """Require aligned, nonnegative cell counts."""
    counts = finite_vector(n_cells, "n_cells")
    if len(counts) != n or (counts < 0).any():
        raise ValueError("n_cells must be nonnegative and match the number of rows.")
    return counts


def cell_count_qc(n_cells: ArrayLike, response_magnitude: ArrayLike,
                  fitness: ArrayLike) -> dict[str, dict]:
    """Calculate the three pairwise cell-count, magnitude and fitness associations."""
    magnitude = finite_vector(response_magnitude, "response_magnitude")
    counts = validate_cell_counts(n_cells, len(magnitude))
    return {
        "cell_count_vs_rna_magnitude": spearman_association(counts, magnitude),
        "cell_count_vs_fitness": spearman_association(counts, fitness),
        "rna_magnitude_vs_fitness": spearman_association(magnitude, fitness),
    }


def cell_count_sensitivity(n_cells: ArrayLike, response_magnitude: ArrayLike,
                           fitness: ArrayLike,
                           thresholds: Sequence[int] = (30, 50, 75, 100, 150)) -> pd.DataFrame:
    """Calculate associations after each inclusive minimum-cell filter."""
    cell_count_qc(n_cells, response_magnitude, fitness)
    counts, magnitude, y = map(np.asarray, (n_cells, response_magnitude, fitness))
    rows = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Cell-count thresholds must be finite and nonnegative.")
        keep = counts >= threshold
        mag = spearman_association(magnitude[keep], y[keep])
        cells = spearman_association(counts[keep], magnitude[keep])
        rows.append({"min_cells": threshold, "n_perturbations": int(keep.sum()),
                     "rna_magnitude_vs_fitness_rho": mag["rho"],
                     "rna_magnitude_vs_fitness_p": mag["p_value"],
                     "cell_count_vs_rna_magnitude_rho": cells["rho"],
                     "cell_count_vs_rna_magnitude_p": cells["p_value"]})
    return pd.DataFrame(rows, columns=[
        "min_cells", "n_perturbations", "rna_magnitude_vs_fitness_rho",
        "rna_magnitude_vs_fitness_p", "cell_count_vs_rna_magnitude_rho",
        "cell_count_vs_rna_magnitude_p",
    ])


def adjust_response_magnitude_for_cell_count(n_cells: ArrayLike,
                                             response_magnitude: ArrayLike) -> NDArray:
    """Residualize magnitude against log1p cell count with an intercept (QC only)."""
    magnitude = finite_vector(response_magnitude, "response_magnitude")
    counts = validate_cell_counts(n_cells, len(magnitude))
    if len(magnitude) < 2:
        raise ValueError("Residualization requires at least two observations.")
    log_cells = np.log1p(counts).reshape(-1, 1)
    model = LinearRegression().fit(log_cells, magnitude)
    return magnitude - model.predict(log_cells)
