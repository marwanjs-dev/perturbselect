"""Mean perturbation minus control profiles from already normalized expression."""
from collections.abc import Sequence
from dataclasses import dataclass

import anndata as ad
import numpy as np
from numpy.typing import NDArray

from .io import validate_expression_schema


@dataclass
class ResponseMatrix:
    """Responses and their explicit row/column identities."""
    X: NDArray
    targets: NDArray
    genes: NDArray
    n_cells: NDArray
    n_controls: int


def build_response_matrix(adata: ad.AnnData, targets: Sequence[str],
                          genes: Sequence[str], *, perturbation_column: str,
                          control_label: str, chunk_size: int = 5000) -> ResponseMatrix:
    """Accumulate contiguous disk chunks without normalizing or densifying all cells."""
    validate_expression_schema(adata, perturbation_column=perturbation_column,
                               control_label=control_label)
    targets, genes = np.asarray(targets, dtype=str), np.asarray(genes, dtype=str)
    for name, values in (("targets", targets), ("genes", genes)):
        if values.ndim != 1 or not len(values) or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty unique list.")
    if control_label in targets:
        raise ValueError("Targets must not include the control label.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    indices = adata.var_names.get_indexer(genes)
    if (indices < 0).any():
        raise ValueError("Requested expression features are missing from the dataset.")
    groups = [control_label, *targets]
    lookup = {group: i for i, group in enumerate(groups)}
    sums = np.zeros((len(groups), len(genes)), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.int64)
    labels = adata.obs[perturbation_column].astype(str).to_numpy()
    for start in range(0, adata.n_obs, chunk_size):
        block = adata.X[start:start + chunk_size][:, indices]
        block_labels = labels[start:start + chunk_size]
        relevant = np.isin(block_labels, groups)
        block, block_labels = block[relevant], block_labels[relevant]
        for group in np.unique(block_labels):
            mask = block_labels == group
            sums[lookup[group]] += np.asarray(block[mask].sum(axis=0)).ravel()
            counts[lookup[group]] += mask.sum()
    if (counts == 0).any():
        raise ValueError("Every requested target and control must have cells.")
    means = sums / counts[:, None]
    X = means[1:] - means[0]
    if not np.isfinite(X).all():
        raise ValueError("Response matrix contains nonfinite expression values.")
    return ResponseMatrix(X, targets, genes, counts[1:], int(counts[0]))


def mask_target_expression(X: NDArray, targets: Sequence[str],
                           genes: Sequence[str]) -> tuple[NDArray, int]:
    """Zero each target's own feature in a copy; leave absent targets unchanged."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape != (len(targets), len(genes)):
        raise ValueError("X dimensions must match targets and genes.")
    if len(set(targets)) != len(targets) or len(set(genes)) != len(genes):
        raise ValueError("Target and feature identifiers must be unique.")
    result = X.copy()
    lookup = {gene: i for i, gene in enumerate(genes)}
    count = 0
    for row, target in enumerate(targets):
        if target in lookup:
            result[row, lookup[target]] = 0
            count += 1
    return result, count
