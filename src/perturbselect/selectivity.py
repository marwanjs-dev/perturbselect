"""Identity-aligned cross-context RNA and functional-selectivity endpoints."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .qc import finite_vector, response_magnitude, validate_cell_counts
from .signatures import mask_target_expression


def unique_identifiers(values: Sequence[str], name: str) -> NDArray:
    """Require nonempty, unique string identifiers, without case/name conversion."""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a nonempty one-dimensional identifier list.")
    if any(not isinstance(v, str) or not v.strip() for v in array):
        raise ValueError(f"{name} must contain nonempty gene-name strings.")
    if len(set(array)) != len(array):
        raise ValueError(f"{name} contains duplicate identifiers.")
    return array.astype(str)


@dataclass
class ContextData:
    """Saved responses with named axes, per-row cell counts and gene-indexed fitness."""
    X: NDArray
    targets: NDArray
    genes: NDArray
    n_cells: NDArray
    fitness: pd.Series

    def __post_init__(self) -> None:
        self.targets = unique_identifiers(self.targets, "targets")
        self.genes = unique_identifiers(self.genes, "genes")
        self.X = np.asarray(self.X)
        response_magnitude(self.X)
        if self.X.shape != (len(self.targets), len(self.genes)):
            raise ValueError("Response dimensions must match the named target and feature axes.")
        self.n_cells = validate_cell_counts(self.n_cells, len(self.targets))
        if (self.n_cells <= 0).any() or (self.n_cells != np.floor(self.n_cells)).any():
            raise ValueError("Response cell counts must be positive integers.")
        unique_identifiers(self.fitness.index.to_numpy(), "fitness genes")
        finite_vector(self.fitness.to_numpy(), "fitness")


@dataclass
class AlignedContexts:
    """Both contexts explicitly reindexed to identical target and feature axes."""
    k562: ContextData
    rpe1: ContextData

    def __post_init__(self) -> None:
        for attribute in ("targets", "genes"):
            if not np.array_equal(getattr(self.k562, attribute), getattr(self.rpe1, attribute)):
                raise ValueError(f"Aligned context {attribute} must have identical identities and order.")
        for context in (self.k562, self.rpe1):
            if not np.array_equal(context.targets, context.fitness.index.to_numpy()):
                raise ValueError("Aligned fitness rows must match response target identities and order.")


def find_common_targets(k562_targets: Sequence[str], rpe1_targets: Sequence[str],
                        k562_fitness_genes: Sequence[str],
                        rpe1_fitness_genes: Sequence[str]) -> NDArray:
    """Return the sorted four-way intersection; never assume a common-universe size."""
    lists = (k562_targets, rpe1_targets, k562_fitness_genes, rpe1_fitness_genes)
    sets = [set(unique_identifiers(values, "target universe")) for values in lists]
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError("No common targets have RNA and functional phenotypes in both contexts.")
    return np.asarray(common, dtype=str)


def _reindex_context(context: ContextData, targets: NDArray, genes: NDArray) -> ContextData:
    """Select rows/columns by identity and bind fitness to the selected target order."""
    rows = pd.Index(context.targets).get_indexer(targets)
    columns = pd.Index(context.genes).get_indexer(genes)
    if (rows < 0).any() or (columns < 0).any():
        raise ValueError("Requested targets or expression genes are missing from a context.")
    return ContextData(context.X[np.ix_(rows, columns)], context.targets[rows],
                       context.genes[columns], context.n_cells[rows],
                       context.fitness.loc[targets].copy())


def align_contexts(k562: ContextData, rpe1: ContextData) -> AlignedContexts:
    """Align all four target sources and require identical feature sets.

    Feature order follows K562's saved order. Unequal feature universes raise an
    error instead of silently shrinking the intended shared-expression space.
    """
    if set(k562.genes) != set(rpe1.genes):
        raise ValueError("Expression gene sets differ: missing genes in RPE1 "
                         f"{sorted(set(k562.genes) - set(rpe1.genes))[:10]}; missing genes in K562 "
                         f"{sorted(set(rpe1.genes) - set(k562.genes))[:10]}.")
    targets = find_common_targets(k562.targets, rpe1.targets,
                                  k562.fitness.index, rpe1.fitness.index)
    return AlignedContexts(_reindex_context(k562, targets, k562.genes),
                           _reindex_context(rpe1, targets, k562.genes))


def orient_dependency(values: ArrayLike, *, higher_is_dependency: bool) -> NDArray:
    """Orient raw phenotype values without rescaling: higher means dependency."""
    values = finite_vector(values, "raw fitness").astype(np.float64)
    return values.copy() if higher_is_dependency else -values


def dependency_percentile(values: ArrayLike) -> NDArray:
    """Average-tie percentile ranks over the supplied final common universe."""
    values = finite_vector(values, "dependency")
    if len(values) < 2:
        raise ValueError("Dependency transformations require at least two common targets.")
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def dependency_zscore(values: ArrayLike) -> NDArray:
    """Population z-score (ddof=0) over common targets; reject constant phenotypes."""
    values = finite_vector(values, "dependency").astype(np.float64)
    if len(values) < 2 or np.std(values, ddof=0) == 0:
        raise ValueError("Z-scores require at least two common targets and nonconstant dependency.")
    return (values - values.mean()) / values.std(ddof=0)


def build_selectivity_endpoints(aligned: AlignedContexts) -> pd.DataFrame:
    """Primary rank(-gamma) minus rank(BAGEL), plus secondary z-score difference."""
    k, r = aligned.k562, aligned.rpe1
    table = pd.DataFrame({"gene": k.targets, "k562_raw_fitness": k.fitness.to_numpy(),
                          "rpe1_raw_fitness": r.fitness.to_numpy()})
    for prefix, context, higher in (("k562", k, False), ("rpe1", r, True)):
        dependency = orient_dependency(context.fitness.to_numpy(), higher_is_dependency=higher)
        table[f"{prefix}_dependency"] = dependency
        table[f"{prefix}_dependency_percentile"] = dependency_percentile(dependency)
        table[f"{prefix}_dependency_z"] = dependency_zscore(dependency)
        table[f"{prefix}_n_cells"] = context.n_cells
        table[f"{prefix}_response_magnitude"] = response_magnitude(context.X)
    table["functional_selectivity"] = table.k562_dependency_percentile - table.rpe1_dependency_percentile
    table["z_selectivity"] = table.k562_dependency_z - table.rpe1_dependency_z
    table["magnitude_difference"] = table.k562_response_magnitude - table.rpe1_response_magnitude
    table["cell_count_difference"] = k.n_cells.astype(float) - r.n_cells
    table["log_cell_count_difference"] = np.log1p(k.n_cells) - np.log1p(r.n_cells)
    return table


def differential_responses(aligned: AlignedContexts, *, masked: bool = False) -> NDArray:
    """K562 minus RPE1 control-relative responses; optionally mask BOTH contexts."""
    k, r = aligned.k562, aligned.rpe1
    if masked:
        Xk, _ = mask_target_expression(k.X, k.targets, k.genes)
        Xr, _ = mask_target_expression(r.X, r.targets, r.genes)
        return Xk - Xr
    return k.X - r.X


def joint_responses(aligned: AlignedContexts) -> NDArray:
    """Concatenate aligned K562 then RPE1 features without subtracting them."""
    return np.concatenate([aligned.k562.X, aligned.rpe1.X], axis=1)


def cell_count_features(aligned: AlignedContexts) -> NDArray:
    """Two technical predictors: log1p cell counts in each context."""
    return np.column_stack([np.log1p(aligned.k562.n_cells), np.log1p(aligned.rpe1.n_cells)])
