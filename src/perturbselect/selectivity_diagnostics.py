"""Lightweight diagnostics of fixed saved OOF predictions; no model fitting."""
import argparse
from collections.abc import Callable
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import pearsonr

from .qc import finite_vector, spearman_association


def _metric(name: str, y: np.ndarray, prediction: np.ndarray) -> float:
    """Undefined correlations or constant-outcome R2 return NaN, never zero."""
    if name == "spearman_rho":
        return float(spearman_association(y, prediction)["rho"])
    if name == "r2":
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - np.sum((y - prediction) ** 2) / total) if total > 0 else float("nan")
    raise ValueError("metric must be spearman_rho or r2.")


def paired_bootstrap_metric_difference(
    y: ArrayLike, candidate: ArrayLike, baseline: ArrayLike, *,
    metric: str = "spearman_rho", n_bootstrap: int = 5000, random_seed: int = 42,
) -> dict[str, float | int]:
    """Resample genes jointly and summarize candidate-minus-baseline metric deltas.

    Predictions remain fixed. Percentile intervals describe uncertainty over
    sampled genes conditional on the existing OOF fits, not retraining uncertainty.
    Degenerate replicates are omitted per metric and their count is reported.
    """
    arrays = [finite_vector(v, name).astype(float) for v, name in
              ((y, "y"), (candidate, "candidate"), (baseline, "baseline"))]
    y, candidate, baseline = arrays
    if len(y) < 2 or any(len(v) != len(y) for v in arrays):
        raise ValueError("All vectors must have equal lengths with at least two genes.")
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer.")
    score: Callable = lambda a, b: _metric(metric, a, b)
    observed = score(y, candidate) - score(y, baseline)
    rng = np.random.default_rng(random_seed)
    deltas = np.full(n_bootstrap, np.nan)
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        deltas[iteration] = score(y[indices], candidate[indices]) - score(y[indices], baseline[indices])
    valid = deltas[np.isfinite(deltas)]
    mean = median = low = high = proportion = float("nan")
    if len(valid):
        mean, median = float(valid.mean()), float(np.median(valid))
        low, high = np.percentile(valid, [2.5, 97.5])
        proportion = float(np.mean(valid > 0))
    return {"observed_delta": float(observed), "bootstrap_mean": mean,
            "bootstrap_median": median, "ci_2_5": float(low), "ci_97_5": float(high),
            "proportion_gt_zero": proportion, "n_valid_bootstrap": len(valid),
            "n_requested_bootstrap": n_bootstrap,
            "n_invalid_bootstrap": n_bootstrap - len(valid)}


def _require(table: pd.DataFrame, columns: list[str]) -> None:
    """Validate required numeric columns without modifying a saved table."""
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    for column in columns:
        finite_vector(table[column].to_numpy(), column)


def run_incremental_information_bootstrap(
    predictions: pd.DataFrame, *, n_bootstrap: int = 5000, random_seed: int = 42,
) -> pd.DataFrame:
    """Compare joint and differential predictions with K562-only on identical draws."""
    _require(predictions, ["functional_selectivity", "pred_k562_only", "pred_joint", "pred_differential"])
    rows = []
    for name, column in (("Joint RNA", "pred_joint"), ("Differential RNA", "pred_differential")):
        for metric in ("spearman_rho", "r2"):
            # Resetting to the same seed gives identical gene draws across all comparisons.
            result = paired_bootstrap_metric_difference(
                predictions.functional_selectivity, predictions[column], predictions.pred_k562_only,
                metric=metric, n_bootstrap=n_bootstrap, random_seed=random_seed)
            rows.append({"comparison": f"{name} vs K562 RNA only", "metric": metric, **result})
    return pd.DataFrame(rows)


def decompose_selectivity_endpoint(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations and sample spread of the unchanged percentile-based endpoint.

    Continuous dependency associations are secondary and within-assay; raw scores
    are never subtracted. Component correlations reflect mathematical coupling.
    """
    columns = ["functional_selectivity", "k562_dependency_percentile", "rpe1_dependency_percentile",
               "k562_dependency", "rpe1_dependency"]
    _require(endpoints, columns)
    if len(endpoints) < 2:
        raise ValueError("Endpoint decomposition requires at least two genes.")
    expected = endpoints.k562_dependency_percentile - endpoints.rpe1_dependency_percentile
    if not np.allclose(endpoints.functional_selectivity, expected, rtol=0, atol=1e-12):
        raise ValueError("Saved functional_selectivity does not equal the percentile difference.")
    pairs = [(columns[0], columns[1], "primary"), (columns[0], columns[2], "primary"),
             (columns[1], columns[2], "primary"), (columns[0], columns[3], "secondary"),
             (columns[0], columns[4], "secondary")]
    rows = []
    for left, right, role in pairs:
        x, y = endpoints[left].to_numpy(), endpoints[right].to_numpy()
        association = spearman_association(x, y)
        pearson = p = float("nan")
        if np.ptp(x) != 0 and np.ptp(y) != 0:
            pearson, p = pearsonr(x, y)
        rows.append({"x": left, "y": right, "role": role, "n": len(x),
                     "spearman_rho": association["rho"], "spearman_p_value": association["p_value"],
                     "pearson_r": float(pearson), "pearson_p_value": float(p)})
    spread = []
    for column in columns[1:3] + [columns[0]]:
        values = endpoints[column]
        spread.append({"component": column, "n": len(values), "mean": values.mean(),
                       "std": values.std(ddof=1), "std_ddof": 1, "min": values.min(),
                       "median": values.median(), "max": values.max(),
                       "iqr": values.quantile(.75) - values.quantile(.25)})
    return pd.DataFrame(rows), pd.DataFrame(spread)


def _index_genes(table: pd.DataFrame) -> pd.DataFrame:
    """Reject missing/duplicate identities before aligning saved tables."""
    if "gene" not in table:
        raise ValueError("Saved diagnostic inputs require a gene column.")
    genes = table.gene
    if genes.isna().any() or genes.duplicated().any() or any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError("Gene identities must be nonempty and unique.")
    return table.set_index("gene")


def run_selectivity_diagnostics(directory: str | Path = "results/selectivity", *,
                                 n_bootstrap: int = 5000, random_seed: int = 42) -> dict[str, pd.DataFrame]:
    """Read two saved CSVs, align the exact universe, and write only new diagnostics."""
    directory = Path(directory)
    paths = [directory / "functional_selectivity.csv", directory / "selectivity_predictions.csv"]
    # Preserve written float precision: tiny parser changes can split/merge rank ties.
    endpoints, predictions = [_index_genes(pd.read_csv(path, float_precision="round_trip")) for path in paths]
    if set(endpoints.index) != set(predictions.index):
        raise ValueError("Endpoint and prediction gene universes must match exactly; no intersection is taken.")
    predictions = predictions.loc[endpoints.index]
    _require(predictions, ["functional_selectivity"])
    _require(endpoints, ["functional_selectivity"])
    if not np.allclose(endpoints.functional_selectivity, predictions.functional_selectivity, rtol=0, atol=1e-12):
        raise ValueError("Saved prediction and endpoint values disagree after gene alignment.")
    decomposition, summary = decompose_selectivity_endpoint(endpoints)
    bootstrap = run_incremental_information_bootstrap(predictions, n_bootstrap=n_bootstrap, random_seed=random_seed)
    tables = {"selectivity_incremental_bootstrap": bootstrap,
              "selectivity_endpoint_decomposition": decomposition, "selectivity_endpoint_summary": summary}
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    metadata = {"n_genes": len(endpoints), "n_bootstrap": n_bootstrap, "random_seed": random_seed,
                "endpoint": "functional_selectivity", "interval": "paired percentile bootstrap, 2.5th/97.5th percentiles",
                "scope": "fixed saved OOF fits; gene resampling only, no retraining; proportion_gt_zero is not a p-value",
                "inputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    (directory / "selectivity_diagnostics_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tables


def main() -> None:
    """Command-line wrapper for the Makefile diagnostic target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="results/selectivity")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, table in run_selectivity_diagnostics(args.directory, n_bootstrap=args.n_bootstrap, random_seed=args.seed).items():
        print(name)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
