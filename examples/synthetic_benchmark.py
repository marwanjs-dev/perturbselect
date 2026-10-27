"""A synthetic sensor-calibration study using only the generic benchmark API.

Run this file from a notebook/IDE with the package available. It never trains a
model; predictions are simulated to demonstrate the reporting data contract.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from perturbselect.benchmarking.results import evaluate_predictions, write_results
from perturbselect.benchmarking.comparisons import compare_to_baseline
from perturbselect.benchmarking.provenance import capture_provenance
from perturbselect.benchmarking.reporting import ArtifactLayout, generate_presentation


def main(root: str | Path = ".") -> Path:
    """Produce a complete, independently named synthetic study with paired statistics."""
    rng = np.random.default_rng(42)
    truth = rng.normal(size=60)
    predictions = pd.concat([pd.DataFrame({
        "sample_id": [f"sensor-{i:03}" for i in range(60)], "study": "sensor_demo",
        "task": "calibration", "dataset": "simulation", "method": method,
        "split": "held_out", "seed": 42, "y_true": truth,
        "y_pred": truth + rng.normal(scale=scale, size=60)})
        for method, scale in (("reference", 1.), ("adjusted", .3))], ignore_index=True)
    results = evaluate_predictions(predictions)
    comparisons = compare_to_baseline(predictions, "reference", ["mae", "r2"], n_bootstrap=200, seed=42)
    layout = ArtifactLayout(Path(root), "sensor_demo")
    write_results(results, layout.results("raw") / "metrics.csv")
    predictions.to_csv(layout.results("raw") / "predictions.csv", index=False)
    layout.results("statistics").mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(layout.results("statistics") / "comparisons.csv", index=False)
    config = {"metrics": ["mae", "r2"], "method_order": ["reference", "adjusted"],
              "display_labels": {}, "output_formats": ["svg", "png", "pdf"], "table_precision": 3}
    return generate_presentation(results, layout, config, comparisons=comparisons,
        metadata={"study": "Synthetic sensor calibration", "n_samples": 60},
        provenance=capture_provenance(seeds=[42], packages=["numpy", "pandas", "scipy"]))


if __name__ == "__main__":
    print(main())
