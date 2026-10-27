import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from perturbselect.benchmarking.plotting import (plot_model_comparison, plot_forest,
    plot_stability, plot_coverage, plot_performance_vs_coverage,
    plot_metric_correlation, plot_ranked_methods, save_figure)
from perturbselect.benchmarking.tables import performance_table, format_table, render_table, save_table
from perturbselect.benchmarking.reporting import ArtifactLayout, generate_presentation, load_presentation_config
from perturbselect.benchmark_adapters import adapt_metrics, adapt_predictions, adapt_incremental_bootstrap


@pytest.fixture
def results():
    return adapt_metrics(pd.DataFrame({"model": ["A", "B"], "mae": [.123456789, .4], "r2": [.8, .5]}),
                         task="synthetic_task", dataset="simulation", n_samples=10)


def test_all_plot_types_and_exports(results, tmp_path):
    coverage = pd.DataFrame({"method": ["B", "A"], "n_supported": [8, 10], "n_total": [10, 10], "coverage_fraction": [.8, 1.]})
    comparison = pd.DataFrame({"method": ["A"], "baseline": ["B"], "metric": ["mae"],
        "observed_improvement": [.2], "improvement_ci_low": [.1], "improvement_ci_high": [.3]})
    repeats = pd.concat([results.assign(seed=1), results.assign(seed=2, value=results.value + .01)])
    figures = [plot_model_comparison(results, "mae", method_order=["B", "A"]),
               plot_forest(comparison, "mae"), plot_stability(repeats, "mae"), plot_coverage(coverage),
               plot_performance_vs_coverage(results, coverage, "mae"),
               plot_metric_correlation(results, "mae", "r2"), plot_ranked_methods(results, "mae")]
    try:
        for i, (fig, ax) in enumerate(figures):
            assert isinstance(fig, Figure) and isinstance(ax, Axes)
            for path in save_figure(fig, tmp_path / f"plot_{i}"):
                assert path.stat().st_size > 100
        assert [label.get_text() for label in figures[0][1].get_yticklabels()] == ["B", "A"]
        assert [label.get_text() for label in figures[-1][1].get_yticklabels()] == ["A", "B"]
    finally:
        for fig, _ in figures:
            plt.close(fig)


def test_tables_preserve_values_and_escape_labels(results, tmp_path):
    table = performance_table(results)
    original = table.copy(deep=True)
    assert format_table(table, 2).loc[0, "mae"] == "0.12"
    for fmt in ("csv", "markdown", "latex"):
        text = render_table(table, fmt, 2)
        assert isinstance(text, str)
        save_table(table, tmp_path / fmt, fmt, 2)
    assert "0.123456789" in render_table(table, "csv")
    assert "\\begin{tabular}" in render_table(table, "latex")
    pd.testing.assert_frame_equal(table, original)
    with pytest.raises(ValueError, match="Select one"):
        performance_table(pd.concat([results, results.assign(seed=5)]))


def test_adapter_values_are_not_recomputed():
    source = pd.DataFrame({"model": ["Any model"], "r2": [.123456789012345], "mae": [.765432109876543]})
    converted = adapt_metrics(source, task="example", dataset="small", n_samples=2)
    assert converted.set_index("metric").loc["r2", "value"] == source.r2.iloc[0]
    wide = pd.DataFrame({"gene": ["B", "A"], "truth": [3., 4.], "prediction": [2.23456789012345, 5.]})
    long = adapt_predictions(wide, target_column="truth", prediction_columns={"prediction": "Any model"}, task="example", dataset="small")
    np.testing.assert_array_equal(long.y_pred, wide.prediction)
    assert long.sample_id.tolist() == ["B", "A"]
    assert long.fold.isna().all()


def test_saved_bootstrap_adapter_preserves_raw_effect_and_direction():
    source = pd.DataFrame({"comparison": ["A vs B", "A vs B"], "metric": ["r2", "mae"],
        "observed_delta": [.12, -.2], "ci_2_5": [.02, -.3], "ci_97_5": [.22, -.1],
        "proportion_gt_zero": [.98, .04], "n_requested_bootstrap": [100, 100],
        "n_valid_bootstrap": [99, 99], "n_invalid_bootstrap": [1, 1]})
    converted = adapt_incremental_bootstrap(source).set_index("metric")
    assert converted.loc["r2", "observed_delta"] == source.observed_delta.iloc[0]
    assert converted.loc["mae", "observed_improvement"] == .2
    assert converted.loc["mae", "improvement_ci_low"] == .1
    assert np.isnan(converted.loc["mae", "proportion_better"])


def test_configured_report(results, tmp_path):
    path = tmp_path / "presentation.yaml"
    path.write_text("metrics: [mae, r2]\noutput_formats: [svg]\nmethod_order: [B, A]\ntable_precision: 2\n")
    layout = ArtifactLayout(tmp_path, "synthetic")
    comparisons = pd.DataFrame({"method": ["A"], "baseline": ["B"], "metric": ["mae"],
        "observed_improvement": [.2], "improvement_ci_low": [.1], "improvement_ci_high": [.3]})
    robustness = pd.concat([results.assign(seed=1), results.assign(seed=2, value=results.value + .01)])
    report = generate_presentation(results, layout, load_presentation_config(path),
        metadata={"source": "synthetic"}, comparisons=comparisons, robustness=robustness)
    assert report.is_file()
    assert "performance.csv" in report.read_text()
    assert (layout.tables() / "performance.tex").is_file()
    assert (layout.tables("supplementary") / "dataset_summary.csv").is_file()
    assert (layout.tables("supplementary") / "stability_summary.csv").is_file()
    assert "comparison_0.svg" in report.read_text()
    path.write_text("n_bootstrap: 20")
    with pytest.raises(ValueError, match="statistical decisions"):
        load_presentation_config(path)
