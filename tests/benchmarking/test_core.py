"""Only synthetic observations; no scientific datasets or fitted estimators."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perturbselect.benchmarking import schema
from perturbselect.benchmarking.metrics import METRICS, score_metric
from perturbselect.benchmarking.results import evaluate_predictions
from perturbselect.benchmarking.statistics import match_predictions, paired_bootstrap, summarize_repeats
from perturbselect.benchmarking.comparisons import compare_to_baseline
from perturbselect.benchmarking.provenance import capture_provenance


@pytest.fixture
def predictions():
    y = np.linspace(-2, 2, 24)
    return pd.concat([pd.DataFrame({"sample_id": [f"s{i:02}" for i in range(24)],
        "task": "forecast", "dataset": "synthetic", "method": method, "split": "held_out",
        "seed": 42, "y_true": y, "y_pred": pred})
        for method, pred in (("reference", y * .3), ("candidate", y + .02))], ignore_index=True)


def test_prediction_and_result_schemas(predictions):
    results = evaluate_predictions(predictions)
    assert len(results) == 10
    assert set(schema.RESULT_ID) <= set(results)
    pd.testing.assert_frame_equal(results, schema.validate_results(results))
    for table, validator in ((predictions, schema.validate_predictions), (results, schema.validate_results)):
        with pytest.raises(ValueError, match="Duplicate"):
            validator(pd.concat([table, table.iloc[:1]]))
        with pytest.raises(ValueError, match="Missing required"):
            validator(table.drop(columns="method"))
    invalid = results.copy()
    invalid.loc[0, "value"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        schema.validate_results(invalid)
    invalid.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="explicitly marked"):
        schema.validate_results(invalid)
    invalid.loc[0, "value_status"] = "undefined"
    schema.validate_results(invalid)


def test_metric_registry_and_legacy_parity():
    from perturbselect.modeling import evaluate_predictions as original
    y, pred = np.array([1., 2., 3.]), np.array([1., 2., 4.])
    old = original("example", y, pred)
    for metric in METRICS:
        assert score_metric(metric, y, pred) == pytest.approx(old[metric])
    assert METRICS["r2"].higher_is_better
    assert not METRICS["mae"].higher_is_better
    assert score_metric("r2", y, pred) == .5
    assert np.isnan(score_metric("spearman_rho", y, np.ones(3)))
    assert np.isnan(score_metric("r2", np.ones(3), pred, bootstrap=True))


def test_pairing_reordering_and_truth_validation(predictions):
    a = match_predictions(predictions, "candidate", "reference")
    b = match_predictions(predictions.iloc[::-1], "candidate", "reference")
    pd.testing.assert_frame_equal(a, b)
    altered = predictions.copy()
    altered.loc[0, "y_true"] += .001
    with pytest.raises(ValueError, match="ground truths"):
        match_predictions(altered, "candidate", "reference")
    with pytest.raises(ValueError, match="universes"):
        match_predictions(predictions.iloc[1:], "candidate", "reference")


def test_bootstrap_reproducibility_direction_and_legacy_parity(predictions):
    a = paired_bootstrap(predictions, "candidate", "reference", "mae", n_bootstrap=40)
    assert a == paired_bootstrap(predictions.iloc[::-1], "candidate", "reference", "mae", n_bootstrap=40)
    assert a["observed_delta"] < 0 < a["observed_improvement"]
    assert a["proportion_better"] > .9
    assert a["n_valid"] == 40 and a["n_invalid"] == 0
    from perturbselect.selectivity_diagnostics import paired_bootstrap_metric_difference
    paired = match_predictions(predictions, "candidate", "reference")
    old = paired_bootstrap_metric_difference(paired.y_true, paired.candidate, paired.baseline, metric="r2", n_bootstrap=40)
    new = paired_bootstrap(predictions, "candidate", "reference", "r2", n_bootstrap=40)
    assert new["observed_delta"] == pytest.approx(old["observed_delta"])
    assert new["ci_low"] == pytest.approx(old["ci_2_5"])
    assert len(compare_to_baseline(predictions, "reference", ["mae"], n_bootstrap=10)) == 1


def test_bootstrap_same_sample_draws(predictions, monkeypatch):
    import perturbselect.benchmarking.statistics as module
    calls = []
    def score(metric, y, pred, **kwargs):
        calls.append((y.copy(), pred.copy()))
        return float(np.mean(abs(y - pred)))
    monkeypatch.setattr(module, "score_metric", score)
    module.paired_bootstrap(predictions, "candidate", "reference", "mae", n_bootstrap=10)
    for (ya, pa), (yb, pb) in zip(calls[::2], calls[1::2]):
        np.testing.assert_array_equal(ya, yb)
        np.testing.assert_allclose(pa, ya + .02)
        np.testing.assert_allclose(pb, yb * .3)


def test_repeat_summary(predictions):
    results = evaluate_predictions(predictions, ["mae"])
    second = results.assign(seed=11, value=results.value + 2)
    summary = summarize_repeats(pd.concat([results, second]))
    assert summary.n_valid.tolist() == [2, 2]
    np.testing.assert_allclose(summary.sd, np.sqrt(2))


def test_provenance_is_json_serializable(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("known source")
    captured = capture_provenance(sources=[source], config=source, seeds=[42],
                                  packages=["numpy", "nonexistent_distribution_xyz"], git_directory=tmp_path)
    json.dumps(captured)
    assert len(captured["config"]["sha256"]) == 64
    assert captured["package_versions"]["nonexistent_distribution_xyz"] is None


def test_generic_import_boundary():
    root = Path(schema.__file__).parent
    for file in root.glob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, f"Domain-relative import in {file}"
                assert not (node.module or "").startswith("perturbselect.")
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("perturbselect.") for name in node.names)
        for word in ("K562", "RPE1", "CRISPR", "BAGEL", "functional_selectivity", "virtual cell"):
            assert word not in file.read_text(encoding="utf-8")


def test_supplied_pvalue_adjustment():
    from perturbselect.benchmarking.statistics import adjust_pvalues
    np.testing.assert_allclose(adjust_pvalues([.01, .04, .03, np.nan]), [.03, .04, .04, np.nan], equal_nan=True)
