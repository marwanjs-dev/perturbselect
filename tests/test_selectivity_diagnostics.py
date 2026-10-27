"""Synthetic diagnostics of saved predictions; no training or H5AD inputs."""
import numpy as np
import pandas as pd
import pytest

from perturbselect.selectivity_diagnostics import (paired_bootstrap_metric_difference,
    decompose_selectivity_endpoint, run_selectivity_diagnostics)


@pytest.mark.parametrize("metric", ["spearman_rho", "r2"])
def test_better_model_and_reproducibility(metric):
    y = np.arange(40, dtype=float)
    first = paired_bootstrap_metric_difference(y, y, -y, metric=metric, n_bootstrap=100)
    assert first == paired_bootstrap_metric_difference(y, y, -y, metric=metric, n_bootstrap=100)
    assert first["proportion_gt_zero"] > .9
    assert first["n_valid_bootstrap"] == 100
    identical = paired_bootstrap_metric_difference(y, y, y, metric=metric, n_bootstrap=100)
    assert identical["observed_delta"] == pytest.approx(0)
    assert identical["bootstrap_mean"] == pytest.approx(0)
    assert identical["proportion_gt_zero"] == 0


def test_same_indices_for_both_models(monkeypatch):
    import perturbselect.selectivity_diagnostics as module
    calls = []

    def score(name, y, pred):
        calls.append((y.copy(), pred.copy()))
        return float(pred.mean())

    monkeypatch.setattr(module, "_metric", score)
    y = np.arange(8, dtype=float)
    paired_bootstrap_metric_difference(y, y + 10, y + 20, n_bootstrap=10)
    for (yc, pc), (yb, pb) in zip(calls[::2], calls[1::2]):
        np.testing.assert_array_equal(yc, yb)
        np.testing.assert_array_equal(pc, yc + 10)
        np.testing.assert_array_equal(pb, yb + 20)
    assert any(len(np.unique(yc)) < len(yc) for yc, _ in calls[2:])


@pytest.mark.parametrize("metric", ["spearman_rho", "r2"])
def test_degenerate_samples_recorded(metric):
    result = paired_bootstrap_metric_difference([0., 1.], [0., 1.], [1., 0.], metric=metric, n_bootstrap=100)
    assert 0 < result["n_valid_bootstrap"] < 100
    assert result["n_valid_bootstrap"] + result["n_invalid_bootstrap"] == 100
    undefined = paired_bootstrap_metric_difference([1., 1.], [1., 1.], [1., 1.], metric=metric, n_bootstrap=10)
    assert undefined["n_valid_bootstrap"] == 0
    assert np.isnan(undefined["bootstrap_mean"])


@pytest.fixture
def endpoints():
    return pd.DataFrame({"gene": list("ABCD"), "k562_dependency_percentile": [.25, .5, .75, 1.],
                         "rpe1_dependency_percentile": [1., .75, .5, .25],
                         "functional_selectivity": [-.75, -.25, .25, .75],
                         "k562_dependency": [1., 2., 3., 4.],
                         "rpe1_dependency": [4000., 3000., 2000., 1000.]})


def test_decomposition_signs_and_no_raw_score_subtraction(endpoints):
    original = endpoints.copy(deep=True)
    associations, summary = decompose_selectivity_endpoint(endpoints)
    np.testing.assert_allclose(associations.spearman_rho, [1, -1, -1, 1, -1])
    np.testing.assert_allclose(associations.pearson_r, [1, -1, -1, 1, -1])
    assert associations.role.tolist() == ["primary"] * 3 + ["secondary"] * 2
    pd.testing.assert_frame_equal(endpoints, original)
    altered = endpoints.copy()
    altered.rpe1_dependency *= 100
    other_associations, other_summary = decompose_selectivity_endpoint(altered)
    pd.testing.assert_frame_equal(summary, other_summary)
    np.testing.assert_allclose(associations.pearson_r, other_associations.pearson_r)


def test_saved_inputs_align_without_changing_universe_or_files(tmp_path, endpoints):
    prediction = endpoints[["gene", "functional_selectivity"]].copy()
    prediction["pred_joint"] = prediction.functional_selectivity
    prediction["pred_differential"] = prediction.functional_selectivity * .9
    prediction["pred_k562_only"] = -prediction.functional_selectivity
    endpoint_path = tmp_path / "functional_selectivity.csv"
    prediction_path = tmp_path / "selectivity_predictions.csv"
    endpoints.to_csv(endpoint_path, index=False)
    prediction.iloc[::-1].to_csv(prediction_path, index=False)
    before = [path.read_bytes() for path in (endpoint_path, prediction_path)]
    result = run_selectivity_diagnostics(tmp_path, n_bootstrap=20)
    assert len(result["selectivity_incremental_bootstrap"]) == 4
    assert before == [path.read_bytes() for path in (endpoint_path, prediction_path)]
    for name in result:
        assert (tmp_path / f"{name}.csv").is_file()
    prediction.iloc[1:].to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="universes must match"):
        run_selectivity_diagnostics(tmp_path, n_bootstrap=10)


def test_validation(endpoints):
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_metric_difference([1, 2], [1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_metric_difference([1, 2], [1, np.nan], [1, 2])
    endpoints.loc[0, "functional_selectivity"] = 7
    with pytest.raises(ValueError, match="percentile difference"):
        decompose_selectivity_endpoint(endpoints)
