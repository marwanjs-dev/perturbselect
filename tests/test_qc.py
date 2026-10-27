import numpy as np
import pytest

from perturbselect.qc import (adjust_response_magnitude_for_cell_count, cell_count_qc,
                              cell_count_sensitivity, response_magnitude, spearman_association)


def test_exact_magnitude():
    np.testing.assert_array_equal(response_magnitude([[1., -3.], [0., 4.]]), [2., 2.])


def test_spearman_and_undefined_cases():
    result = spearman_association([1, 2, 3, 4], [8, 6, 4, 2])
    assert result["rho"] == pytest.approx(-1)
    assert result["n"] == 4
    assert result["p_value"] <= 0.05
    assert np.isnan(spearman_association([1, 1], [2, 3])["rho"])
    assert np.isnan(spearman_association([], [])["rho"])
    assert np.isnan(spearman_association([1], [2])["p_value"])


def test_sensitivity_inclusive_thresholds_and_empty_subset():
    result = cell_count_sensitivity([30, 50, 75, 100], [4, 3, 2, 1], [1, 2, 3, 4], [30, 50, 101])
    assert result.n_perturbations.tolist() == [4, 3, 0]
    assert result.iloc[0].rna_magnitude_vs_fitness_rho == pytest.approx(-1)
    assert np.isnan(result.iloc[2].rna_magnitude_vs_fitness_rho)


def test_residualization_removes_log_cell_trend():
    cells = np.arange(1, 30)
    residual = adjust_response_magnitude_for_cell_count(cells, 2 + 3 * np.log1p(cells))
    np.testing.assert_allclose(residual, 0, atol=1e-13)


@pytest.mark.parametrize("X", [[1, 2], np.empty((2, 0)), [[np.inf]], [[np.nan]]])
def test_invalid_response(X):
    with pytest.raises(ValueError, match="2D|finite"):
        response_magnitude(X)


def test_useful_vector_errors():
    with pytest.raises(ValueError, match="equal lengths"):
        spearman_association([1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        spearman_association([np.nan], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        spearman_association([[1]], [1])
    with pytest.raises(ValueError, match="nonnegative"):
        cell_count_qc([-1, 2], [1, 2], [3, 4])
