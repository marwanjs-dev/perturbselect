import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbselect.signatures import build_response_matrix, mask_target_expression


@pytest.mark.parametrize("sparse_input", [False, True])
def test_chunked_means_and_order(tmp_path, sparse_input):
    X = np.array([[1., 2., 3.], [3., 4., 5.], [7., 8., 9.], [5., 6., 7.]])
    data = ad.AnnData(sparse.csr_matrix(X) if sparse_input else X,
                     obs=pd.DataFrame({"label": ["ctrl", "A", "B", "ctrl"]}, index=list("1234")),
                     var=pd.DataFrame(index=["A", "B", "C"]))
    path = tmp_path / "small.h5ad"
    data.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        result = build_response_matrix(backed, ["B", "A"], ["C", "A"],
            perturbation_column="label", control_label="ctrl", chunk_size=2)
        np.testing.assert_allclose(result.X, [[4, 4], [0, 0]])
        assert result.targets.tolist() == ["B", "A"]
        assert result.genes.tolist() == ["C", "A"]
        assert result.n_controls == 2
        assert result.n_cells.tolist() == [1, 1]
        with pytest.raises(ValueError, match="missing"):
            build_response_matrix(backed, ["A"], ["absent"], perturbation_column="label", control_label="ctrl")
        with pytest.raises(ValueError, match="must have cells"):
            build_response_matrix(backed, ["absent"], ["A"], perturbation_column="label", control_label="ctrl")
    finally:
        backed.file.close()


def test_mask_only_matching_row_feature():
    X = np.ones((3, 2))
    masked, count = mask_target_expression(X, ["B", "absent", "A"], ["A", "B"])
    np.testing.assert_array_equal(masked, [[1, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(X, np.ones((3, 2)))
    assert count == 2
