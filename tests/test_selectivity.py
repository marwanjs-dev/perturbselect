"""Fast synthetic tests of identities, phenotype scales and cross-context features."""
import numpy as np
import pandas as pd
import pytest

from perturbselect.selectivity import (AlignedContexts, ContextData, align_contexts,
    build_selectivity_endpoints, cell_count_features, dependency_percentile,
    dependency_zscore, differential_responses, find_common_targets,
    joint_responses, orient_dependency)


@pytest.fixture
def contexts():
    k = ContextData(np.array([[3., 4.], [1., 2.], [9., 8.]]),
                    np.array(["B", "A", "K_only"]), np.array(["A", "B"]),
                    np.array([20, 10, 30]),
                    pd.Series([-10., 0., -1000.], index=["A", "B", "K_only"]))
    # RPE1 target and feature order both differ. Noncommon genes are extreme.
    r = ContextData(np.array([[1., 0.], [7., 8.], [6., 5.]]),
                    np.array(["A", "R_only", "B"]), np.array(["B", "A"]),
                    np.array([40, 60, 50]),
                    pd.Series([10., 0., 1000.], index=["B", "A", "R_only"]))
    return k, r


def test_four_way_intersection_and_deterministic_order():
    result = find_common_targets(["C", "B", "A", "D"], ["A", "B", "C"],
                                  ["B", "A", "D"], ["A", "B", "C"])
    assert result.tolist() == ["A", "B"]
    with pytest.raises(ValueError, match="No common targets"):
        find_common_targets(["A"], ["B"], ["A"], ["B"])


def test_named_alignment_not_position(contexts):
    aligned = align_contexts(*contexts)
    assert aligned.k562.targets.tolist() == aligned.rpe1.targets.tolist() == ["A", "B"]
    assert aligned.k562.genes.tolist() == aligned.rpe1.genes.tolist() == ["A", "B"]
    assert aligned.k562.fitness.index.tolist() == ["A", "B"]
    np.testing.assert_array_equal(aligned.k562.X, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(aligned.rpe1.X, [[0, 1], [5, 6]])
    np.testing.assert_array_equal(aligned.k562.n_cells, [10, 20])
    np.testing.assert_array_equal(aligned.rpe1.n_cells, [40, 50])


def test_dependency_direction_ranks_and_ties():
    np.testing.assert_array_equal(orient_dependency([-2., -1., 1.], higher_is_dependency=False), [2, 1, -1])
    np.testing.assert_array_equal(orient_dependency([3., 7.], higher_is_dependency=True), [3, 7])
    np.testing.assert_allclose(dependency_percentile([2, 2, 1, 3]), [.625, .625, .25, 1.])


def test_endpoints_use_only_common_targets_and_have_correct_sign(contexts):
    endpoints = build_selectivity_endpoints(align_contexts(*contexts))
    assert endpoints.gene.tolist() == ["A", "B"]
    np.testing.assert_array_equal(endpoints.k562_dependency, [10, 0])
    np.testing.assert_array_equal(endpoints.rpe1_dependency, [0, 10])
    np.testing.assert_array_equal(endpoints.k562_dependency_percentile, [1, .5])
    np.testing.assert_array_equal(endpoints.rpe1_dependency_percentile, [.5, 1])
    np.testing.assert_array_equal(endpoints.functional_selectivity, [.5, -.5])
    np.testing.assert_array_equal(endpoints.z_selectivity, [2, -2])
    # The raw-score difference is deliberately not the selectivity endpoint.
    assert not np.array_equal(endpoints.functional_selectivity,
                              endpoints.k562_raw_fitness - endpoints.rpe1_raw_fitness)


def test_zscores_population_convention_and_constant_error():
    z = dependency_zscore([1., 2., 3.])
    assert z.mean() == pytest.approx(0)
    assert z.std(ddof=0) == pytest.approx(1)
    with pytest.raises(ValueError, match="nonconstant"):
        dependency_zscore([1, 1])
    with pytest.raises(ValueError, match="finite"):
        dependency_percentile([1, np.nan])


def test_exact_delta_joint_and_cell_features(contexts):
    aligned = align_contexts(*contexts)
    np.testing.assert_array_equal(differential_responses(aligned), [[1, 1], [-2, -2]])
    np.testing.assert_array_equal(joint_responses(aligned), [[1, 2, 0, 1], [3, 4, 5, 6]])
    np.testing.assert_allclose(cell_count_features(aligned), np.log1p([[10, 40], [20, 50]]))


def test_masking_reuses_both_contexts_and_does_not_mutate(contexts, monkeypatch):
    import perturbselect.selectivity as module
    aligned = align_contexts(*contexts)
    original = module.mask_target_expression
    calls = []

    def capture(X, targets, genes):
        calls.append(X.copy())
        return original(X, targets, genes)

    monkeypatch.setattr(module, "mask_target_expression", capture)
    np.testing.assert_array_equal(differential_responses(aligned, masked=True), [[0, 1], [-2, 0]])
    assert len(calls) == 2
    np.testing.assert_array_equal(aligned.k562.X, calls[0])
    np.testing.assert_array_equal(aligned.rpe1.X, calls[1])


def test_absent_target_feature_remains_unmasked():
    fitness = pd.Series([1., 2.], index=["A", "absent"])
    k = ContextData(np.array([[2.], [3.]]), np.array(["A", "absent"]), np.array(["A"]), np.array([3, 4]), fitness)
    r = ContextData(np.array([[1.], [1.]]), k.targets, k.genes, k.n_cells, fitness)
    np.testing.assert_array_equal(differential_responses(align_contexts(k, r), masked=True), [[0], [2]])


def test_missing_feature_and_invalid_dimensions(contexts):
    k, r = contexts
    r.genes = np.array(["B", "different"])
    with pytest.raises(ValueError, match="gene sets differ"):
        align_contexts(k, r)
    with pytest.raises(ValueError, match="dimensions"):
        ContextData(np.zeros((2, 2)), np.array(["A"]), np.array(["A", "B"]), np.array([1]), pd.Series([1], index=["A"]))
    with pytest.raises(ValueError, match="duplicate"):
        find_common_targets(["A", "A"], ["A"], ["A"], ["A"])


def test_unaligned_container_rejected(contexts):
    with pytest.raises(ValueError, match="identities and order"):
        AlignedContexts(*contexts)
