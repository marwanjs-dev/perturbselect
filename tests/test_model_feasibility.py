"""Synthetic metadata-only feasibility tests; no models, H5AD or network access."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from perturbselect.model_feasibility import (BenchmarkRequirements, assess_model,
    estimate_coverage, load_benchmark_requirements, run_feasibility_audit)


@pytest.fixture
def benchmark():
    return BenchmarkRequirements(np.array(["A", "B", "C"]), np.array(["X", "Y"]), {})


@pytest.fixture
def ready():
    return {"name": "synthetic candidate", "license": "synthetic permissive license",
            "gene_perturbation_compatible": True, "k562_support": True, "rpe1_support": True,
            "supported_perturbations": ["C", "A", "B"], "output_gene_space": ["Y", "X"],
            "prediction_download_available": False, "inference_available": True,
            "checkpoint_available": True, "training_required": False,
            "gpu_required": False, "approximate_ram": 4,
            "expression_output": True, "output_compatible": True, "gene_space_compatible": True,
            "context_fairness": True, "preprocessing_documented": True,
            "reproducibility_documented": True, "evidence_sources": ["Synthetic test specification"]}


def test_ready_candidate(benchmark, ready):
    original = deepcopy(ready)
    report = assess_model(ready, benchmark, hardware={"ram_gb": 8})
    assert report["feasibility_status"] == "READY"
    assert all(d["status"] == "PASS" for d in report["dimensions"].values())
    assert ready == original


def test_multiple_scientific_incompatibilities_preserved(benchmark, ready):
    ready.update(rpe1_support=False, gene_perturbation_compatible=False,
                 expression_output=False, gene_space_compatible=False)
    report = assess_model(ready, benchmark)
    assert report["feasibility_status"] == "INCOMPATIBLE"
    for flag in ("missing_rpe1_support", "no_gene_perturbation_prediction", "no_expression_output", "incompatible_gene_space"):
        assert report["flags"][flag] is True
    assert len(report["scientific_failures"]) >= 3


def test_coverage_by_identifier_and_thresholds(benchmark, ready):
    assert estimate_coverage(["A", "B"], ["B", "B", "A", "extra"])["coverage_fraction"] == 1
    ready.update(supported_perturbations=["B"], output_gene_space=["Y"])
    report = assess_model(ready, benchmark, hardware={"ram_gb": 8})
    assert report["target_coverage"]["n_supported"] == 1
    assert report["target_coverage"]["missing_examples"] == ["A", "C"]
    assert report["flags"]["insufficient_target_coverage"] is True
    assert report["flags"]["insufficient_feature_coverage"] is True
    conditional = assess_model(ready, benchmark, hardware={"ram_gb": 8}, min_target_coverage=.3, min_feature_coverage=.5)
    assert conditional["feasibility_status"] == "PROBABLY_FEASIBLE"


def test_checkpoint_retraining_and_compute_are_not_scientific_failures(benchmark, ready):
    ready.update(checkpoint_available=False, training_required=True)
    report = assess_model(ready, benchmark, hardware={"ram_gb": 2})
    assert report["flags"]["checkpoint_unavailable"] is True
    assert report["flags"]["requires_full_retraining"] is True
    assert report["flags"]["prohibitively_large_compute"] is True
    assert report["scientific_failures"] == []
    assert report["feasibility_status"] == "HIGH_RISK"


def test_unknowns_remain_unknown_and_repository_is_not_readiness(benchmark):
    report = assess_model({"name": "unresearched", "repository": "https://example.invalid/model"}, benchmark)
    assert report["feasibility_status"] == "NEEDS_INVESTIGATION"
    assert report["flags"]["missing_rpe1_support"] is None
    assert report["flags"]["checkpoint_unavailable"] is None
    assert report["target_coverage"]["coverage_fraction"] is None
    assert all(d["status"] == "UNKNOWN" for d in report["dimensions"].values())
    assert estimate_coverage(["A"], [])["coverage_fraction"] == 0


def test_download_route_can_work_without_checkpoint(benchmark, ready):
    ready.update(prediction_download_available=True, checkpoint_available=False, inference_available=False)
    report = assess_model(ready, benchmark, hardware={"ram_gb": 8})
    assert report["flags"]["checkpoint_unavailable"] is True
    assert report["feasibility_status"] == "READY"


def test_missing_preprocessing_is_engineering_risk(benchmark, ready):
    ready["preprocessing_documented"] = False
    report = assess_model(ready, benchmark, hardware={"ram_gb": 8})
    assert report["feasibility_status"] == "HIGH_RISK"
    assert report["scientific_failures"] == []
    assert report["flags"]["preprocessing_unknown"] is True


def test_load_authoritative_identifiers_without_response_matrix(tmp_path):
    np.savez(tmp_path / "differential_responses.npz", targets=np.array(["B", "A"]), genes=np.array(["X", "Y", "Z"]))
    (tmp_path / "selectivity_summary.json").write_text(json.dumps({"n_common_targets": 2, "n_features": 3}))
    pd.DataFrame({"gene": ["A", "B"]}).to_csv(tmp_path / "common_targets.csv", index=False)
    result = load_benchmark_requirements(tmp_path)
    assert result.metadata["n_benchmark_targets"] == 2
    assert result.metadata["n_benchmark_features"] == 3
    assert result.targets.tolist() == ["B", "A"]
    (tmp_path / "selectivity_summary.json").write_text(json.dumps({"n_features": 99}))
    with pytest.raises(ValueError, match="disagrees"):
        load_benchmark_requirements(tmp_path)


def test_empty_registry_does_not_invent_candidates(tmp_path):
    np.savez(tmp_path / "differential_responses.npz", targets=np.array(["A"]), genes=np.array(["X"]))
    config = {"project_root": ".", "benchmark_directory": ".", "output_directory": "audit", "models": []}
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run_feasibility_audit(path).empty
    report = json.loads((tmp_path / "audit/model_feasibility_report.json").read_text())
    assert report["reports"] == []
    assert report["benchmark_requirements"]["n_benchmark_targets"] == 1


def test_boolean_validation(benchmark, ready):
    ready["rpe1_support"] = "unknown"
    with pytest.raises(ValueError, match="true, false or null"):
        assess_model(ready, benchmark)
