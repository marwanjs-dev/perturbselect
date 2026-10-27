"""Evidence-based feasibility gate; no model imports, downloads, or inference."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .selectivity import unique_identifiers


STATUSES = {"READY", "PROBABLY_FEASIBLE", "NEEDS_INVESTIGATION", "HIGH_RISK", "INCOMPATIBLE"}
FLAGS = ("no_gene_perturbation_prediction", "no_relevant_context_support",
         "missing_k562_support", "missing_rpe1_support", "no_expression_output",
         "requires_full_retraining", "checkpoint_unavailable", "insufficient_target_coverage",
         "insufficient_feature_coverage", "incompatible_gene_space", "preprocessing_unknown",
         "prohibitively_large_compute")
DIMENSIONS = {
    "perturbation_compatibility": "scientific_compatibility",
    "k562_compatibility": "scientific_compatibility",
    "rpe1_compatibility": "scientific_compatibility",
    "target_coverage": "scientific_compatibility",
    "feature_coverage": "scientific_compatibility",
    "prediction_accessibility": "data_availability",
    "compute_feasibility": "compute_difficulty",
    "output_compatibility": "scientific_compatibility",
    "context_fairness": "scientific_compatibility",
    "reproducibility": "engineering_difficulty",
}
BOOLEAN_FIELDS = (
    "gene_perturbation_compatible", "k562_support", "rpe1_support",
    "pretrained_model_available", "checkpoint_available", "zero_shot_perturbation_support",
    "unseen_cell_context_support", "gpu_required", "inference_available", "training_required",
    "prediction_download_available", "expression_output", "output_compatible", "gene_space_compatible",
    "context_fairness", "preprocessing_documented", "reproducibility_documented",
)


@dataclass
class BenchmarkRequirements:
    """Authoritative saved target/feature identities, with provenance metadata."""
    targets: np.ndarray
    genes: np.ndarray
    metadata: dict[str, Any]


def load_benchmark_requirements(directory: str | Path) -> BenchmarkRequirements:
    """Read identifiers only from the differential NPZ; never materialize X_delta."""
    directory = Path(directory)
    path = directory / "differential_responses.npz"
    with np.load(path, allow_pickle=False) as archive:
        if not {"targets", "genes"} <= set(archive.files):
            raise ValueError("Benchmark NPZ must contain targets and genes.")
        targets = unique_identifiers(archive["targets"], "benchmark targets")
        genes = unique_identifiers(archive["genes"], "benchmark genes")
    sources = [str(path.resolve())]
    summary_path = directory / "selectivity_summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, count in (("n_common_targets", len(targets)), ("n_features", len(genes))):
            if key in summary and summary[key] != count:
                raise ValueError(f"Saved benchmark {key} disagrees with NPZ identifiers.")
        sources.append(str(summary_path.resolve()))
    target_path = directory / "common_targets.csv"
    if target_path.is_file():
        table = pd.read_csv(target_path)
        if "gene" not in table:
            raise ValueError("common_targets.csv must have a gene column.")
        listed = unique_identifiers(table.gene.to_numpy(), "common targets")
        if set(listed) != set(targets):
            raise ValueError("Saved common target list disagrees with NPZ identifiers.")
        sources.append(str(target_path.resolve()))
    return BenchmarkRequirements(targets, genes, {
        "n_benchmark_targets": len(targets), "n_benchmark_features": len(genes),
        "sources": sources, "saved_selectivity_metadata": summary,
        "required_contexts": ["K562", "RPE1"],
        "perturbation_reference": "Replogle CRISPRi; comparability requires scientific review",
        "response_reference": "perturbation mean minus context-specific control mean",
        "output_review": "Verify gene-level numerical responses with named target/feature axes, normalization and expression scale against the actual adapter contract.",
    })


def estimate_coverage(benchmark: list[str] | np.ndarray,
                      supported: list[str] | None, *, missing_examples: int = 10) -> dict:
    """Exact, case-sensitive set overlap; unknown support is not zero coverage.

    Candidate lists must describe support jointly applicable to both contexts.
    Duplicates in candidate lists do not inflate coverage; no alias mapping occurs.
    """
    required = set(unique_identifiers(benchmark, "benchmark identifiers"))
    if supported is None:
        return {"n_benchmark": len(required), "n_supported": None,
                "coverage_fraction": None, "missing_examples": None}
    if not isinstance(supported, list) or any(not isinstance(v, str) or not v.strip() for v in supported):
        raise ValueError("Supported identifiers must be a list of nonempty strings or null.")
    present = required & set(supported)
    return {"n_benchmark": len(required), "n_supported": len(present),
            "coverage_fraction": len(present) / len(required),
            "missing_examples": sorted(required - present)[:missing_examples]}


def _all_known(values: list[bool | None]) -> bool | None:
    if False in values:
        return False
    return None if None in values else True


def _compute_feasibility(candidate: dict, hardware: dict) -> bool | None:
    """Compare stated inference requirements with explicit hardware; never guess hardware."""
    checks = []
    for requirement, capacity in (("approximate_ram", "ram_gb"),):
        need, available = candidate.get(requirement), hardware.get(capacity)
        checks.append(None if need is None or available is None else need <= available)
    gpu = candidate.get("gpu_required")
    if gpu is None:
        checks.append(None)
    elif gpu:
        available_gpu = hardware.get("gpu_available")
        checks.append(available_gpu)
        need, available = candidate.get("approximate_vram"), hardware.get("vram_gb")
        checks.append(None if need is None or available is None else need <= available)
    return _all_known(checks)


def assess_model(candidate: dict, benchmark: BenchmarkRequirements, *,
                 hardware: dict | None = None, min_target_coverage: float = 1.0,
                 min_feature_coverage: float = 1.0) -> dict:
    """Report ten dimensions and independent tri-state flags, without an opaque score.

    Scientific failures yield INCOMPATIBLE. Access, retraining, reproducibility,
    or resource barriers yield HIGH_RISK, not scientific incompatibility. READY
    requires every dimension to pass plus evidence sources and a known license.
    """
    candidate = dict(candidate)
    if not isinstance(candidate.get("name"), str) or not candidate["name"].strip():
        raise ValueError("Each candidate requires a nonempty name.")
    for key in BOOLEAN_FIELDS:
        if candidate.get(key) is not None and type(candidate[key]) is not bool:
            raise ValueError(f"{key} must be true, false or null.")
    for threshold in (min_target_coverage, min_feature_coverage):
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("Coverage thresholds must be between 0 and 1.")
    hardware = hardware or {}
    if hardware.get("gpu_available") is not None and type(hardware["gpu_available"]) is not bool:
        raise ValueError("hardware.gpu_available must be true, false or null.")
    for table, keys in ((candidate, ("approximate_ram", "approximate_vram")),
                        (hardware, ("ram_gb", "vram_gb"))):
        for key in keys:
            value = table.get(key)
            if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be a nonnegative number in GB or null.")
    declared = candidate.get("feasibility_status")
    if declared is not None and declared not in STATUSES:
        raise ValueError("Unknown declared feasibility_status.")
    targets = estimate_coverage(benchmark.targets, candidate.get("supported_perturbations"))
    features = estimate_coverage(benchmark.genes, candidate.get("output_gene_space"))
    dimensions = {}

    def record(name: str, value: bool | None, reason: str, *, conditional: bool = False) -> None:
        dimensions[name] = {"category": DIMENSIONS[name],
                            "status": "UNKNOWN" if value is None else ("FAIL" if not value else ("CONDITIONAL" if conditional else "PASS")),
                            "reason": reason}

    for dimension, field in (("perturbation_compatibility", "gene_perturbation_compatible"),
                             ("k562_compatibility", "k562_support"), ("rpe1_compatibility", "rpe1_support"),
                             ("context_fairness", "context_fairness")):
        record(dimension, candidate.get(field), f"Reviewed metadata: {field}={candidate.get(field)}")
    for name, coverage, minimum in (("target_coverage", targets, min_target_coverage),
                                    ("feature_coverage", features, min_feature_coverage)):
        fraction = coverage["coverage_fraction"]
        record(name, None if fraction is None else fraction >= minimum,
               f"Identifier coverage={fraction}; configured minimum={minimum}",
               conditional=fraction is not None and fraction < 1)
    download = candidate.get("prediction_download_available")
    infer = _all_known([candidate.get("inference_available"), candidate.get("checkpoint_available")])
    access = True if download is True or infer is True else (False if download is False and infer is False else None)
    retrain = candidate.get("training_required")
    if retrain is True:
        access = False
    record("prediction_accessibility", access, f"download={download}; checkpoint inference={infer}; full retraining={retrain}")
    record("compute_feasibility", _compute_feasibility(candidate, hardware),
           "Compare documented inference RAM/GPU/VRAM requirements with the configured hardware; unknown requirements stay unknown.")
    record("output_compatibility", _all_known([candidate.get("expression_output"), candidate.get("output_compatible"),
                                               candidate.get("gene_space_compatible")]),
           "Requires expression output, compatible gene identifiers and a reviewed response-contract mapping.")
    record("reproducibility", _all_known([candidate.get("reproducibility_documented"), candidate.get("preprocessing_documented")]),
           "Manual review of versioned code/checkpoint or prediction artifacts, configuration and preprocessing.")
    negate = lambda value: None if value is None else not value
    coverage_failed = lambda coverage, minimum: None if coverage["coverage_fraction"] is None else coverage["coverage_fraction"] < minimum
    contexts = [candidate.get("k562_support"), candidate.get("rpe1_support")]
    flags = {
        "no_gene_perturbation_prediction": negate(candidate.get("gene_perturbation_compatible")),
        "no_relevant_context_support": None if None in contexts and True not in contexts else not any(contexts),
        "missing_k562_support": negate(contexts[0]), "missing_rpe1_support": negate(contexts[1]),
        "no_expression_output": negate(candidate.get("expression_output")),
        "requires_full_retraining": retrain,
        "checkpoint_unavailable": negate(candidate.get("checkpoint_available")),
        "insufficient_target_coverage": coverage_failed(targets, min_target_coverage),
        "insufficient_feature_coverage": coverage_failed(features, min_feature_coverage),
        "incompatible_gene_space": negate(candidate.get("gene_space_compatible")),
        "preprocessing_unknown": negate(candidate.get("preprocessing_documented")),
        "prohibitively_large_compute": negate(_compute_feasibility(candidate, hardware)),
    }
    scientific_failures = [key for key, value in dimensions.items()
                           if value["category"] == "scientific_compatibility" and value["status"] == "FAIL"]
    evidence = candidate.get("evidence_sources")
    if evidence is not None and (not isinstance(evidence, list) or any(not isinstance(v, str) or not v.strip() for v in evidence)):
        raise ValueError("evidence_sources must be a list of source references or null.")
    states = [d["status"] for d in dimensions.values()]
    if scientific_failures:
        status = "INCOMPATIBLE"
    elif "FAIL" in states:
        status = "HIGH_RISK"
    elif "UNKNOWN" in states or not evidence or not candidate.get("license") or retrain is None:
        status = "NEEDS_INVESTIGATION"
    elif "CONDITIONAL" in states:
        status = "PROBABLY_FEASIBLE"
    else:
        status = "READY"
    return {"name": candidate["name"], "feasibility_status": status,
            "declared_status": declared, "dimensions": dimensions, "flags": flags,
            "target_coverage": targets, "feature_coverage": features,
            "scientific_failures": scientific_failures, "candidate_metadata": candidate,
            "evidence_sources": evidence, "hardware": hardware,
            "policy": {"min_target_coverage": min_target_coverage, "min_feature_coverage": min_feature_coverage}}


def run_feasibility_audit(registry_path: str | Path = "configs/virtual_models.yaml") -> pd.DataFrame:
    """Evaluate local registry metadata only and write separate audit reports."""
    path = Path(registry_path).resolve()
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), list):
        raise ValueError("Registry must contain a models list (empty is allowed).")
    if any(not isinstance(model, dict) for model in registry["models"]):
        raise ValueError("Every registry model must be a metadata mapping.")
    root = (path.parent / registry.get("project_root", ".")).resolve()
    benchmark = load_benchmark_requirements(root / registry["benchmark_directory"])
    names = [model.get("name") for model in registry["models"]]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")
    reports = [assess_model(model, benchmark, hardware=registry.get("hardware"),
                            **registry.get("policy", {})) for model in registry["models"]]
    rows = []
    for report in reports:
        row = {key: report[key] for key in ("name", "feasibility_status", "declared_status")}
        for name, result in report["dimensions"].items():
            row[name] = result["status"]
        for kind in ("target", "feature"):
            for key, value in report[f"{kind}_coverage"].items():
                row[f"{kind}_{key}"] = json.dumps(value) if isinstance(value, list) else value
        row.update({f"flag_{key}": value for key, value in report["flags"].items()})
        rows.append(row)
    columns = ["name", "feasibility_status", "declared_status", *DIMENSIONS]
    columns += [f"{kind}_{field}" for kind in ("target", "feature")
                for field in ("n_benchmark", "n_supported", "coverage_fraction", "missing_examples")]
    columns += [f"flag_{flag}" for flag in FLAGS]
    table = pd.DataFrame(rows, columns=columns)
    output = (root / registry["output_directory"]).resolve()
    if output == (root / registry["benchmark_directory"]).resolve():
        raise ValueError("Audit output directory must be separate from validated benchmark outputs.")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "model_feasibility.csv", index=False, na_rep="UNKNOWN")
    dimension_rows = [{"name": report["name"], "dimension": name, **dimension}
                      for report in reports for name, dimension in report["dimensions"].items()]
    pd.DataFrame(dimension_rows, columns=["name", "dimension", "category", "status", "reason"]).to_csv(
        output / "model_feasibility_dimensions.csv", index=False)
    content = {"benchmark_requirements": benchmark.metadata, "registry": str(path),
               "reports": reports, "note": "An empty registry means no real candidates have been assessed. Null flag values mean unknown, not false."}
    (output / "model_feasibility_report.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    return table


def main() -> None:
    """Thin local-metadata-only audit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/virtual_models.yaml")
    args = parser.parse_args()
    print(run_feasibility_audit(args.registry).to_string(index=False))


if __name__ == "__main__":
    main()
