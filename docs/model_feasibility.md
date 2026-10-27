# Model feasibility audit

Phase 0 is complete. This stage reduces project risk before committing to full
virtual-cell integrations: are enough candidates practically evaluable against
the existing K562/RPE1 benchmark? This is a feasibility gate, not a performance
benchmark. No real model has been researched or declared compatible by this task.

The framework reads only saved identifier metadata and a local human-edited
registry. It does not install packages, fetch repositories/artifacts, perform
inference, train models, load H5AD, or change existing benchmark logic/results.

## Registry and evidence

`configs/virtual_models.yaml` contains an empty `models` list and an ignored
`template` mapping. Copy the template into `models` to begin a real review.
All unresearched facts must remain YAML `null`, including boolean facts.
`false` means a verified negative finding; `[]` means a verified empty identifier
list. Do not write `unknown` as a string in boolean fields. Descriptive fields
preserve provenance but do not silently establish compatibility.

Record paper/repository references, license, proposed prediction-access route,
perturbation type, supported targets and contexts, input/output requirements,
pretrained/checkpoint availability, zero-shot/unseen-context claims, documented
inference resources, normalization/scale, limitations and notes. Cite versioned
sources and a review date in `evidence_sources`. Explicit boolean review conclusions
must be backed by that research. A repository URL alone never implies readiness.
The optional reviewer `feasibility_status` is preserved separately from the
computed status; it cannot override evidence gaps or failures.

Review both contexts under the same proposed evaluation setup. Identifier lists
in `supported_perturbations` and `output_gene_space` must apply to BOTH K562 and
RPE1; intersect context-specific support lists before entering them. Do not assume
a list of training genes establishes unseen-perturbation inference support.
Expected coverage estimates are descriptive only; actual coverage requires lists.

## Requirements and coverage

`load_benchmark_requirements` reads only `targets` and `genes` from the saved
`differential_responses.npz`; it never loads `X_delta`. When present,
`common_targets.csv` and `selectivity_summary.json` are checked for agreement.
The currently saved metadata specifies 1,440 targets and 7,226 features. These
numbers are read from saved outputs, not hard-coded as audit requirements.
Missing required artifacts or contradictory metadata cause a readable error.

`estimate_coverage` matches exact case-sensitive identifiers, independent of order.
It reports the benchmark count, number of overlapping supported identifiers,
fraction covered and up to ten missing examples. Candidate duplicates do not
inflate coverage. There is no automatic gene-alias conversion and no requirement
to supply prediction matrices. An unknown list returns null counts/fractions,
not zero. Extra candidate identifiers are harmless.

Coverage defaults to requiring the full unchanged benchmark. Explicitly lowering
registry policy thresholds permits a conditional feasibility result for a subset,
but does not change the benchmark or certify an equivalent comparison.

No virtual-benchmark adapter implementation was found in this checkout during
inspection. The explicit `output_compatible` review must check the actual proposed
adapter/NPZ contract, named target and gene axes, numerical gene-level expression
responses, control subtraction and expression scale. Merely predicting embeddings
or providing a gene list is insufficient evidence of output compatibility.

## Dimensions and decisions

| Dimension | Category |
| --- | --- |
| Perturbation compatibility | Scientific compatibility |
| K562 compatibility | Scientific compatibility |
| RPE1 compatibility | Scientific compatibility |
| Target coverage | Scientific compatibility |
| Expression-feature coverage | Scientific compatibility |
| Prediction accessibility | Data availability |
| Compute feasibility | Compute difficulty |
| Output compatibility | Scientific compatibility |
| Context fairness | Scientific compatibility |
| Reproducibility | Engineering difficulty |

Each dimension reports PASS, CONDITIONAL, FAIL or UNKNOWN with an explanation.
There is no weighted aggregate score. Hardware is explicitly configured as available
RAM, GPU availability and VRAM. Resource estimates are in GB for the proposed
inference or downloaded-output processing workload. Unknown hardware/requirements
stay unknown. GPU incompatibility or inadequate memory is a compute barrier.

Computed status follows these transparent rules:

- `INCOMPATIBLE`: at least one verified scientific failure under the stated gate,
  including insufficient coverage at configured thresholds.
- `HIGH_RISK`: a verified access, retraining, compute or reproducibility barrier.
  This does not mean the scientific task itself is incompatible.
- `NEEDS_INVESTIGATION`: unresolved dimensions, absent evidence references,
  unknown license or unknown retraining requirement.
- `PROBABLY_FEASIBLE`: all requirements are documented/met at the selected thresholds,
  but coverage is partial (explicitly permitted by policy).
- `READY`: all ten dimensions pass, full coverage, evidence references, a known
  license and no retraining requirement. This is a documented feasibility decision,
  not proof that an integration or performance evaluation has succeeded. License
  suitability for the intended use still needs human review.

Independently visible tri-state flags are:
`no_gene_perturbation_prediction`, `no_relevant_context_support`,
`missing_k562_support`, `missing_rpe1_support`, `no_expression_output`,
`requires_full_retraining`, `checkpoint_unavailable`, `insufficient_target_coverage`,
`insufficient_feature_coverage`, `incompatible_gene_space`, `preprocessing_unknown`,
and `prohibitively_large_compute`.

All flags are retained, even when multiple conditions hold. Flag true does not
necessarily veto an alternative route: a missing checkpoint can coexist with
downloadable predictions. `training_required` specifically means full retraining
is required for the proposed route, not that the original model was trained at
some point. Missing preprocessing documentation is a reproducibility/engineering
barrier rather than proof of scientific incompatibility. Null flags mean that the
relevant condition has not yet been determined.

## Usage and outputs

```python
from perturbselect.model_feasibility import run_feasibility_audit
summary = run_feasibility_audit("configs/virtual_models.yaml")
```

`make model-feasibility` runs the same local audit. `MODEL_REGISTRY` overrides the
registry path. The output directory defaults to `results/model_feasibility`:

- `model_feasibility.csv`: candidate comparison with dimensions, flags and coverage.
- `model_feasibility_dimensions.csv`: one explained dimension per candidate per row.
- `model_feasibility_report.json`: full candidate metadata, evidence, policies,
  hardware, independent flags and saved benchmark provenance.

With an empty registry, reports contain no candidate rows. CSV missing values are
written as `UNKNOWN`; JSON retains explicit null. No compatibility data is invented.
The reports are generated when the audit is run, not prepopulated with fictional
models. Original registry and saved benchmark files remain unchanged.

Use `tests/test_model_feasibility.py` in VS Code Testing first. Synthetic tests cover
ready/conditional/unknown candidates, independent simultaneous failures, both
coverage dimensions, downloads versus checkpoint access, retraining and compute
barriers, unknown handling, identifier alignment, and loading authoritative saved
metadata without a response matrix. No tests were executed by the coding agent.
