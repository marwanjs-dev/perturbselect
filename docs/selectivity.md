# Functional selectivity from saved context responses

The context-specific analyses have passed according to the completed K562/RPE1
runs. This next experiment tests whether differential RNA responses predict
differential functional dependency. No selectivity success is claimed before
examining its results.

## Inputs and common universe

The default `configs/selectivity.yaml` explicitly uses the available saved K562
notebook artifacts in `results/phase0` and RPE1 pipeline artifacts in `results/rpe1`.
`results/k562` was empty when this extension was implemented. No automatic fallback
or rerun of single-cell processing occurs. To use newly saved K562 pipeline outputs,
change the three K562 paths and its fitness column from `k562_gamma` to `fitness`,
and update the provenance file list in the selectivity config.

Inspection of both saved NPZ target lists and both gene fitness tables found
**1,440 common targets**. Both NPZ files contain **7,226 expression features**.
The code computes this intersection again each run; it never fixes the target
count at 1,440. Rows use sorted common gene identifiers. Features retain the K562
NPZ order and RPE1 columns are explicitly reindexed by name. Unequal feature sets,
duplicate names, missing metadata, nonfinite values, or inconsistent NPZ/metadata
cell counts raise errors. Exact identifiers are used without alias mapping.

The K562 notebook NPZ has `X`, `targets`, `genes`. Its cell counts are read from
`k562_response_metadata.csv` by gene. The RPE1 NPZ additionally has `n_cells` and
`n_controls`; its counts are cross-checked against `response_metadata.csv`.
NPZ files are opened with `allow_pickle=False`. No H5AD input is needed.

## Endpoint and interpretation

For the final common target universe only:

```text
k562_dependency = -k562_raw_fitness       # CRISPRi gamma
rpe1_dependency =  rpe1_raw_fitness       # CRISPR-KO BAGEL Bayes Factor

functional_selectivity = percentile(k562_dependency) - percentile(rpe1_dependency)
```

Percentiles use `pandas.rank(method="average", pct=True)` independently in each
context on the same genes. Positive values indicate stronger relative dependency
in K562; negative values indicate stronger relative dependency in RPE1. Ties receive
average ranks. Raw assay scores are never subtracted from one another.

The secondary endpoint is `z(k562_dependency) - z(rpe1_dependency)`, using population
standard deviations (`ddof=0`) within that same common universe. Constant dependency
vectors make z-scores undefined and raise a clear error. Rank selectivity remains
the primary endpoint. CRISPRi gamma and CRISPR-KO BAGEL measure different aspects
of dependency, and this contrast describes relative positions in these assays,
not an absolute growth-effect difference.

## Features and models

Both inputs already represent perturbation mean minus their context's control
mean. `X_delta = X_k562 - X_rpe1` therefore compares perturbational responses, not
baseline expression. The secondary joint matrix concatenates K562 then RPE1
response columns. Each modeling row is one target gene.

Primary-endpoint OOF models are:

- Mean baseline.
- Cell-count baseline using both `log1p(k562_n_cells)` and `log1p(rpe1_n_cells)`.
- Difference in context mean absolute RNA response magnitudes.
- **K562 RNA only** and **RPE1 RNA only**: single-context ablations predicting
  the same selectivity endpoint, not their respective context fitness endpoints.
- **Differential RNA (K562 - RPE1)**: primary PCA + Ridge model.
- **Joint RNA (K562 + RPE1)**: secondary PCA + Ridge model.

All reuse existing model factories and OOF evaluation. Scaling and randomized PCA
stay inside each training-fold pipeline. Defaults are 100 PCs, the existing 17
RidgeCV alphas, five shuffled folds, seed 42, and `n_jobs=1`. Input NPZ precision is
preserved (float32 in the existing saved outputs). All four RNA models use the
same common targets in the same order, the same rank endpoint, and identical fold
assignments and PCA + Ridge settings. Compare them in
`selectivity_context_ablation_metrics.csv`, also included in the full model metrics.
Differential or joint performance must be compared with both single-context models
before attributing predictive information specifically to cross-context comparison.

Robustness masks the target-gene feature in **both** contexts using the existing
masking function before subtraction. Stability seeds 11/22/33/44/55 and the single
seed-42 masked permutation control use this masked differential model with the primary
rank endpoint; PCA's seed stays fixed. An additional unmasked control is labeled
`Differential RNA (shuffled selectivity)`. Both controls use identical shuffled
labels and folds, and score against the shuffled endpoint. Predictions retain
`pred_shuffled` for the masked control and add `pred_shuffled_differential` for the
unmasked control. A separate unmasked differential model predicts the secondary
z-score endpoint. MAE/RMSE across these endpoints use different units and should
not be directly compared. A single shuffled-label run is a control, not an
empirical permutation p-value.

Additional Spearman associations report magnitude difference, raw cell-count
difference, log-cell-count difference, and the z-score endpoint against primary
functional selectivity.

## API and Makefile

```python
from perturbselect.selectivity_pipeline import prepare_selectivity, run_selectivity_pipeline

prepared = prepare_selectivity("configs/selectivity.yaml")  # no model fitting or output writes
result = run_selectivity_pipeline("configs/selectivity.yaml")
```

Use `../configs/selectivity.yaml` from a notebook in `notebooks/`, or an absolute
path. The same project-root-relative YAML path convention as context configs is
used. The dedicated selectivity config loader leaves context config handling unchanged.

Makefile targets:

| Target | Action |
| --- | --- |
| `make selectivity-config-check` | Validate config structure and resolve paths |
| `make selectivity-audit` | Read saved arrays/tables, align them, report common count and endpoint definition |
| `make run-selectivity` | Run all models and robustness checks, save outputs |
| `make test-selectivity` | Run synthetic selectivity tests |
| `make test-selectivity-integration` | Opt into checks using saved context outputs |

`SELECTIVITY_CONFIG` overrides the selectivity YAML path; `PYTHON` and `TEST_ARGS`
work as in existing targets. Analysis targets require the package to be installed
in the chosen environment. They never invoke context-pipeline aggregation.

## Outputs and provenance

The default output directory is `results/selectivity`. A completed run writes:

- `common_targets.csv`
- `functional_selectivity.csv` (raw phenotypes, oriented dependencies, percentiles,
  z-scores, both selectivity endpoints, cell counts, magnitudes and differences)
- `selectivity_model_metrics.csv`
- `selectivity_context_ablation_metrics.csv` (K562 only, RPE1 only, differential, joint)
- `selectivity_predictions.csv` (named genes, both endpoints, all OOF predictions,
  shuffled endpoint and its predictions)
- `selectivity_cv_stability.csv`
- `selectivity_permutation_metrics.csv`
- `selectivity_robustness_metrics.csv`
- `selectivity_associations.csv`
- `differential_responses.npz` (`X_delta`, `targets`, `genes`)
- `selectivity_summary.json`
- `resolved_config.json`

Summary metadata includes the exact computed common count, definitions, ordering,
assays, model-input precision, library versions, creation time, source paths and
SHA-256 hashes. Configured source JSON provenance is embedded as well. Existing
files with these output names are replaced on rerun; context artifacts are not
modified. Differential NPZ remains gitignored like other NPZ files.

## Incremental context diagnostics

K562-only prediction is already strong (Spearman about 0.366). Joint and
differential RNA add modest observed performance. The lightweight diagnostic
entry point uses the existing `functional_selectivity.csv` and
`selectivity_predictions.csv`; it never loads H5AD data or fits models.

```python
from perturbselect.selectivity_diagnostics import run_selectivity_diagnostics
tables = run_selectivity_diagnostics("results/selectivity")
```

`make selectivity-diagnostics` runs the same analysis; `SELECTIVITY_RESULTS`
can select another directory. `make test-selectivity-diagnostics` is available,
and these tests also belong to the existing synthetic selectivity test target.
For VS Code Testing, select `tests/test_selectivity_diagnostics.py`.

Both comparisons (joint versus K562-only, differential versus K562-only) use
5,000 paired resamples of genes with replacement, seed 42. Each sampled index
vector is shared across the compared models; the same draws also apply across
comparisons and metrics. Saved OOF predictions are fixed. Spearman ranks are
recomputed in each resample. Undefined correlations and constant-outcome R²
replicates are excluded per metric, and valid/invalid sample counts are recorded.
The reported 95% interval is the 2.5th–97.5th percentile of paired metric deltas.
The proportion above zero is descriptive bootstrap support, not a p-value.
These intervals condition on the existing OOF fits; they do not capture
retraining/fold uncertainty or dependence induced by overlapping training folds.

Endpoint decomposition reports Spearman and Pearson associations for the two
percentile components and their difference, plus clearly labeled secondary
associations with each oriented continuous dependency. It never subtracts raw
assay scores. Component spread uses sample standard deviation (`ddof=1`).
Strong endpoint–component correlations are partly mathematical: the endpoint
contains both components. Since percentile ranks have nearly equal spread unless
ties differ substantially, inspect both spreads and the between-context
association before attributing endpoint variation to one context.

Only new diagnostic outputs are written (replaced on diagnostic reruns):

- `selectivity_incremental_bootstrap.csv`
- `selectivity_endpoint_decomposition.csv`
- `selectivity_endpoint_summary.csv`
- `selectivity_diagnostics_metadata.json` (seed, sample count, interpretation and input hashes)

Original endpoints, predictions and model outputs remain unchanged. Statistical
support should be judged from the generated intervals, not assumed from the
positive observed differences.

Observed diagnostics on the saved 1,440-gene run (5,000 valid resamples for each
comparison/metric, seed 42):

| Comparison against K562 only | Metric | Observed delta | 95% paired interval | Proportion > 0 |
| --- | --- | ---: | --- | ---: |
| Joint RNA | Spearman | 0.016905 | [-0.012431, 0.045666] | 0.8694 |
| Joint RNA | R² | 0.032346 | [0.013530, 0.050359] | 0.9998 |
| Differential RNA | Spearman | 0.002528 | [-0.035100, 0.038096] | 0.5522 |
| Differential RNA | R² | 0.021691 | [-0.001967, 0.045012] | 0.9652 |

Only joint RNA's R² gain excludes zero at this interval level. The two Spearman
gains and differential R² gain remain uncertain. Thus these diagnostics support
a modest joint-model R² improvement conditional on the saved fits, rather than
a broad claim that context comparison improves every performance measure.
Both percentile components have sample SD about 0.288775. Selectivity's Spearman
correlations with K562 and RPE1 percentiles are 0.579184 and -0.571122, respectively;
between-context percentile Spearman is 0.311499. This gives no indication that
one percentile component numerically dominates the endpoint. It does not establish
equal biological contributions or identical assay reliability.

CSV inputs are parsed with round-trip floating-point precision to preserve rank
ties and reproduce the existing observed metrics. The diagnostic module was run
successfully; the new pytest tests remain unexecuted by the coding agent.

## Tests and scientific acceptance

In VS Code Testing, run `tests/test_selectivity.py` first, then
`tests/test_selectivity_pipeline.py`. These cover shuffled axes and metadata,
four-way intersection, ranking only after intersection, ties, dependency signs,
z-score convention, exact delta/joint matrices, masking, legacy NPZ support,
validation errors, synthetic end-to-end output and provenance.

The opt-in heavy test is `tests/integration/test_selectivity_saved_outputs.py`.
It uses saved artifacts only and writes to pytest's temporary directory. It checks
alignment, finite values, the shared feature count, both endpoint signs, and a
computed rather than assumed target count. A separate scientific acceptance test
requires absolute shuffled Spearman below 0.15, real masked Spearman to exceed
absolute shuffled Spearman by more than 0.10, and real R² to exceed shuffled R².
Both masked and unmasked controls must have absolute Spearman below 0.15 and R² ≤ 0.
These are proposed checks before seeing the new results, not established reference
performance or formal significance thresholds. If they fail, inspect the results;
do not weaken them simply to claim successful selectivity prediction.

The coding agent inspected saved artifact identifiers but did not execute the new
pipeline or tests, following the user's instruction. Numerical selectivity results
are pending execution.
