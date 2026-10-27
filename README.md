# PerturbSelect

PerturbSelect studies whether perturbation transcriptomic responses preserve
functional information that can identify context-selective interventions.
The current Phase-0 pipeline predicts **measured functional fitness from measured
perturbation RNA responses**, with one gene perturbation per prediction unit.

The validated claim is: **Measured K562 perturbation transcriptomic responses
contain reproducible information about independent K562 functional growth
phenotypes.** Expression and fitness come from independent experiments, so this
is orthogonal validation rather than a perfectly matched assay. Virtual-cell
models have not yet been benchmarked, and therapeutic selectivity has not been
demonstrated.

Context validation is complete (reported successful context pipeline runs):
K562 measured RNA → K562 functional fitness **PASS**; RPE1 measured RNA → RPE1
functional fitness **PASS**. The measured-RNA functional-selectivity analysis and
its incremental-information diagnostics complete Phase 0; the detailed results
and their limitations are documented in the selectivity guide.

Phase 0 is complete. The current development stage is a **model feasibility audit**
before investing in virtual-cell integrations. The human-editable registry is
`configs/virtual_models.yaml`; it intentionally contains no researched candidates
yet. `make model-feasibility` reports separate scientific, data-access, engineering
and compute findings using saved benchmark identifiers. It does not run or install
virtual-cell models. See [the audit guide](docs/model_feasibility.md) for evidence
requirements, decision rules and outputs. No real model compatibility is claimed.

The internal [generic benchmarking guide](docs/benchmarking.md) describes the
standardized prediction/result schemas, paired statistics, presentation exports,
and the thin adapter for existing selectivity results. The adapter reads saved
outputs without rerunning the validated analysis. `make benchmark-export` writes
canonical tables, and `make benchmark-report` adds configured figures, tables,
and a linked Markdown report. Its synthetic tests are in `tests/benchmarking`
for VS Code Testing.

## Structure

- `src/perturbselect/io.py`: backed expression/table loading and schema checks.
- `fitness.py`: replicate aggregation, construct collapse and target alignment.
- `signatures.py`: chunked mean response construction and target-feature masking.
- `qc.py`: magnitude, associations, cell-count sensitivity and residualization.
- `modeling.py`: fold-safe benchmarks, CV stability and permutation control.
- `config.py`, `pipeline.py`: validated configuration and numerical orchestration.
- `selectivity.py`, `selectivity_pipeline.py`: named cross-context alignment,
  rank-based selectivity endpoints, and modeling from saved response artifacts.
- `configs/k562.yaml`: notebook-compatible K562 analysis.
- `configs/example.yaml`: template for another expression/functional assay pair.
- `configs/selectivity.yaml`: explicit saved K562/RPE1 inputs and selectivity settings.
- `notebooks/`: original exploratory audit and validated K562 analysis.
- `results/phase0`: preserved notebook reference tables.
- `tests/`: fast synthetic tests plus opt-in local-data regressions.

## Callable workflow

Use a notebook or the VS Code Python interface with the project environment:

```python
from perturbselect.pipeline import run_pipeline

result = run_pipeline("configs/k562.yaml")
result.tables["model_metrics"]
```

The config argument may also be a dictionary or the result of `load_config`.
For a notebook in `notebooks/`, use `../configs/k562.yaml` or an absolute path.
Relative paths inside YAML use `project_root`, itself relative to the YAML file.
Without that field, the YAML directory is used. For dictionaries, the default
root is the caller's current directory.

`pyproject.toml` defines an installable `src` package and runtime dependencies.
PyYAML is already in the original environment snapshot (`requirements.txt`). The
`test` extra declares pytest; the optional `excel` extra supplies Excel readers.
Pytest's configured `pythonpath` permits VS Code discovery without a custom
`sys.path` modification or editable installation. Notebook/API usage requires
the package to be available in the selected Python environment.

## Optional Makefile shortcuts

The root `Makefile` provides optional command shortcuts alongside the callable API
and VS Code Testing panel. It requires **GNU Make** on your PATH, including when
launched from PowerShell; it is not an NMake file. It prefers the existing
`.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Unix, falling back
to `python`. Override `PYTHON` to choose another interpreter.

Run these from the repository root:

```text
make help
make install-dev
make test
make config-check
make run
```

`install-dev` installs the package in editable mode and pytest into the selected
Python environment. `install` installs only the package's runtime requirements;
`install-excel` also includes optional Excel readers. Installation is explicit:
test and pipeline targets never install dependencies automatically.

`make test` and `make test-unit` exclude integration checks. Heavy checks require
an explicit `make test-integration`, `make test-k562`, or `make test-all`; these
targets set the integration opt-in for their own Python process, without editing
your VS Code environment. Missing local data still causes the relevant tests to
skip. For example, `make test TEST_ARGS="-q -x"` passes options to pytest.

`make config-check` validates configuration structure and resolves paths without
opening datasets or running analysis. `make run` executes the complete pipeline;
both default to `configs/k562.yaml`. Choose another configuration with
`make run CONFIG="configs/my_dataset.yaml"`. Pipeline runs replace the named
output files in the configured output directory. The API targets require the
package to be installed in the selected interpreter first.

For the configured RPE1 dataset, use `make run-rpe1`. This calls the same pipeline
with `configs/rpe1.yaml` and saves outputs to `results/rpe1`. It uses the configured
Hart/ORCS functional scores, saved RPE1 target order, and shared K562/RPE1 expression
features. To inspect that configuration first, use
`make config-check CONFIG="configs/rpe1.yaml"`.

## Current experiment: functional selectivity

Use `make selectivity-audit` to inspect the aligned saved inputs and
`make run-selectivity` to run the new experiment. These use
`configs/selectivity.yaml`, independently configurable through `SELECTIVITY_CONFIG`.
`make selectivity-config-check`, `make test-selectivity`, and
`make test-selectivity-integration` provide the related configuration and test targets.

Saved-artifact inspection found **1,440 common targets** with **7,226 RNA features**.
The config explicitly uses the available K562 notebook artifacts in `results/phase0`
and the RPE1 outputs in `results/rpe1`; no context pipeline is rerun. Target rows and
feature columns are aligned by gene identity, and the common target count is
recomputed each run.

The primary endpoint is `percentile(-K562 gamma) - percentile(RPE1 BAGEL)`, with
average-tie percentile ranks computed independently within that same common target
universe. Positive values mean stronger relative K562 dependency. Raw assay scores
are never subtracted. Population z-score differences (`ddof=0`) form a secondary
sensitivity endpoint; CRISPRi gamma and CRISPR-KO BAGEL remain distinct assays.

Differential RNA is the primary model; concatenated joint RNA is secondary. Both
reuse the existing fold-safe PCA + Ridge estimator. Technical baselines include
both context cell counts and the RNA magnitude difference. Robustness covers
target masking in both contexts, CV seeds, shuffled labels, and the z-score endpoint.

The same run also fits **K562 RNA only** and **RPE1 RNA only** to the same rank
selectivity endpoint with identical targets, folds and PCA + Ridge settings.
`selectivity_context_ablation_metrics.csv` contains these two ablations beside
differential and joint RNA. All seven benchmark rows remain in
`selectivity_model_metrics.csv`; their OOF predictions are in
`selectivity_predictions.csv`. `selectivity_permutation_metrics.csv` contains both
masked and unmasked differential controls. The existing `make run-selectivity`
command regenerates these outputs; previously saved results do not automatically
gain the newly added models.

`make selectivity-diagnostics` reads the saved OOF predictions and endpoints to
produce paired incremental-performance bootstrap intervals and endpoint
decomposition. It does not refit models or replace validated analysis outputs.
See [Incremental context diagnostics](docs/selectivity.md#incremental-context-diagnostics).
Synthetic coverage is in `tests/test_selectivity_diagnostics.py` for VS Code Testing.

Outputs go to `results/selectivity` and include aligned targets, both endpoints,
OOF predictions, model/robustness metrics, differential-response NPZ and source
provenance. See [the selectivity guide](docs/selectivity.md) for exact filenames,
assumptions and API usage. In VS Code Testing, start with `test_selectivity.py`,
then `test_selectivity_pipeline.py`; run the opt-in
`integration/test_selectivity_saved_outputs.py` afterward. The new selectivity
pipeline and tests have not been executed by the coding agent.

## Scientific and configuration choices

Input `.h5ad` expression must already be normalized. The pipeline never
renormalizes it. Each response is perturbation mean minus control mean; this is
a mean response profile, not raw-count pseudobulk. Disk-backed contiguous chunks
are accumulated into float64 group sums. QC uses float64 responses, while the
modeling matrix and saved NPZ use float32, matching notebooks 03 and 04.

The K562 config explicitly preserves two reference choices:

1. Targets and their order come from the identifier column of
   `results/phase0/k562_gene_fitness.csv`. Phenotypes are rebuilt from the raw
   screen: complete dual-guide replicates, their mean, removal of `non-targeting`,
   then the median across constructs. This preserves notebook row order and seeded
   folds. The original order came from sorting the pilot fitness table; it is
   frozen for regression reproduction, not recomputed for new datasets.
2. Features are the intersection with the RPE1 `.h5ad`, retaining K562 feature
   order: 7,226 features, not all 8,563 K562 genes. Both expression files are needed,
   but only the secondary dataset's feature metadata is used.

The K562 raw screen contains 18,902 unique gene labels (`raw_unique_genes`).
After requiring complete dual-guide replicates, 18,728 remain
(`retained_construct_unique_genes`, before removing `non-targeting` or intersecting
targets). These are distinct stages of notebook 02's audit; the final pilot has
1,641 genes.

For other datasets, omit `targets.path` to use the sorted intersection of fitness
and expression targets, or supply an ordered CSV/TSV identifier column. Explicit
target lists must all be available; missing targets raise errors. `features.mode`
must explicitly be `all`, `intersection` (secondary `.h5ad` path), or `list`
(CSV/TSV path and optional `column`, default `gene`). Lists preserve their own order
and reject unavailable/duplicate identifiers. Fitness and expression row identities
are checked before modeling.

Phenotype columns, aggregation rules, controls and assay semantics are configurable.
Raw phenotype signs and scales are preserved. In K562, more negative gamma means
stronger growth impairment. `configs/rpe1.yaml` uses the Hart/ORCS table's
`OFFICIAL_SYMBOL` identifiers and `28-SCORE.1` phenotype column, preserving its
raw score direction as documented in the configuration. Use the example template
for additional datasets and assays.

Benchmarks use shuffled five-fold KFold (seed 42): training-fold mean,
log1p cell count plus RidgeCV, RNA magnitude plus RidgeCV, and StandardScaler →
randomized PCA (100 components, seed 42) → RidgeCV. Scaling, PCA and RidgeCV
are inside training-fold pipelines. PCA components must fit the smallest training
fold; the implementation raises an error rather than silently reducing them.
Splits are at the perturbation level, never the cell level. Stability varies only
the fold seed, leaving PCA settings fixed. Permutation uses `default_rng(seed)`
and scores against shuffled labels. Cell-count residualization is descriptive QC.
Constant inputs or fewer than two observations produce NaN correlations;
nonfinite observations are rejected rather than silently discarded.

## Saved outputs

The default K562 destination is `results/k562`, leaving notebook references intact.
Each run replaces named output files in its configured destination after analysis:

- `dataset_summary.json`, `resolved_config.json`: dimensions, counts, precision,
  phenotype semantics, selection sources and resolved configuration.
- `gene_fitness.csv`: aligned gene phenotypes, retaining configured column names.
- `response_metadata.csv`: gene, cell count, magnitude and adjusted magnitude.
- `qc_associations.csv`, `qc_cell_count_sensitivity.csv`: effects, p-values and counts.
- `model_metrics.csv`, `target_masked_metrics.csv`, `cv_stability.csv`,
  `permutation_metrics.csv`: numerical predictive evaluations.
- `predictions.csv`: gene, observed fitness, cell counts, all baseline predictions,
  target-masked predictions, shuffled fitness and shuffled predictions.
- `measured_responses.npz`: float32 `X`, ordered `targets`, ordered `genes`,
  `n_cells`, and `n_controls`; NPZ files remain gitignored.

No plots are generated automatically. Backed expression handles close on normal
completion and analysis failures. Returned `result.responses.X` retains the
float64 QC matrix; the saved modeling matrix intentionally uses float32.

## Validation through VS Code Testing

Select the project Python environment, enable pytest through **Python: Configure
Tests**, choose `tests`, and refresh the Testing panel. No terminal test commands
are required.

Run `test_qc.py`, `test_signatures.py`, `test_modeling.py`, and `test_config.py`
first, then `test_pipeline.py`. These use small synthetic arrays or temporary
backed h5ad files. Ordinary fitness and IO tests also avoid local K562 data,
except for explicitly marked integration cases.

All `integration` tests are discoverable but skipped by default, including the
older K562 IO/fitness tests. To opt in, set `PERTURBSELECT_RUN_INTEGRATION=1` in
the environment file used by the VS Code Python extension (`python.envFile`,
normally the workspace `.env`). Refresh tests and select integration tests in the
Testing panel. Remove that setting to return to the fast default. No existing
workspace editor settings are replaced.

`tests/integration/test_k562_pipeline.py` runs the full pipeline once per module,
including stability fits, and writes to pytest's temporary directory. It can take
substantial time and needs both local expression files and the raw K562 screen.
It checks 308,646 cells, 10,691 controls, 1,641 targets, a (1,641, 7,226) matrix,
30/130/992 min/median/max perturbation cells, mean absolute response near
0.03653514, QC associations, and predictive performance. Notebook reference
PCA+Ridge Spearman is 0.700177 and R² 0.407387; target-masked Spearman is 0.699898,
and shuffled-label Spearman is 0.043326. Tolerances permit small numerical-library
differences, not material scientific changes. Investigate ordering, precision,
features and library versions before changing tolerances after any failure.

The refactor's tests are authored but have not been executed by the coding agent,
as requested. Reference metrics above describe the validated notebooks, not a
newly executed pipeline run.
