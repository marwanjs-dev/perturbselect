# Generic benchmarking layer

The internal `perturbselect.benchmarking` package provides a reusable data contract
for predictions, scores, paired statistics, tables, figures, and reports. It has no
knowledge of cell lines, assays, perturbations, or selectivity. The
`perturbselect.benchmark_adapters` module is the boundary that knows the saved
PerturbSelect column names and method labels. Existing analysis modules still
produce the validated scientific outputs.

```text
data/model adapters
    -> standardized predictions (one row per sample and method)
    -> generic evaluation (optional when scores are already saved)
    -> paired statistics and baseline comparisons (optional)
    -> standardized long metric results
    -> tables, figures, and Markdown reports
```

## Data contracts

A result row requires `task`, `dataset`, `method`, `split`, `metric`, `value`,
and `n_samples`. It can also carry `study`, `context`, `method_variant`,
`representation`, `baseline`, `fold`, `seed`, `n_features`, and arbitrary
metadata. The full identity is the named axes plus `metric`. Duplicate
identities fail validation. Numeric values must be finite; an undefined score
is allowed only as `NaN` with `value_status: undefined`. Counts are
nonnegative integers. An unknown feature count remains null.

A prediction row requires `sample_id`, `task`, `dataset`, `method`, `split`,
`y_true`, and `y_pred`. The same optional axes apply; the full identity uses
`sample_id` in place of `metric`. Truth and prediction must be finite. A
sample ID is an independent unit for paired resampling, so an adapter must
choose it according to the study design. Comparing methods requires exactly
the same sample IDs, truths, and fold assignments in one study/task/dataset/
context/split/seed slice. Missing or duplicated samples raise errors.

The metric registry has Spearman rho, Pearson r, R² (higher is better), MAE,
and RMSE (lower is better). Its implementations use the same SciPy and
scikit-learn primitives as the existing scientific evaluation and are checked
against it by synthetic tests. `evaluate_predictions` scores standardized
predictions without fitting models. A project with saved validated scores can
adapt them directly and skip evaluation.

`paired_bootstrap` resamples matched sample IDs once per replicate for both
methods. It reports the raw candidate-minus-baseline difference, percentile
interval, direction-adjusted improvement, seed, and requested/valid/invalid
replicate counts. Its `proportion_better` describes bootstrap support, **not**
a p-value. These intervals condition on fixed predictions; they do not include
model refitting uncertainty. `summarize_repeats` reports mean, median, sample
SD, IQR, and valid/invalid counts for explicit seed or fold observations.
`adjust_pvalues` applies Benjamini–Hochberg correction only to supplied
p-values in a caller-defined family; it never interprets bootstrap support as
a p-value.

## Presentation and files

Plot functions accept saved standardized tables and return Matplotlib
`Figure, Axes`. Available views are model comparison with supplied intervals,
baseline forest plot, seed/fold stability, coverage, performance versus
coverage, correlation between two stored metrics, and ranked methods. They do
not evaluate or fit models or call `show()`. `save_figure` supports SVG, PNG,
and PDF. Table builders return numeric pandas DataFrames for performance,
baseline comparisons, stability, coverage, and dataset/task summary. CSV
keeps numeric precision; Markdown and LaTeX round only display copies.

`ArtifactLayout` reserves `results/<study>/{raw,summaries,statistics}`,
`figures/<study>/{manuscript,supplementary}`, and
`tables/<study>/{manuscript,supplementary}`. The report is Markdown and links
to generated artifacts. Provenance includes a timestamp, available Git
commit, hashed configuration and source artifacts, Python and package
versions, and seeds. Missing Git metadata is recorded as unavailable.

The presentation YAML may set shown metrics, method order, baseline, display
labels, figure formats, and display precision. It cannot set an endpoint,
cross-validation scheme, bootstrap count, or other statistical decisions.
`configs/benchmark_presentation.yaml` is the PerturbSelect presentation
example. `make benchmark-export` reads the existing selectivity files and
writes new standardized raw tables. `make benchmark-report` also creates
tables, figures, and a linked report. `SELECTIVITY_RESULTS` and
`PRESENTATION_CONFIG` select their inputs. These commands do not run either
context pipeline, refit models, or replace files under `results/selectivity`.
Matplotlib is provided by the optional `presentation` or `test` dependency.

The adapter maps saved `model` to `method`, saved `endpoint` to `task`,
`gene` to `sample_id`, and `functional_selectivity`/`pred_*` to `y_true`/
`y_pred`. It records `split=out_of_fold`. The seven saved model rows become
long metric rows; the saved incremental bootstrap and seed stability files
become separate statistics files when present. Stored metric and prediction
numbers are copied at round-trip CSV precision. Fold IDs are not present in
the wide saved predictions and remain unknown. The existing source files stay
in their original locations. Later model adapters can emit these same two
tables without adding presentation code.

## Unrelated synthetic example

`examples/synthetic_benchmark.py` demonstrates simulated sensor calibration.
It generates 60 named sensor samples, two fixed prediction sets, long metric
results, matched paired comparisons, provenance, and a Markdown report. It
imports only the generic benchmarking API, NumPy, and pandas. It can be called
from an IDE or notebook after selecting an environment with this package:

```python
from examples.synthetic_benchmark import main
report_path = main("/path/to/a/writable/demo")
```

The example writes to a separate `sensor_demo` study layout. It does not use
biological data or reproduce a scientific analysis. The small synthetic tests
in `tests/benchmarking` are discoverable in VS Code Testing and require no
H5AD files.
