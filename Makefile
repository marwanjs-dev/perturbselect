# GNU Make shortcuts. Run from the repository root.
# Prefer the existing virtual environment; override with PYTHON=/path/to/python.
.DEFAULT_GOAL := help

ifneq ($(wildcard .venv/Scripts/python.exe),)
PYTHON ?= .venv/Scripts/python.exe
else ifneq ($(wildcard .venv/bin/python),)
PYTHON ?= .venv/bin/python
else
PYTHON ?= python
endif

CONFIG ?= configs/k562.yaml
SELECTIVITY_CONFIG ?= configs/selectivity.yaml
SELECTIVITY_RESULTS ?= results/selectivity
MODEL_REGISTRY ?= configs/virtual_models.yaml
PRESENTATION_CONFIG ?= configs/benchmark_presentation.yaml
TEST_ARGS ?=

.PHONY: help install install-dev install-excel test test-unit test-integration test-k562 test-rpe1 test-all config-check run run-rpe1
.PHONY: run-selectivity selectivity-audit selectivity-config-check test-selectivity test-selectivity-integration
.PHONY: selectivity-diagnostics test-selectivity-diagnostics
.PHONY: model-feasibility test-model-feasibility
.PHONY: benchmark-export benchmark-report test-benchmarking

help:
	@echo PerturbSelect commands - GNU Make
	@echo   make install           Install the package in editable mode
	@echo   make install-dev       Install the editable package and pytest
	@echo   make install-excel     Install the editable package and Excel readers
	@echo   make test              Run fast synthetic tests only
	@echo   make test-unit         Alias for make test
	@echo   make test-integration  Opt in to all local-data regression checks
	@echo   make test-k562         Opt in to the full K562 pipeline regression
	@echo   make test-rpe1         Opt in to the full RPE1 pipeline regression
	@echo   make test-all          Run both unit and integration tests
	@echo   make config-check      Validate CONFIG without running the pipeline
	@echo   make run               Run the pipeline with CONFIG
	@echo   make run-rpe1          Run configs/rpe1.yaml and save its configured outputs
	@echo   make run-selectivity   Run selectivity, single-context ablations and permutation controls
	@echo   make selectivity-diagnostics Paired bootstrap and endpoint decomposition from saved predictions
	@echo   make test-selectivity-diagnostics Run synthetic diagnostic tests
	@echo   make model-feasibility Audit local model metadata and identifier coverage only
	@echo   make test-model-feasibility Run synthetic feasibility tests
	@echo   make benchmark-export Export saved selectivity outputs to canonical tables
	@echo   make benchmark-report Export canonical tables and generate presentation artifacts
	@echo   make test-benchmarking Run the synthetic generic benchmarking tests
	@echo   make selectivity-audit Align saved inputs and report the common universe, without model fitting
	@echo   make selectivity-config-check Validate SELECTIVITY_CONFIG without loading arrays
	@echo   make test-selectivity  Run synthetic selectivity tests
	@echo   make test-selectivity-integration Opt in to selectivity checks on saved context data
	@echo Variables: PYTHON, CONFIG, SELECTIVITY_CONFIG, SELECTIVITY_RESULTS, MODEL_REGISTRY, PRESENTATION_CONFIG, TEST_ARGS
	@echo Default CONFIG: configs/k562.yaml
	@echo Pipeline and integration targets may take substantial time.

install:
	"$(PYTHON)" -m pip install -e .

install-dev:
	"$(PYTHON)" -m pip install -e ".[test]"

install-excel:
	"$(PYTHON)" -m pip install -e ".[excel]"

# Explicitly exclude integration tests even if opt-in is set in the environment.
test: test-unit

test-unit:
	"$(PYTHON)" -m pytest tests -m "not integration" $(TEST_ARGS)

# Set opt-in inside Python for compatibility with Windows and POSIX shells.
# Propagate pytest's exit status so Make reports failures correctly.
test-integration:
	"$(PYTHON)" -c "import os, sys, pytest; os.environ['PERTURBSELECT_RUN_INTEGRATION'] = '1'; sys.exit(pytest.main(sys.argv[1:]))" tests -m integration $(TEST_ARGS)

test-k562:
	"$(PYTHON)" -c "import os, sys, pytest; os.environ['PERTURBSELECT_RUN_INTEGRATION'] = '1'; sys.exit(pytest.main(sys.argv[1:]))" tests/integration/test_k562_pipeline.py $(TEST_ARGS)

test-rpe1:
	"$(PYTHON)" -c "import os, sys, pytest; os.environ['PERTURBSELECT_RUN_INTEGRATION'] = '1'; sys.exit(pytest.main(sys.argv[1:]))" tests/integration/test_rpe1_pipeline.py $(TEST_ARGS)

test-all:
	"$(PYTHON)" -c "import os, sys, pytest; os.environ['PERTURBSELECT_RUN_INTEGRATION'] = '1'; sys.exit(pytest.main(sys.argv[1:]))" tests $(TEST_ARGS)

# These API targets require the package to be installed in the chosen interpreter.
# Pass CONFIG as an argument rather than interpolating it into Python source.
config-check:
	"$(PYTHON)" -c "import json, sys; from perturbselect.config import load_config; print(json.dumps(load_config(sys.argv[1]), indent=2, default=str))" "$(CONFIG)"

run:
	"$(PYTHON)" -c "import sys; from perturbselect.pipeline import run_pipeline; result = run_pipeline(sys.argv[1]); print(result.tables['model_metrics'].to_string(index=False)); print('Outputs:', result.output_directory)" "$(CONFIG)"

# Reuse the generic runner while explicitly selecting the RPE1 configuration.
run-rpe1:
	$(MAKE) run CONFIG="configs/rpe1.yaml"

# Saved-output selectivity analysis: never reruns either context pipeline.
run-selectivity:
	"$(PYTHON)" -m perturbselect.selectivity_pipeline run --config "$(SELECTIVITY_CONFIG)"

selectivity-audit:
	"$(PYTHON)" -m perturbselect.selectivity_pipeline audit --config "$(SELECTIVITY_CONFIG)"

selectivity-config-check:
	"$(PYTHON)" -m perturbselect.selectivity_pipeline check-config --config "$(SELECTIVITY_CONFIG)"

test-selectivity:
	"$(PYTHON)" -m pytest tests/test_selectivity.py tests/test_selectivity_pipeline.py tests/test_selectivity_diagnostics.py -m "not integration" $(TEST_ARGS)

selectivity-diagnostics:
	"$(PYTHON)" -m perturbselect.selectivity_diagnostics --directory "$(SELECTIVITY_RESULTS)"

test-selectivity-diagnostics:
	"$(PYTHON)" -m pytest tests/test_selectivity_diagnostics.py $(TEST_ARGS)

test-selectivity-integration:
	"$(PYTHON)" -c "import os, sys, pytest; os.environ['PERTURBSELECT_RUN_INTEGRATION'] = '1'; sys.exit(pytest.main(sys.argv[1:]))" tests/integration/test_selectivity_saved_outputs.py $(TEST_ARGS)

model-feasibility:
	"$(PYTHON)" -m perturbselect.model_feasibility --registry "$(MODEL_REGISTRY)"

test-model-feasibility:
	"$(PYTHON)" -m pytest tests/test_model_feasibility.py $(TEST_ARGS)

benchmark-export:
	"$(PYTHON)" -m perturbselect.benchmark_adapters --input "$(SELECTIVITY_RESULTS)"

benchmark-report:
	"$(PYTHON)" -m perturbselect.benchmark_adapters --input "$(SELECTIVITY_RESULTS)" --presentation "$(PRESENTATION_CONFIG)"

test-benchmarking:
	"$(PYTHON)" -m pytest tests/benchmarking $(TEST_ARGS)
