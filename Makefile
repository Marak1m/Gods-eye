# ps10-scd — HANDOFF.md section 3.1. Run from WSL/Linux. `make setup` first.
UV        ?= uv
PY        := $(UV) run python
SMOKE_DATA ?= /tmp/ps10-scd-smoke
SMOKE_ENV := SCD_DATA=$(SMOKE_DATA) SCD_RUNS=$(SMOKE_DATA)/runs
SMOKE_CFG := --config configs/smoke.yaml
CONFIG    ?= configs/default.yaml
STAGE     ?= all
RUN_NAME  ?= v1_seed42

.PHONY: setup test smoke-data smoke lint notebook clean help

help:
	@echo "setup | test | smoke-data | smoke | lint | notebook STAGE=all | clean"

setup:
	$(UV) python install 3.11
	$(UV) sync --extra dev
	@if [ -n "$$SCD_DATA" ]; then \
	  case "$$SCD_DATA" in /mnt/*) echo "WARNING: SCD_DATA=$$SCD_DATA is on a 9p mount; put the data root on the Linux filesystem (README, WSL notes)";; esac; \
	  mkdir -p "$$SCD_DATA"; \
	  if [ ! -e data ]; then ln -s "$$SCD_DATA" data && echo "data -> $$SCD_DATA"; fi; \
	else echo "SCD_DATA is unset: the data symlink was not created (HANDOFF.md section 2.2)"; fi

test:
	$(PY) -m pytest -q -m "not gpu and not network"

# M4 gate: fixtures -> pairs -> labels -> chips -> object library -> sampler report (no model)
smoke-data:
	rm -rf $(SMOKE_DATA)
	$(SMOKE_ENV) $(PY) tests/fixtures/make_smoke_data.py --data-root $(SMOKE_DATA)
	$(SMOKE_ENV) $(PY) scripts/02_preprocess_pairs.py $(SMOKE_CFG)
	$(SMOKE_ENV) $(PY) scripts/06_rasterize.py $(SMOKE_CFG)
	$(SMOKE_ENV) $(PY) scripts/07_build_chips.py $(SMOKE_CFG) --version 1 --sampler-report
	$(SMOKE_ENV) $(PY) scripts/08_build_object_library.py $(SMOKE_CFG) --version 1

# M7 gate: smoke-data + 2 training steps + stub calibration + inference + the invariance tests
smoke: smoke-data
	$(SMOKE_ENV) $(PY) scripts/09_train.py $(SMOKE_CFG) --run smoke
	$(SMOKE_ENV) $(PY) scripts/10_calibrate.py $(SMOKE_CFG) --run smoke --stub
	$(SMOKE_ENV) $(PY) scripts/11_infer.py $(SMOKE_CFG) --run smoke \
	  --t1 $(SMOKE_DATA)/pairs/A_smoke_01/p01/t1.tif --t2 $(SMOKE_DATA)/pairs/A_smoke_01/p01/t2.tif \
	  --out $(SMOKE_DATA)/outputs/smoke
	$(SMOKE_ENV) $(PY) -m pytest -q -m "not gpu" tests/test_grid_symmetry.py tests/test_coreg_symmetry.py \
	  tests/test_preprocess_symmetry.py tests/test_fusion_symmetry.py tests/test_net_symmetry.py \
	  tests/test_identity_guard.py tests/test_pipeline_swap.py tests/test_export.py

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy scd

notebook:
	$(PY) -m papermill notebooks/00_pipeline.ipynb notebooks/00_pipeline.out.ipynb \
	  -p STAGES "$(STAGE)" -p CONFIG "$(CONFIG)" -p RUN_NAME "$(RUN_NAME)"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
