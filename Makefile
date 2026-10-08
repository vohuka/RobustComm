# RobustComm — developer convenience targets.
# All targets run offline on synthetic data. No dataset is downloaded.

PYTHON ?= python3
PKG = robustcomm

.PHONY: help install install-dev test demo plots lint typecheck clean

help:
	@echo "RobustComm make targets:"
	@echo "  install       Install the package (editable)."
	@echo "  install-dev   Install with dev extras (pytest, ruff, mypy)."
	@echo "  test          Run the pytest suite (synthetic data only)."
	@echo "  demo          Run the synthetic smoke-test demo."
	@echo "  plots         Regenerate demo plots from the demo outputs."
	@echo "  lint          Run ruff (if installed)."
	@echo "  typecheck     Run mypy on src (if installed)."
	@echo "  clean         Remove caches and generated demo outputs."

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml

plots:
	$(PYTHON) scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/plots

lint:
	ruff check . || echo "ruff not installed; skipping"

typecheck:
	mypy src || echo "mypy not installed; skipping"

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__ outputs/synthetic_demo
	@echo "cleaned caches and demo outputs (kept outputs/.gitkeep)"
