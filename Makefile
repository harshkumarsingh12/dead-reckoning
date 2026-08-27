# Thin wrappers so local and CI run byte-identical commands. Windows users: the same
# targets exist in tasks.ps1 (.\tasks.ps1 test). If you change one, change the other.
.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test frames cov all serve web clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

lint:  ## Ruff lint
	ruff check .

format:  ## Ruff format in place
	ruff format .

typecheck:  ## Mypy, strict
	mypy

test:  ## Full test suite
	pytest -q

frames:  ## Coordinate-frame invariants only
	pytest -m frames -v

cov:  ## Tests with a coverage report
	pytest -q --cov --cov-report=term-missing

all: lint typecheck test  ## What CI runs

serve:  ## Start the gateway on the hotspot interface
	python -m services.gateway --host 0.0.0.0 --port 8000

web:  ## Start the web UI dev server
	cd apps/web && npm run dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
