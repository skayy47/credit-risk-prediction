# Credit Risk Prediction Pipeline
# Run targets from the project root.
# Usage: make <target>

.PHONY: install test lint ingest train scenarios all clean serve

# ── environment ───────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev,app]"

# ── pipeline ──────────────────────────────────────────────────────────────────
ingest:
	python -m credit_risk.cli ingest-validate

dashboard-tables:
	python -m credit_risk.cli make-dashboard-tables

train:
	python -m credit_risk.cli train-simulate

scenarios:
	python -m credit_risk.cli simulate-scenarios

all: ingest dashboard-tables train scenarios
	@echo "Full pipeline complete."

# ── quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

# ── app ───────────────────────────────────────────────────────────────────────
serve:
	streamlit run app.py

# ── cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
