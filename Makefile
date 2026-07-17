.PHONY: install dev lint format-check format type-check test coverage smoke reproduce clean quality security pre-commit

install:
	uv sync

dev:
	uv sync --extra dev

lint:
	uv run ruff check src tests scripts

format-check:
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts

type-check:
	uv run mypy src tests scripts --ignore-missing-imports

test:
	uv run coverage erase
	uv run coverage run -m pytest tests -v --tb=short
	uv run coverage run --append scripts/reproduce.py --smoke
	uv run coverage report --fail-under=80

coverage:
	uv run python -m pytest tests -v --cov=src --cov-branch --cov-report=html --cov-report=term-missing --tb=short

smoke:
	uv run python scripts/reproduce.py --smoke

reproduce:
	uv run python scripts/reproduce.py

security:
	uv run bandit -r src -x tests && uv run pip-audit

quality: dev lint format-check type-check test security pre-commit
	@echo "All quality checks passed."

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf reports/metrics reports/figures reports/tables reports/final_report.md reports/smoke
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf data/raw/* data/interim/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
