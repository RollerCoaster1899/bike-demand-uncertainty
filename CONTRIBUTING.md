# Contributing

Contributions are welcome. Please follow these guidelines:

## Development Setup

```bash
git clone <repository-url>
cd bike-demand-uncertainty
uv sync --extra dev
```

## Code Quality

All checks must pass before merging:

```bash
make quality
```

This runs lint, format-check, type-check, tests with coverage, and security scan.

## Testing

- Write tests for new functionality.
- Maintain >=80% branch coverage on source code.
- Use `make test` to run tests.
- Use `make coverage` for HTML coverage report.

## Pre-commit

Install pre-commit hooks:

```bash
pre-commit install
```

## Commit Messages

Use conventional commits:

```
feat: add new feature
fix: correct bug
docs: update documentation
test: add tests
refactor: restructure code without changing behavior
chore: maintenance tasks
```
