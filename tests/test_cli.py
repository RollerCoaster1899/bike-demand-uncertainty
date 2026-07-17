"""Tests for the CLI entry point."""

from __future__ import annotations

from bike_demand_forecast.cli import main


def test_cli_smoke_exit_code():
    """Running with --smoke should return 0."""
    exit_code = main(["--smoke"])
    assert exit_code == 0


def test_cli_config_not_found():
    exit_code = main(["--config", "/nonexistent/path.yaml"])
    assert exit_code == 1
