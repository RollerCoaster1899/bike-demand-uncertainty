"""Command-line interface for the bike demand forecasting pipeline."""

from __future__ import annotations

import argparse
import sys

from bike_demand_forecast.config import load_config
from bike_demand_forecast.models import METHOD_NAMES
from bike_demand_forecast.pipeline import run_experiment


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``bike-demand-forecast`` CLI."""
    parser = argparse.ArgumentParser(
        description="Bike demand forecasting with calibrated split-conformal uncertainty intervals."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional; uses defaults if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Root output directory (default: reports).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (synthetic data, reduced config) instead of full evaluation.",
    )
    args = parser.parse_args(argv)

    mode = "smoke" if args.smoke else "full"

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    try:
        results = run_experiment(
            cfg,
            output_dir=args.output_dir,
            mode=mode,
            config_path=args.config or "",
        )
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 1

    print(f"Experiment completed. Mode: {mode.upper()}")
    print(f"  Run ID:   {results.metadata.run_id}")
    print(f"  Config hash: {results.metadata.config_hash}")
    print(f"  Source:   {results.metadata.source_label}")
    print(
        f"  Train/Cal/Test: {results.metadata.n_train}/{results.metadata.n_cal}/{results.metadata.n_test}"
    )
    print(f"  Selected method: {results.selected_method} (by calibration MASE)")
    for method in METHOD_NAMES:
        m = results.method_metrics[method]
        print(f"    {method}: MASE={m['mase']:.4f}, MAE={m['mae']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
