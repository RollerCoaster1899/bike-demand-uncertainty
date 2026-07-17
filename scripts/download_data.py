#!/usr/bin/env python3
"""Download the UCI Bike Sharing dataset and extract hour.csv.

Usage:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from bike_demand_forecast.config import ExperimentConfig  # noqa: E402
from bike_demand_forecast.data import DataError, download_and_extract  # noqa: E402


def main() -> int:
    cfg = ExperimentConfig()
    print(f"Downloading UCI Bike Sharing dataset from {cfg.data.url}")

    try:
        hour_csv_path = download_and_extract(cfg)
    except DataError as e:
        print(f"Data error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    print(f"Dataset extracted to: {hour_csv_path}")
    print("Ready for experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
