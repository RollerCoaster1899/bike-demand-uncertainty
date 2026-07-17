"""Integration tests for the experiment pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.models import METHOD_NAMES
from bike_demand_forecast.pipeline import run_experiment


def test_smoke_pipeline_completes():
    """Smoke pipeline runs without error using synthetic data."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)
    object.__setattr__(cfg.hgb, "max_iter", 50)
    object.__setattr__(cfg.random_forest, "n_estimators", 20)
    object.__setattr__(cfg.random_forest, "max_depth", 8)
    object.__setattr__(cfg.features, "rolling_windows_hours", (24,))
    object.__setattr__(cfg.features, "lag_hours", (24, 48))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        results = run_experiment(cfg, output_dir=str(out), mode="smoke", config_path="")

        assert results.metadata.mode == "smoke"
        assert results.metadata.source_label == "synthetic fixture"
        assert results.metadata.n_train > 0
        assert results.metadata.n_cal > 0
        assert results.metadata.n_test > 0
        assert results.metadata.status == "completed"
        assert results.metadata.schema_version == 1

        # All four methods should have predictions
        for method in METHOD_NAMES:
            assert method in results.method_predictions, f"Missing predictions for {method}"
            assert method in results.method_metrics, f"Missing metrics for {method}"
            assert method in results.method_per_hour, f"Missing per-hour for {method}"
            assert results.method_lower[method] is not None
            assert results.method_upper[method] is not None

        # Selected method must be one of the methods
        assert results.selected_method in METHOD_NAMES

        # All metric values should be non-negative
        for method in METHOD_NAMES:
            m = results.method_metrics[method]
            assert m["mase"] >= 0 or (m["mase"] != m["mase"])
            assert m["mae"] >= 0
            assert m["rmse"] >= 0

        # Check artifacts exist
        assert (out / "smoke" / "metrics" / "summary.json").exists()
        assert (out / "smoke" / "metrics" / "run_metadata.json").exists()
        assert (out / "smoke" / "tables" / "predictions.csv").exists()
        assert (out / "smoke" / "tables" / "per_hour_metrics.csv").exists()


def test_smoke_pipeline_artifacts_content():
    """Verify summary.json has expected keys for 4 methods."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)
    object.__setattr__(cfg.hgb, "max_iter", 50)
    object.__setattr__(cfg.random_forest, "n_estimators", 20)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_experiment(cfg, output_dir=str(out), mode="smoke", config_path="")

        summary_path = out / "smoke" / "metrics" / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        assert "methods" in summary
        assert "comparisons" in summary
        assert "calibration_mase" in summary
        assert summary["selected_method"] in METHOD_NAMES
        for method in METHOD_NAMES:
            assert method in summary["methods"], f"Missing {method} in summary methods"

        meta_path = out / "smoke" / "metrics" / "run_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert "run_id" in meta
        assert "config_hash" in meta
        assert "schema_version" in meta
        assert meta["schema_version"] == 1
        assert "dataset_sha256" in meta
        assert "peak_rss_mb" in meta
        assert "git_commit" in meta
        assert "scipy_version" in meta
        assert "matplotlib_version" in meta
        assert meta["status"] == "completed"


def test_predictions_csv_has_timestamps():
    """Predictions CSV should contain timestamp and horizon_hour columns."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)
    object.__setattr__(cfg.hgb, "max_iter", 50)
    object.__setattr__(cfg.random_forest, "n_estimators", 20)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_experiment(cfg, output_dir=str(out), mode="smoke")
        pred_path = out / "smoke" / "tables" / "predictions.csv"
        content = pred_path.read_text()
        assert "timestamp" in content
        assert "horizon_hour" in content
        for method in METHOD_NAMES:
            assert f"{method}_prediction" in content


def test_selection_by_calibration_not_test():
    """The selected method should be chosen based on calibration, not test."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)
    object.__setattr__(cfg.hgb, "max_iter", 50)
    object.__setattr__(cfg.random_forest, "n_estimators", 20)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        results = run_experiment(cfg, output_dir=str(out), mode="smoke")

        # The selected method must be one of the methods
        assert results.selected_method in METHOD_NAMES

        # Verify that the method with best calibration MASE is the selected one
        best_cal = min(results.cal_mase, key=lambda k: results.cal_mase[k])
        assert results.selected_method == best_cal, (
            f"Selected {results.selected_method} but calibration best is {best_cal}"
        )
