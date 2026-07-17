"""Experiment pipeline orchestration.

Runs data loading, feature engineering, temporal splitting, model training,
prediction, conformal interval computation, metric evaluation, bootstrap
analysis, and artifact serialisation.
"""

from __future__ import annotations

import csv
import json
import os
import platform as plat
import sys
import time as time_module
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.conformal import compute_absolute_residual_quantile, predict_intervals
from bike_demand_forecast.data import load_data
from bike_demand_forecast.features import create_features
from bike_demand_forecast.metrics import (
    bootstrap_paired_mase_improvement,
    compute_all_metrics,
    compute_per_hour_metrics,
    compute_seasonal_scale,
)
from bike_demand_forecast.models import (
    METHOD_NAMES,
    TRAINED_METHODS,
    predict_method,
    train_method,
)
from bike_demand_forecast.split import create_temporal_split

# Schema version for artifacts
ARTIFACT_SCHEMA_VERSION = 1


def _peak_rss_mb() -> float | None:
    """Return peak RSS in MB on Unix-like platforms, or None."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return round(usage.ru_maxrss / divisor, 1)
    except (AttributeError, ImportError, ValueError):
        return None


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _display_config_path(config_path: str) -> str:
    if not config_path:
        return "(defaults)"
    path = Path(config_path).resolve()
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return path.name


def _git_commit() -> str:
    ci_sha = os.environ.get("GITHUB_SHA")
    if ci_sha:
        return ci_sha[:7]

    git_dir = Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head.removeprefix("ref: ")
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:7]
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0][:7]
    except OSError:
        pass
    return "unknown"


def _compute_dataset_sha256(cfg: ExperimentConfig, is_smoke: bool) -> str:
    """Return dataset SHA-256: real hash for full run, fixture digest for smoke."""
    if is_smoke:
        # Deterministic fixture based on config seed
        return f"fixture_seed_{cfg.hgb.random_state}"
    try:
        path = Path(cfg.data.raw_dir) / "hour.csv"
        if path.exists():
            import hashlib

            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
    except OSError:
        pass
    return "unknown"


def _override_smoke(cfg: ExperimentConfig) -> ExperimentConfig:
    """Override config for smoke mode with minimal parameters."""
    from copy import deepcopy

    new = deepcopy(cfg)
    object.__setattr__(new.temporal, "train_fraction", 0.5)
    object.__setattr__(new.temporal, "cal_fraction", 0.25)
    object.__setattr__(new.temporal, "test_fraction", 0.25)
    object.__setattr__(new.temporal, "min_train_rows", 10)
    object.__setattr__(new.temporal, "min_cal_rows", 5)
    object.__setattr__(new.temporal, "min_test_rows", 5)
    object.__setattr__(new.hgb, "max_iter", 50)
    object.__setattr__(new.hgb, "max_leaf_nodes", 16)
    object.__setattr__(new.random_forest, "n_estimators", 20)
    object.__setattr__(new.random_forest, "max_depth", 8)
    object.__setattr__(new.reporting, "bootstrap_samples", 100)
    return new


# ------------------------------------------------------------------ #
#  Data containers
# ------------------------------------------------------------------ #
@dataclass
class RunMetadata:
    """Reproducibility metadata for a pipeline run."""

    run_id: str
    config_hash: str
    config_path: str
    mode: str
    source_label: str
    dataset_sha256: str
    timestamp_start: str
    timestamp_end: str
    duration_seconds: float
    n_train: int
    n_cal: int
    n_test: int
    n_features: int
    seed: int
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    git_commit: str = ""
    python_version: str = ""
    platform: str = ""
    numpy_version: str = ""
    pandas_version: str = ""
    scipy_version: str = ""
    sklearn_version: str = ""
    matplotlib_version: str = ""
    peak_rss_mb: float | None = None
    status: str = "completed"


@dataclass
class ExperimentResults:
    """Container for all experiment outputs."""

    config: ExperimentConfig
    metadata: RunMetadata
    # Test index for CSV with timestamps
    test_index: pd.DatetimeIndex
    y_test: np.ndarray
    seasonal_scale: float
    # Per-method predictions and intervals
    method_predictions: dict[str, np.ndarray]
    method_lower: dict[str, np.ndarray | None]
    method_upper: dict[str, np.ndarray | None]
    method_metrics: dict[str, dict[str, float]]
    method_per_hour: dict[str, list[dict[str, Any]]]
    conformal_quantiles: dict[str, float | None]
    # Calibration MASE for selecting best method (without looking at test)
    cal_mase: dict[str, float]
    selected_method: str  # best by calibration MASE
    # Comparisons vs seasonal_naive
    comparisons: dict[str, dict[str, float] | None]


# ------------------------------------------------------------------ #
#  Pipeline
# ------------------------------------------------------------------ #
def run_experiment(
    cfg: ExperimentConfig,
    output_dir: str = "reports",
    mode: str = "full",
    config_path: str = "",
) -> ExperimentResults:
    """Run the full or smoke experiment pipeline.

    Parameters
    ----------
    cfg : ExperimentConfig
        Validated experiment configuration.
    output_dir : str
        Root directory for artifacts.
    mode : str
        ``"smoke"`` for fast validation or ``"full"`` for full evaluation.
    config_path : str
        Path to config file (for metadata).
    """
    start_ts = datetime.now(UTC)
    run_id = _make_run_id()
    t0 = time_module.perf_counter()
    is_smoke = mode == "smoke"

    if is_smoke:
        cfg = _override_smoke(cfg)

    base_dir = Path(output_dir)
    out_dir = base_dir / "smoke" if is_smoke else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    for d in [metrics_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    source_label = "UCI Bike Sharing" if not is_smoke else "synthetic fixture"

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    df = load_data(cfg, synthetic=is_smoke)

    # ------------------------------------------------------------------ #
    # 2. Create features
    # ------------------------------------------------------------------ #
    X, y = create_features(df, cfg)

    # ------------------------------------------------------------------ #
    # 3. Temporal split
    # ------------------------------------------------------------------ #
    split = create_temporal_split(X, cfg)

    X_train = X.loc[split.idx_train]
    y_train = y.loc[split.idx_train]
    X_cal = X.loc[split.idx_cal]
    y_cal = y.loc[split.idx_cal]
    X_test = X.loc[split.idx_test]
    y_test = y.loc[split.idx_test]

    n_train = len(X_train)
    n_cal = len(X_cal)
    n_test = len(X_test)
    n_features = X_train.shape[1]

    # ------------------------------------------------------------------ #
    # 4. Compute seasonal scale on training data (aligned via lag_24)
    # ------------------------------------------------------------------ #
    seasonal_scale = compute_seasonal_scale(y_train.values, X_train["lag_24"].values)

    # ------------------------------------------------------------------ #
    # 5. Train models and get predictions
    # ------------------------------------------------------------------ #
    models: dict[str, Any] = {}
    cal_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}

    for method in METHOD_NAMES:
        models[method] = train_method(method, X_train, y_train, cfg)
        cal_preds[method] = predict_method(method, models[method], X_cal)
        test_preds[method] = predict_method(method, models[method], X_test)

    # ------------------------------------------------------------------ #
    # 6. Compute calibration MASE for selection (no test data used)
    # ------------------------------------------------------------------ #
    cal_mase: dict[str, float] = {}
    for method in METHOD_NAMES:
        m = compute_all_metrics(y_cal.values, cal_preds[method], seasonal_scale=seasonal_scale)
        cal_mase[method] = m.mase

    # Select best method by calibration MASE (lower is better)
    best_cal_method = min(cal_mase, key=lambda k: cal_mase[k])
    selected_method = best_cal_method

    # ------------------------------------------------------------------ #
    # 7. Conformal prediction intervals (all methods)
    # ------------------------------------------------------------------ #
    method_lower: dict[str, np.ndarray | None] = {}
    method_upper: dict[str, np.ndarray | None] = {}
    conformal_quantiles: dict[str, float | None] = {}

    for method in METHOD_NAMES:
        q = compute_absolute_residual_quantile(
            y_cal.values,
            cal_preds[method],
            significance_level=cfg.conformal.significance_level,
            method=cfg.conformal.method,
        )
        lower, upper = predict_intervals(test_preds[method], q)
        method_lower[method] = lower
        method_upper[method] = upper
        conformal_quantiles[method] = float(q)

    # ------------------------------------------------------------------ #
    # 8. Compute test metrics for all methods
    # ------------------------------------------------------------------ #
    alpha = cfg.conformal.significance_level
    method_metrics: dict[str, dict[str, float]] = {}
    method_per_hour: dict[str, list[dict[str, Any]]] = {}

    for method in METHOD_NAMES:
        fm = compute_all_metrics(
            y_test.values,
            test_preds[method],
            y_lower=method_lower[method],
            y_upper=method_upper[method],
            seasonal_scale=seasonal_scale,
            alpha=alpha,
        )
        method_metrics[method] = {
            "mase": fm.mase,
            "mae": fm.mae,
            "rmse": fm.rmse,
            "smape": fm.smape,
            "interval_coverage": fm.interval_coverage,
            "mean_interval_width": fm.mean_interval_width,
            "winkler_score": fm.winkler_score,
        }

    # Per-hour metrics
    test_hours = X_test.index.hour.values
    for method in METHOD_NAMES:
        ph = compute_per_hour_metrics(
            y_test.values,
            test_preds[method],
            test_hours,
            seasonal_scale=seasonal_scale,
        )
        method_per_hour[method] = [_ph_dict(m) for m in ph]

    # ------------------------------------------------------------------ #
    # 9. Bootstrap comparisons (each trained method vs seasonal naive)
    # ------------------------------------------------------------------ #
    comparisons: dict[str, dict[str, float] | None] = {}
    unique_test_days = sorted(set(X_test.index.normalize()))

    for method in TRAINED_METHODS:
        comp = _compute_day_block_bootstrap(
            y_test.values,
            test_preds["seasonal_naive"],
            test_preds[method],
            unique_test_days,
            seasonal_scale,
            X_test,
            cfg,
        )
        comparisons[method] = comp

    # ------------------------------------------------------------------ #
    # 10. Metadata
    # ------------------------------------------------------------------ #
    end_ts = datetime.now(UTC)
    elapsed = time_module.perf_counter() - t0

    import matplotlib as mpl_ver
    import numpy as np_ver
    import pandas as pd_ver
    import scipy as sp_ver
    import sklearn as sk_ver

    metadata = RunMetadata(
        run_id=run_id,
        config_hash=cfg.config_hash(),
        config_path=_display_config_path(config_path),
        mode=mode,
        source_label=source_label,
        dataset_sha256=_compute_dataset_sha256(cfg, is_smoke),
        timestamp_start=start_ts.isoformat(),
        timestamp_end=end_ts.isoformat(),
        duration_seconds=elapsed,
        n_train=n_train,
        n_cal=n_cal,
        n_test=n_test,
        n_features=n_features,
        seed=cfg.hgb.random_state,
        git_commit=_git_commit(),
        python_version=sys.version,
        platform=plat.platform(),
        numpy_version=np_ver.__version__,
        pandas_version=pd_ver.__version__,
        scipy_version=sp_ver.__version__,
        sklearn_version=sk_ver.__version__,
        matplotlib_version=mpl_ver.__version__,
        peak_rss_mb=_peak_rss_mb(),
        status="completed",
    )

    results = ExperimentResults(
        config=cfg,
        metadata=metadata,
        test_index=X_test.index,
        y_test=y_test.values,
        seasonal_scale=seasonal_scale,
        method_predictions=test_preds,
        method_lower=method_lower,
        method_upper=method_upper,
        method_metrics=method_metrics,
        method_per_hour=method_per_hour,
        conformal_quantiles=conformal_quantiles,
        cal_mase=cal_mase,
        selected_method=selected_method,
        comparisons=comparisons,
    )

    # ------------------------------------------------------------------ #
    # 11. Save artifacts
    # ------------------------------------------------------------------ #
    _save_artifacts(results, out_dir)

    return results


def _compute_day_block_bootstrap(
    y_test_vals: np.ndarray,
    pred_baseline: np.ndarray,
    pred_advanced: np.ndarray,
    unique_test_days: list[pd.Timestamp],
    seasonal_scale: float,
    X_test: pd.DataFrame,
    cfg: ExperimentConfig,
) -> dict[str, float] | None:
    """Compute day-block bootstrap for paired MASE improvement."""
    if seasonal_scale <= 0 or len(unique_test_days) < 4:
        return None

    day_mase_base: list[float] = []
    day_mase_adv: list[float] = []
    for day in unique_test_days:
        day_mask = X_test.index.normalize() == day
        if day_mask.sum() < 2:
            continue
        yt_day = y_test_vals[day_mask]
        yb_day = pred_baseline[day_mask]
        ya_day = pred_advanced[day_mask]
        day_mase_base.append(float(np.mean(np.abs(yt_day - yb_day)) / seasonal_scale))
        day_mase_adv.append(float(np.mean(np.abs(yt_day - ya_day)) / seasonal_scale))

    if len(day_mase_base) < 4:
        return None

    boot = bootstrap_paired_mase_improvement(
        np.array(day_mase_base),
        np.array(day_mase_adv),
        n_samples=cfg.reporting.bootstrap_samples,
        ci_level=cfg.reporting.bootstrap_ci_level,
        rng_seed=cfg.hgb.random_state + 1,
    )
    return {
        "improvement_mean": boot.improvement_mean,
        "ci_lower": boot.ci_lower,
        "ci_upper": boot.ci_upper,
        "ci_level": boot.ci_level,
        "n_samples": boot.n_samples,
        "bootstrap_probability_improvement": boot.bootstrap_probability_improvement,
    }


def _ph_dict(m: Any) -> dict[str, Any]:
    return {
        "hour": m.hour,
        "count": m.count,
        "mae": m.mae,
        "rmse": m.rmse,
        "mase": m.mase,
        "bias": m.bias,
    }


# ------------------------------------------------------------------ #
#  Artifact saving
# ------------------------------------------------------------------ #
def _atomic_write_json(data: Any, path: Path) -> None:
    """Write JSON atomically via temp file and rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_round_floats(data), f, indent=2, default=_json_default)
        f.write("\n")
        f.flush()
    tmp.rename(path)


def _save_artifacts(results: ExperimentResults, out_dir: Path) -> None:
    """Save all experiment artifacts to disk."""
    summary = _build_summary(results)
    _atomic_write_json(summary, out_dir / "metrics" / "summary.json")

    meta = _build_metadata_dict(results.metadata)
    _atomic_write_json(meta, out_dir / "metrics" / "run_metadata.json")

    _save_predictions_csv(results, out_dir / "tables" / "predictions.csv")
    _save_per_hour_csv(results, out_dir / "tables" / "per_hour_metrics.csv")


def _build_summary(results: ExperimentResults) -> dict[str, Any]:
    methods_out: dict[str, Any] = {}
    for method in METHOD_NAMES:
        m = results.method_metrics[method]
        methods_out[method] = {
            "mase": m["mase"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "smape": m["smape"],
            "interval_coverage": m["interval_coverage"],
            "mean_interval_width": m["mean_interval_width"],
            "winkler_score": m["winkler_score"],
        }

    comparisons_out: dict[str, Any] = {}
    for method in TRAINED_METHODS:
        c = results.comparisons.get(method)
        if c is not None:
            comparisons_out[method] = c
        else:
            comparisons_out[method] = None

    summary: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": results.metadata.mode,
        "source": results.metadata.source_label,
        "n_train": results.metadata.n_train,
        "n_cal": results.metadata.n_cal,
        "n_test": results.metadata.n_test,
        "n_features": results.metadata.n_features,
        "seasonal_scale": results.seasonal_scale,
        "selected_method": results.selected_method,
        "methods": methods_out,
        "comparisons": comparisons_out,
        "calibration_mase": results.cal_mase,
    }
    return summary


def _build_metadata_dict(meta: RunMetadata) -> dict[str, Any]:
    return {
        "schema_version": meta.schema_version,
        "run_id": meta.run_id,
        "config_hash": meta.config_hash,
        "config_path": meta.config_path,
        "mode": meta.mode,
        "source_label": meta.source_label,
        "dataset_sha256": meta.dataset_sha256,
        "timestamp_start": meta.timestamp_start,
        "timestamp_end": meta.timestamp_end,
        "duration_seconds": meta.duration_seconds,
        "n_train": meta.n_train,
        "n_cal": meta.n_cal,
        "n_test": meta.n_test,
        "n_features": meta.n_features,
        "seed": meta.seed,
        "git_commit": meta.git_commit,
        "python_version": meta.python_version,
        "platform": meta.platform,
        "numpy_version": meta.numpy_version,
        "pandas_version": meta.pandas_version,
        "scipy_version": meta.scipy_version,
        "sklearn_version": meta.sklearn_version,
        "matplotlib_version": meta.matplotlib_version,
        "peak_rss_mb": meta.peak_rss_mb,
        "status": meta.status,
    }


def _save_predictions_csv(results: ExperimentResults, path: Path) -> None:
    """Save test set predictions as CSV with timestamp and horizon_hour.

    horizon_hour: 1-24, where hour 1 corresponds to 00:00-01:00 (first hour
    after forecast origin at midnight).
    """
    test_idx = results.test_index
    n = len(results.y_test)
    records: list[dict[str, Any]] = []

    for i in range(n):
        ts = test_idx[i]
        row: dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "horizon_hour": ts.hour + 1,  # 1-24
            "actual": round(float(results.y_test[i]), 8),
        }
        for method in METHOD_NAMES:
            row[f"{method}_prediction"] = round(float(results.method_predictions[method][i]), 8)
            lo = results.method_lower[method]
            up = results.method_upper[method]
            if lo is not None and up is not None:
                row[f"{method}_lower"] = round(float(lo[i]), 8)
                row[f"{method}_upper"] = round(float(up[i]), 8)
        records.append(row)

    if not records:
        path.write_text("")
        return

    fieldnames = list(records[0].keys())
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    tmp.rename(path)


def _save_per_hour_csv(results: ExperimentResults, path: Path) -> None:
    """Save per-hour metrics as CSV with all methods."""
    fieldnames = ["hour", "count", "method", "mae", "rmse", "mase", "bias"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for method in METHOD_NAMES:
            for row in results.method_per_hour[method]:
                writer.writerow(
                    {
                        "hour": row["hour"],
                        "count": row["count"],
                        "method": method,
                        "mae": round(float(row["mae"]), 8),
                        "rmse": round(float(row["rmse"]), 8),
                        "mase": round(float(row["mase"]), 8),
                        "bias": round(float(row["bias"]), 8),
                    }
                )
    tmp.rename(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _round_floats(obj: Any) -> Any:
    """Round artifact floats to stable, analysis-grade precision."""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 12)
    if isinstance(obj, dict):
        return {key: _round_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_round_floats(value) for value in obj)
    return obj
