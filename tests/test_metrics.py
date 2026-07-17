"""Tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from bike_demand_forecast.metrics import (
    MetricError,
    bootstrap_paired_mase_improvement,
    compute_all_metrics,
    compute_interval_coverage,
    compute_mae,
    compute_mase,
    compute_mean_interval_width,
    compute_per_hour_metrics,
    compute_rmse,
    compute_seasonal_scale,
    compute_smape,
    compute_winkler_score,
)


def test_compute_seasonal_scale() -> None:
    rng = np.random.default_rng(42)
    y_train = 100 + rng.normal(0, 10, size=200)
    lag_24 = 100 + rng.normal(0, 10, size=200)
    scale = compute_seasonal_scale(y_train, lag_24)
    assert scale > 0
    assert np.isfinite(scale)


def test_seasonal_scale_empty() -> None:
    with pytest.raises(MetricError, match="empty"):
        compute_seasonal_scale(np.array([]), np.array([]))


def test_seasonal_scale_mismatched_lengths() -> None:
    with pytest.raises(MetricError, match="length"):
        compute_seasonal_scale(np.array([1.0]), np.array([1.0, 2.0]))


def test_seasonal_scale_zero() -> None:
    with pytest.raises(MetricError, match="positive"):
        compute_seasonal_scale(np.array([5.0, 5.0]), np.array([5.0, 5.0]))


def test_compute_mase() -> None:
    y_true = np.array([10.0, 12.0, 15.0, 11.0])
    y_pred = np.array([10.0, 12.0, 15.0, 11.0])
    mase = compute_mase(y_true, y_pred, seasonal_scale=5.0)
    assert mase == 0.0


def test_compute_mase_zero_scale() -> None:
    y_true = np.array([5.0, 5.0])
    y_pred = np.array([5.0, 5.0])
    mase = compute_mase(y_true, y_pred, seasonal_scale=0.0)
    assert mase == 0.0


def test_compute_mae() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    assert compute_mae(y_true, y_pred) == pytest.approx(0.3333, abs=1e-3)


def test_compute_rmse() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(y_true, y_pred) == 0.0


def test_compute_smape() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 29.0])
    smape = compute_smape(y_true, y_pred)
    assert smape > 0
    assert smape < 100


def test_compute_smape_zero_division() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    smape = compute_smape(y_true, y_pred)
    assert np.isfinite(smape)


def test_interval_coverage() -> None:
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([3.0, 8.0, 12.0])
    upper = np.array([7.0, 12.0, 18.0])
    cov = compute_interval_coverage(y_true, lower, upper)
    assert cov == 1.0


def test_mean_interval_width() -> None:
    lower = np.array([3.0, 8.0])
    upper = np.array([7.0, 12.0])
    width = compute_mean_interval_width(lower, upper)
    assert width == 4.0


def test_winkler_score() -> None:
    y_true = np.array([5.0, 10.0])
    lower = np.array([3.0, 8.0])
    upper = np.array([7.0, 12.0])
    score = compute_winkler_score(y_true, lower, upper, alpha=0.1)
    assert score > 0
    assert np.isfinite(score)


def test_compute_all_metrics() -> None:
    rng = np.random.default_rng(42)
    y_true = np.abs(rng.normal(100, 20, size=100))
    y_pred = y_true + rng.normal(0, 10, size=100)
    y_lower = y_pred - 20
    y_upper = y_pred + 20

    metrics = compute_all_metrics(
        y_true,
        y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        seasonal_scale=15.0,
        alpha=0.1,
    )
    assert metrics.mase > 0
    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert metrics.smape > 0
    assert np.isfinite(metrics.interval_coverage)
    assert np.isfinite(metrics.mean_interval_width)
    assert np.isfinite(metrics.winkler_score)


def test_per_hour_metrics() -> None:
    rng = np.random.default_rng(42)
    y_true = np.abs(rng.normal(100, 20, size=240))
    y_pred = y_true + rng.normal(0, 10, size=240)
    hours = np.tile(np.arange(24), 10)

    results = compute_per_hour_metrics(y_true, y_pred, hours, seasonal_scale=15.0)
    assert len(results) == 24
    for r in results:
        assert r.count == 10
        assert r.mae > 0
        assert np.isfinite(r.bias)


def test_bootstrap_paired_mase_improvement() -> None:
    rng = np.random.default_rng(42)
    base_mase = 1.5 + rng.normal(0, 0.2, size=50)
    adv_mase = 1.0 + rng.normal(0, 0.2, size=50)

    boot = bootstrap_paired_mase_improvement(base_mase, adv_mase, n_samples=500)
    assert boot.improvement_mean > 0
    assert boot.ci_lower < boot.ci_upper
    assert boot.bootstrap_probability_improvement > 0.5


def test_bootstrap_mismatched_sizes() -> None:
    with pytest.raises(MetricError, match="Mismatched"):
        bootstrap_paired_mase_improvement(
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )


def test_bootstrap_deterministic() -> None:
    rng = np.random.default_rng(42)
    base = 1.5 + rng.normal(0, 0.2, size=50)
    adv = 1.0 + rng.normal(0, 0.2, size=50)

    boot1 = bootstrap_paired_mase_improvement(base, adv, n_samples=200, rng_seed=42)
    boot2 = bootstrap_paired_mase_improvement(base, adv, n_samples=200, rng_seed=42)

    assert boot1.improvement_mean == pytest.approx(boot2.improvement_mean)
    assert boot1.ci_lower == pytest.approx(boot2.ci_lower)
    assert boot1.bootstrap_probability_improvement == boot2.bootstrap_probability_improvement


def test_bootstrap_field_name() -> None:
    """Verify the field is named bootstrap_probability_improvement, not p_improvement."""
    rng = np.random.default_rng(42)
    base = 1.5 + rng.normal(0, 0.2, size=50)
    adv = 1.0 + rng.normal(0, 0.2, size=50)
    boot = bootstrap_paired_mase_improvement(base, adv, n_samples=100)
    assert hasattr(boot, "bootstrap_probability_improvement")
    assert not hasattr(boot, "p_improvement_positive")
