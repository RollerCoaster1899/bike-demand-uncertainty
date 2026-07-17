"""Tests for conformal prediction interval computation."""

from __future__ import annotations

import numpy as np
import pytest

from bike_demand_forecast.conformal import (
    ConformalError,
    compute_absolute_residual_quantile,
    predict_intervals,
)


def test_compute_quantile_basic():
    y_cal = np.array([10.0, 12.0, 15.0, 11.0, 13.0, 14.0])
    y_pred_cal = np.array([9.0, 13.0, 14.0, 12.0, 12.5, 14.5])
    q = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.1)
    assert q > 0
    assert np.isfinite(q)


def test_compute_quantile_perfect_predictions():
    y_cal = np.array([10.0, 20.0, 30.0])
    y_pred_cal = np.array([10.0, 20.0, 30.0])
    q = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.1)
    assert q == 0.0


def test_compute_quantile_too_small():
    with pytest.raises(ConformalError, match="Calibration set too small"):
        compute_absolute_residual_quantile(
            np.array([1.0]),
            np.array([1.0]),
        )


def test_compute_quantile_invalid_method():
    y_cal = np.array([10.0, 12.0, 15.0])
    y_pred_cal = np.array([9.0, 13.0, 14.0])
    with pytest.raises(ConformalError, match="Unknown quantile method"):
        compute_absolute_residual_quantile(y_cal, y_pred_cal, method="invalid")


def test_predict_intervals():
    y_pred = np.array([10.0, 20.0, 30.0])
    quantile = 5.0
    lower, upper = predict_intervals(y_pred, quantile)
    assert np.allclose(lower, [5.0, 15.0, 25.0])
    assert np.allclose(upper, [15.0, 25.0, 35.0])


def test_predict_intervals_clipped():
    y_pred = np.array([2.0, 10.0, 50.0])
    quantile = 5.0
    lower, upper = predict_intervals(y_pred, quantile)
    assert lower[0] == 0.0
    assert lower[1] == 5.0
    assert lower[2] == 45.0


def test_conformal_quantile_deterministic():
    y_cal = np.abs(np.random.default_rng(42).normal(100, 20, size=100))
    y_pred_cal = y_cal + np.random.default_rng(42).normal(0, 10, size=100)
    q1 = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.1)
    q2 = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.1)
    assert q1 == q2


def test_different_significance_levels():
    y_cal = np.array([10.0, 12.0, 15.0, 11.0, 13.0, 14.0, 16.0, 9.0, 18.0, 7.0])
    y_pred_cal = np.array([9.0, 13.0, 14.0, 12.0, 12.5, 14.5, 15.0, 10.0, 17.0, 8.0])
    q90 = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.1)
    q80 = compute_absolute_residual_quantile(y_cal, y_pred_cal, significance_level=0.2)
    assert q80 >= q90
