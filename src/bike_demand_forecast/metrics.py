"""Metrics computation for bike demand forecasting.

Primary metric: MASE (seasonal scale computed on training demand via aligned
lag-24).  Secondary: MAE, RMSE, sMAPE, interval coverage, mean interval width,
Winkler score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MetricError(Exception):
    """Raised on metric computation errors."""


@dataclass
class ForecastMetrics:
    """Metrics for a single forecast method on one dataset split."""

    mase: float
    mae: float
    rmse: float
    smape: float
    interval_coverage: float = float("nan")
    mean_interval_width: float = float("nan")
    winkler_score: float = float("nan")


@dataclass
class PerHourMetrics:
    """Per-hour breakdown of forecast errors."""

    hour: int
    count: int
    mae: float
    rmse: float
    mase: float
    bias: float


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for paired MASE improvement.

    ``bootstrap_probability_improvement`` is the fraction of resamples where
    improvement > 0.  This is NOT a hypothesis-test p-value; it describes
    the proportion of bootstrap resamples with positive improvement.
    """

    improvement_mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_samples: int
    bootstrap_probability_improvement: float


def compute_seasonal_scale(
    y_train: np.ndarray,
    lag_24_train: np.ndarray,
) -> float:
    """Compute the MASE seasonal scale.

    The scale is the in-sample MAE of the seasonal naive (lag-24) forecast,
    computed on **training data only** using temporally aligned values.
    Both arrays must be non-empty, finite, and have the same length.
    Raises ``MetricError`` on invalid inputs.
    """
    y_train = np.asarray(y_train, dtype=float)
    lag_24_train = np.asarray(lag_24_train, dtype=float)

    if len(y_train) == 0:
        raise MetricError("y_train is empty")
    if len(y_train) != len(lag_24_train):
        raise MetricError(
            f"y_train length ({len(y_train)}) != lag_24 length ({len(lag_24_train)})"
        )
    finite_mask = np.isfinite(y_train) & np.isfinite(lag_24_train)
    if not finite_mask.any():
        raise MetricError("No finite values in training arrays")

    errors = np.abs(y_train[finite_mask] - lag_24_train[finite_mask])
    scale = float(np.mean(errors))
    if scale <= 0:
        raise MetricError(
            f"Seasonal scale must be positive, got {scale}. "
            "Check that training data has variation."
        )
    return scale


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_scale: float,
) -> float:
    """Compute the Mean Absolute Scaled Error.

    The ``seasonal_scale`` must be pre-computed (via ``compute_seasonal_scale``)
    on training data only.
    """
    if seasonal_scale <= 0:
        return 0.0
    errors = np.abs(y_true - y_pred)
    return float(np.mean(errors) / seasonal_scale)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100)


def compute_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage of prediction intervals."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_mean_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean width of prediction intervals."""
    return float(np.mean(upper - lower))


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Winkler interval score (lower is better)."""
    width = upper - lower
    below = lower - y_true
    above = y_true - upper
    penalty = (2.0 / alpha) * (np.sum(np.maximum(below, 0)) + np.sum(np.maximum(above, 0)))
    return float(np.sum(width) + penalty) / len(y_true)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    seasonal_scale: float | None = None,
    alpha: float = 0.1,
) -> ForecastMetrics:
    """Compute all evaluation metrics for a forecast.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Point predictions.
    y_lower, y_upper : np.ndarray | None
        Prediction interval bounds.
    seasonal_scale : float | None
        Pre-computed seasonal scale (required for MASE).
    alpha : float
        Significance level for Winkler score.

    Returns
    -------
    ForecastMetrics
    """
    mase = (
        compute_mase(y_true, y_pred, seasonal_scale)
        if (seasonal_scale and seasonal_scale > 0)
        else float("nan")
    )
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)

    interval_coverage = float("nan")
    mean_interval_width = float("nan")
    winkler_score = float("nan")

    if y_lower is not None and y_upper is not None:
        interval_coverage = compute_interval_coverage(y_true, y_lower, y_upper)
        mean_interval_width = compute_mean_interval_width(y_lower, y_upper)
        winkler_score = compute_winkler_score(y_true, y_lower, y_upper, alpha)

    return ForecastMetrics(
        mase=mase,
        mae=mae,
        rmse=rmse,
        smape=smape,
        interval_coverage=interval_coverage,
        mean_interval_width=mean_interval_width,
        winkler_score=winkler_score,
    )


def compute_per_hour_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
    seasonal_scale: float | None = None,
) -> list[PerHourMetrics]:
    """Compute per-hour breakdown of forecast errors."""
    results: list[PerHourMetrics] = []
    for h in range(24):
        mask = hours == h
        count = int(mask.sum())
        if count == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse_val = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mase = (
            compute_mase(yt, yp, seasonal_scale)
            if (seasonal_scale and seasonal_scale > 0)
            else float("nan")
        )
        bias = float(np.mean(yp - yt))
        results.append(
            PerHourMetrics(
                hour=h,
                count=count,
                mae=mae,
                rmse=rmse_val,
                mase=mase,
                bias=bias,
            )
        )
    return results


def bootstrap_paired_mase_improvement(
    mase_baseline: np.ndarray,
    mase_advanced: np.ndarray,
    n_samples: int = 10000,
    ci_level: float = 0.95,
    rng_seed: int = 123,
) -> BootstrapCI:
    """Bootstrap CI for paired MASE improvement (baseline - advanced).

    This function resamples paired aggregate values. The caller is responsible
    for aggregating observations into dependence-preserving blocks, such as
    daily MASE values, before calling it. The returned
    ``bootstrap_probability_improvement`` is the proportion of resamples where
    the mean improvement is positive; it is not a hypothesis-test p-value.
    """
    if len(mase_baseline) != len(mase_advanced):
        raise MetricError(f"Mismatched arrays: {len(mase_baseline)} vs {len(mase_advanced)}")

    improvements = mase_baseline - mase_advanced

    rng = np.random.default_rng(rng_seed)
    n_obs = len(improvements)
    boot_means = np.zeros(n_samples)

    for i in range(n_samples):
        idx = rng.integers(0, n_obs, size=n_obs)
        boot_means[i] = float(np.mean(improvements[idx]))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    prob_pos = float(np.mean(boot_means > 0))

    return BootstrapCI(
        improvement_mean=float(np.mean(improvements)),
        ci_lower=lower,
        ci_upper=upper,
        ci_level=ci_level,
        n_samples=n_samples,
        bootstrap_probability_improvement=prob_pos,
    )
