"""Split-conformal prediction intervals using absolute residual quantiles.

Uses a chronologically separate calibration set to compute the quantile
of absolute residuals, then applies finite-sample correction.
"""

from __future__ import annotations

import numpy as np


class ConformalError(Exception):
    """Raised on conformal prediction computation errors."""


def compute_absolute_residual_quantile(
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    significance_level: float = 0.1,
    method: str = "higher",
) -> float:
    """Compute the finite-sample corrected quantile of absolute residuals.

    For split-conformal prediction intervals, the quantile is
    ``ceil((n + 1) * (1 - alpha)) / n``-th empirical quantile of
    absolute residuals on the calibration set.

    Parameters
    ----------
    y_cal : np.ndarray
        True values in the calibration set.
    y_pred_cal : np.ndarray
        Predicted values on the calibration set.
    significance_level : float
        Desired significance level (e.g., 0.1 for 90% intervals).
    method : str
        Quantile interpolation method: "higher", "lower", or "nearest".

    Returns
    -------
    float
        The conformal quantile value.

    Raises
    ------
    ConformalError
        If calibration set is too small or method is invalid.
    """
    n = len(y_cal)
    if n < 2:
        raise ConformalError(f"Calibration set too small: {n} observations (need at least 2)")

    residuals = np.abs(y_cal - y_pred_cal)
    alpha = significance_level

    # Finite-sample corrected quantile index
    # Use the (ceil((n+1)*(1-alpha)) / n) -th quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    # Clip to valid range
    q_level = min(q_level, 1.0)

    if method == "higher":
        q = float(np.quantile(residuals, q_level, method="higher"))
    elif method == "lower":
        q = float(np.quantile(residuals, q_level, method="lower"))
    elif method == "nearest":
        q = float(np.quantile(residuals, q_level, method="nearest"))
    else:
        raise ConformalError(f"Unknown quantile method: {method}")

    return q


def predict_intervals(
    y_pred: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals around point predictions.

    Intervals are [y_pred - quantile, y_pred + quantile], clipped at zero.

    Parameters
    ----------
    y_pred : np.ndarray
        Point predictions.
    quantile : float
        Conformal quantile of absolute residuals.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds of prediction intervals.
    """
    lower = np.clip(y_pred - quantile, 0, None)
    upper = y_pred + quantile
    return lower, upper
