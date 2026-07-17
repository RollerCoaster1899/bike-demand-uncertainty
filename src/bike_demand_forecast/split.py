"""Chronological train/calibration/test splitting at day boundaries.

No random split -- all partitions are temporal and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bike_demand_forecast.config import ExperimentConfig


class SplitError(Exception):
    """Raised on temporal split validation failures."""


@dataclass
class TemporalSplit:
    """Results of a chronological temporal split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    cal_start: pd.Timestamp
    cal_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    idx_train: pd.DatetimeIndex
    idx_cal: pd.DatetimeIndex
    idx_test: pd.DatetimeIndex


def create_temporal_split(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> TemporalSplit:
    """Create a chronological train/calibration/test split.

    Splits are at day boundaries (midnight). The DataFrame must have a
    DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        Data with DatetimeIndex.
    cfg : ExperimentConfig
        Experiment configuration with temporal fractions.

    Returns
    -------
    TemporalSplit
        Split indices and date boundaries.

    Raises
    ------
    SplitError
        If minimum row counts are not met.
    """
    temporal = cfg.temporal
    # Get sorted unique dates
    dates = sorted(df.index.normalize().unique())
    n_dates = len(dates)

    if n_dates < 3:
        raise SplitError(f"Temporal split requires at least 3 distinct days, got {n_dates}")

    # Reserve at least one complete day for calibration and test.
    train_n_dates = min(max(1, int(n_dates * temporal.train_fraction)), n_dates - 2)
    remaining_dates = n_dates - train_n_dates
    cal_n_dates = min(max(1, int(n_dates * temporal.cal_fraction)), remaining_dates - 1)
    train_dates = dates[:train_n_dates]
    cal_dates = dates[train_n_dates : train_n_dates + cal_n_dates]
    test_dates = dates[train_n_dates + cal_n_dates :]

    # Build indices
    idx_train = df.index[df.index.normalize().isin(train_dates)]
    idx_cal = df.index[df.index.normalize().isin(cal_dates)]
    idx_test = df.index[df.index.normalize().isin(test_dates)]

    # Check minimum row counts
    if len(idx_train) < temporal.min_train_rows:
        raise SplitError(
            f"Training set has {len(idx_train)} rows, minimum is {temporal.min_train_rows}"
        )
    if len(idx_cal) < temporal.min_cal_rows:
        raise SplitError(
            f"Calibration set has {len(idx_cal)} rows, minimum is {temporal.min_cal_rows}"
        )
    if len(idx_test) < temporal.min_test_rows:
        raise SplitError(f"Test set has {len(idx_test)} rows, minimum is {temporal.min_test_rows}")

    return TemporalSplit(
        train_start=train_dates[0],
        train_end=train_dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        cal_start=cal_dates[0],
        cal_end=cal_dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        test_start=test_dates[0],
        test_end=test_dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        idx_train=idx_train,
        idx_cal=idx_cal,
        idx_test=idx_test,
    )
