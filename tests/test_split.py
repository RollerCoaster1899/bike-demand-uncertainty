"""Tests for temporal splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.split import SplitError, create_temporal_split


def _make_test_data(n_hours: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"cnt": range(n_hours)}, index=idx)


def test_create_temporal_split_basic():
    df = _make_test_data(1000)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)

    split = create_temporal_split(df, cfg)
    assert len(split.idx_train) > 0
    assert len(split.idx_cal) > 0
    assert len(split.idx_test) > 0

    train_set = set(split.idx_train)
    cal_set = set(split.idx_cal)
    test_set = set(split.idx_test)
    assert train_set.isdisjoint(cal_set)
    assert train_set.isdisjoint(test_set)
    assert cal_set.isdisjoint(test_set)

    assert split.idx_train.max() < split.idx_cal.min()
    assert split.idx_cal.max() < split.idx_test.min()


def test_split_preserves_order():
    df = _make_test_data(500)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)

    split = create_temporal_split(df, cfg)
    assert split.idx_train.is_monotonic_increasing
    assert split.idx_cal.is_monotonic_increasing
    assert split.idx_test.is_monotonic_increasing


def test_split_date_boundaries():
    df = _make_test_data(1000)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 10)
    object.__setattr__(cfg.temporal, "min_cal_rows", 5)
    object.__setattr__(cfg.temporal, "min_test_rows", 5)

    split = create_temporal_split(df, cfg)
    assert split.idx_cal[0].hour == 0
    assert split.idx_test[0].hour == 0


def test_split_min_rows_enforced():
    df = _make_test_data(100)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 1000)
    object.__setattr__(cfg.temporal, "min_cal_rows", 1)
    object.__setattr__(cfg.temporal, "min_test_rows", 1)

    with pytest.raises(SplitError, match="Training set has"):
        create_temporal_split(df, cfg)


def test_split_calibration_min_rows():
    df = _make_test_data(200)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 1)
    object.__setattr__(cfg.temporal, "min_cal_rows", 1000)
    object.__setattr__(cfg.temporal, "min_test_rows", 1)

    with pytest.raises(SplitError, match="Calibration set has"):
        create_temporal_split(df, cfg)


def test_split_with_exact_day_boundaries():
    df = _make_test_data(72)
    cfg = ExperimentConfig()
    object.__setattr__(cfg.temporal, "min_train_rows", 1)
    object.__setattr__(cfg.temporal, "min_cal_rows", 1)
    object.__setattr__(cfg.temporal, "min_test_rows", 1)
    object.__setattr__(cfg.temporal, "train_fraction", 0.5)
    object.__setattr__(cfg.temporal, "cal_fraction", 0.25)
    object.__setattr__(cfg.temporal, "test_fraction", 0.25)

    split = create_temporal_split(df, cfg)
    assert len(split.idx_train) > 0
    assert len(split.idx_cal) > 0
    assert len(split.idx_test) > 0
