"""Tests for feature engineering with strict leakage prevention."""

from __future__ import annotations

import pandas as pd
import pytest

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.data import _make_synthetic_fixture
from bike_demand_forecast.features import (
    FeatureEngineeringError,
    _add_calendar_features,
    _add_lag_features,
    _add_rolling_features,
    _validate_feature_columns,
    create_features,
    reindex_to_complete_hourly,
)


def _make_test_df(n_hours: int = 500, seed: int = 42) -> pd.DataFrame:
    df = _make_synthetic_fixture(n_hours=n_hours, seed=seed)
    df["dteday"] = df.index.date
    df["hr"] = df.index.hour.astype(int)
    df["instant"] = range(len(df))
    df["weathersit"] = 1
    df["temp"] = 0.5
    df["atemp"] = 0.4
    df["hum"] = 0.6
    df["windspeed"] = 0.3
    return df


def test_reindex_to_complete_hourly_fills_gaps() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-01 02:00:00"),
            pd.Timestamp("2020-01-01 04:00:00"),
        ]
    )
    df = pd.DataFrame({"cnt": [10, 20, 30]}, index=idx)
    result = reindex_to_complete_hourly(df)
    assert len(result) == 5
    assert result["cnt"].isna().sum() == 2


def test_add_calendar_features() -> None:
    df = _make_test_df(100)
    cfg = ExperimentConfig()
    result = _add_calendar_features(df, cfg)
    assert "hour" in result.columns
    assert "weekday" in result.columns
    assert result["hour"].iloc[0] == df.index[0].hour


def test_add_lag_features() -> None:
    df = _make_test_df(200)
    cfg = ExperimentConfig()
    result = _add_lag_features(df, cfg)
    assert "lag_24" in result.columns
    assert pd.isna(result["lag_24"].iloc[0])
    assert not pd.isna(result["lag_24"].iloc[24])


def test_add_rolling_features() -> None:
    df = _make_test_df(200)
    cfg = ExperimentConfig()
    result = _add_rolling_features(df, cfg)
    assert "rolling_mean_24" in result.columns


def test_validate_feature_columns_passes() -> None:
    cols = ["lag_24", "hour", "weekday", "month", "season"]
    _validate_feature_columns(cols)


def test_validate_feature_columns_raises() -> None:
    cols = ["lag_24", "cnt", "casual", "registered"]
    with pytest.raises(FeatureEngineeringError, match="Forbidden columns"):
        _validate_feature_columns(cols)


def test_validate_feature_columns_weather() -> None:
    cols = ["lag_24", "weathersit", "temp"]
    with pytest.raises(FeatureEngineeringError, match="Forbidden columns"):
        _validate_feature_columns(cols)


def test_create_features_drops_forbidden_columns() -> None:
    df = _make_test_df(500)
    cfg = ExperimentConfig()
    X, y = create_features(df, cfg)

    forbidden = {"casual", "registered", "cnt", "weathersit", "temp", "atemp", "hum", "windspeed"}
    intersection = forbidden & set(X.columns)
    assert len(intersection) == 0, f"Forbidden columns found: {intersection}"


def test_create_features_no_cnt_in_x() -> None:
    """Critical: cnt must NOT be in the final feature matrix X."""
    df = _make_test_df(500)
    cfg = ExperimentConfig()
    X, y = create_features(df, cfg)
    assert "cnt" not in X.columns, "cnt found in X -- direct target leakage!"
    assert "casual" not in X.columns
    assert "registered" not in X.columns
    assert "weathersit" not in X.columns
    assert "temp" not in X.columns


def test_create_features_output_types() -> None:
    df = _make_test_df(500)
    cfg = ExperimentConfig()
    X, y = create_features(df, cfg)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_create_features_reindexes() -> None:
    df = _make_test_df(500)
    df_dropped = df.drop(df.index[100])
    cfg = ExperimentConfig()
    X, y = create_features(df_dropped, cfg)
    assert len(X) > 0
    assert X.isna().sum().sum() == 0


def test_create_features_contains_lag24() -> None:
    df = _make_test_df(500)
    cfg = ExperimentConfig()
    X, y = create_features(df, cfg)
    assert "lag_24" in X.columns


def test_leakage_regression() -> None:
    """Regression test: cnt, casual, registered, weather all excluded from X."""
    df = _make_test_df(500)
    cfg = ExperimentConfig()
    X, y = create_features(df, cfg)

    forbidden = {"cnt", "casual", "registered", "weathersit", "temp", "atemp", "hum", "windspeed"}
    found = forbidden & set(X.columns)
    assert not found, f"Leakage detected: {found}"
