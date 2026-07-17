"""Tests for model training and prediction (all four methods)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.models import (
    METHOD_NAMES,
    ModelError,
    predict_method,
    predict_seasonal_naive,
    train_method,
)


def _make_feature_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cnt = np.abs(rng.poisson(100, size=n).astype(float))
    df = pd.DataFrame(
        {
            "cnt": cnt,
            "lag_24": np.concatenate([np.full(24, np.nan), cnt[:-24]]),
            "lag_48": np.concatenate([np.full(48, np.nan), cnt[:-48]]),
            "hour": np.arange(n) % 24,
            "weekday": np.zeros(n),
            "season": np.ones(n),
            "month": np.ones(n),
            "year": np.zeros(n),
            "workingday": np.ones(n),
            "holiday": np.zeros(n),
            "rolling_mean_24": pd.Series(cnt).shift(24).rolling(24).mean().values,
            "rolling_std_24": pd.Series(cnt).shift(24).rolling(24).std().values,
        }
    )
    return df


def test_predict_seasonal_naive_uses_lag24() -> None:
    df = _make_feature_df(200).dropna(subset=["lag_24"])
    preds = predict_seasonal_naive(df)
    assert len(preds) == len(df)
    assert np.all(preds >= 0)


def test_predict_seasonal_naive_with_nan_lag() -> None:
    df = _make_feature_df(50)
    df["lag_24"] = np.nan
    with pytest.raises(ModelError, match="non-finite lag_24"):
        predict_seasonal_naive(df)


def test_predict_seasonal_naive_clipped() -> None:
    df = _make_feature_df(100)
    df["lag_24"] = -10.0
    preds = predict_seasonal_naive(df)
    assert np.all(preds >= 0)


def test_train_ridge_returns_ridge() -> None:
    df = _make_feature_df(500)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]
    model = train_method("ridge", X, y, cfg)
    assert isinstance(model, Ridge)


def test_train_random_forest_returns_rf() -> None:
    df = _make_feature_df(500)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]
    model = train_method("random_forest", X, y, cfg)
    assert isinstance(model, RandomForestRegressor)


def test_train_hgb_returns_hgb() -> None:
    df = _make_feature_df(500)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]
    model = train_method("hist_gradient_boosting", X, y, cfg)
    assert isinstance(model, HistGradientBoostingRegressor)


def test_seasonal_naive_no_model() -> None:
    df = _make_feature_df(100)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]
    model = train_method("seasonal_naive", X, y, cfg)
    assert model is None


def test_all_methods_predict_clipped() -> None:
    df = _make_feature_df(500)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]

    for method in METHOD_NAMES:
        model = train_method(method, X, y, cfg)
        preds = predict_method(method, model, X)
        assert np.all(preds >= 0), f"{method} has negative predictions"
        assert len(preds) == len(X)


def test_all_methods_deterministic() -> None:
    df = _make_feature_df(500)
    cfg = ExperimentConfig()
    feature_cols = [c for c in df.columns if c != "cnt"]
    X = df[feature_cols].dropna()
    y = df["cnt"].loc[X.index]

    for method in METHOD_NAMES:
        model1 = train_method(method, X, y, cfg)
        model2 = train_method(method, X, y, cfg)
        p1 = predict_method(method, model1, X)
        p2 = predict_method(method, model2, X)
        np.testing.assert_array_almost_equal(p1, p2, err_msg=f"{method} not deterministic")


def test_unknown_method() -> None:
    cfg = ExperimentConfig()
    with pytest.raises(ModelError):
        train_method("nonexistent", pd.DataFrame(), pd.Series(), cfg)
