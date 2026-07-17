"""Model training, prediction, and baseline implementations.

Exactly four methods with fixed config-driven hyperparameters:
  1. seasonal_naive  --  lag-24 prediction (baseline)
  2. ridge           --  Ridge regression
  3. random_forest   --  RandomForestRegressor
  4. hist_gradient_boosting  --  HistGradientBoostingRegressor

All predictions are clipped at zero.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from bike_demand_forecast.config import ExperimentConfig


class ModelError(Exception):
    """Raised on model training or prediction errors."""


# --------------------------------------------------------------------------- #
#  Method dispatch table
# --------------------------------------------------------------------------- #
METHOD_NAMES = (
    "seasonal_naive",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
)
TRAINED_METHODS = METHOD_NAMES[1:]

Model: TypeAlias = Ridge | RandomForestRegressor | HistGradientBoostingRegressor
Trainer: TypeAlias = Callable[[pd.DataFrame, pd.Series, ExperimentConfig], Model]
Predictor: TypeAlias = Callable[[Model, pd.DataFrame], np.ndarray]


def _clip(preds: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(preds, 0, None), dtype=float)


# --------------------------------------------------------------------------- #
#  Seasonal naive (no training required)
# --------------------------------------------------------------------------- #
def predict_seasonal_naive(X: pd.DataFrame) -> np.ndarray:
    """Predict using lag-24 value.

    The feature matrix must contain a ``lag_24`` column.
    """
    if "lag_24" not in X.columns:
        raise ModelError("Seasonal naive forecasting requires the lag_24 feature")
    preds = X["lag_24"].to_numpy(dtype=float, copy=True)
    if not np.isfinite(preds).all():
        invalid_count = int((~np.isfinite(preds)).sum())
        raise ModelError(
            f"Seasonal naive forecasting received {invalid_count} non-finite lag_24 values"
        )

    return _clip(preds)


# --------------------------------------------------------------------------- #
#  Ridge regression
# --------------------------------------------------------------------------- #
def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Ridge:
    """Train a Ridge regression model."""
    rcfg = cfg.ridge
    model = Ridge(
        alpha=rcfg.alpha,
        random_state=rcfg.random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_ridge(model: Ridge, X: pd.DataFrame) -> np.ndarray:
    """Predict using trained Ridge model, clipped at zero."""
    return _clip(model.predict(X))


# --------------------------------------------------------------------------- #
#  Random Forest
# --------------------------------------------------------------------------- #
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> RandomForestRegressor:
    """Train a Random Forest regression model."""
    rcfg = cfg.random_forest
    model = RandomForestRegressor(
        n_estimators=rcfg.n_estimators,
        max_depth=rcfg.max_depth,
        min_samples_leaf=rcfg.min_samples_leaf,
        n_jobs=rcfg.n_jobs,
        random_state=rcfg.random_state,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def predict_random_forest(
    model: RandomForestRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict using trained Random Forest model, clipped at zero."""
    return _clip(model.predict(X))


# --------------------------------------------------------------------------- #
#  HistGradientBoosting
# --------------------------------------------------------------------------- #
def train_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> HistGradientBoostingRegressor:
    """Train a HistGradientBoostingRegressor model."""
    mcfg = cfg.hgb
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=mcfg.learning_rate,
        max_iter=mcfg.max_iter,
        max_leaf_nodes=mcfg.max_leaf_nodes,
        min_samples_leaf=mcfg.min_samples_leaf,
        l2_regularization=mcfg.l2_regularization,
        random_state=mcfg.random_state,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def predict_hgb(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict using trained HGB model, clipped at zero."""
    return _clip(model.predict(X))


# --------------------------------------------------------------------------- #
#  Unified train / predict  (dispatch by method name)
# --------------------------------------------------------------------------- #
_TRAIN_DISPATCH: dict[str, tuple[Trainer, Predictor]] = {
    "ridge": (train_ridge, predict_ridge),
    "random_forest": (train_random_forest, predict_random_forest),
    "hist_gradient_boosting": (train_hgb, predict_hgb),
}


def train_method(
    method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Any:
    """Train a single method by name."""
    if method == "seasonal_naive":
        return None  # no model object needed
    if method not in _TRAIN_DISPATCH:
        raise ModelError(f"Unknown method: {method}")
    trainer, _ = _TRAIN_DISPATCH[method]
    return trainer(X_train, y_train, cfg)


def predict_method(
    method: str,
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict using a trained method by name."""
    if method == "seasonal_naive":
        return predict_seasonal_naive(X)
    if method not in _TRAIN_DISPATCH:
        raise ModelError(f"Unknown method: {method}")
    _, predictor = _TRAIN_DISPATCH[method]
    return predictor(cast(Model, model), X)
