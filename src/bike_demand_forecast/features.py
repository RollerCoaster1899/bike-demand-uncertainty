"""Feature engineering with strict leakage prevention.

Critical rules:
- casual and registered are excluded from features (casual + registered = cnt).
- cnt itself is excluded from final X after building lag/rolling values.
- Target-hour weather variables are excluded (not known at forecast origin).
- Time-based shifts use a complete hourly index to avoid positional leakage.
- Lags and rolling summaries are shifted by 24 hours so they are known
  at forecast origin (preceding midnight).
"""

from __future__ import annotations

import pandas as pd

from bike_demand_forecast.config import ExperimentConfig


class FeatureEngineeringError(Exception):
    """Raised on feature engineering validation failures."""


_FORBIDDEN_IN_X = {
    "instant",
    "dteday",
    "casual",
    "registered",
    "cnt",
    # Target-hour weather (not known without a weather forecast)
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
}

# Columns to drop before feature building but AFTER extracting target.
# cnt is kept during lag/rolling computation then dropped afterwards.
_EARLY_DROP = {
    "instant",
    "dteday",
    "casual",
    "registered",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
}


def reindex_to_complete_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex DataFrame to a complete hourly index.

    This ensures that lagged features computed via ``shift()`` use the
    correct temporal offset and do not accidentally skip missing hours.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (may have gaps).

    Returns
    -------
    pd.DataFrame
        DataFrame with a complete hourly DatetimeIndex. New rows are NaN.
    """
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
    )
    return df.reindex(full_idx)


def _add_calendar_features(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add calendar-based features from the datetime index."""
    result = df.copy()
    cal = cfg.features.calendar

    if "hour" in cal:
        result["hour"] = result.index.hour.astype(int)
    if "weekday" in cal:
        result["weekday"] = result.index.weekday.astype(int)
    if "month" in cal:
        result["month"] = result.index.month.astype(int)
    if "season" in cal and "season" not in result.columns:
        result["season"] = ((result.index.month - 1) // 3 % 4 + 1).astype(int)
    if "year" in cal:
        result["year"] = result.index.year.astype(int)
    if "workingday" in cal and "workingday" not in result.columns:
        result["workingday"] = (result.index.weekday < 5).astype(int)
    if "holiday" in cal and "holiday" not in result.columns:
        result["holiday"] = 0

    return result


def _add_lag_features(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add demand lag features based on the 'cnt' column."""
    result = df.copy()
    for lag in cfg.features.lag_hours:
        result[f"lag_{lag}"] = result["cnt"].shift(lag)
    return result


def _add_rolling_features(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add rolling window summary features.

    All windows are based on the 'cnt' column and shifted by 24 hours
    so they are known at forecast origin.
    """
    result = df.copy()
    for window in cfg.features.rolling_windows_hours:
        result[f"rolling_mean_{window}"] = (
            result["cnt"].shift(24).rolling(window=window, min_periods=1).mean()
        )
        result[f"rolling_std_{window}"] = (
            result["cnt"].shift(24).rolling(window=window, min_periods=1).std()
        )
    return result


def _validate_feature_columns(features: list[str]) -> None:
    """Validate that no forbidden columns exist in the final feature set."""
    found = _FORBIDDEN_IN_X & set(features)
    if found:
        raise FeatureEngineeringError(
            f"Forbidden columns in feature set: {sorted(found)}. "
            f"cnt, casual, registered, and target-hour weather variables "
            f"must be excluded from X."
        )


def create_features(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target vector with strict leakage prevention.

    Steps:
    1. Reindex to complete hourly grid.
    2. Extract target (cnt) before any drops.
    3. Drop early forbidden columns (casual, registered, target-hour weather).
    4. Add calendar features.
    5. Add lag features using time-based shifts (uses cnt column).
    6. Add rolling features shifted by 24 hours (uses cnt column).
    7. Remove cnt from X (final forbidden column).
    8. Drop rows with NaN features (from incomplete lags/windows).
    9. Validate no forbidden columns remain.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with datetime index and at least a 'cnt' column.
    cfg : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix (X) and target series (y = cnt).
    """
    # Step 1: Reindex to complete hourly grid
    df_complete = reindex_to_complete_hourly(df)

    # Step 2: Extract target before dropping columns
    y = df_complete["cnt"].copy()

    # Step 3: Drop early forbidden columns (keep cnt for lag/rolling computation)
    cols_to_drop = [c for c in _EARLY_DROP if c in df_complete.columns]
    X = df_complete.drop(columns=cols_to_drop)

    # Step 4: Add calendar features
    X = _add_calendar_features(X, cfg)

    # Step 5: Add lag features (uses cnt)
    X = _add_lag_features(X, cfg)

    # Step 6: Add rolling features (uses cnt)
    X = _add_rolling_features(X, cfg)

    # Step 7: Remove cnt from X (final forbidden column)
    if "cnt" in X.columns:
        X = X.drop(columns=["cnt"])

    # Step 8: Drop rows with NaN features (from incomplete lags/windows)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # Step 9: Validate no forbidden columns remain
    _validate_feature_columns(list(X.columns))

    # Ensure y is numeric
    y = y.astype(float)

    return X, y
