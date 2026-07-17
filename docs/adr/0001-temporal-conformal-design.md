# ADR 0001: Temporal Conformal Design

## Status

Accepted

## Context

The project requires:

1. A chronological train/calibration/test split for time series forecasting.
2. Split-conformal prediction intervals using a separate calibration set.
3. Strict leakage prevention (no casual/registered, no target-hour weather).
4. Reproducible evaluation with deterministic seeds.

## Decision

1. **Chronological split at day boundaries.** Data is split by unique
   calendar dates, not by row count. Default: 60% train, 15% calibration,
   25% test. This preserves temporal order and prevents lookahead.

2. **Split-conformal on calibration set.** The calibration set is a
   contiguous block chronologically between train and test. Absolute
   residuals on calibration are used to compute the conformal quantile.
   The finite-sample corrected quantile (method='higher') ensures
   valid coverage for finite samples.

3. **Complete hourly reindexing.** Before computing time-based shifts, the
   DataFrame is reindexed to a complete hourly grid. This prevents
   positional shift errors when the raw data has missing hours.

4. **Lags and rolling windows shifted by 24 hours.** All lag and rolling
   features are based only on data known at the forecast origin (midnight).
   Rolling windows use a 24-hour shift to ensure no future information leaks.

5. **Forbidden column exclusion.** Columns `casual`, `registered`, and
   target-hour weather variables (`weathersit`, `temp`, `atemp`, `hum`,
   `windspeed`) are dropped before feature creation. Validation checks
   confirm they never appear in the final feature set.

## Consequences

### Positive

- No temporal leakage: test data is always future relative to train.
- Conformal intervals use a proper calibration set, avoiding overfitting.
- Reindexing ensures mathematically correct lag features.
- Validation confirms leakage prevention.

### Negative

- The chronological split may produce imbalanced sets if the time series
  has strong seasonal patterns relative to the split boundaries.
- Conformal intervals may undercover if the calibration set does not
  represent the test distribution (distribution drift).
- Complete-hour reindexing introduces NaN rows that must be dropped.

### Mitigation

- Configurable split fractions allow adjusting the train/cal/test balance.
- The conformal method is clearly documented as a simple approach.
- NaN rows from reindexing and lag computation are explicitly dropped,
  with the number of valid rows reported in metadata.
