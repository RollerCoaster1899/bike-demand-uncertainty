# Model Card: Bike Demand Forecasting

## Model Purpose

Provide day-ahead hourly bike rental demand forecasts with calibrated 90%
prediction intervals for bike-share operations planning.

## Models

### Baseline: Seasonal Naive

- **Type:** Deterministic rule.
- **Inputs:** Lag-24 value of cnt.
- **Outputs:** Point forecast = cnt from 24 hours earlier.
- **Decision horizon:** Single step.

### Advanced: HistGradientBoostingRegressor

- **Type:** Gradient-boosted regression tree ensemble.
- **Implementation:** scikit-learn ``HistGradientBoostingRegressor``.
- **Inputs:** Calendar features (hour, weekday, month, season, year,
  workingday, holiday), lagged demand (24h, 48h, 168h), rolling summaries
  (24h, 48h, 168h means and standard deviations, shifted by 24h).
- **Outputs:** Point forecast (clipped at zero).
- **Hyperparameters:** learning_rate=0.1, max_iter=500, max_leaf_nodes=64,
  min_samples_leaf=20, l2_regularization=0.1, random_state=42.

### Conformal Prediction Intervals

- **Method:** Split-conformal absolute residual quantile.
- **Calibration:** Chronologically separate calibration set.
- **Quantile:** Finite-sample corrected (method='higher').
- **Coverage target:** 90%.

## Intended Use

- Day-ahead operational planning for bike-share systems.
- Staffing and rebalancing schedule optimisation.
- Benchmark for comparing forecast methods.

## Out-of-Scope Use

- Real-time or high-frequency decision-making.
- Safety-critical applications without human oversight.
- Financial investment decisions.
- Generalisation to other cities or time periods without validation.

## Performance Summary

(Verified results are in `reports/metrics/summary.json` and final_report.md.)

## Limitations

1. Limited to two years of data from one city.
2. No weather forecast integration (uses only lag and calendar features).
3. Simple conformal method assumes exchangeability.
4. Single advanced method evaluated.
5. Point forecast only; intervals are post-hoc.
