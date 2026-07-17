# Project Specification: Bike Demand Forecasting with Calibrated Uncertainty

## Project Title

Bike Demand Forecasting with Calibrated Uncertainty

## Problem Statement

Bike-share operators must schedule bicycle rebalancing and staffing on a
day-ahead basis, which requires accurate hourly demand forecasts with
reliable uncertainty quantification. Underestimating demand leads to
empty stations and lost revenue; overestimating leads to wasted
staff and repositioning costs.

This project develops a day-ahead hourly rental demand forecast (horizons
1-24 hours from the preceding midnight) with calibrated 90% split-conformal
prediction intervals. A seasonal naive baseline is compared against Ridge,
Random Forest, and HistGradientBoosting alternatives on identical splits.

## Target User

Bike-share operations planners scheduling bicycle rebalancing and staffing.

## Primary Output

- Hourly point forecasts for the next 24 hours.
- 90% conformal prediction intervals.
- Reproducible evaluation comparing one baseline with three alternatives.
- Metrics: MASE (primary), MAE, RMSE, sMAPE, interval coverage, Winkler score.

## Dataset

UCI Bike Sharing dataset (dataset id 275).

- **Source:** UCI Machine Learning Repository
- **DOI:** 10.24432/C5W894
- **License:** CC BY 4.0
- **Records:** 17,389 hourly records from 2011-01-01 to 2012-12-31
- **Location:** Washington, D.C., USA

See `docs/data_card.md` for full documentation.

## Prediction Target

Hourly count of total rental bikes (`cnt`), forecast for horizons 1-24 hours
from the preceding midnight.

## Baseline Method

Seasonal naive: predict using the value from 24 hours earlier (lag 24).

## Alternative Methods

Three alternatives test distinct hypotheses:

- **Ridge:** whether a regularized linear relationship captures the dominant
  lag and calendar effects.
- **Random Forest:** whether bagged nonlinear trees improve robustness to
  demand regimes and feature interactions.
- **HistGradientBoosting:** whether sequentially boosted trees reduce residual
  error beyond the bagged and linear alternatives.

All three methods use the same feature set:

- Calendar features: hour, weekday, month, season, year, workingday, holiday
- Demand lags: 24h, 48h, 168h
- Rolling summaries: 24h, 48h, 168h means and standard deviations, shifted
  by 24 hours
- Clipped predictions at zero

## Primary Evaluation Metric

MASE (Mean Absolute Scaled Error) using seasonal scale computed on training
demand only. The seasonal scale is the in-sample MAE of the seasonal naive
(lag-24) forecast on training data.

## Secondary Metrics

- MAE, RMSE, sMAPE
- Interval empirical coverage (90% target)
- Mean interval width
- Winkler interval score

## Validation Strategy

Chronological train/calibration/test split at day boundaries.
No random shuffle.

- Train: 60% of days
- Calibration: 15% of days
- Test: 25% of days

## Expected Repository Deliverables

- Python package `bike_demand_forecast`
- CLI entry point
- Data download script
- Reproduce script
- Configs (full and smoke)
- Tests (unit, integration, smoke)
- Generated report with figures and tables
- Full README with results block
- Project documentation (spec, methodology, data card, model card, limitations, ADR)

## Known Constraints

- CPU-only (no GPU required).
- No weather forecast integration (uses only lag and calendar features).
- Single city (Washington, D.C.).
- Two years of data (2011-2012).

## Major Risks

- Dataset is relatively small (17k hourly records).
- Weather variables are not available at forecast time without a weather
  forecast service.
- Conformal intervals assume exchangeability; temporal dependence may reduce
  calibration quality.

## Idea Score

| Dimension | Score (0-5) | Justification |
|---|---|---|
| Problem value | 4 | Real operational planning problem with clear stakeholder (bike-share operations). |
| Technical depth | 4 | Requires temporal validation, leakage prevention, conformal intervals, and rigorous metrics. |
| Data credibility | 5 | Real public dataset (UCI ML Repo), CC BY 4.0, well-documented, 17k hourly records. |
| Evaluation quality | 5 | Chronological split, MASE with proper seasonal scale, per-hour analysis, bootstrap CI for paired improvement, conformal calibration. |
| Engineering depth | 4 | Full package, CLI, tests, CI, config system, artifact generation, strict leakage prevention. |
| Differentiation | 4 | Conformal uncertainty on bike demand is not a common tutorial; rigorous leakage prevention adds value. |
| Feasibility | 5 | CPU-only, small dataset, scikit-learn model, quick to run. |
| **Total** | **31 / 35** | Exceeds 25 threshold; all dimensions >= 3. |
