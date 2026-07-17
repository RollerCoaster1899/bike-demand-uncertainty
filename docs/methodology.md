# Methodology

## Problem Formulation

Given historical hourly bike rental counts up to midnight of day $D$,
forecast the demand for each hour $h \in \{0, 1, \ldots, 23\}$ of day
$D+1$ (horizons 1-24). The forecast is issued at midnight.

### Notation

| Symbol | Description |
|---|---|
| $y_{d,h}$ | Bike rental count on day $d$, hour $h$ |
| $\hat{y}_{d,h}$ | Point forecast |
| $\mathcal{T}_{\text{train}}$ | Training period (chronological) |
| $\mathcal{T}_{\text{cal}}$ | Calibration period |
| $\mathcal{T}_{\text{test}}$ | Test period |

### Leakage Prevention

Four critical rules prevent data leakage:

1. **Target column excluded:** ``cnt`` is removed from the feature matrix
   after computing lagged and rolling values.

2. **Decomposition columns excluded:** ``casual`` and ``registered`` are
   removed because ``casual + registered = cnt``. Including them would leak
   the target.

3. **Target-hour weather excluded:** ``weathersit``, ``temp``, ``atemp``,
   ``hum``, ``windspeed`` are removed because these values would not be
   known at the forecast origin without a weather forecast.

4. **Time-based shifts on complete index:** Data is reindexed to a complete
   hourly grid before computing shifts. This ensures that ``shift(24)``
   correctly aligns with data from 24 hours earlier, even if intermediate
   hours are missing in the raw data.

### Features

**Calendar features** (extracted from datetime index):
- hour (0-23)
- weekday (0-6)
- month (1-12)
- season (1-4)
- year
- workingday (0/1)
- holiday (0/1)

**Lagged demand** (based on hourly cnt):
- `lag_24`: value from 24 hours ago
- `lag_48`: value from 48 hours ago
- `lag_168`: value from 168 hours ago (same hour, 7 days ago)

**Rolling summaries** (shifted by 24 hours to be known at forecast origin):
- `rolling_mean_24`, `rolling_std_24`
- `rolling_mean_48`, `rolling_std_48`
- `rolling_mean_168`, `rolling_std_168`

### Temporal Split

Data is split chronologically at day boundaries:

```
|------------ TRAIN ------------|---- CAL ----|------ TEST ------|
2011-01-01 -- ... -- (split 1)  (split 2)      (split 3) -- 2012-12-31
```

Default proportions: 60% train, 15% calibration, 25% test.

## Methods

Four methods are compared on the same feature set and temporal splits.

### 1. Seasonal Naive

$$
\hat{y}_{d,h} = y_{d-1,h}
$$

The forecast for hour $h$ on day $d$ is the observed value from the same
hour on the previous day (lag 24). This is the baseline.

### 2. Ridge Regression

A linear model with L2 regularisation trained on the full feature set.

### 3. Random Forest

An ensemble of decision trees (RandomForestRegressor) trained on the full
feature set. Default: 300 trees, max depth 32, min samples leaf 5.

### 4. HistGradientBoostingRegressor

A gradient-boosted regression tree ensemble from scikit-learn. The model
minimises squared error loss with the following parameters:

- learning_rate: 0.1
- max_iter: 500
- max_leaf_nodes: 64
- min_samples_leaf: 20
- l2_regularization: 0.1
- random_state: 42

All predictions are clipped at zero to respect the nonnegative demand constraint.

## Conformal Prediction Intervals

Split-conformal prediction intervals are computed as:

1. Fit the model on $\mathcal{T}_{\text{train}}$.
2. Compute absolute residuals on $\mathcal{T}_{\text{cal}}$:
   $R_i = |y_i - \hat{y}_i|$.
3. Compute the finite-sample corrected quantile:
   $q = \text{Quantile}\left(R, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$
   with method='higher'.
4. Intervals on test data: $\hat{y}_i \pm q$.

The same procedure is applied for both baseline and advanced predictions.

## Evaluation Metrics

### MASE (Primary)

$$
\text{MASE} = \frac{\frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|}
{\frac{1}{T-24} \sum_{t=25}^T |y_t - y_{t-24}|}
$$

The denominator (seasonal scale) is computed on training data only using
the same seasonal naive (lag-24) forecast.

### Secondary Metrics

- **MAE:** $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **RMSE:** $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
- **sMAPE:** $\frac{100\%}{n} \sum \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$
- **Interval coverage:** $\frac{1}{n} \sum \mathbb{1}(y_i \in [L_i, U_i])$
- **Mean interval width:** $\frac{1}{n} \sum (U_i - L_i)$
- **Winkler score:** $\frac{1}{n} \sum \left[(U_i - L_i) + \frac{2}{\alpha}
   \max(0, L_i - y_i) + \frac{2}{\alpha} \max(0, y_i - U_i) \right]$

### Bootstrap CI for Paired MASE Improvement

Day-block bootstrap (10,000 resamples, sampling days with replacement) is
used to compute 95% confidence intervals for the paired MASE improvement
(baseline MASE - advanced MASE).
