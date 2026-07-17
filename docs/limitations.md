# Limitations

## Technical Limitations

1. **Limited dataset.** The UCI Bike Sharing dataset covers only two years
   (2011-2012) from a single city (Washington, D.C.). Results may not
   generalise to other locations, time periods, or bike-share systems with
   different usage patterns.

2. **No weather forecast integration.** Target-hour weather variables are
   explicitly excluded because they would not be available at the forecast
   origin without a numerical weather prediction service. A production
   system would benefit from integrating weather forecasts.

3. **Simple conformal method.** The absolute residual quantile method assumes
   exchangeability of residuals and constant uncertainty across the prediction
   space. Adaptive or locally-weighted conformal methods may provide better
   interval calibration.

4. **Limited method set.** The benchmark evaluates Ridge, Random Forest, and
   HistGradientBoosting against seasonal naive. A broader comparison could
   include SARIMA, LightGBM, neural networks, and explicit probabilistic
   forecasters.

5. **Point forecast model.** The model produces point forecasts; prediction
   intervals are derived post-hoc via conformal prediction. An interval
   forecasting method (e.g., quantile regression) might produce better
   calibrated intervals.

## Methodological Limitations

1. **MASE limitations.** MASE can be sensitive to the choice of seasonal
   period and may not be well-defined when the seasonal naive error is zero.

2. **Conformal exchangeability.** Split-conformal prediction assumes
   exchangeability between calibration and test data. Temporal dependence
   in the time series may reduce the validity of this assumption.

3. **Fixed forecast horizon.** The forecast is issued at midnight for the
   next 24 hours. A rolling forecast updated more frequently would be
   more operationally useful.

4. **No uncertainty in features.** The conformal method only quantifies
   uncertainty in the residuals, not uncertainty from feature measurement
   or availability.

## Operational Limitations

1. **No deployment.** The code is a research benchmark and forecasting
   prototype, not a production system. No real-time data pipeline, API,
   or monitoring is implemented.

2. **Offline evaluation.** Metrics are computed on historical data. No
   online or A/B evaluation has been performed.

3. **Single station.** The dataset aggregates demand across the entire
   system. Station-level forecasting would require additional data and
   modeling.
