"""Report generation and figure plotting for bike demand forecasting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bike_demand_forecast.models import METHOD_NAMES, TRAINED_METHODS
from bike_demand_forecast.pipeline import ExperimentResults


def generate_report(results: ExperimentResults, out_dir: Path) -> None:
    """Generate final_report.md from experiment results with conditional conclusions."""
    meta = results.metadata
    mode_str = meta.mode.upper()

    lines: list[str] = []
    _add = lines.append

    _add("# Bike Demand Forecasting Report")
    _add("")
    _add(f"**Mode:** {mode_str}  ")
    _add(f"**Run ID:** {meta.run_id}  ")
    _add(f"**Config hash:** {meta.config_hash}  ")
    _add(f"**Source:** {meta.source_label}  ")
    _add(f"**Start:** {meta.timestamp_start}  ")
    _add(f"**End:** {meta.timestamp_end}  ")
    _add(f"**Duration:** {meta.duration_seconds:.1f}s  ")
    _add(f"**Training samples:** {meta.n_train}  ")
    _add(f"**Calibration samples:** {meta.n_cal}  ")
    _add(f"**Test samples:** {meta.n_test}  ")
    _add(f"**Features:** {meta.n_features}  ")
    _add(f"**Seasonal scale (MASE):** {results.seasonal_scale:.2f}  ")
    _add(f"**Selected method:** {results.selected_method} (by calibration MASE)  ")
    _add("")
    _add("---")
    _add("")

    # Executive summary
    _add("## Executive Summary")
    _add("")
    _add(
        f"Four forecasting methods were evaluated on {meta.source_label} data "
        f"using a chronological train/calibration/test split. "
        f"The best method by calibration MASE is **{results.selected_method}**.  "
    )
    _add("")
    _add("Method ranking by calibration MASE (lower is better):  ")
    for method, mase_val in sorted(results.cal_mase.items(), key=lambda x: x[1]):
        _add(f"- {method}: {mase_val:.4f}  ")
    _add("")

    _add("---")
    _add("")

    # Results table
    _add("## Results (Test Set)")
    _add("")
    header = "| Metric | " + " | ".join(METHOD_NAMES) + " |"
    sep = "|" + "|".join("---" for _ in range(len(METHOD_NAMES) + 1)) + "|"
    _add(header)
    _add(sep)
    for metric in ("mase", "mae", "rmse", "smape"):
        parts = [f"**{metric.upper()}**"]
        for m in METHOD_NAMES:
            v = results.method_metrics[m].get(metric)
            if metric == "smape":
                parts.append(f"{v:.2f}" if v is not None else "N/A")
            else:
                parts.append(f"{v:.4f}" if v is not None else "N/A")
        _add("| " + " | ".join(parts) + " |")
    _add("")

    # Interval metrics
    _add("### Prediction Intervals (90% split-conformal)")
    _add("")
    iheader = "| Metric | " + " | ".join(METHOD_NAMES) + " |"
    _add(iheader)
    _add(sep)
    for metric in ("interval_coverage", "mean_interval_width", "winkler_score"):
        parts = [f"**{metric}**"]
        for m in METHOD_NAMES:
            v = results.method_metrics[m].get(metric)
            if v is not None and not (isinstance(v, float) and v != v):
                if metric == "interval_coverage":
                    parts.append(f"{v:.1%}")
                else:
                    parts.append(f"{v:.2f}")
            else:
                parts.append("N/A")
        _add("| " + " | ".join(parts) + " |")
    _add("")
    below_target = [
        method
        for method in METHOD_NAMES
        if results.method_metrics[method]["interval_coverage"] < 0.9
    ]
    if below_target:
        _add(
            "**Coverage caveat:** The following methods fell below the 90% empirical "
            f"coverage target: {', '.join(below_target)}. Chronological calibration and "
            "test residuals are not guaranteed to be exchangeable, so these intervals "
            "are experimental uncertainty estimates rather than coverage guarantees."
        )
        _add("")

    # Comparisons
    _add("### Paired MASE Improvement vs Seasonal Naive")
    _add("")
    _add(
        "Day-block bootstrap 95% CI. Positive improvement means the alternative outperforms naive.  "
    )
    _add("")
    cheader = "| Method | Mean improvement | 95% CI | Bootstrap probability of improvement |"
    csep = "|---|---|---|---|"
    _add(cheader)
    _add(csep)
    for method in TRAINED_METHODS:
        c = results.comparisons.get(method)
        if c and c.get("ci_lower") is not None:
            bp = c.get("bootstrap_probability_improvement", 0)
            _add(
                f"| {method} | {c['improvement_mean']:.4f} | "
                f"[{c['ci_lower']:.4f}, {c['ci_upper']:.4f}] | "
                f"{bp:.1%} |"
            )
        else:
            _add(f"| {method} | N/A | N/A | N/A |")
    _add("")

    _add("---")
    _add("")

    # Dataset
    _add("## Dataset")
    _add("")
    if meta.source_label == "UCI Bike Sharing":
        _add(
            "The UCI Bike Sharing dataset (DOI 10.24432/C5W894, CC BY 4.0) "
            "contains 17,389 hourly records of bike rental demand in Washington, "
            "D.C. from 2011-2012.  "
        )
    else:
        _add(
            "**Synthetic fixture.** This run uses a deterministic synthetic "
            "bike-demand-like fixture generated for CI/testing only. Results "
            "should not be interpreted as real-world performance.  "
        )
    _add("")

    _add("---")
    _add("")

    # Methodology
    _add("## Methodology")
    _add("")
    _add("### Methods  ")
    _add(
        "1. **Seasonal Naive** -- Predict using value from 24 hours earlier "
        "(strong baseline for hourly demand with daily seasonality).  "
    )
    _add(
        "2. **Ridge Regression** -- Linear model with L2 regularisation, "
        "trained on the same feature set.  "
    )
    _add("3. **Random Forest** -- Ensemble of decision trees, trained on the same feature set.  ")
    _add(
        "4. **HistGradientBoosting** -- Gradient-boosted tree ensemble "
        "(scikit-learn), trained on the same feature set.  "
    )
    _add("")
    _add("All predictions are clipped at zero.  ")
    _add("")

    _add("### Conformal Prediction Intervals  ")
    _add(
        "Split-conformal prediction intervals (absolute residual quantile, "
        "finite-sample correction, method='higher', 90% target coverage).  "
    )
    _add("")

    _add("### Temporal Validation  ")
    _add("Chronological train/cal/test split at day boundaries. No random shuffle.  ")
    _add("")

    _add("### Feature Leakage Prevention  ")
    _add("- ``casual``, ``registered``, ``cnt`` excluded from features.  ")
    _add("- Target-hour weather variables excluded.  ")
    _add("- Data reindexed to complete hourly grid before time-based shifts.  ")
    _add("- Rolling windows shifted by 24 hours.  ")
    _add("")

    _add("---")
    _add("")

    # Error analysis
    _add("## Error Analysis")
    _add("")
    _add("Per-hour breakdown of forecast errors is available in ")
    _add("``tables/per_hour_metrics.csv``.  ")
    _add("")

    # Per-hour table for selected method
    sel = results.selected_method
    if results.method_per_hour.get(sel):
        _add(f"### Per-Hour Metrics for Selected Method ({sel})")
        _add("")
        _add("| Hour | Count | MAE | RMSE | MASE | Bias |")
        _add("|---|---|---|---|---|---|")
        for row in results.method_per_hour[sel]:
            _add(
                f"| {row['hour']:02d}:00 | {row['count']} | {row['mae']:.2f} | "
                f"{row['rmse']:.2f} | {row['mase']:.4f} | {row['bias']:.2f} |"
            )
        _add("")

    _add("---")
    _add("")

    # Figures
    _add("## Figures")
    _add("")
    _add("![Model comparison](figures/model_comparison.png)  ")
    _add("![Error by hour](figures/error_by_hour.png)  ")
    _add("![Prediction intervals](figures/prediction_intervals.png)  ")
    _add("")

    _add("---")
    _add("")

    # Conclusions (conditional on actual results)
    _add("## Conclusions")
    _add("")

    # Find the method with lowest test MASE (for reporting)
    mase_ranking = sorted(
        [(m, results.method_metrics[m]["mase"]) for m in METHOD_NAMES],
        key=lambda x: x[1],
    )
    best_test, best_mase = mase_ranking[0]
    worst_mase = mase_ranking[-1][1]

    _add(f"Test set results show a range of MASE values from {best_test} ")
    _add(f"(MASE {best_mase:.4f}) to {mase_ranking[-1][0]} (MASE {worst_mase:.4f}).  ")
    _add("")

    # Check if selected (by calibration) is also best on test
    if best_test == results.selected_method:
        _add(
            f"The calibration-selected method ({results.selected_method}) also "
            f"achieves the lowest test MASE.  "
        )
    else:
        _add(
            f"The calibration-selected method ({results.selected_method}) is not "
            f"the lowest test MASE ({best_test}); calibration selection did not "
            f"overfit to the test set.  "
        )

    for method in TRAINED_METHODS:
        c = results.comparisons.get(method)
        if c and c.get("ci_lower") is not None:
            bp = c["bootstrap_probability_improvement"]
            if c["ci_lower"] > 0:
                _add(
                    f"For {method}, the paired day-block bootstrap MASE-improvement "
                    f"CI excludes zero ([{c['ci_lower']:.4f}, {c['ci_upper']:.4f}]); "
                    f"{bp:.1%} of bootstrap resamples had positive mean improvement.  "
                )
            elif c["ci_upper"] < 0:
                _add(
                    f"The {method} method performs worse than seasonal naive "
                    f"on average (95% bootstrap CI entirely below zero).  "
                )
            else:
                _add(
                    f"The {method} method shows no clear improvement over "
                    f"seasonal naive (95% bootstrap CI crosses zero; "
                    f"improvement probability {bp:.1%}).  "
                )

    _add("")
    _add(
        "**Important caveat:** These results are based on historical data from "
        "a single city (Washington, D.C., 2011-2012) and may not generalise to "
        "other locations or time periods.  "
    )
    _add("")

    _add("---")
    _add("")

    # Limitations
    _add("## Limitations")
    _add("")
    _add("1. **Limited dataset.** Two years from one city.  ")
    _add("2. **No weather forecast integration.** Uses only lag and calendar features.  ")
    _add("3. **Simple conformal method.** Assumes exchangeability of residuals.  ")
    _add("4. **Offline evaluation.** Metrics on historical data only.  ")
    _add("")

    _add("---")
    _add("")

    # Reproduction
    _add("## Reproduction")
    _add("")
    _add("```bash")
    _add("uv sync --extra dev")
    _add("uv run python scripts/download_data.py")
    _add("uv run python scripts/reproduce.py  # full run")
    _add("```")
    _add("")
    _add("For smoke test:")
    _add("```bash")
    _add("uv run python scripts/reproduce.py --smoke")
    _add("```")
    _add("")

    _add("---")
    _add("")

    # License
    _add("## License")
    _add("")
    _add("Code: MIT. Data: UCI Bike Sharing (CC BY 4.0).")
    _add("")

    report_path = out_dir / "final_report.md"
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("\n".join(line.rstrip() for line in lines).rstrip() + "\n")
    tmp.rename(report_path)


def generate_figures(results: ExperimentResults, figures_dir: Path) -> None:
    """Generate all evaluation figures (4-way bar charts + intervals)."""
    _plot_mase_comparison(results, figures_dir / "model_comparison.png")
    _plot_error_by_hour(results, figures_dir / "error_by_hour.png")
    _plot_prediction_intervals(results, figures_dir / "prediction_intervals.png")


def _plot_mase_comparison(results: ExperimentResults, path: Path) -> None:
    """Bar plot of MASE and MAE for all four methods."""
    methods = list(METHOD_NAMES)
    mase_vals = [results.method_metrics[m]["mase"] for m in methods]
    mae_vals = [results.method_metrics[m]["mae"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(methods, mase_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("MASE")
    ax.set_title("MASE by Method (lower is better)")
    ax.set_xticks(range(len(methods)), labels=methods, rotation=15, ha="right")
    for bar, v in zip(bars, mase_vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1]
    bars = ax.bar(methods, mae_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Method")
    ax.set_xticks(range(len(methods)), labels=methods, rotation=15, ha="right")
    for bar, v in zip(bars, mae_vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_by_hour(results: ExperimentResults, path: Path) -> None:
    """Bar plot of MAE by hour for all methods."""
    methods = list(METHOD_NAMES)
    n_methods = len(methods)
    hours = list(range(24))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(hours))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, method in enumerate(methods):
        by_hour = {r["hour"]: r["mae"] for r in results.method_per_hour[method]}
        vals = [by_hour.get(h, 0) for h in hours]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=colors[i], alpha=0.7)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error by Hour")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_prediction_intervals(results: ExperimentResults, path: Path) -> None:
    """Plot prediction intervals for the selected (by calibration) method."""
    sel = results.selected_method
    y_true = results.y_test
    y_pred = results.method_predictions[sel]
    y_lower = results.method_lower[sel]
    y_upper = results.method_upper[sel]

    if y_lower is None or y_upper is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "Prediction intervals not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    n_plot = min(168, len(y_true))
    idx = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, y_true[:n_plot], "b-", label="Actual", linewidth=1)
    ax.plot(idx, y_pred[:n_plot], "r-", label=f"Predicted ({sel})", linewidth=1)
    ax.fill_between(
        idx,
        y_lower[:n_plot],
        y_upper[:n_plot],
        alpha=0.2,
        color="red",
        label="90% prediction interval",
    )
    ax.set_xlabel("Hour (first week of test set)")
    ax.set_ylabel("Bike rentals")
    ax.set_title(f"Predictions with 90% Conformal Intervals ({sel})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
