"""Configuration loading, schema validation, and defaults for bike demand forecasting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data source and download parameters."""

    url: str = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    raw_dir: str = "data/raw"
    expected_hour_csv_sha256: str = (
        "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
    )
    expected_zip_sha256: str = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal train/calibration/test split parameters."""

    train_fraction: float = 0.6
    cal_fraction: float = 0.15
    test_fraction: float = 0.25
    min_train_rows: int = 5000
    min_cal_rows: int = 1000
    min_test_rows: int = 1000


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature engineering parameters."""

    lag_hours: tuple[int, ...] = (24, 48, 168)
    rolling_windows_hours: tuple[int, ...] = (24, 48, 168)
    calendar: tuple[str, ...] = (
        "hour",
        "weekday",
        "month",
        "season",
        "year",
        "workingday",
        "holiday",
    )


@dataclass(frozen=True)
class RidgeConfig:
    """Ridge regression parameters."""

    alpha: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class RandomForestConfig:
    """RandomForestRegressor parameters."""

    n_estimators: int = 300
    max_depth: int = 32
    min_samples_leaf: int = 5
    n_jobs: int = -1
    random_state: int = 42


@dataclass(frozen=True)
class HGBConfig:
    """HistGradientBoostingRegressor parameters."""

    random_state: int = 42
    learning_rate: float = 0.1
    max_iter: int = 500
    max_leaf_nodes: int = 64
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1


@dataclass(frozen=True)
class ConformalConfig:
    """Split-conformal prediction interval parameters."""

    significance_level: float = 0.1  # 90% prediction interval
    method: str = "higher"  # finite-sample corrected quantile


@dataclass(frozen=True)
class ReportingConfig:
    """Reporting and bootstrap parameters."""

    bootstrap_samples: int = 10000
    bootstrap_ci_level: float = 0.95


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    hgb: HGBConfig = field(default_factory=HGBConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def config_hash(self) -> str:
        """Return a SHA-256 hex digest of the config values."""
        raw = (
            f"{self.data}{self.temporal}{self.features}"
            f"{self.ridge}{self.random_forest}{self.hgb}"
            f"{self.conformal}{self.reporting}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


_KNOWN_SECTIONS = {
    "data",
    "temporal",
    "features",
    "ridge",
    "random_forest",
    "hgb",
    "conformal",
    "reporting",
}

_KNOWN_KEYS: dict[str, set[str]] = {
    "data": {"url", "raw_dir", "expected_hour_csv_sha256", "expected_zip_sha256"},
    "temporal": {
        "train_fraction",
        "cal_fraction",
        "test_fraction",
        "min_train_rows",
        "min_cal_rows",
        "min_test_rows",
    },
    "features": {"lag_hours", "rolling_windows_hours", "calendar"},
    "ridge": {"alpha", "random_state"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "n_jobs",
        "random_state",
    },
    "hgb": {
        "random_state",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    },
    "conformal": {"significance_level", "method"},
    "reporting": {"bootstrap_samples", "bootstrap_ci_level"},
}


def _validate_range(value: float, name: str, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Validate all config values, raising ValueError on bad values."""
    frac_sum = cfg.temporal.train_fraction + cfg.temporal.cal_fraction + cfg.temporal.test_fraction
    if abs(frac_sum - 1.0) > 1e-4:
        raise ValueError(f"temporal fractions must sum to 1.0, got {frac_sum}")

    _validate_range(cfg.temporal.train_fraction, "train_fraction", 0.1, 0.9)
    _validate_range(cfg.temporal.cal_fraction, "cal_fraction", 0.05, 0.5)
    _validate_range(cfg.temporal.test_fraction, "test_fraction", 0.05, 0.5)
    _validate_range(cfg.temporal.min_train_rows, "min_train_rows", 10, 100000)
    _validate_range(cfg.temporal.min_cal_rows, "min_cal_rows", 10, 100000)
    _validate_range(cfg.temporal.min_test_rows, "min_test_rows", 10, 100000)

    _validate_range(cfg.ridge.alpha, "ridge alpha", 0.0, 10000.0)
    _validate_range(cfg.ridge.random_state, "ridge random_state", 0, 2**31 - 1)

    _validate_range(cfg.random_forest.n_estimators, "rf n_estimators", 10, 5000)
    _validate_range(cfg.random_forest.max_depth, "rf max_depth", 2, 500)
    _validate_range(cfg.random_forest.min_samples_leaf, "rf min_samples_leaf", 1, 500)
    _validate_range(cfg.random_forest.n_jobs, "rf n_jobs", -16, 64)
    _validate_range(cfg.random_forest.random_state, "rf random_state", 0, 2**31 - 1)

    _validate_range(cfg.hgb.random_state, "hgb random_state", 0, 2**31 - 1)
    _validate_range(cfg.hgb.learning_rate, "hgb learning_rate", 0.01, 1.0)
    _validate_range(cfg.hgb.max_iter, "hgb max_iter", 10, 10000)
    _validate_range(cfg.hgb.max_leaf_nodes, "hgb max_leaf_nodes", 2, 1000)
    _validate_range(cfg.hgb.min_samples_leaf, "hgb min_samples_leaf", 1, 1000)
    _validate_range(cfg.hgb.l2_regularization, "hgb l2_regularization", 0.0, 10.0)

    _validate_range(cfg.conformal.significance_level, "significance_level", 0.01, 0.5)
    if cfg.conformal.method not in ("higher", "lower", "nearest"):
        raise ValueError(
            f"conformal method must be 'higher', 'lower', or 'nearest', got {cfg.conformal.method}"
        )

    _validate_range(cfg.reporting.bootstrap_samples, "bootstrap_samples", 100, 1000000)
    _validate_range(cfg.reporting.bootstrap_ci_level, "bootstrap_ci_level", 0.5, 0.999)

    if cfg.features.lag_hours:
        for lag in cfg.features.lag_hours:
            if lag <= 0:
                raise ValueError(f"lag_hours entries must be positive, got {lag}")

    if cfg.features.rolling_windows_hours:
        for rw in cfg.features.rolling_windows_hours:
            if rw <= 0:
                raise ValueError(f"rolling_windows_hours entries must be positive, got {rw}")

    return cfg


def _validate_yaml_structure(raw: Any, source: str) -> None:
    """Validate YAML root is a mapping and reject unknown section/keys."""
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a YAML mapping, got {type(raw).__name__}")
    for section, keys in raw.items():
        if section.startswith("_"):
            continue
        if section not in _KNOWN_SECTIONS:
            raise ValueError(
                f"Unknown config section '{section}' in {source}. "
                f"Known sections: {sorted(_KNOWN_SECTIONS)}"
            )
        if not isinstance(keys, dict):
            raise ValueError(f"Section '{section}' must be a mapping in {source}")
        known = _KNOWN_KEYS.get(section, set())
        for key in keys:
            if key not in known:
                raise ValueError(
                    f"Unknown key '{key}' in section '{section}' in {source}. "
                    f"Known keys: {sorted(known)}"
                )


def load_config(path: str | None = None) -> ExperimentConfig:
    """Load and validate config from a YAML file.

    Returns a validated ExperimentConfig.
    """
    raw: dict[str, Any] = {}
    source = path or "(defaults)"

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p) as f:
            raw = yaml.safe_load(f)
            if raw is None:
                raw = {}
        _validate_yaml_structure(raw, source)

    return validate_config(dict_to_config(raw))


def dict_to_config(d: dict[str, Any]) -> ExperimentConfig:
    """Convert nested dict to ExperimentConfig, filling missing fields with defaults."""
    data = d.get("data", {})
    temporal = d.get("temporal", {})
    features = d.get("features", {})
    ridge = d.get("ridge", {})
    rf = d.get("random_forest", {})
    hgb = d.get("hgb", {})
    conformal = d.get("conformal", {})
    report = d.get("reporting", {})

    return ExperimentConfig(
        data=DataConfig(
            url=str(data.get("url", DataConfig.url)),
            raw_dir=str(data.get("raw_dir", DataConfig.raw_dir)),
            expected_hour_csv_sha256=str(
                data.get("expected_hour_csv_sha256", DataConfig.expected_hour_csv_sha256)
            ),
            expected_zip_sha256=str(
                data.get("expected_zip_sha256", DataConfig.expected_zip_sha256)
            ),
        ),
        temporal=TemporalConfig(
            train_fraction=float(temporal.get("train_fraction", TemporalConfig.train_fraction)),
            cal_fraction=float(temporal.get("cal_fraction", TemporalConfig.cal_fraction)),
            test_fraction=float(temporal.get("test_fraction", TemporalConfig.test_fraction)),
            min_train_rows=int(temporal.get("min_train_rows", TemporalConfig.min_train_rows)),
            min_cal_rows=int(temporal.get("min_cal_rows", TemporalConfig.min_cal_rows)),
            min_test_rows=int(temporal.get("min_test_rows", TemporalConfig.min_test_rows)),
        ),
        features=FeaturesConfig(
            lag_hours=tuple(features.get("lag_hours", list(FeaturesConfig.lag_hours))),
            rolling_windows_hours=tuple(
                features.get("rolling_windows_hours", list(FeaturesConfig.rolling_windows_hours))
            ),
            calendar=tuple(features.get("calendar", list(FeaturesConfig.calendar))),
        ),
        ridge=RidgeConfig(
            alpha=float(ridge.get("alpha", RidgeConfig.alpha)),
            random_state=int(ridge.get("random_state", RidgeConfig.random_state)),
        ),
        random_forest=RandomForestConfig(
            n_estimators=int(rf.get("n_estimators", RandomForestConfig.n_estimators)),
            max_depth=int(rf.get("max_depth", RandomForestConfig.max_depth)),
            min_samples_leaf=int(rf.get("min_samples_leaf", RandomForestConfig.min_samples_leaf)),
            n_jobs=int(rf.get("n_jobs", RandomForestConfig.n_jobs)),
            random_state=int(rf.get("random_state", RandomForestConfig.random_state)),
        ),
        hgb=HGBConfig(
            random_state=int(hgb.get("random_state", HGBConfig.random_state)),
            learning_rate=float(hgb.get("learning_rate", HGBConfig.learning_rate)),
            max_iter=int(hgb.get("max_iter", HGBConfig.max_iter)),
            max_leaf_nodes=int(hgb.get("max_leaf_nodes", HGBConfig.max_leaf_nodes)),
            min_samples_leaf=int(hgb.get("min_samples_leaf", HGBConfig.min_samples_leaf)),
            l2_regularization=float(hgb.get("l2_regularization", HGBConfig.l2_regularization)),
        ),
        conformal=ConformalConfig(
            significance_level=float(
                conformal.get("significance_level", ConformalConfig.significance_level)
            ),
            method=str(conformal.get("method", ConformalConfig.method)),
        ),
        reporting=ReportingConfig(
            bootstrap_samples=int(
                report.get("bootstrap_samples", ReportingConfig.bootstrap_samples)
            ),
            bootstrap_ci_level=float(
                report.get("bootstrap_ci_level", ReportingConfig.bootstrap_ci_level)
            ),
        ),
    )
