"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from bike_demand_forecast.config import (
    ExperimentConfig,
    load_config,
    validate_config,
)


def test_default_config_is_valid():
    cfg = ExperimentConfig()
    validate_config(cfg)


def test_temporal_fractions_sum_to_one():
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        cfg = ExperimentConfig()
        object.__setattr__(cfg.temporal, "train_fraction", 0.5)
        object.__setattr__(cfg.temporal, "cal_fraction", 0.2)
        object.__setattr__(cfg.temporal, "test_fraction", 0.2)
        validate_config(cfg)


def test_invalid_conformal_method():
    with pytest.raises(ValueError, match="conformal method"):
        cfg = ExperimentConfig()
        object.__setattr__(cfg.conformal, "method", "invalid")
        validate_config(cfg)


def test_load_config_from_yaml():
    data = {
        "data": {"url": "http://example.com/data.zip", "raw_dir": "data/raw"},
        "temporal": {
            "train_fraction": 0.5,
            "cal_fraction": 0.25,
            "test_fraction": 0.25,
            "min_train_rows": 100,
            "min_cal_rows": 50,
            "min_test_rows": 50,
        },
        "features": {
            "lag_hours": [24, 48],
            "rolling_windows_hours": [24],
            "calendar": ["hour", "weekday"],
        },
        "ridge": {"alpha": 0.5, "random_state": 42},
        "random_forest": {
            "n_estimators": 50,
            "max_depth": 16,
            "min_samples_leaf": 5,
            "n_jobs": 1,
            "random_state": 42,
        },
        "hgb": {"random_state": 42, "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 32},
        "conformal": {"significance_level": 0.1, "method": "higher"},
        "reporting": {"bootstrap_samples": 500, "bootstrap_ci_level": 0.95},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        fpath = f.name

    try:
        cfg = load_config(fpath)
        assert cfg.temporal.train_fraction == 0.5
        assert cfg.ridge.alpha == 0.5
        assert cfg.conformal.significance_level == 0.1
    finally:
        os.unlink(fpath)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_unknown_section_rejected():
    data = {"unknown_section": {"key": 1}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        fpath = f.name
    try:
        with pytest.raises(ValueError, match="Unknown config section"):
            load_config(fpath)
    finally:
        os.unlink(fpath)


def test_unknown_key_rejected():
    data = {
        "hgb": {"unknown_key": 10},
        "temporal": {
            "train_fraction": 0.6,
            "cal_fraction": 0.15,
            "test_fraction": 0.25,
            "min_train_rows": 100,
            "min_cal_rows": 50,
            "min_test_rows": 50,
        },
        "conformal": {"significance_level": 0.1, "method": "higher"},
        "reporting": {"bootstrap_samples": 500, "bootstrap_ci_level": 0.95},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        fpath = f.name
    try:
        with pytest.raises(ValueError, match="Unknown key"):
            load_config(fpath)
    finally:
        os.unlink(fpath)


def test_negative_lag_rejected():
    cfg = ExperimentConfig()
    object.__setattr__(cfg.features, "lag_hours", (-24,))
    with pytest.raises(ValueError, match="lag_hours"):
        validate_config(cfg)


def test_significance_level_range():
    with pytest.raises(ValueError, match="significance_level"):
        cfg = ExperimentConfig()
        object.__setattr__(cfg.conformal, "significance_level", 0.6)
        validate_config(cfg)


def test_ridge_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.ridge.alpha == 1.0
    assert cfg.ridge.random_state == 42


def test_rf_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.random_forest.n_estimators == 300
    assert cfg.random_forest.n_jobs == 1
