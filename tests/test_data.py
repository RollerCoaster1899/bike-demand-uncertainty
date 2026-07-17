"""Tests for data loading and validation."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from bike_demand_forecast.config import ExperimentConfig
from bike_demand_forecast.data import (
    DataError,
    _compute_file_sha256,
    _make_synthetic_fixture,
    _validate_raw_data,
    download_and_extract,
    load_data,
    load_raw_data,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert _compute_file_sha256(f) == expected


def test_make_synthetic_fixture_shape() -> None:
    df = _make_synthetic_fixture(n_hours=500, seed=42)
    assert len(df) == 500
    assert "cnt" in df.columns
    assert "casual" in df.columns
    assert "registered" in df.columns
    assert "instant" in df.columns
    assert "dteday" in df.columns
    assert df.index.name == "datetime"
    assert df.index.is_monotonic_increasing


def test_make_synthetic_fixture_deterministic() -> None:
    df1 = _make_synthetic_fixture(n_hours=100, seed=42)
    df2 = _make_synthetic_fixture(n_hours=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_make_synthetic_fixture_nonnegative() -> None:
    df = _make_synthetic_fixture(n_hours=500, seed=42)
    assert (df["cnt"] >= 0).all()
    assert (df["casual"] >= 0).all()
    assert (df["registered"] >= 0).all()


def test_make_synthetic_fixture_decomposition() -> None:
    df = _make_synthetic_fixture(n_hours=500, seed=42)
    assert (df["casual"] + df["registered"] == df["cnt"]).all()


def test_validate_raw_data_passes() -> None:
    df = _make_synthetic_fixture(n_hours=100, seed=42)
    df["dteday"] = df.index.date
    df["hr"] = df.index.hour.astype(int)
    _validate_raw_data(df)


def test_validate_raw_data_missing_column() -> None:
    df = _make_synthetic_fixture(n_hours=100, seed=42)
    df = df.drop(columns=["season"])
    df["dteday"] = df.index.date
    df["hr"] = df.index.hour.astype(int)
    with pytest.raises(DataError, match="Missing required columns"):
        _validate_raw_data(df)


def test_validate_raw_data_bad_decomposition() -> None:
    df = _make_synthetic_fixture(n_hours=100, seed=42)
    df.loc[df.index[0], "cnt"] = 999
    df["dteday"] = df.index.date
    df["hr"] = df.index.hour.astype(int)
    with pytest.raises(DataError, match="cnt != casual"):
        _validate_raw_data(df)


def test_validate_raw_data_negative_target() -> None:
    df = _make_synthetic_fixture(n_hours=100, seed=42)
    df.loc[df.index[0], "cnt"] = -5
    df["dteday"] = df.index.date
    df["hr"] = df.index.hour.astype(int)
    with pytest.raises(DataError, match="Negative cnt"):
        _validate_raw_data(df)


def test_load_data_synthetic() -> None:
    cfg = ExperimentConfig()
    df = load_data(cfg, synthetic=True)
    assert len(df) > 0
    assert "cnt" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_raw_data_file_not_found(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    object.__setattr__(cfg.data, "raw_dir", str(tmp_path))
    with pytest.raises(DataError, match="hour.csv not found"):
        load_raw_data(cfg)


def test_download_with_missing_zip(tmp_path: Path) -> None:
    """Test that download_and_extract fails gracefully with bad URL."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.data, "url", "http://nonexistent.invalid/data.zip")
    object.__setattr__(cfg.data, "raw_dir", str(tmp_path))
    with pytest.raises(DataError):
        download_and_extract(cfg)


def test_zip_extraction_with_hour_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extraction logic with a valid zip containing hour.csv."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.data, "raw_dir", str(tmp_path))
    csv_content = b"a,b,c\n1,2,3\n"
    csv_sha256 = hashlib.sha256(csv_content).hexdigest()

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("hour.csv", csv_content)
        zf.writestr("day.csv", b"x,y\n1,2\n")

    zip_sha256 = hashlib.sha256(zip_buffer.getvalue()).hexdigest()
    object.__setattr__(cfg.data, "url", "https://example.test/data.zip")
    object.__setattr__(cfg.data, "expected_hour_csv_sha256", csv_sha256)
    object.__setattr__(cfg.data, "expected_zip_sha256", zip_sha256)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: BytesIO(zip_buffer.getvalue()),
    )

    result = download_and_extract(cfg)
    assert result.exists()
    assert result.read_bytes() == csv_content


def test_hour_csv_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hash mismatch after extraction raises error."""
    cfg = ExperimentConfig()
    object.__setattr__(cfg.data, "raw_dir", str(tmp_path))
    csv_content = b"a,b,c\n1,2,3\n"

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("hour.csv", csv_content)

    zip_sha256 = hashlib.sha256(zip_buffer.getvalue()).hexdigest()
    object.__setattr__(cfg.data, "url", "https://example.test/data.zip")
    object.__setattr__(cfg.data, "expected_hour_csv_sha256", "0000" + "0" * 60)  # wrong
    object.__setattr__(cfg.data, "expected_zip_sha256", zip_sha256)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: BytesIO(zip_buffer.getvalue()),
    )

    with pytest.raises(DataError, match="SHA-256 mismatch"):
        download_and_extract(cfg)
