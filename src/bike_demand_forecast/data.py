"""Data acquisition, validation, and loading for the UCI Bike Sharing dataset."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd

from bike_demand_forecast.config import ExperimentConfig


class DataError(Exception):
    """Raised on data acquisition or validation failures."""


_REQUIRED_RAW_COLUMNS = {
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
}


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def download_and_extract(cfg: ExperimentConfig) -> Path:
    """Download the UCI zip and extract hour.csv safely.

    Uses SHA-256 verification, safe extraction via reading the archive
    member directly (no ZipFile.extract to prevent path traversal), and
    atomic write via temp file and rename. Removes the downloaded zip
    after verified extraction.

    Returns the path to the extracted ``hour.csv`` file.
    Raises ``DataError`` on failure.
    """
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    hour_csv_path = raw_dir / "hour.csv"

    # Skip download if hour.csv already exists and hash matches
    if (
        hour_csv_path.exists()
        and _compute_file_sha256(hour_csv_path) == cfg.data.expected_hour_csv_sha256
    ):
        return hour_csv_path

    # Download
    parsed_url = urllib.parse.urlparse(cfg.data.url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise DataError("Dataset URL must be an HTTPS URL without embedded credentials")
    try:
        # The URL scheme and authority are validated immediately above.
        with urllib.request.urlopen(cfg.data.url, timeout=120) as resp:  # nosec B310
            raw_data = resp.read()
    except urllib.error.URLError as e:
        raise DataError(f"Failed to download dataset: {e}") from e

    # Verify zip SHA-256
    actual_zip_hash = _compute_bytes_sha256(raw_data)
    if actual_zip_hash != cfg.data.expected_zip_sha256:
        raise DataError(
            f"Downloaded zip SHA-256 mismatch: expected {cfg.data.expected_zip_sha256}, "
            f"got {actual_zip_hash}"
        )

    # Save zip temporarily (for reference, will be cleaned)
    zip_path = raw_dir / "bike+sharing+dataset.zip"
    zip_path.write_bytes(raw_data)

    # Safely extract hour.csv: read member bytes, write atomically
    try:
        with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
            names = zf.namelist()
            hour_member = None
            for n in names:
                if n.endswith("hour.csv"):
                    hour_member = n
                    break
            if hour_member is None:
                raise DataError("hour.csv not found in zip archive")

            # Read the file bytes directly (safe: no path traversal)
            csv_bytes = zf.read(hour_member)
    except zipfile.BadZipFile as e:
        raise DataError(f"Corrupted zip file: {e}") from e

    # Atomic write to hour_csv_path using temp file
    tmp_path = hour_csv_path.with_suffix(".tmp_download")
    try:
        tmp_path.write_bytes(csv_bytes)
        tmp_path.rename(hour_csv_path)
    except OSError as e:
        raise DataError(f"Failed to write hour.csv: {e}") from e

    # Remove the downloaded zip after verified extraction
    with suppress(OSError):
        zip_path.unlink()

    # Verify extracted file SHA-256
    if not hour_csv_path.exists():
        raise DataError(f"hour.csv not found after extraction at {hour_csv_path}")

    actual_hash = _compute_file_sha256(hour_csv_path)
    if actual_hash != cfg.data.expected_hour_csv_sha256:
        raise DataError(
            f"hour.csv SHA-256 mismatch: expected {cfg.data.expected_hour_csv_sha256}, "
            f"got {actual_hash}"
        )

    return hour_csv_path


def _validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw DataFrame has expected columns and properties.

    Raises ``DataError`` on validation failure.
    """
    missing = _REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise DataError(f"Missing required columns: {sorted(missing)}")

    # Check nonnegative target
    if (df["cnt"] < 0).any():
        raise DataError("Negative cnt values found")

    # Check cnt = casual + registered
    if not (df["cnt"] == df["casual"] + df["registered"]).all():
        raise DataError("Target decomposition check failed: cnt != casual + registered")

    # Check no nulls in required fields
    required_cols = {"season", "yr", "mnth", "hr", "weekday", "workingday", "cnt"}
    nulls = df[list(required_cols & set(df.columns))].isnull().sum()
    if nulls.any():
        raise DataError(f"Null values found in required columns: {nulls[nulls > 0].to_dict()}")

    # Check integer target
    if not pd.api.types.is_integer_dtype(df["cnt"]):
        raise DataError("cnt column is not integer type")


def load_raw_data(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load raw hour.csv, parse dates vectorially, and validate.

    Returns a DataFrame with a proper datetime index.
    """
    hour_csv_path = Path(cfg.data.raw_dir) / "hour.csv"
    if not hour_csv_path.exists():
        raise DataError(
            f"hour.csv not found at {hour_csv_path}. Run 'python scripts/download_data.py' first."
        )

    df = pd.read_csv(hour_csv_path)
    _validate_raw_data(df)

    # Construct datetime vectorially: convert dteday to datetime, add hours as timedelta
    df["datetime"] = pd.to_datetime(df["dteday"], format="%Y-%m-%d") + pd.to_timedelta(
        df["hr"], unit="h"
    )

    # Check timestamp order and uniqueness
    if not df["datetime"].is_monotonic_increasing:
        df = df.sort_values("datetime")
    if df["datetime"].duplicated().any():
        raise DataError("Duplicate timestamps found in raw data")

    df = df.set_index("datetime")
    return df


def _make_synthetic_fixture(
    n_hours: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic bike-demand-like fixture for smoke tests.

    This is clearly synthetic and should not be used for real evaluation.
    cnt == casual + registered is guaranteed by construction.
    All required raw columns are included.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-01 00:00:00", tz=None)
    idx = pd.date_range(start, periods=n_hours, freq="h")

    hour_of_day = idx.hour
    base_pattern = 100 + 50 * np.sin(np.pi * (hour_of_day - 6) / 12)
    base_pattern = np.clip(base_pattern, 10, None)
    weekday_factor = np.where(idx.weekday < 5, 1.2, 0.8)
    noise = rng.normal(0, 15, size=n_hours)
    cnt = (base_pattern * weekday_factor + noise).astype(int)
    cnt = np.clip(cnt, 0, None)
    casual = (cnt * 0.3).astype(int)
    registered = cnt - casual  # guarantee cnt == casual + registered

    df = pd.DataFrame(
        {
            "instant": range(1, n_hours + 1),
            "dteday": idx.date,
            "season": ((idx.month - 1) // 3 % 4 + 1).astype(int),
            "yr": (idx.year - 2011).astype(int),
            "mnth": idx.month.astype(int),
            "hr": hour_of_day.astype(int),
            "holiday": np.zeros(n_hours, dtype=int),
            "weekday": idx.weekday.astype(int),
            "workingday": (idx.weekday < 5).astype(int),
            "weathersit": np.ones(n_hours, dtype=int),
            "temp": 0.5 + 0.3 * np.sin(np.pi * (hour_of_day - 8) / 14),
            "atemp": 0.4 + 0.25 * np.sin(np.pi * (hour_of_day - 8) / 14),
            "hum": 0.6 + 0.2 * np.sin(np.pi * hour_of_day / 12),
            "windspeed": 0.3 + 0.15 * rng.random(n_hours),
            "casual": casual,
            "registered": registered,
            "cnt": cnt,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def load_data(cfg: ExperimentConfig, synthetic: bool = False) -> pd.DataFrame:
    """Load the primary dataset or a synthetic fixture.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    synthetic : bool
        If True, generate a synthetic fixture for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and raw columns.
    """
    if synthetic:
        return _make_synthetic_fixture(n_hours=500, seed=cfg.hgb.random_state)

    return load_raw_data(cfg)
