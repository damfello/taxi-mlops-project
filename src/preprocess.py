"""Data loading and preprocessing utilities for NYC Green Taxi trip duration."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

FEATURE_COLUMNS: list[str] = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET_COLUMN: str = "duration"
TLC_BASE_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_taxi_data(
    year: int = 2024,
    month: int = 1,
    color: str = "green",
    data_dir: Path = Path("data"),
) -> Path:
    """Download TLC trip data parquet file locally if it does not already exist."""
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"

    if not file_path.exists():
        url = f"{TLC_BASE_URL}/{color}_tripdata_{year}-{month:02d}.parquet"
        urlretrieve(url, file_path)

    return file_path


def load_taxi_data(data_path: Path) -> pd.DataFrame:
    """Load a taxi Parquet dataset from disk."""
    return pd.read_parquet(data_path)


def preprocess_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trip duration in minutes, filter outliers, and format zone IDs."""
    prepared = df.copy()
    prepared[TARGET_COLUMN] = (
        prepared["lpep_dropoff_datetime"] - prepared["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    prepared = prepared[
        (prepared[TARGET_COLUMN] >= 1) & (prepared[TARGET_COLUMN] <= 60)
    ]
    prepared = prepared[prepared["trip_distance"].between(0.1, 100)]

    prepared[["PULocationID", "DOLocationID"]] = prepared[
        ["PULocationID", "DOLocationID"]
    ].astype(str)

    return prepared[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()