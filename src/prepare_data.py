"""Prepare the NYC Green Taxi dataset used by the training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess import download_taxi_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def prepare_data(
    year: int = 2024,
    month: int = 1,
    color: str = "green",
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Ensure the taxi trip dataset is available locally and return its path."""
    return download_taxi_data(
        year=year,
        month=month,
        color=color,
        data_dir=data_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare NYC taxi dataset.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--color", type=str, default="green")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = prepare_data(
        year=args.year,
        month=args.month,
        color=args.color,
    )
    print(f"Prepared dataset at {output_path}")