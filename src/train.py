"""Train the NYC Green Taxi duration model and log runs/artifacts with MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    download_taxi_data,
    load_taxi_data,
    preprocess_taxi_data,
)

logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
EXPERIMENT_NAME = "green-taxi-monitoring"
REGISTERED_MODEL_NAME = "green-taxi-ride-duration"
MODEL_ALIAS = "production"


def _get_or_create_experiment_id(experiment_name: str, developer: str) -> str:
    """Create or retrieve the MLflow experiment ID."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(
            name=experiment_name,
            tags={"developer": developer},
        )
    return exp.experiment_id


def train_model(
    year: int = 2024,
    month: int = 1,
    color: str = "green",
    test_size: float = 0.2,
    random_state: int = 42,
    developer: str = "student",
    data_path: Path | None = None,
) -> dict[str, float | int | str]:
    """Train the taxi trip duration pipeline, save local artifact, and log to MLflow."""
    load_dotenv(ROOT_DIR / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)

    resolved_data_path = (
        data_path
        if data_path is not None
        else download_taxi_data(
            year=year,
            month=month,
            color=color,
            data_dir=DEFAULT_DATA_DIR,
        )
    )

    raw_df = load_taxi_data(resolved_data_path)
    df_processed = preprocess_taxi_data(raw_df)

    y = df_processed[TARGET_COLUMN]
    X = df_processed.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    experiment_id = _get_or_create_experiment_id(EXPERIMENT_NAME, developer=developer)
    mlflow.set_experiment(experiment_id=experiment_id)

    pipeline = make_pipeline(DictVectorizer(), LinearRegression())

    with mlflow.start_run() as run:
        tags = {
            "model": "linear regression",
            "developer": developer,
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "scope": "historical_replay_taximeter_trip_distance",
        }
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "year": year,
                "month": month,
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        pipeline.fit(X_train.to_dict(orient="records"), y_train)
        y_pred = pipeline.predict(X_test.to_dict(orient="records"))
        rmse = float(root_mean_squared_error(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        metadata = {
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "rmse": round(rmse, 4),
            "scope": "historical_replay_taximeter_trip_distance",
        }
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="mlflow-model",
            serialization_format="skops",
        )

        client = MlflowClient()
        try:
            client.create_registered_model(
                name=REGISTERED_MODEL_NAME,
                tags={"framework": "scikit-learn"},
                description="Model to predict ride duration in NYC",
            )
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_info.model_uri,
            model_id=model_info.model_id,
            run_id=run.info.run_id,
        )
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            alias=MODEL_ALIAS,
        )

        return {
            "run_id": run.info.run_id,
            "rmse": round(rmse, 4),
            "registered_model": REGISTERED_MODEL_NAME,
            "model_version": str(mv.version),
            "alias": MODEL_ALIAS,
            "artifact_path": str(MODEL_PATH),
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train NYC taxi duration model.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--color", type=str, default="green")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--developer", type=str, default="damfello")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = train_model(
        year=args.year,
        month=args.month,
        color=args.color,
        test_size=args.test_size,
        random_state=args.random_state,
        developer=args.developer,
    )
    print(json.dumps(summary, indent=2))