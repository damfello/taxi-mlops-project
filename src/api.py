"""Prediction API for the NYC Green Taxi duration model."""

from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
MODEL_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"

PREDICTION_REQUESTS = Counter(
    "api_prediction_requests_total",
    "Total number of prediction requests handled by the API.",
)
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds",
    "Latency of prediction requests in seconds.",
)

REQUEST_LOG_LOCK = threading.Lock()

app = FastAPI(title="NYC Green Taxi Duration API", version="1.0.0")
logger = logging.getLogger(__name__)


class TaxiPredictionRequest(BaseModel):
    """Input schema for a single taxi trip duration prediction."""

    PULocationID: int = Field(..., ge=1, examples=[74])
    DOLocationID: int = Field(..., ge=1, examples=[41])
    trip_distance: float = Field(..., gt=0, allow_inf_nan=False, examples=[2.5])


def _load_model() -> tuple[Any, dict[str, Any]]:
    """Load the trained model pipeline and metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `uv run dvc repro` first."
        )
    model = joblib.load(MODEL_PATH)
    metadata: dict[str, Any] = {}
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


MODEL, MODEL_METADATA = _load_model()


def _build_model_input(payload: TaxiPredictionRequest) -> list[dict[str, Any]]:
    """Build feature dictionary matching DictVectorizer expectations."""
    return [
        {
            "PU_DO": f"{payload.PULocationID}_{payload.DOLocationID}",
            "trip_distance": payload.trip_distance,
        }
    ]


def _append_request_log(payload: TaxiPredictionRequest, duration_minutes: float) -> None:
    """Append prediction request and result to CSV for monitoring."""
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "PULocationID": payload.PULocationID,
        "DOLocationID": payload.DOLocationID,
        "trip_distance": payload.trip_distance,
        "predicted_duration_minutes": round(duration_minutes, 4),
    }

    with REQUEST_LOG_LOCK:
        file_exists = REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: TaxiPredictionRequest) -> dict[str, Any]:
    """Predict trip duration in minutes and log the request."""
    try:
        start_time = perf_counter()

        features = _build_model_input(payload)
        prediction = float(MODEL.predict(features)[0])

        _append_request_log(payload, prediction)
        PREDICTION_REQUESTS.inc()
        PREDICTION_LATENCY.observe(perf_counter() - start_time)

        return {"predicted_duration_minutes": round(prediction, 2)}
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc