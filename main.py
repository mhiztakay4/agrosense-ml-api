from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional


# ==================================================
# AGROSENSE NIGERIA — MACHINE LEARNING API
# ==================================================

app = FastAPI(
    title="AgroSense Nigeria API",
    description=(
        "API for future soil nutrient predictions. "
        "Predictions remain unavailable until suitable "
        "trained models have been supplied."
    ),
    version="0.1.0",
)

# Temporary development configuration.
# Before production, replace "*" with your actual website origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MODEL_DIR = Path(__file__).parent / "models"

MODEL_FILES = {
    "nitrate": MODEL_DIR / "nitrate_linear_regression.joblib",
    "ammonium": MODEL_DIR / "ammonium_linear_regression.joblib",
    "phosphorus": MODEL_DIR / "phosphorus_linear_regression.joblib",
}

FEATURES = [
    "nitrate_sensor",
    "ammonium_sensor",
    "phosphorus_sensor",
    "soil_moisture_percent",
    "soil_temperature_c",
    "soil_ph",
]


class SensorReadings(BaseModel):
    nitrate_sensor: Optional[float] = Field(default=None)
    ammonium_sensor: Optional[float] = Field(default=None)
    phosphorus_sensor: Optional[float] = Field(default=None)
    soil_moisture_percent: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    soil_temperature_c: Optional[float] = Field(default=None)
    soil_ph: Optional[float] = Field(default=None, ge=0, le=14)


@app.get("/")
def home():
    return {
        "project": "AgroSense Nigeria",
        "service": "Soil Nutrient Prediction API",
        "status": "online",
        "model_status": get_model_status(),
        "message": (
            "API is running. Scientific predictions require "
            "suitable trained models and validation."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "agrosense-ml-api",
    }


def get_model_status():
    available = {
        name: path.is_file()
        for name, path in MODEL_FILES.items()
    }

    if all(available.values()):
        return {
            "ready": True,
            "models": available,
        }

    return {
        "ready": False,
        "models": available,
        "message": "One or more trained model files are missing.",
    }


@app.get("/model-status")
def model_status():
    return get_model_status()


@app.post("/predict")
def predict(readings: SensorReadings):
    status = get_model_status()

    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_available",
                "message": (
                    "Predictions are disabled because the required "
                    "trained models are not available. Collect suitable "
                    "real sensor and laboratory data, train the models, "
                    "and evaluate them before enabling predictions."
                ),
                "model_status": status,
            },
        )

    values = readings.model_dump()

    # Require every feature used by the model.
    missing = [
        feature for feature in FEATURES
        if values.get(feature) is None
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required sensor readings are missing.",
                "missing_fields": missing,
            },
        )

    row = np.array(
        [[values[feature] for feature in FEATURES]],
        dtype=float,
    )

    if not np.isfinite(row).all():
        raise HTTPException(
            status_code=422,
            detail="Sensor readings must be finite numeric values.",
        )

    results = {}

    try:
        for nutrient, path in MODEL_FILES.items():
            model = joblib.load(path)
            estimate = float(model.predict(row)[0])

            if not np.isfinite(estimate):
                raise ValueError("Model returned a non-finite result.")

            results[nutrient] = estimate

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "The model files could not be loaded or used. "
                "Predictions have been disabled."
            ),
        )

    return {
        "status": "prediction_generated",
        "estimates": results,
        "units": {
            "nitrate": "mg/kg",
            "ammonium": "mg/kg",
            "phosphorus": "mg/kg",
        },
        "warning": (
            "Research estimates only. Verify model calibration, "
            "laboratory methods, units, and independent validation "
            "before interpreting results."
        ),
  }
