from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models.schemas import HealthResponse
from api.routes.metrics import router as metrics_router
from api.routes.predict import router as predict_router
from src.ingestion.loader import DataLoader
from src.ml.cancellation import CancellationPredictor
from src.ml.churn import ChurnPredictor
from src.ml.demand import DemandForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_rides_with_fallback() -> pd.DataFrame:
    """Load rides from DB first, then fall back to local CSV if DB/env is unavailable."""
    try:
        logger.info("Loading ride data via DataLoader.load_all_rides()")
        loader = DataLoader()
        df = loader.load_all_rides()
        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info("Loaded %s ride rows from database.", len(df))
            return df
        logger.warning("Database returned no rows. Trying CSV fallback.")
    except Exception as exc:
        logger.exception("Failed to load rides from DB: %s", exc)

    csv_path = Path("DATA/ncr_ride_bookings.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)

            # Normalize historical CSV schema to expected API/model schema.
            rename_map = {
                "Booking Value": "Booking Value (INR)",
                "Ride Distance": "Ride Distance (km)",
            }
            df = df.rename(columns=rename_map)

            if "Pickup Time" not in df.columns:
                if "Date" in df.columns and "Time" in df.columns:
                    df["Pickup Time"] = pd.to_datetime(
                        df["Date"].astype(str) + " " + df["Time"].astype(str),
                        errors="coerce",
                    )
                elif "Date" in df.columns:
                    df["Pickup Time"] = pd.to_datetime(df["Date"], errors="coerce")

            # Align status semantics expected by analytics/ML modules.
            if "Booking Status" in df.columns:
                status_clean = df["Booking Status"].astype(str).str.strip().str.lower()
                normalized_status = np.where(
                    status_clean.eq("completed"),
                    "Success",
                    df["Booking Status"].astype(str).str.strip(),
                )
                df["Booking Status"] = normalized_status

            logger.info("Loaded %s ride rows from CSV fallback: %s", len(df), csv_path)
            return df
        except Exception as exc:
            logger.exception("Failed to load rides from CSV fallback: %s", exc)
    else:
        logger.warning("CSV fallback not found at %s", csv_path)

    return pd.DataFrame()


def _load_or_train_cancellation_model(df: pd.DataFrame) -> CancellationPredictor | None:
    """Load persisted cancellation model, or train and persist it if missing."""
    model_dir = Path("models/cancellation_model")
    try:
        logger.info("Loading CancellationPredictor from %s", model_dir)
        model = CancellationPredictor.load(str(model_dir))
        logger.info("CancellationPredictor loaded successfully.")
        return model
    except Exception as exc:
        logger.warning("Cancellation model load failed: %s", exc)

    if df is None or df.empty:
        logger.warning("Skipping cancellation model training because data is empty.")
        return None

    try:
        logger.info("Training CancellationPredictor from loaded data...")
        model = CancellationPredictor()
        model.train(df)
        model.save(str(model_dir))
        logger.info("CancellationPredictor trained and saved to %s", model_dir)
        return model
    except Exception as exc:
        logger.exception("Failed to train/save CancellationPredictor: %s", exc)
        return None


def _load_or_train_churn_model(df: pd.DataFrame) -> ChurnPredictor | None:
    """Load persisted churn model, or train and persist it if missing."""
    model_dir = Path("models/churn_model")
    try:
        logger.info("Loading ChurnPredictor from %s", model_dir)
        model = ChurnPredictor.load(str(model_dir))
        logger.info("ChurnPredictor loaded successfully.")
        return model
    except Exception as exc:
        logger.warning("Churn model load failed: %s", exc)

    if df is None or df.empty:
        logger.warning("Skipping churn model training because data is empty.")
        return None

    try:
        logger.info("Training ChurnPredictor from loaded data...")
        model = ChurnPredictor()
        model.train(df)
        model.save(str(model_dir))
        logger.info("ChurnPredictor trained and saved to %s", model_dir)
        return model
    except Exception as exc:
        logger.exception("Failed to train/save ChurnPredictor: %s", exc)
        return None


tags_metadata = [
    {
        "name": "Predictions",
        "description": "Model scoring endpoints for cancellation, churn, and demand forecasts.",
    },
    {
        "name": "Business Metrics",
        "description": "Operational and analytical KPIs for rides, cohorts, retention, and revenue.",
    },
    {
        "name": "System",
        "description": "Service health and platform diagnostics.",
    },
]

app = FastAPI(
    title="Uber Analytics API",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return structured JSON for validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Request validation failed.",
            "details": exc.errors(),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler for unhandled server errors."""
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load models and data into application state on service startup."""
    logger.info("Starting Uber Analytics API bootstrap...")

    app.state.cancellation_model = None
    app.state.churn_model = None
    app.state.demand_forecaster = None
    app.state.df = pd.DataFrame()

    app.state.df = _load_rides_with_fallback()

    app.state.cancellation_model = _load_or_train_cancellation_model(app.state.df)
    app.state.churn_model = _load_or_train_churn_model(app.state.df)

    try:
        logger.info("Initializing DemandForecaster")
        app.state.demand_forecaster = DemandForecaster()
        logger.info("DemandForecaster initialized.")
    except Exception as exc:
        logger.exception("Failed to initialize DemandForecaster: %s", exc)

    logger.info("Startup bootstrap complete.")


# curl "http://localhost:8000/health"
@app.get("/health", tags=["System"], response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service and model readiness status."""
    model_loaded = bool(
        app.state.cancellation_model is not None
        and app.state.churn_model is not None
        and app.state.demand_forecaster is not None
    )
    row_count = int(len(app.state.df)) if isinstance(app.state.df, pd.DataFrame) else 0

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        row_count=row_count,
        timestamp=datetime.utcnow(),
    )


app.include_router(predict_router)
app.include_router(metrics_router)
