from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.ml.cancellation import CancellationPredictor, build_features


@pytest.fixture
def synthetic_cancellation_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 100

    base_time = pd.Timestamp("2026-01-01 06:00:00")
    pickup_times = [base_time + pd.Timedelta(hours=int(i)) for i in rng.integers(0, 24 * 45, size=n_rows)]

    statuses = rng.choice(
        ["Success", "Cancelled by Customer", "Cancelled by Driver"],
        size=n_rows,
        p=[0.7, 0.2, 0.1],
    )

    df = pd.DataFrame(
        {
            "Booking ID": [f"B{i:04d}" for i in range(n_rows)],
            "Customer ID": [f"C{int(i):03d}" for i in rng.integers(1, 16, size=n_rows)],
            "Booking Status": statuses,
            "Vehicle Type": rng.choice(["Auto", "Bike", "Sedan", "SUV", "Prime Sedan"], size=n_rows),
            "Payment Method": rng.choice(["Cash", "UPI", "Card", "Wallet"], size=n_rows),
            "Ride Distance (km)": rng.uniform(1.0, 40.0, size=n_rows),
            "Booking Value (INR)": rng.uniform(80.0, 1200.0, size=n_rows),
            "Driver Ratings": rng.uniform(3.0, 5.0, size=n_rows),
            "Customer Rating": rng.uniform(3.0, 5.0, size=n_rows),
            "Pickup Time": pickup_times,
            "Drop Time": [t + pd.Timedelta(minutes=int(m)) for t, m in zip(pickup_times, rng.integers(10, 90, size=n_rows))],
            "Cancellation Reason (if applicable)": [None] * n_rows,
        }
    )
    return df


def _build_fitted_predictor(constant_class: int = 0) -> CancellationPredictor:
    predictor = CancellationPredictor()
    predictor.feature_names = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "ride_distance_km",
        "booking_value_inr",
        "vehicle_type_encoded",
        "payment_method_encoded",
        "customer_ride_count",
    ]
    predictor.vehicle_categories = ["Auto", "Bike", "Prime Sedan", "SUV", "Sedan", "unknown"]
    predictor.payment_categories = ["Card", "Cash", "UPI", "Wallet", "unknown"]
    predictor.threshold = 0.5

    X_fit = np.array(
        [
            [8, 1, 0, 1, 5.0, 120.0, 0, 2, 0],
            [19, 5, 1, 1, 18.0, 700.0, 3, 1, 3],
            [14, 2, 0, 0, 9.0, 240.0, 2, 0, 1],
            [23, 6, 1, 0, 3.0, 90.0, 1, 3, 5],
        ],
        dtype=float,
    )
    y_fit = np.array([constant_class, constant_class, constant_class, constant_class], dtype=int)

    model = DummyClassifier(strategy="constant", constant=constant_class)
    model.fit(X_fit, y_fit)

    predictor.model = model
    predictor._fitted = True
    return predictor


def test_build_features_returns_correct_columns(synthetic_cancellation_df: pd.DataFrame) -> None:
    features = build_features(synthetic_cancellation_df)

    expected = {
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "ride_distance_km",
        "booking_value_inr",
        "vehicle_type_encoded",
        "payment_method_encoded",
        "customer_ride_count",
        "booking_label",
    }
    assert set(features.columns) == expected


def test_build_features_label_is_binary(synthetic_cancellation_df: pd.DataFrame) -> None:
    features = build_features(synthetic_cancellation_df)
    unique_values = set(features["booking_label"].unique().tolist())

    assert unique_values.issubset({0, 1})


def test_predict_proba_returns_probability_between_0_and_1(
    synthetic_cancellation_df: pd.DataFrame,
) -> None:
    predictor = _build_fitted_predictor(constant_class=0)
    row = synthetic_cancellation_df.iloc[0].to_dict()

    result = predictor.predict_proba(row)

    assert 0.0 <= result["probability_cancelled"] <= 1.0
    assert 0.0 <= result["probability_completed"] <= 1.0


def test_risk_level_low_for_low_probability(synthetic_cancellation_df: pd.DataFrame) -> None:
    predictor = _build_fitted_predictor(constant_class=0)
    row = synthetic_cancellation_df.iloc[1].to_dict()

    result = predictor.predict_proba(row)

    assert result["risk_level"] == "low"


def test_risk_level_high_for_high_probability(synthetic_cancellation_df: pd.DataFrame) -> None:
    predictor = _build_fitted_predictor(constant_class=1)
    row = synthetic_cancellation_df.iloc[2].to_dict()

    result = predictor.predict_proba(row)

    assert result["risk_level"] == "high"


def test_save_and_load_model_preserves_predictions(
    synthetic_cancellation_df: pd.DataFrame, tmp_path: Path
) -> None:
    predictor = _build_fitted_predictor(constant_class=0)
    sample_row = synthetic_cancellation_df.iloc[3].to_dict()

    before = predictor.predict_proba(sample_row)

    model_dir = tmp_path / "cancellation_model"
    predictor.save(str(model_dir))

    loaded = CancellationPredictor.load(str(model_dir))
    after = loaded.predict_proba(sample_row)

    assert pytest.approx(before["probability_cancelled"], rel=1e-9) == after["probability_cancelled"]
    assert pytest.approx(before["probability_completed"], rel=1e-9) == after["probability_completed"]
    assert before["risk_level"] == after["risk_level"]
