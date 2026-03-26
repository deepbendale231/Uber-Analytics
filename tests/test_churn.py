from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.ml.churn import ChurnPredictor


@pytest.fixture
def churn_source_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Customer C001: churned (last ride very old)
    for i, dt in enumerate(["2025-11-01", "2025-11-10", "2025-11-20", "2025-12-01"]):
        rows.append(
            {
                "Booking ID": f"B1{i}",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Vehicle Type": "Sedan",
                "Payment Method": "UPI",
                "Ride Distance (km)": 10 + i,
                "Booking Value (INR)": 200 + i * 10,
                "Driver Ratings": 4.5,
                "Customer Rating": 4.2,
                "Pickup Time": dt,
                "Drop Time": dt,
                "Cancellation Reason (if applicable)": None,
            }
        )

    # Customer C002: active (recent rides)
    for i, dt in enumerate(["2026-02-15", "2026-02-20", "2026-03-01", "2026-03-15"]):
        rows.append(
            {
                "Booking ID": f"B2{i}",
                "Customer ID": "C002",
                "Booking Status": "Success",
                "Vehicle Type": "Auto",
                "Payment Method": "Cash",
                "Ride Distance (km)": 6 + i,
                "Booking Value (INR)": 120 + i * 12,
                "Driver Ratings": 4.0,
                "Customer Rating": 4.0,
                "Pickup Time": dt,
                "Drop Time": dt,
                "Cancellation Reason (if applicable)": None,
            }
        )

    # Customer C003: < 3 rides should be excluded
    for i, dt in enumerate(["2026-03-05", "2026-03-10"]):
        rows.append(
            {
                "Booking ID": f"B3{i}",
                "Customer ID": "C003",
                "Booking Status": "Cancelled by Customer" if i == 1 else "Success",
                "Vehicle Type": "Bike",
                "Payment Method": "Card",
                "Ride Distance (km)": 3 + i,
                "Booking Value (INR)": 90 + i * 5,
                "Driver Ratings": 3.9,
                "Customer Rating": 3.8,
                "Pickup Time": dt,
                "Drop Time": dt,
                "Cancellation Reason (if applicable)": "Changed plan" if i == 1 else None,
            }
        )

    # Customer C004: active enough rides
    for i, dt in enumerate(["2026-02-01", "2026-02-09", "2026-03-03", "2026-03-18"]):
        rows.append(
            {
                "Booking ID": f"B4{i}",
                "Customer ID": "C004",
                "Booking Status": "Success",
                "Vehicle Type": "SUV",
                "Payment Method": "Wallet",
                "Ride Distance (km)": 14 + i,
                "Booking Value (INR)": 300 + i * 15,
                "Driver Ratings": 4.6,
                "Customer Rating": 4.5,
                "Pickup Time": dt,
                "Drop Time": dt,
                "Cancellation Reason (if applicable)": None,
            }
        )

    return pd.DataFrame(rows)


def _build_dummy_fitted_predictor() -> ChurnPredictor:
    predictor = ChurnPredictor(churn_days=30)
    predictor.feature_names = [
        "days_since_last_ride",
        "total_rides",
        "rides_last_30d",
        "rides_last_60d",
        "avg_booking_value",
        "avg_rating_given",
        "preferred_vehicle",
        "preferred_payment",
        "completion_rate",
        "avg_ride_distance",
    ]

    X_fit = np.array(
        [
            [120, 5, 0, 0, 220, 4.2, 2, 1, 1.0, 11.0],
            [5, 6, 3, 5, 280, 4.5, 3, 3, 0.9, 15.0],
            [70, 4, 1, 2, 150, 4.0, 0, 0, 0.8, 7.0],
            [10, 7, 4, 6, 330, 4.6, 4, 2, 0.95, 18.0],
        ],
        dtype=float,
    )
    y_fit = np.array([1, 0, 1, 0], dtype=int)

    model = DummyClassifier(strategy="prior")
    model.fit(X_fit, y_fit)

    predictor.model = model
    predictor._fitted = True
    return predictor


def test_churn_label_assigned_correctly(churn_source_df: pd.DataFrame) -> None:
    predictor = ChurnPredictor(churn_days=30)
    features = predictor.build_churn_features(churn_source_df, reference_date=pd.Timestamp("2026-03-20"))

    c001 = features[features["Customer ID"] == "C001"].iloc[0]
    assert int(c001["is_churned"]) == 1


def test_active_customer_not_churned(churn_source_df: pd.DataFrame) -> None:
    predictor = ChurnPredictor(churn_days=30)
    features = predictor.build_churn_features(churn_source_df, reference_date=pd.Timestamp("2026-03-20"))

    c002 = features[features["Customer ID"] == "C002"].iloc[0]
    assert int(c002["is_churned"]) == 0


def test_customers_with_less_than_3_rides_excluded(churn_source_df: pd.DataFrame) -> None:
    predictor = ChurnPredictor(churn_days=30)
    features = predictor.build_churn_features(churn_source_df, reference_date=pd.Timestamp("2026-03-20"))

    assert "C003" not in set(features["Customer ID"].tolist())


def test_churn_probability_between_0_and_1(churn_source_df: pd.DataFrame) -> None:
    predictor = _build_dummy_fitted_predictor()
    scores = predictor.predict_churn_scores(churn_source_df)

    assert not scores.empty
    assert ((scores["churn_probability"] >= 0.0) & (scores["churn_probability"] <= 1.0)).all()


def test_save_load_preserves_predictions(churn_source_df: pd.DataFrame, tmp_path: Path) -> None:
    predictor = _build_dummy_fitted_predictor()
    before = predictor.predict_churn_scores(churn_source_df)

    model_dir = tmp_path / "churn_model"
    predictor.save(str(model_dir))

    loaded = ChurnPredictor.load(str(model_dir))
    after = loaded.predict_churn_scores(churn_source_df)

    merged = before.merge(after, on="Customer ID", suffixes=("_before", "_after"))
    assert not merged.empty
    assert np.allclose(merged["churn_probability_before"], merged["churn_probability_after"])
