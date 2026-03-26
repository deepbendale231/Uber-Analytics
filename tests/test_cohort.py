from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.cohort import CohortAnalyzer


@pytest.fixture
def sample_cohort_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Booking ID": "B001",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Pickup Time": "2026-01-05 08:00:00",
                "Drop Time": "2026-01-05 08:35:00",
                "Booking Value (INR)": 200.0,
                "Vehicle Type": "Sedan",
                "Payment Method": "UPI",
                "Driver Ratings": 4.8,
                "Customer Rating": 4.7,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B002",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Pickup Time": "2026-02-10 09:00:00",
                "Drop Time": "2026-02-10 09:40:00",
                "Booking Value (INR)": 250.0,
                "Vehicle Type": "Sedan",
                "Payment Method": "Card",
                "Driver Ratings": 4.5,
                "Customer Rating": 4.6,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B003",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Pickup Time": "2026-03-12 10:00:00",
                "Drop Time": "2026-03-12 10:42:00",
                "Booking Value (INR)": 300.0,
                "Vehicle Type": "SUV",
                "Payment Method": "UPI",
                "Driver Ratings": 4.6,
                "Customer Rating": 4.5,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B004",
                "Customer ID": "C002",
                "Booking Status": "Success",
                "Pickup Time": "2026-01-20 11:15:00",
                "Drop Time": "2026-01-20 11:40:00",
                "Booking Value (INR)": 180.0,
                "Vehicle Type": "Auto",
                "Payment Method": "Cash",
                "Driver Ratings": 4.2,
                "Customer Rating": 4.1,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B005",
                "Customer ID": "C002",
                "Booking Status": "Success",
                "Pickup Time": "2026-02-18 12:00:00",
                "Drop Time": "2026-02-18 12:28:00",
                "Booking Value (INR)": 220.0,
                "Vehicle Type": "Auto",
                "Payment Method": "UPI",
                "Driver Ratings": 4.3,
                "Customer Rating": 4.2,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B006",
                "Customer ID": "C003",
                "Booking Status": "Success",
                "Pickup Time": "2026-02-03 08:20:00",
                "Drop Time": "2026-02-03 08:50:00",
                "Booking Value (INR)": 310.0,
                "Vehicle Type": "Bike",
                "Payment Method": "Wallet",
                "Driver Ratings": 4.7,
                "Customer Rating": 4.8,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B007",
                "Customer ID": "C003",
                "Booking Status": "Success",
                "Pickup Time": "2026-03-09 07:30:00",
                "Drop Time": "2026-03-09 07:55:00",
                "Booking Value (INR)": 330.0,
                "Vehicle Type": "Bike",
                "Payment Method": "Wallet",
                "Driver Ratings": 4.8,
                "Customer Rating": 4.9,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B008",
                "Customer ID": "C004",
                "Booking Status": "Success",
                "Pickup Time": "2026-03-14 19:00:00",
                "Drop Time": "2026-03-14 19:35:00",
                "Booking Value (INR)": 500.0,
                "Vehicle Type": "Prime Sedan",
                "Payment Method": "Card",
                "Driver Ratings": 4.9,
                "Customer Rating": 4.8,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B009",
                "Customer ID": "C004",
                "Booking Status": "Cancelled by Customer",
                "Pickup Time": "2026-03-16 20:00:00",
                "Drop Time": None,
                "Booking Value (INR)": 0.0,
                "Vehicle Type": "Prime Sedan",
                "Payment Method": "Card",
                "Driver Ratings": None,
                "Customer Rating": None,
                "Cancellation Reason (if applicable)": "Changed plans",
            },
            {
                "Booking ID": "B010",
                "Customer ID": "C002",
                "Booking Status": "Success",
                "Pickup Time": "2026-03-25 17:00:00",
                "Drop Time": "2026-03-25 17:22:00",
                "Booking Value (INR)": 210.0,
                "Vehicle Type": "Auto",
                "Payment Method": "Cash",
                "Driver Ratings": 4.4,
                "Customer Rating": 4.3,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B011",
                "Customer ID": "C003",
                "Booking Status": "Success",
                "Pickup Time": "2026-01-01 13:00:00",
                "Drop Time": "2026-01-01 13:20:00",
                "Booking Value (INR)": 270.0,
                "Vehicle Type": "Bike",
                "Payment Method": "UPI",
                "Driver Ratings": 4.5,
                "Customer Rating": 4.4,
                "Cancellation Reason (if applicable)": None,
            },
            {
                "Booking ID": "B012",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Pickup Time": "not-a-date",
                "Drop Time": None,
                "Booking Value (INR)": 0.0,
                "Vehicle Type": "Sedan",
                "Payment Method": "UPI",
                "Driver Ratings": None,
                "Customer Rating": None,
                "Cancellation Reason (if applicable)": None,
            },
        ]
    )


def test_build_cohorts_returns_cohort_columns(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    cohort_df = analyzer.build_cohorts()

    assert "CohortMonth" in cohort_df.columns
    assert "CohortIndex" in cohort_df.columns


def test_cohort_index_formula_correct(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    cohort_df = analyzer.build_cohorts()

    customer_row = cohort_df[
        (cohort_df["Customer ID"] == "C001") & (cohort_df["RideMonth"] == pd.Period("2026-03", freq="M"))
    ].iloc[0]

    assert customer_row["CohortMonth"] == pd.Period("2026-01", freq="M")
    assert int(customer_row["CohortIndex"]) == 3


def test_retention_matrix_month1_is_always_1(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    retention = analyzer.get_retention_matrix()

    assert 1 in retention.columns
    assert retention[1].eq(1.0).all()


def test_retention_values_between_0_and_1(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    retention = analyzer.get_retention_matrix()

    assert ((retention >= 0.0) & (retention <= 1.0)).all().all()


def test_invalid_dates_are_dropped(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    cohort_df = analyzer.build_cohorts()

    assert "B012" not in cohort_df["Booking ID"].values


def test_empty_df_raises_value_error() -> None:
    with pytest.raises(ValueError):
        CohortAnalyzer(pd.DataFrame())


def test_get_retention_summary_returns_dict(sample_cohort_df: pd.DataFrame) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    summary = analyzer.get_retention_summary()

    assert isinstance(summary, dict)
    assert "avg_month1_retention" in summary
    assert "best_cohort" in summary
    assert "worst_cohort" in summary
    assert "overall_avg_retention" in summary


def test_export_creates_csv_files(sample_cohort_df: pd.DataFrame, tmp_path: Path) -> None:
    analyzer = CohortAnalyzer(sample_cohort_df)
    analyzer.export_csvs(str(tmp_path))

    counts_path = tmp_path / "cohort_counts.csv"
    retention_path = tmp_path / "cohort_retention_fraction.csv"

    assert counts_path.exists()
    assert retention_path.exists()
