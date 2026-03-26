import pandas as pd
import pytest

from src.ingestion.validator import RideBookingsSchema


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Booking ID": "B001",
                "Customer ID": "C001",
                "Booking Status": "Success",
                "Vehicle Type": "Sedan",
                "Payment Method": "UPI",
                "Ride Distance (km)": 12.5,
                "Booking Value (INR)": 350.0,
                "Driver Ratings": 4.8,
                "Customer Rating": 4.6,
                "Pickup Time": "2026-03-01 10:00:00",
                "Drop Time": "2026-03-01 10:35:00",
                "Cancellation Reason (if applicable)": None,
                "Incomplete Rides Reason": None,
            }
        ]
    )


def test_valid_row_passes(sample_valid_df: pd.DataFrame) -> None:
    valid_rows, invalid_rows = RideBookingsSchema.validate(sample_valid_df)
    assert len(valid_rows) == 1
    assert invalid_rows.empty


def test_invalid_rating_fails(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df.loc[0, "Driver Ratings"] = 6.0

    valid_rows, invalid_rows = RideBookingsSchema.validate(df)
    assert valid_rows.empty
    assert len(invalid_rows) == 1


def test_invalid_status_fails(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df.loc[0, "Booking Status"] = "Unknown"

    valid_rows, invalid_rows = RideBookingsSchema.validate(df)
    assert valid_rows.empty
    assert len(invalid_rows) == 1


def test_negative_distance_fails(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df.loc[0, "Ride Distance (km)"] = -5.0

    valid_rows, invalid_rows = RideBookingsSchema.validate(df)
    assert valid_rows.empty
    assert len(invalid_rows) == 1


def test_returns_two_dataframes(sample_valid_df: pd.DataFrame) -> None:
    result = RideBookingsSchema.validate(sample_valid_df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)


def test_empty_df_returns_empty_tuple(sample_valid_df: pd.DataFrame) -> None:
    empty_df = sample_valid_df.iloc[0:0].copy()

    valid_rows, invalid_rows = RideBookingsSchema.validate(empty_df)
    assert isinstance(valid_rows, pd.DataFrame)
    assert isinstance(invalid_rows, pd.DataFrame)
    assert valid_rows.empty
    assert invalid_rows.empty
