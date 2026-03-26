import pandas as pd
import pytest

from src.ingestion.loader import DataLoader


REAL_CSV_PATH = "DATA/ncr_ride_bookings.csv"


def test_load_from_csv_returns_dataframe() -> None:
    loader = DataLoader()

    df = loader.load_from_csv(REAL_CSV_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_from_csv_missing_file_raises() -> None:
    loader = DataLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_from_csv("DATA/does_not_exist.csv")


def test_completed_rides_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = DataLoader()
    sample_df = pd.DataFrame(
        [
            {"Booking ID": "B1", "Booking Status": "Success"},
            {"Booking ID": "B2", "Booking Status": "Cancelled by Driver"},
            {"Booking ID": "B3", "Booking Status": "Success"},
        ]
    )

    monkeypatch.setattr(loader, "load_all_rides", lambda: sample_df)

    completed = loader.load_completed_rides()

    assert isinstance(completed, pd.DataFrame)
    assert len(completed) == 2
    assert completed["Booking Status"].eq("Success").all()
