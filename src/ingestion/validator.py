from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)


class RideBookingsSchema:
    schema = pa.DataFrameSchema(
        {
            "Booking ID": pa.Column(str, nullable=False, unique=True),
            "Customer ID": pa.Column(str, nullable=False),
            "Booking Status": pa.Column(
                str,
                checks=pa.Check(
                    lambda s: s.isin(
                        [
                            "Success",
                            "Cancelled by Customer",
                            "Cancelled by Driver",
                            "Incomplete",
                        ]
                    )
                ),
                nullable=False,
            ),
            "Vehicle Type": pa.Column(
                str,
                checks=pa.Check(
                    lambda s: s.isin(["Auto", "Bike", "Sedan", "SUV", "Prime Sedan"])
                ),
                nullable=False,
            ),
            "Payment Method": pa.Column(
                str,
                checks=pa.Check(lambda s: s.isin(["Cash", "UPI", "Card", "Wallet"])),
                nullable=False,
            ),
            "Ride Distance (km)": pa.Column(
                float,
                checks=pa.Check(lambda s: (s >= 0) & (s <= 200)),
                nullable=False,
            ),
            "Booking Value (INR)": pa.Column(
                float,
                checks=pa.Check(lambda s: (s >= 0) & (s <= 10000)),
                nullable=False,
            ),
            "Driver Ratings": pa.Column(
                float,
                checks=pa.Check(lambda s: s.isna() | ((s >= 1) & (s <= 5))),
                nullable=True,
            ),
            "Customer Rating": pa.Column(
                float,
                checks=pa.Check(lambda s: s.isna() | ((s >= 1) & (s <= 5))),
                nullable=True,
            ),
            "Pickup Time": pa.Column(pa.DateTime, coerce=True, nullable=False),
            "Drop Time": pa.Column(pa.DateTime, coerce=True, nullable=True),
        },
        strict=False,
        coerce=True,
    )

    @classmethod
    def validate(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            logger.info("Validation summary: total=0 valid=0 invalid=0 pass_rate=0.00%%")
            empty_valid = df.copy()
            empty_invalid = df.copy()
            return empty_valid, empty_invalid

        try:
            valid_rows = cls.schema.validate(df, lazy=True)
            invalid_rows = df.iloc[0:0].copy()
        except SchemaErrors as exc:
            failure_cases = exc.failure_cases
            invalid_indices: list[int] = []
            if "index" in failure_cases.columns:
                index_values = pd.to_numeric(
                    failure_cases["index"], errors="coerce"
                ).dropna()
                invalid_indices = sorted(set(index_values.astype(int).tolist()))

            # Some pandera check errors are schema-level and don't map to specific row indices.
            if invalid_indices:
                invalid_rows = df.loc[invalid_indices].copy()
                valid_rows = df.drop(index=invalid_indices).copy()
            else:
                invalid_rows = df.copy()
                valid_rows = df.iloc[0:0].copy()

            if not invalid_rows.empty:
                output_dir = Path("DATA")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"validation_failures_{datetime.now().strftime('%Y%m%d')}.csv"
                invalid_rows.to_csv(output_path, index=False)
                logger.info("Saved %s invalid rows to %s", len(invalid_rows), output_path)

        total = len(df)
        valid_count = len(valid_rows)
        invalid_count = len(invalid_rows)
        pass_rate = (valid_count / total * 100.0) if total else 0.0
        logger.info(
            "Validation summary: total=%s valid=%s invalid=%s pass_rate=%.2f%%",
            total,
            valid_count,
            invalid_count,
            pass_rate,
        )
        return valid_rows, invalid_rows
