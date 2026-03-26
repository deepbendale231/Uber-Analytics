from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, table_name: str | None = None, engine=None) -> None:
        self._engine = engine
        self._table_name = table_name

    def _get_engine(self):
        if self._engine is None:
            settings = get_settings()
            self._engine = create_engine(settings.get_db_url(), pool_pre_ping=True)
        return self._engine

    def _get_table_name(self) -> str:
        if self._table_name:
            return self._table_name
        return get_settings().db_table

    def load_from_db(self, query: str) -> pd.DataFrame:
        try:
            engine = self._get_engine()
            df = pd.read_sql_query(query, con=engine)
            logger.info("Loaded %s rows from database query.", len(df))
            return df
        except SQLAlchemyError as exc:
            logger.exception("Database load failed for query: %s", query)
            raise RuntimeError(f"Failed to load data from database: {exc}") from exc

    def load_from_csv(self, path: str) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            message = f"CSV file not found at path: {csv_path}"
            logger.error(message)
            raise FileNotFoundError(message)

        try:
            df = pd.read_csv(csv_path)
            logger.info("Loaded %s rows from CSV: %s", len(df), csv_path)
            return df
        except pd.errors.ParserError as exc:
            logger.exception("CSV parse failed for path: %s", csv_path)
            raise ValueError(f"Unable to parse CSV file at {csv_path}: {exc}") from exc
        except OSError as exc:
            logger.exception("CSV read failed for path: %s", csv_path)
            raise RuntimeError(f"Unable to read CSV file at {csv_path}: {exc}") from exc

    def load_all_rides(self) -> pd.DataFrame:
        table_name = self._get_table_name()
        query = f"SELECT * FROM `{table_name}`"
        df = self.load_from_db(query)
        logger.info("Loaded %s total rides from table '%s'.", len(df), table_name)
        return df

    def load_completed_rides(self) -> pd.DataFrame:
        all_rides = self.load_all_rides()
        completed = all_rides[all_rides["Booking Status"] == "Success"].copy()
        logger.info("Loaded %s completed rides (Booking Status='Success').", len(completed))
        return completed
