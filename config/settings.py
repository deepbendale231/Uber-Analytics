from __future__ import annotations

from functools import lru_cache
import os
from dataclasses import dataclass
from typing import ClassVar

from dotenv import load_dotenv
from sqlalchemy.engine import URL

# Load environment variables once when this module is imported.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_user: str
    db_password: str
    db_name: str
    db_table: str
    report_email: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    mlflow_tracking_uri: str
    environment: str

    REQUIRED_ENV_VARS: ClassVar[tuple[str, ...]] = (
        "DB_HOST",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME",
        "DB_TABLE",
        "REPORT_EMAIL",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
        "MLFLOW_TRACKING_URI",
        "ENVIRONMENT",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        missing = [key for key in cls.REQUIRED_ENV_VARS if not os.getenv(key)]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables: {missing_list}. "
                "Create a .env file from .env.example and set all required values."
            )

        smtp_port_raw = os.getenv("SMTP_PORT", "")
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError as exc:
            raise ValueError(
                f"SMTP_PORT must be an integer, got: {smtp_port_raw!r}"
            ) from exc

        return cls(
            db_host=os.getenv("DB_HOST", ""),
            db_user=os.getenv("DB_USER", ""),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_name=os.getenv("DB_NAME", ""),
            db_table=os.getenv("DB_TABLE", ""),
            report_email=os.getenv("REPORT_EMAIL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=smtp_port,
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            environment=os.getenv("ENVIRONMENT", ""),
        )

    def get_db_url(self) -> str:
        db_url = URL.create(
            drivername="mysql+pymysql",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            database=self.db_name,
        )
        return db_url.render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
