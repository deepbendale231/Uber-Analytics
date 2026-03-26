from __future__ import annotations

import logging
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

try:
    from prophet import Prophet
except ModuleNotFoundError:
    Prophet = None

logger = logging.getLogger(__name__)


class DemandForecaster:
    """Forecast ride demand using Prophet with MLflow tracking."""

    def __init__(self, experiment_name: str = "demand_forecasting") -> None:
        """Initialize forecaster and configure MLflow experiment if available."""
        self.experiment_name = experiment_name
        self.model: Any | None = None
        self.forecast_start: pd.Timestamp | None = None

        if mlflow is not None:
            mlflow.set_experiment(experiment_name)
        else:
            logger.warning("MLflow is not installed. Training logs will be skipped.")

    @staticmethod
    def _manual_india_holidays(years: list[int]) -> pd.DataFrame:
        """Return a manual holiday table for key Indian festivals and public holidays."""
        holi_dates = {
            2024: "2024-03-25",
            2025: "2025-03-14",
            2026: "2026-03-03",
            2027: "2027-03-22",
            2028: "2028-03-11",
            2029: "2029-03-01",
            2030: "2030-03-20",
        }
        diwali_dates = {
            2024: "2024-11-01",
            2025: "2025-10-20",
            2026: "2026-11-08",
            2027: "2027-10-29",
            2028: "2028-10-17",
            2029: "2029-11-05",
            2030: "2030-10-26",
        }

        rows: list[dict[str, Any]] = []
        for year in years:
            rows.extend(
                [
                    {"holiday": "new_year", "ds": pd.Timestamp(f"{year}-01-01")},
                    {"holiday": "republic_day", "ds": pd.Timestamp(f"{year}-01-26")},
                    {
                        "holiday": "holi",
                        "ds": pd.Timestamp(holi_dates.get(year, f"{year}-03-15")),
                    },
                    {
                        "holiday": "independence_day",
                        "ds": pd.Timestamp(f"{year}-08-15"),
                    },
                    {
                        "holiday": "diwali",
                        "ds": pd.Timestamp(diwali_dates.get(year, f"{year}-11-01")),
                    },
                    {"holiday": "christmas", "ds": pd.Timestamp(f"{year}-12-25")},
                ]
            )

        holidays = pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"]).sort_values("ds")
        holidays["lower_window"] = 0
        holidays["upper_window"] = 0
        return holidays

    def prepare_prophet_df(self, daily_df: pd.DataFrame, column: str = "ride_count") -> pd.DataFrame:
        """Convert daily time-series data into Prophet format with holiday attributes in attrs."""
        if daily_df.empty:
            result = pd.DataFrame(columns=["ds", "y"])
            result.attrs["holidays"] = self._manual_india_holidays([pd.Timestamp.today().year])
            return result

        if column not in daily_df.columns:
            raise ValueError(f"Column '{column}' not found in daily dataframe.")

        work = daily_df.copy()
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={work.index.name or "index": "ds"})
        elif "date" in work.columns:
            work = work.rename(columns={"date": "ds"})
        elif "ds" not in work.columns:
            raise ValueError("daily_df must have a DatetimeIndex, 'date', or 'ds' column.")

        work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
        work["y"] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(subset=["ds", "y"]).sort_values("ds")

        if work.empty:
            result = pd.DataFrame(columns=["ds", "y"])
            result.attrs["holidays"] = self._manual_india_holidays([pd.Timestamp.today().year])
            return result

        years = list(range(int(work["ds"].dt.year.min()) - 1, int(work["ds"].dt.year.max()) + 3))
        holidays = self._manual_india_holidays(years)

        prophet_df = work[["ds", "y"]].copy()
        if "is_weekend" in daily_df.columns:
            weekend_values = pd.to_numeric(daily_df["is_weekend"], errors="coerce").fillna(0).astype(int)
            prophet_df["is_weekend"] = weekend_values.values[: len(prophet_df)]

        prophet_df.attrs["holidays"] = holidays
        return prophet_df

    def train(self, daily_df: pd.DataFrame, periods: int = 30) -> dict[str, Any]:
        """Train Prophet model when available, else use a naive fallback forecaster."""
        prophet_df = self.prepare_prophet_df(daily_df, column="ride_count")
        if prophet_df.empty:
            raise ValueError("No usable rows found for demand training.")

        if Prophet is None:
            logger.warning("Prophet is not installed. Using naive demand forecast fallback.")
            history = prophet_df.copy().sort_values("ds")
            last_date = pd.to_datetime(history["ds"]).max()
            recent_len = min(56, len(history))
            recent_window = history.tail(recent_len).copy()
            recent_y = pd.to_numeric(recent_window["y"], errors="coerce").fillna(0.0).astype(float)

            base = float(recent_y.mean()) if not recent_y.empty else 0.0
            spread = float(recent_y.std(ddof=0)) if len(recent_y) > 1 else max(base * 0.1, 1.0)

            # Learn day-of-week demand profile from recent history.
            recent_window = recent_window.assign(dow=pd.to_datetime(recent_window["ds"]).dt.dayofweek)
            dow_means = recent_window.groupby("dow")["y"].mean()
            if dow_means.empty:
                seasonal_index = np.ones(7, dtype=float)
            else:
                overall = float(dow_means.mean()) if float(dow_means.mean()) > 0 else max(base, 1.0)
                seasonal_index = np.ones(7, dtype=float)
                for i in range(7):
                    seasonal_index[i] = float(dow_means.get(i, overall)) / overall
                # Keep seasonality reasonable and avoid unstable spikes.
                seasonal_index = np.clip(seasonal_index, 0.80, 1.25)

            # Estimate recent trend slope (rides/day) from last ~8 weeks.
            if len(recent_y) >= 7:
                x = np.arange(len(recent_y), dtype=float)
                slope, _ = np.polyfit(x, recent_y.values, 1)
                slope = float(np.clip(slope, -3.0, 3.0))
            else:
                slope = 0.0

            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")
            horizon = np.arange(1, periods + 1, dtype=float)
            trend_component = np.clip(base + (slope * horizon), a_min=0.0, a_max=None)
            dow_factor = seasonal_index[future_dates.dayofweek.to_numpy()]

            yhat = np.clip(trend_component * dow_factor, a_min=0.0, a_max=None)

            # Uncertainty increases with horizon.
            horizon_spread = spread * (1.0 + 0.02 * np.sqrt(horizon))
            yhat_lower = np.clip(yhat - horizon_spread, a_min=0.0, a_max=None)
            yhat_upper = np.clip(yhat + horizon_spread, a_min=0.0, a_max=None)

            history_fc = pd.DataFrame(
                {
                    "ds": history["ds"],
                    "yhat": history["y"],
                    "yhat_lower": history["y"],
                    "yhat_upper": history["y"],
                    "trend": history["y"].rolling(window=7, min_periods=1).mean(),
                    "weekly": 0.0,
                    "yearly": 0.0,
                }
            )
            future_fc = pd.DataFrame(
                {
                    "ds": future_dates,
                    "yhat": yhat,
                    "yhat_lower": yhat_lower,
                    "yhat_upper": yhat_upper,
                    "trend": trend_component,
                    "weekly": yhat - trend_component,
                    "yearly": np.zeros(periods),
                }
            )
            forecast = pd.concat([history_fc, future_fc], ignore_index=True)
            self.model = None
            self.forecast_start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

            pseudo_window = history.tail(min(30, len(history)))
            mae = 0.0 if pseudo_window.empty else float(mean_absolute_error(pseudo_window["y"], pseudo_window["y"]))
            rmse = 0.0 if pseudo_window.empty else float(np.sqrt(mean_squared_error(pseudo_window["y"], pseudo_window["y"])))
            return {
                "forecast_df": forecast,
                "mae": mae,
                "rmse": rmse,
            }

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
            holidays=prophet_df.attrs.get("holidays"),
        )

        if "is_weekend" in prophet_df.columns:
            model.add_regressor("is_weekend")

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods, freq="D")
        if "is_weekend" in prophet_df.columns:
            weekend_map = prophet_df.set_index("ds")["is_weekend"]
            future["is_weekend"] = future["ds"].map(weekend_map)
            future["is_weekend"] = future["is_weekend"].fillna((future["ds"].dt.dayofweek >= 5).astype(int))

        forecast = model.predict(future)
        self.model = model
        self.forecast_start = pd.to_datetime(prophet_df["ds"].max()) + pd.Timedelta(days=1)

        joined = forecast.merge(prophet_df[["ds", "y"]], on="ds", how="left")
        pseudo_window = joined.dropna(subset=["y"]).tail(min(30, len(joined.dropna(subset=["y"]))))
        if pseudo_window.empty:
            mae = 0.0
            rmse = 0.0
        else:
            mae = float(mean_absolute_error(pseudo_window["y"], pseudo_window["yhat"]))
            rmse = float(np.sqrt(mean_squared_error(pseudo_window["y"], pseudo_window["yhat"])))

        if mlflow is not None:
            with mlflow.start_run(run_name="prophet_demand_v1"):
                mlflow.log_param("periods", int(periods))
                mlflow.log_param("changepoint_prior_scale", 0.05)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                with tempfile.NamedTemporaryFile(suffix="_forecast.csv", delete=False) as tmp:
                    forecast.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, artifact_path="forecast")

        return {
            "forecast_df": forecast,
            "mae": mae,
            "rmse": rmse,
        }

    def plot_forecast(self, forecast_df: pd.DataFrame, actual_df: pd.DataFrame | None = None) -> go.Figure:
        """Plot historical and forecasted demand with uncertainty intervals."""
        fig = go.Figure()

        if actual_df is not None and not actual_df.empty:
            actual = actual_df.copy()
            if isinstance(actual.index, pd.DatetimeIndex):
                x_actual = actual.index
            elif "ds" in actual.columns:
                x_actual = pd.to_datetime(actual["ds"], errors="coerce")
            elif "date" in actual.columns:
                x_actual = pd.to_datetime(actual["date"], errors="coerce")
            else:
                x_actual = pd.RangeIndex(start=0, stop=len(actual))
            y_actual = (
                pd.to_numeric(actual.get("ride_count", actual.get("y", pd.Series(dtype=float))), errors="coerce")
                .fillna(0.0)
                .values
            )
            fig.add_trace(
                go.Scatter(x=x_actual, y=y_actual, mode="lines", line=dict(color="blue"), name="Actual")
            )

        if not forecast_df.empty:
            x_fc = pd.to_datetime(forecast_df["ds"], errors="coerce")
            yhat = pd.to_numeric(forecast_df["yhat"], errors="coerce").fillna(0.0)
            yhat_lower = pd.to_numeric(forecast_df["yhat_lower"], errors="coerce").fillna(yhat)
            yhat_upper = pd.to_numeric(forecast_df["yhat_upper"], errors="coerce").fillna(yhat)

            fig.add_trace(
                go.Scatter(x=x_fc, y=yhat, mode="lines", line=dict(color="orange"), name="Forecast")
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([x_fc, x_fc[::-1]]),
                    y=pd.concat([yhat_upper, yhat_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,165,0,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="Uncertainty",
                )
            )

            forecast_start = self.forecast_start
            if forecast_start is None and actual_df is not None and not actual_df.empty:
                if isinstance(actual_df.index, pd.DatetimeIndex):
                    forecast_start = pd.to_datetime(actual_df.index.max()) + pd.Timedelta(days=1)
            if forecast_start is not None:
                fig.add_vline(x=forecast_start, line_dash="dash", line_color="gray")

        fig.update_layout(
            template="plotly_white",
            title="30-Day Ride Demand Forecast",
            xaxis_title="Date",
            yaxis_title="Ride Count",
        )
        return fig

    def plot_components(self, forecast_df: pd.DataFrame) -> go.Figure:
        """Plot trend, weekly, and yearly components from forecast output."""
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Trend", "Weekly Seasonality", "Yearly Seasonality"],
        )

        if forecast_df.empty:
            for row in [1, 2, 3]:
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines"), row=row, col=1)
        else:
            x_axis = pd.to_datetime(forecast_df["ds"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=pd.to_numeric(forecast_df.get("trend", pd.Series(index=forecast_df.index, dtype=float)), errors="coerce"),
                    mode="lines",
                    name="Trend",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=pd.to_numeric(forecast_df.get("weekly", pd.Series(index=forecast_df.index, dtype=float)), errors="coerce"),
                    mode="lines",
                    name="Weekly",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=pd.to_numeric(forecast_df.get("yearly", pd.Series(index=forecast_df.index, dtype=float)), errors="coerce"),
                    mode="lines",
                    name="Yearly",
                ),
                row=3,
                col=1,
            )

        fig.update_layout(template="plotly_white", height=900, title="Forecast Components", showlegend=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        return fig

    def get_peak_forecast_days(self, forecast_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Return top forecasted demand days with uncertainty bounds."""
        if forecast_df.empty:
            return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

        required = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        missing = [col for col in required if col not in forecast_df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"forecast_df is missing columns: {missing_str}")

        peaks = forecast_df[required].copy()
        peaks["ds"] = pd.to_datetime(peaks["ds"], errors="coerce")
        peaks = peaks.dropna(subset=["ds", "yhat"]).sort_values("yhat", ascending=False)
        return peaks.head(max(int(top_n), 1)).reset_index(drop=True)
