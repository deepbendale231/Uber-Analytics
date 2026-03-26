from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL


def prepare_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed rides into a daily time series with gap-filled dates."""
    columns = ["ride_count", "total_revenue", "avg_revenue", "avg_rating"]
    if df.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    required = {"Booking Status", "Pickup Time", "Booking Value (INR)"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    work = df.copy()
    work["Pickup Time"] = pd.to_datetime(work["Pickup Time"], errors="coerce")
    work["Booking Value (INR)"] = pd.to_numeric(work["Booking Value (INR)"], errors="coerce")

    rating_cols = [c for c in ["Driver Ratings", "Customer Rating"] if c in work.columns]
    if rating_cols:
        for col in rating_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work["_avg_rating_row"] = work[rating_cols].mean(axis=1, skipna=True)
    else:
        work["_avg_rating_row"] = np.nan

    completed = work[work["Booking Status"].eq("Success")].dropna(subset=["Pickup Time"])
    if completed.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    daily = (
        completed.assign(date=completed["Pickup Time"].dt.floor("D"))
        .groupby("date")
        .agg(
            ride_count=("Pickup Time", "count"),
            total_revenue=("Booking Value (INR)", "sum"),
            avg_revenue=("Booking Value (INR)", "mean"),
            avg_rating=("_avg_rating_row", "mean"),
        )
        .sort_index()
    )

    full_index = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    daily["ride_count"] = daily["ride_count"].fillna(0).astype(int)
    daily["total_revenue"] = daily["total_revenue"].fillna(0.0)
    daily["avg_revenue"] = daily["avg_revenue"].fillna(0.0)
    daily["avg_rating"] = daily["avg_rating"].fillna(0.0)

    return daily


def decompose_series(
    daily_df: pd.DataFrame, column: str = "ride_count", period: int = 7
) -> Dict[str, pd.Series]:
    """Run STL decomposition on a daily metric and return original/trend/seasonal/residual series."""
    empty_series = pd.Series(dtype=float)
    if daily_df.empty or column not in daily_df.columns:
        return {
            "original": empty_series,
            "trend": empty_series,
            "seasonal": empty_series,
            "residual": empty_series,
        }

    series = pd.to_numeric(daily_df[column], errors="coerce").fillna(0.0)
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        index = pd.to_datetime(daily_df.index, errors="coerce")
        series.index = index
    series = series.dropna()

    if series.empty:
        return {
            "original": empty_series,
            "trend": empty_series,
            "seasonal": empty_series,
            "residual": empty_series,
        }

    if len(series) < max(period, 2):
        zeros = pd.Series(np.zeros(len(series)), index=series.index)
        return {
            "original": series,
            "trend": series.copy(),
            "seasonal": zeros,
            "residual": zeros,
        }

    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return {
        "original": series,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }


def plot_decomposition(decomp_dict: Dict[str, pd.Series]) -> go.Figure:
    """Create a four-panel decomposition plot for original, trend, seasonal, and residual components."""
    names = ["original", "trend", "seasonal", "residual"]
    titles = ["Original", "Trend", "Seasonal", "Residual"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=titles)
    for row_idx, (name, title) in enumerate(zip(names, titles), start=1):
        series = decomp_dict.get(name, pd.Series(dtype=float))
        if series is None or series.empty:
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=title), row=row_idx, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, mode="lines", name=title),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        height=900,
        template="plotly_white",
        title_text="STL Decomposition",
        showlegend=False,
    )
    return fig


def detect_anomalies(
    daily_df: pd.DataFrame, column: str = "ride_count", threshold: float = 2.5
) -> pd.DataFrame:
    """Detect anomalies using absolute z-score threshold and label direction."""
    if daily_df.empty or column not in daily_df.columns:
        return pd.DataFrame(columns=list(daily_df.columns) + ["z_score", "anomaly_direction"]) if not daily_df.empty else pd.DataFrame(columns=["z_score", "anomaly_direction"])

    work = daily_df.copy()
    values = pd.to_numeric(work[column], errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)

    if pd.isna(std) or std == 0:
        work["z_score"] = 0.0
        work["anomaly_direction"] = ""
        return work.iloc[0:0].copy()

    work["z_score"] = (values - mean) / std
    anomalies = work[work["z_score"].abs() > threshold].copy()
    if anomalies.empty:
        anomalies["anomaly_direction"] = pd.Series(dtype=str)
        return anomalies

    anomalies["anomaly_direction"] = np.where(anomalies["z_score"] > 0, "spike", "drop")
    return anomalies


def rolling_stats(daily_df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add rolling statistics for ride_count and total_revenue with min_periods=1."""
    if daily_df.empty:
        return daily_df.copy()

    result = daily_df.copy()
    w = max(int(window), 1)

    for col in ["ride_count", "total_revenue"]:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
        roll = result[col].rolling(window=w, min_periods=1)
        result[f"{col}_rolling_mean"] = roll.mean()
        result[f"{col}_rolling_std"] = roll.std().fillna(0.0)
        result[f"{col}_rolling_min"] = roll.min()
        result[f"{col}_rolling_max"] = roll.max()

    return result
