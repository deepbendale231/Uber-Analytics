from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _is_cancelled(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for cancellation statuses."""
    status = df.get("Booking Status", pd.Series(index=df.index, dtype=object)).astype(str)
    return status.str.contains("Cancelled", case=False, na=False)


def cancellation_funnel(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute cancellation diagnostics across reasons, vehicle, hour, payment, and weekday-hour heatmap."""
    empty = {
        "by_reason": pd.DataFrame(columns=["Cancellation Reason", "count"]),
        "by_vehicle": pd.DataFrame(columns=["Vehicle Type", "cancellation_rate", "total_rides"]),
        "by_hour": pd.DataFrame(columns=["hour", "cancellation_rate", "total_rides"]),
        "by_payment": pd.DataFrame(columns=["Payment Method", "cancellation_rate", "total_rides"]),
        "heatmap_data": pd.DataFrame(
            np.zeros((24, len(WEEKDAY_ORDER))), index=range(24), columns=WEEKDAY_ORDER
        ),
    }

    if df.empty:
        return empty

    work = df.copy()
    work["is_cancelled"] = _is_cancelled(work)
    work["Pickup Time"] = pd.to_datetime(work.get("Pickup Time"), errors="coerce")

    cancelled_only = work[work["is_cancelled"]]
    reason_col = "Cancellation Reason (if applicable)"
    if reason_col in cancelled_only.columns:
        by_reason = (
            cancelled_only[reason_col]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Cancellation Reason")
            .reset_index(name="count")
        )
    else:
        by_reason = empty["by_reason"]

    if "Vehicle Type" in work.columns:
        by_vehicle = (
            work.groupby("Vehicle Type", as_index=False)
            .agg(cancellation_rate=("is_cancelled", "mean"), total_rides=("is_cancelled", "size"))
            .sort_values("cancellation_rate", ascending=False)
        )
    else:
        by_vehicle = empty["by_vehicle"]

    work["hour"] = work["Pickup Time"].dt.hour
    by_hour = (
        work.dropna(subset=["hour"])
        .groupby("hour", as_index=False)
        .agg(cancellation_rate=("is_cancelled", "mean"), total_rides=("is_cancelled", "size"))
        .sort_values("hour")
    )
    if by_hour.empty:
        by_hour = empty["by_hour"]

    if "Payment Method" in work.columns:
        by_payment = (
            work.groupby("Payment Method", as_index=False)
            .agg(cancellation_rate=("is_cancelled", "mean"), total_rides=("is_cancelled", "size"))
            .sort_values("cancellation_rate", ascending=False)
        )
    else:
        by_payment = empty["by_payment"]

    heat_df = work.dropna(subset=["Pickup Time"]).copy()
    heat_df["weekday"] = heat_df["Pickup Time"].dt.day_name().str[:3]
    heat_df["hour"] = heat_df["Pickup Time"].dt.hour
    heatmap_data = (
        heat_df.pivot_table(
            index="hour",
            columns="weekday",
            values="is_cancelled",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=range(24), fill_value=0.0)
        .reindex(columns=WEEKDAY_ORDER, fill_value=0.0)
    )

    return {
        "by_reason": by_reason,
        "by_vehicle": by_vehicle,
        "by_hour": by_hour,
        "by_payment": by_payment,
        "heatmap_data": heatmap_data,
    }


def plot_cancellation_heatmap(funnel_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Plot cancellation rate heatmap by hour and weekday."""
    heatmap_data = funnel_dict.get("heatmap_data", pd.DataFrame())
    if heatmap_data.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Cancellation Heatmap (No data)")
        return fig

    fig = px.imshow(
        heatmap_data,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Reds",
        template="plotly_white",
        labels={"x": "Weekday", "y": "Hour", "color": "Cancellation Rate"},
        title="Cancellation Rate Heatmap",
    )
    return fig


def plot_cancellation_by_reason(funnel_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Plot cancellation count by reason as a descending horizontal bar chart."""
    by_reason = funnel_dict.get("by_reason", pd.DataFrame())
    if by_reason.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Cancellation by Reason (No data)")
        return fig

    data = by_reason.sort_values("count", ascending=True)
    fig = px.bar(
        data,
        x="count",
        y="Cancellation Reason",
        orientation="h",
        template="plotly_white",
        title="Cancellation Reasons",
    )
    return fig


def revenue_breakdown(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return revenue summaries by vehicle, payment, hour, and date."""
    empty = {
        "by_vehicle": pd.DataFrame(columns=["Vehicle Type", "total_revenue", "avg_revenue"]),
        "by_payment": pd.DataFrame(columns=["Payment Method", "total_revenue"]),
        "by_hour": pd.DataFrame(columns=["hour", "avg_booking_value"]),
        "daily_revenue": pd.DataFrame(columns=["date", "total_revenue"]),
    }

    if df.empty:
        return empty

    work = df.copy()
    work["Booking Value (INR)"] = pd.to_numeric(work.get("Booking Value (INR)"), errors="coerce")
    work["Pickup Time"] = pd.to_datetime(work.get("Pickup Time"), errors="coerce")
    if "Booking Status" in work.columns:
        completed = work[work["Booking Status"].eq("Success")].copy()
    else:
        completed = work.iloc[0:0].copy()

    if completed.empty:
        return empty

    if "Vehicle Type" in completed.columns:
        by_vehicle = (
            completed.groupby("Vehicle Type", as_index=False)
            .agg(
                total_revenue=("Booking Value (INR)", "sum"),
                avg_revenue=("Booking Value (INR)", "mean"),
            )
            .sort_values("total_revenue", ascending=False)
        )
    else:
        by_vehicle = empty["by_vehicle"]

    if "Payment Method" in completed.columns:
        by_payment = (
            completed.groupby("Payment Method", as_index=False)
            .agg(total_revenue=("Booking Value (INR)", "sum"))
            .sort_values("total_revenue", ascending=False)
        )
    else:
        by_payment = empty["by_payment"]

    by_hour = (
        completed.assign(hour=completed["Pickup Time"].dt.hour)
        .dropna(subset=["hour"])
        .groupby("hour", as_index=False)
        .agg(avg_booking_value=("Booking Value (INR)", "mean"))
        .sort_values("hour")
    )
    if by_hour.empty:
        by_hour = empty["by_hour"]

    daily_revenue = (
        completed.assign(date=completed["Pickup Time"].dt.date)
        .dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(total_revenue=("Booking Value (INR)", "sum"))
        .sort_values("date")
    )
    if daily_revenue.empty:
        daily_revenue = empty["daily_revenue"]

    return {
        "by_vehicle": by_vehicle,
        "by_payment": by_payment,
        "by_hour": by_hour,
        "daily_revenue": daily_revenue,
    }


def peak_demand_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot ride count heatmap by hour (rows) and weekday (columns)."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Peak Demand Heatmap (No data)")
        return fig

    work = df.copy()
    work["Pickup Time"] = pd.to_datetime(work.get("Pickup Time"), errors="coerce")
    work = work.dropna(subset=["Pickup Time"])
    if work.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Peak Demand Heatmap (No valid time data)")
        return fig

    work["hour"] = work["Pickup Time"].dt.hour
    work["weekday"] = work["Pickup Time"].dt.day_name().str[:3]
    pivot = (
        work.pivot_table(
            index="hour",
            columns="weekday",
            values="Booking ID" if "Booking ID" in work.columns else "Pickup Time",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=range(24), fill_value=0)
        .reindex(columns=WEEKDAY_ORDER, fill_value=0)
    )

    fig = px.imshow(
        pivot,
        x=pivot.columns,
        y=pivot.index,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        template="plotly_white",
        labels={"x": "Weekday", "y": "Hour", "color": "Ride Count"},
        title="Peak Demand Heatmap",
    )
    return fig


def driver_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Build a vehicle-level operational scorecard as a proxy for driver segments."""
    columns = [
        "Vehicle Type",
        "total_rides",
        "completed_rides",
        "completion_rate",
        "avg_driver_rating",
        "avg_customer_rating",
        "avg_revenue_per_ride",
        "total_revenue",
        "cancellation_rate",
    ]
    if df.empty or "Vehicle Type" not in df.columns:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work["is_completed"] = work.get("Booking Status", "").eq("Success")
    work["is_cancelled"] = _is_cancelled(work)
    work["Booking Value (INR)"] = pd.to_numeric(work.get("Booking Value (INR)"), errors="coerce")
    work["Driver Ratings"] = pd.to_numeric(work.get("Driver Ratings"), errors="coerce")
    work["Customer Rating"] = pd.to_numeric(work.get("Customer Rating"), errors="coerce")

    grouped = work.groupby("Vehicle Type", as_index=False).agg(
        total_rides=("Vehicle Type", "size"),
        completed_rides=("is_completed", "sum"),
        avg_driver_rating=("Driver Ratings", "mean"),
        avg_customer_rating=("Customer Rating", "mean"),
        avg_revenue_per_ride=("Booking Value (INR)", "mean"),
        total_revenue=("Booking Value (INR)", "sum"),
        cancellation_rate=("is_cancelled", "mean"),
    )

    grouped["completion_rate"] = np.where(
        grouped["total_rides"] > 0,
        grouped["completed_rides"] / grouped["total_rides"],
        0.0,
    )

    grouped = grouped[
        [
            "Vehicle Type",
            "total_rides",
            "completed_rides",
            "completion_rate",
            "avg_driver_rating",
            "avg_customer_rating",
            "avg_revenue_per_ride",
            "total_revenue",
            "cancellation_rate",
        ]
    ].sort_values("completion_rate", ascending=False)

    return grouped
