from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_rfm_frame() -> pd.DataFrame:
    """Return an empty RFM dataframe with the expected schema."""
    return pd.DataFrame(
        columns=[
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "R",
            "F",
            "M",
            "RFM_Score",
        ]
    )


def _score_series(series: pd.Series, labels: list[int]) -> pd.Series:
    """Score a numeric series into quantiles, with robust fallback for small/constant inputs."""
    if series.empty:
        return pd.Series(dtype=int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series([1] * len(series), index=series.index, dtype=int)

    filled = numeric.fillna(numeric.median())

    if filled.nunique(dropna=True) < 2:
        return pd.Series([labels[len(labels) // 2]] * len(series), index=series.index, dtype=int)

    ranked = filled.rank(method="first")
    try:
        scored = pd.qcut(ranked, q=5, labels=labels)
        return scored.astype(int)
    except ValueError:
        quantile_bins = pd.cut(ranked, bins=5, labels=labels, include_lowest=True)
        return quantile_bins.astype(int)


def compute_rfm(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute RFM metrics and quantile-based scores for each customer.

    Recency: days since last completed ride.
    Frequency: number of completed rides.
    Monetary: total booking value from completed rides.
    """
    required_columns = {
        "Customer ID",
        "Booking Status",
        "Booking Value (INR)",
        "Pickup Time",
    }
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_rfm_frame()

    completed = df[df["Booking Status"].eq("Success")].copy()
    completed = completed.dropna(subset=["Customer ID"]) if not completed.empty else completed

    if completed.empty:
        return _empty_rfm_frame()

    completed["Pickup Time"] = pd.to_datetime(completed["Pickup Time"], errors="coerce")
    completed["Booking Value (INR)"] = pd.to_numeric(
        completed["Booking Value (INR)"], errors="coerce"
    ).fillna(0.0)
    completed = completed.dropna(subset=["Pickup Time"])

    if completed.empty:
        return _empty_rfm_frame()

    if reference_date is None:
        ref_ts = completed["Pickup Time"].max().normalize() + pd.Timedelta(days=1)
    else:
        ref_ts = pd.Timestamp(reference_date)

    grouped = completed.groupby("Customer ID", as_index=False).agg(
        last_ride=("Pickup Time", "max"),
        Frequency=("Booking ID", "count") if "Booking ID" in completed.columns else ("Pickup Time", "count"),
        Monetary=("Booking Value (INR)", "sum"),
    )
    grouped["Recency"] = (ref_ts - grouped["last_ride"]).dt.days.clip(lower=0)

    rfm = grouped[["Customer ID", "Recency", "Frequency", "Monetary"]].copy()
    rfm["R"] = _score_series(rfm["Recency"], labels=[5, 4, 3, 2, 1])
    rfm["F"] = _score_series(rfm["Frequency"], labels=[1, 2, 3, 4, 5])
    rfm["M"] = _score_series(rfm["Monetary"], labels=[1, 2, 3, 4, 5])
    rfm["RFM_Score"] = (
        rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    )

    return rfm


def assign_segment(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assign business-friendly customer segments using R and F/M score rules."""
    if rfm_df.empty:
        output = rfm_df.copy()
        if "Segment" not in output.columns:
            output["Segment"] = pd.Series(dtype=str)
        return output

    result = rfm_df.copy()
    for col in ["R", "F", "M"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(1).astype(int)

    conditions = [
        (result["R"] >= 4) & (result["F"] >= 4) & (result["M"] >= 4),
        (result["R"] >= 3) & (result["F"] >= 4),
        (result["R"] == 5) & (result["F"] == 1),
        (result["R"] >= 4) & (result["F"] <= 2),
        (result["R"] <= 2) & (result["F"] >= 3),
        (result["R"] == 1) & (result["F"] == 1),
    ]
    choices = [
        "Champions",
        "Loyal Customers",
        "New Customers",
        "Potential Loyalists",
        "At Risk",
        "Lost",
    ]
    result["Segment"] = np.select(conditions, choices, default="Regular")
    return result


def plot_rfm_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    """Create an RFM scatter plot (Recency vs Frequency, bubble size=Monetary)."""
    if rfm_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="RFM Scatter (No data)")
        return fig

    data = rfm_df.copy()
    if "Segment" not in data.columns:
        data["Segment"] = "Regular"

    fig = px.scatter(
        data,
        x="Recency",
        y="Frequency",
        size="Monetary",
        color="Segment",
        hover_data=["Customer ID", "RFM_Score"],
        template="plotly_white",
        title="RFM Customer Segments",
    )
    fig.update_layout(legend_title_text="Segment")
    return fig


def plot_segment_distribution(rfm_df: pd.DataFrame) -> go.Figure:
    """Create a donut chart showing distribution of customer segments."""
    if rfm_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Segment Distribution (No data)")
        return fig

    data = rfm_df.copy()
    if "Segment" not in data.columns:
        data["Segment"] = "Regular"

    counts = data["Segment"].value_counts(dropna=False).rename_axis("Segment").reset_index(name="Count")
    fig = px.pie(
        counts,
        names="Segment",
        values="Count",
        hole=0.45,
        template="plotly_white",
        title="Customer Segment Distribution",
    )
    return fig


def get_rfm_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Return segment-level summary metrics for RFM analysis."""
    expected_cols = {"Segment", "Recency", "Frequency", "Monetary"}
    if rfm_df.empty or not expected_cols.issubset(rfm_df.columns):
        return pd.DataFrame(
            columns=[
                "Segment",
                "avg_R",
                "avg_F",
                "avg_M",
                "customer_count",
                "total_revenue",
                "avg_revenue_per_customer",
            ]
        )

    summary = (
        rfm_df.groupby("Segment", as_index=False)
        .agg(
            avg_R=("Recency", "mean"),
            avg_F=("Frequency", "mean"),
            avg_M=("Monetary", "mean"),
            customer_count=("Customer ID", "nunique")
            if "Customer ID" in rfm_df.columns
            else ("Segment", "count"),
            total_revenue=("Monetary", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    summary["avg_revenue_per_customer"] = np.where(
        summary["customer_count"] > 0,
        summary["total_revenue"] / summary["customer_count"],
        0.0,
    )
    return summary
