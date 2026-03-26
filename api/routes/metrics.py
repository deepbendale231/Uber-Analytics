from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from api.models.schemas import (
    CancellationFunnelResponse,
    CohortRetentionResponse,
    DataDateRange,
    FunnelItem,
    RevenueTrendPoint,
    SegmentDistributionItem,
    SummaryResponse,
)
from src.analytics.cohort import CohortAnalyzer
from src.analytics.descriptive import cancellation_funnel
from src.analytics.rfm import assign_segment, compute_rfm

router = APIRouter(prefix="/metrics", tags=["Business Metrics"])
logger = logging.getLogger(__name__)


def _safe_float(value: float | int | None) -> float:
    """Return float with NaN-safe fallback."""
    if value is None:
        return 0.0
    out = float(value)
    return 0.0 if np.isnan(out) else out


# curl "http://localhost:8000/metrics/summary"
@router.get("/summary", response_model=SummaryResponse)
def get_summary_metrics(request: Request) -> SummaryResponse:
    """Return top-level business KPIs from loaded ride data."""
    df = request.app.state.df
    if df is None or df.empty:
        return SummaryResponse(
            total_rides=0,
            completed_rides=0,
            cancelled_rides=0,
            completion_rate=0.0,
            total_revenue_inr=0.0,
            avg_booking_value=0.0,
            avg_driver_rating=0.0,
            avg_customer_rating=0.0,
            avg_ride_distance_km=0.0,
            top_vehicle_type=None,
            top_payment_method=None,
            data_date_range=DataDateRange(min_date=None, max_date=None),
        )

    work = df.copy()
    status = work.get("Booking Status", pd.Series(index=work.index, dtype=object)).astype(str)
    completed_mask = status.eq("Success")
    cancelled_mask = status.str.contains("Cancelled", case=False, na=False)

    work["Booking Value (INR)"] = pd.to_numeric(work.get("Booking Value (INR)"), errors="coerce")
    work["Driver Ratings"] = pd.to_numeric(work.get("Driver Ratings"), errors="coerce")
    work["Customer Rating"] = pd.to_numeric(work.get("Customer Rating"), errors="coerce")
    work["Ride Distance (km)"] = pd.to_numeric(work.get("Ride Distance (km)"), errors="coerce")
    pickup = pd.to_datetime(work.get("Pickup Time"), errors="coerce")

    total_rides = int(len(work))
    completed_rides = int(completed_mask.sum())
    cancelled_rides = int(cancelled_mask.sum())
    completion_rate = float(completed_rides / total_rides) if total_rides else 0.0

    top_vehicle = (
        work["Vehicle Type"].mode().iloc[0]
        if "Vehicle Type" in work.columns and not work["Vehicle Type"].dropna().empty
        else None
    )
    top_payment = (
        work["Payment Method"].mode().iloc[0]
        if "Payment Method" in work.columns and not work["Payment Method"].dropna().empty
        else None
    )

    min_date = pickup.min()
    max_date = pickup.max()

    return SummaryResponse(
        total_rides=total_rides,
        completed_rides=completed_rides,
        cancelled_rides=cancelled_rides,
        completion_rate=completion_rate,
        total_revenue_inr=_safe_float(work["Booking Value (INR)"].sum()),
        avg_booking_value=_safe_float(work["Booking Value (INR)"].mean()),
        avg_driver_rating=_safe_float(work["Driver Ratings"].mean()),
        avg_customer_rating=_safe_float(work["Customer Rating"].mean()),
        avg_ride_distance_km=_safe_float(work["Ride Distance (km)"].mean()),
        top_vehicle_type=str(top_vehicle) if top_vehicle is not None else None,
        top_payment_method=str(top_payment) if top_payment is not None else None,
        data_date_range=DataDateRange(
            min_date=min_date.to_pydatetime() if pd.notna(min_date) else None,
            max_date=max_date.to_pydatetime() if pd.notna(max_date) else None,
        ),
    )


# curl "http://localhost:8000/metrics/rfm-segments"
@router.get("/rfm-segments", response_model=list[SegmentDistributionItem])
def get_rfm_segments(request: Request) -> list[SegmentDistributionItem]:
    """Run RFM and return segment-level distribution metrics."""
    df = request.app.state.df
    if df is None or df.empty:
        return []

    try:
        rfm_df = assign_segment(compute_rfm(df))
        if rfm_df.empty:
            return []

        grouped = (
            rfm_df.groupby("Segment", as_index=False)
            .agg(
                count=("Customer ID", "nunique"),
                avg_revenue=("Monetary", "mean"),
                avg_frequency=("Frequency", "mean"),
            )
            .sort_values("count", ascending=False)
        )
        total_customers = int(grouped["count"].sum())
    except Exception as exc:
        logger.exception("RFM computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"RFM analysis error: {exc}") from exc

    response: list[SegmentDistributionItem] = []
    for _, row in grouped.iterrows():
        percentage = float(row["count"] / total_customers * 100.0) if total_customers else 0.0
        response.append(
            SegmentDistributionItem(
                segment=str(row["Segment"]),
                count=int(row["count"]),
                percentage=percentage,
                avg_revenue=_safe_float(row["avg_revenue"]),
                avg_frequency=_safe_float(row["avg_frequency"]),
            )
        )
    return response


# curl "http://localhost:8000/metrics/cancellation-funnel"
@router.get("/cancellation-funnel", response_model=CancellationFunnelResponse)
def get_cancellation_funnel(request: Request) -> CancellationFunnelResponse:
    """Return cancellation funnel by vehicle, hour, reason, and payment method."""
    df = request.app.state.df
    if df is None or df.empty:
        return CancellationFunnelResponse(by_vehicle=[], by_hour=[], by_reason=[], by_payment=[])

    try:
        funnel = cancellation_funnel(df)
    except Exception as exc:
        logger.exception("Cancellation funnel failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Cancellation funnel error: {exc}") from exc

    by_vehicle = [
        FunnelItem(
            label=str(row["Vehicle Type"]),
            cancellation_rate=_safe_float(row["cancellation_rate"]),
            total_bookings=int(row["total_rides"]),
        )
        for _, row in funnel.get("by_vehicle", pd.DataFrame()).iterrows()
    ]
    by_hour = [
        FunnelItem(
            label=str(int(row["hour"])),
            cancellation_rate=_safe_float(row["cancellation_rate"]),
            total_bookings=int(row["total_rides"]),
        )
        for _, row in funnel.get("by_hour", pd.DataFrame()).iterrows()
    ]
    by_payment = [
        FunnelItem(
            label=str(row["Payment Method"]),
            cancellation_rate=_safe_float(row["cancellation_rate"]),
            total_bookings=int(row["total_rides"]),
        )
        for _, row in funnel.get("by_payment", pd.DataFrame()).iterrows()
    ]

    reason_df = funnel.get("by_reason", pd.DataFrame())
    total_cancelled = int(reason_df["count"].sum()) if "count" in reason_df.columns else 0
    by_reason = [
        FunnelItem(
            label=str(row["Cancellation Reason"]),
            cancellation_rate=(float(row["count"] / total_cancelled) if total_cancelled else 0.0),
            total_bookings=int(row["count"]),
        )
        for _, row in reason_df.iterrows()
    ]

    return CancellationFunnelResponse(
        by_vehicle=by_vehicle,
        by_hour=by_hour,
        by_reason=by_reason,
        by_payment=by_payment,
    )


# curl "http://localhost:8000/metrics/cohort-retention"
@router.get("/cohort-retention", response_model=CohortRetentionResponse)
def get_cohort_retention(request: Request) -> CohortRetentionResponse:
    """Run cohort retention analysis and return matrix with summary stats."""
    df = request.app.state.df
    if df is None or df.empty:
        return CohortRetentionResponse(retention_matrix=[], cohort_labels=[], index_labels=[], summary={})

    try:
        analyzer = CohortAnalyzer(df)
        retention = analyzer.get_retention_matrix()
        summary = analyzer.get_retention_summary()
    except Exception as exc:
        logger.exception("Cohort retention failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Cohort analysis error: {exc}") from exc

    if retention.empty:
        return CohortRetentionResponse(retention_matrix=[], cohort_labels=[], index_labels=[], summary=summary)

    matrix = retention.fillna(0.0).values.tolist()
    cohort_labels = [str(idx) for idx in retention.index.tolist()]
    index_labels = [str(col) for col in retention.columns.tolist()]

    return CohortRetentionResponse(
        retention_matrix=matrix,
        cohort_labels=cohort_labels,
        index_labels=index_labels,
        summary=summary,
    )


# curl "http://localhost:8000/metrics/revenue-trends?period=daily"
@router.get("/revenue-trends", response_model=list[RevenueTrendPoint])
def get_revenue_trends(
    request: Request,
    period: str = Query("daily", pattern="^(daily|weekly|monthly)$"),
) -> list[RevenueTrendPoint]:
    """Return revenue/volume/value trends for the selected time granularity."""
    df = request.app.state.df
    if df is None or df.empty:
        return []

    work = df.copy()
    work["Pickup Time"] = pd.to_datetime(work.get("Pickup Time"), errors="coerce")
    work["Booking Value (INR)"] = pd.to_numeric(work.get("Booking Value (INR)"), errors="coerce")
    work = work.dropna(subset=["Pickup Time"])
    if work.empty:
        return []

    if period == "daily":
        period_col = work["Pickup Time"].dt.to_period("D").astype(str)
    elif period == "weekly":
        period_col = work["Pickup Time"].dt.to_period("W-MON").astype(str)
    else:
        period_col = work["Pickup Time"].dt.to_period("M").astype(str)

    work = work.assign(period_label=period_col)

    grouped = (
        work.groupby("period_label", as_index=False)
        .agg(
            total_revenue=("Booking Value (INR)", "sum"),
            ride_count=("period_label", "count"),
            avg_value=("Booking Value (INR)", "mean"),
        )
        .sort_values("period_label")
    )

    return [
        RevenueTrendPoint(
            period_label=str(row["period_label"]),
            total_revenue=_safe_float(row["total_revenue"]),
            ride_count=int(row["ride_count"]),
            avg_value=_safe_float(row["avg_value"]),
        )
        for _, row in grouped.iterrows()
    ]
