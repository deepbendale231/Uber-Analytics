from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


VALID_VEHICLES = {
    "Auto",
    "Bike",
    "Sedan",
    "SUV",
    "Prime Sedan",
    "Go Mini",
    "Go Sedan",
    "Premier Sedan",
    "Uber XL",
    "eBike",
}
VALID_PAYMENTS = {"Cash", "UPI", "Card", "Wallet", "Credit Card", "Debit Card", "Uber Wallet"}


class CancellationRequest(BaseModel):
    """Request payload for cancellation risk scoring."""

    vehicle_type: str = Field(..., description="Vehicle type")
    payment_method: str = Field(..., description="Payment method")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    ride_distance_km: float = Field(..., gt=0)
    booking_value_inr: float = Field(..., gt=0)
    customer_ride_count: int = Field(..., ge=0)

    @field_validator("vehicle_type")
    @classmethod
    def validate_vehicle_type(cls, value: str) -> str:
        if value not in VALID_VEHICLES:
            valid = ", ".join(sorted(VALID_VEHICLES))
            raise ValueError(f"vehicle_type must be one of: {valid}")
        return value

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, value: str) -> str:
        if value not in VALID_PAYMENTS:
            valid = ", ".join(sorted(VALID_PAYMENTS))
            raise ValueError(f"payment_method must be one of: {valid}")
        return value


class CancellationResponse(BaseModel):
    """Response payload for cancellation risk scoring."""

    probability_cancelled: float
    probability_completed: float
    risk_level: Literal["low", "medium", "high"]
    top_risk_factors: list[str]
    booking_id: str
    timestamp: datetime


class ChurnRiskItem(BaseModel):
    """Top churn-risk customer record."""

    customer_id: str
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    days_since_last_ride: int
    total_rides: int
    top_risk_factors: list[str]


class ForecastDay(BaseModel):
    """Single-day forecast value with confidence intervals."""

    date: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    """Forecast response container."""

    generated_at: datetime
    periods: int
    forecast: list[ForecastDay]


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    model_loaded: bool
    row_count: int
    timestamp: datetime


class DataDateRange(BaseModel):
    """Minimum and maximum data dates available."""

    min_date: datetime | None
    max_date: datetime | None


class SummaryResponse(BaseModel):
    """Top-level KPI summary response."""

    total_rides: int
    completed_rides: int
    cancelled_rides: int
    completion_rate: float
    total_revenue_inr: float
    avg_booking_value: float
    avg_driver_rating: float
    avg_customer_rating: float
    avg_ride_distance_km: float
    top_vehicle_type: str | None
    top_payment_method: str | None
    data_date_range: DataDateRange


class SegmentDistributionItem(BaseModel):
    """RFM segment aggregate."""

    segment: str
    count: int
    percentage: float
    avg_revenue: float
    avg_frequency: float


class FunnelItem(BaseModel):
    """Cancellation funnel row."""

    label: str
    cancellation_rate: float
    total_bookings: int


class CancellationFunnelResponse(BaseModel):
    """Cancellation funnel grouped outputs."""

    by_vehicle: list[FunnelItem]
    by_hour: list[FunnelItem]
    by_reason: list[FunnelItem]
    by_payment: list[FunnelItem]


class CohortRetentionResponse(BaseModel):
    """Cohort retention payload for frontend heatmap rendering."""

    retention_matrix: list[list[float]]
    cohort_labels: list[str]
    index_labels: list[str]
    summary: dict


class RevenueTrendPoint(BaseModel):
    """Single point in revenue trend series."""

    period_label: str
    total_revenue: float
    ride_count: int
    avg_value: float
