"""
Uber NCR Ride Analytics Dashboard - Professional Streamlit Application

A comprehensive 4-tab analytics dashboard with real-time predictions and business metrics.
Connects to FastAPI backend for data and ML predictions.
"""

import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Uber NCR Analytics",
    layout="wide",
    page_icon="🚕",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Uber NCR Ride Analytics Dashboard v1.0",
        "Get help": "mailto:analytics@uber.com",
    }
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card { 
        background: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-medium { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
    .badge { 
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        color: white;
    }
    .badge-high { background-color: #d32f2f; }
    .badge-medium { background-color: #f57c00; }
    .badge-low { background-color: #388e3c; }
    .api-status { 
        display: flex; 
        align-items: center; 
        gap: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", st.secrets.get("API_BASE_URL", "http://localhost:8000"))
REFRESH_INTERVAL = 60  # seconds
CACHE_TTL = 300  # 5 minutes

# Initialize session state
if "auto_refresh_count" not in st.session_state:
    st.session_state.auto_refresh_count = 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_api(
    endpoint: str,
    method: str = "GET",
    payload: Optional[Dict] = None,
    timeout: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Fetch data from FastAPI backend with error handling.
    
    Args:
        endpoint: API endpoint (e.g., '/metrics/summary')
        method: HTTP method ('GET' or 'POST')
        payload: JSON payload for POST requests
        timeout: Request timeout in seconds
    
    Returns:
        JSON response or None if request fails
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error: Cannot reach {API_BASE_URL}")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout: Request to {endpoint} took too long")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} on {endpoint}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None


@st.cache_data(ttl=CACHE_TTL)
def get_health_status() -> Dict[str, Any]:
    """Get API health status."""
    return fetch_api("/health") or {}


@st.cache_data(ttl=CACHE_TTL)
def get_summary_metrics() -> Dict[str, Any]:
    """Get business metrics summary."""
    return fetch_api("/metrics/summary") or {}


@st.cache_data(ttl=CACHE_TTL)
def get_rfm_segments() -> List[Dict]:
    """Get RFM customer segments."""
    result = fetch_api("/metrics/rfm-segments")
    return result if result else []


@st.cache_data(ttl=CACHE_TTL)
def get_cancellation_funnel() -> Dict[str, Any]:
    """Get cancellation funnel data."""
    return fetch_api("/metrics/cancellation-funnel") or {}


@st.cache_data(ttl=CACHE_TTL)
def get_cohort_retention() -> Dict[str, Any]:
    """Get cohort retention data."""
    return fetch_api("/metrics/cohort-retention") or {}


@st.cache_data(ttl=CACHE_TTL)
def get_revenue_trends(period: str = "daily") -> List[Dict]:
    """Get revenue trends by period."""
    result = fetch_api(f"/metrics/revenue-trends?period={period}")
    return result if result else []


def predict_cancellation(booking_data: Dict[str, Any]) -> Optional[Dict]:
    """Predict booking cancellation risk."""
    return fetch_api("/predict/cancellation", method="POST", payload=booking_data)


def get_churn_risk(top_n: int = 25) -> List[Dict]:
    """Get top churn risk customers."""
    # Larger top_n values can take longer due to model scoring + explanations.
    # Use an adaptive timeout and one retry to reduce false "no data" states.
    adaptive_timeout = max(15, min(90, int(top_n * 0.8)))
    result = fetch_api(f"/predict/churn-risk?top_n={top_n}", timeout=adaptive_timeout)
    if not result and top_n > 25:
        result = fetch_api(f"/predict/churn-risk?top_n={top_n}", timeout=90)
    return result if result else []


def get_demand_forecast(days: int = 30) -> Dict[str, Any]:
    """Get demand forecast."""
    return fetch_api(f"/predict/demand-forecast?days={days}") or {}


def format_inr(value: float) -> str:
    """Format value as Indian Rupees."""
    if pd.isna(value) or value is None:
        return "₹0"
    
    if value >= 1_000_000:
        return f"₹{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"₹{value/1_000:.1f}K"
    else:
        return f"₹{value:.0f}"


def risk_badge(level: str) -> str:
    """Generate HTML for risk level badge."""
    level = level.lower() if level else "unknown"
    color_map = {
        "high": "#d32f2f",
        "medium": "#f57c00",
        "low": "#388e3c",
    }
    color = color_map.get(level, "#999999")
    return f'<span class="badge" style="background-color: {color};">{level.upper()}</span>'


def status_indicator(is_online: bool) -> str:
    """Generate HTML for API status indicator."""
    color = "#388e3c" if is_online else "#d32f2f"
    status = "🟢 Online" if is_online else "🔴 Offline"
    return f'<div class="api-status"><span style="color: {color};">{status}</span></div>'


def calculate_delta(current: float, previous: float) -> Optional[float]:
    """Calculate percentage change from previous to current."""
    if previous == 0 or pd.isna(previous):
        return None
    return ((current - previous) / abs(previous)) * 100


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with filters and status."""
    with st.sidebar:
        st.markdown("## 🚕 Uber Analytics")
        st.markdown("---")
        
        # Date range filter
        st.markdown("### 📅 Filters")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=90),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="end_date"
            )
        
        # Vehicle type filter
        vehicle_types = st.multiselect(
            "Vehicle Types",
            options=["Auto", "Bike", "Sedan", "SUV", "Prime Sedan"],
            default=["Auto", "Bike", "Sedan"],
            key="vehicle_filter"
        )
        
        st.markdown("---")
        
        # Auto-refresh toggle
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=False,
            help=f"Refresh every {REFRESH_INTERVAL} seconds"
        )
        
        if auto_refresh:
            st.info(f"🔄 Auto-refreshing every {REFRESH_INTERVAL}s")
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔗 API Status")
        health = get_health_status()
        is_online = bool(health)
        st.markdown(status_indicator(is_online), unsafe_allow_html=True)
        
        if is_online:
            st.caption(f"✓ Connected to {API_BASE_URL}")
            st.metric("Data Rows Loaded", health.get("row_count", "N/A"))
        else:
            st.error(
                f"Cannot reach API at {API_BASE_URL}\n\n"
                f"**Actions:**\n"
                f"1. Start API: `make run-api`\n"
                f"2. Verify it's running: `curl {API_BASE_URL}/health`\n"
                f"3. Refresh this page (F5)"
            )
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.caption("v1.0 | Uber Analytics © 2026")
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "vehicle_types": vehicle_types,
        "auto_refresh": auto_refresh,
    }


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def render_overview_tab():
    """Render overview tab with KPIs and trends."""
    st.header("📊 Overview")
    
    with st.spinner("Loading overview metrics..."):
        metrics = get_summary_metrics()
    
    if not metrics:
        st.error("Unable to load metrics. Please check API connection.")
        return
    
    # Row 1: KPI Cards
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_rides = metrics.get("total_rides", 0)
    completed_rides = metrics.get("completed_rides", 0)
    completion_rate = metrics.get("completion_rate", 0)
    total_revenue = metrics.get("total_revenue_inr", 0)
    avg_rating = metrics.get("avg_customer_rating", 0)
    avg_distance = metrics.get("avg_ride_distance_km", 0)
    
    # Dummy previous period (would come from params in production)
    prev_total_rides = int(total_rides * 0.95)
    prev_completion_rate = completion_rate * 0.98
    prev_avg_rating = avg_rating * 0.99
    prev_total_revenue = total_revenue * 0.92
    prev_avg_distance = avg_distance * 0.97
    
    with col1:
        delta_rides = calculate_delta(total_rides, prev_total_rides)
        st.metric(
            "Total Rides",
            f"{total_rides:,}",
            delta=f"{delta_rides:+.1f}%" if delta_rides else None,
            help="Total ride bookings"
        )
    
    with col2:
        delta_rate = calculate_delta(completion_rate, prev_completion_rate)
        st.metric(
            "Completion Rate",
            f"{completion_rate:.1%}",
            delta=f"{delta_rate:+.1f}%" if delta_rate else None,
            help="% rides completed successfully"
        )
    
    with col3:
        delta_revenue = calculate_delta(total_revenue, prev_total_revenue)
        st.metric(
            "Total Revenue",
            format_inr(total_revenue),
            delta=f"{delta_revenue:+.1f}%" if delta_revenue else None,
            help="Total booking value"
        )
    
    with col4:
        delta_rating = calculate_delta(avg_rating, prev_avg_rating)
        st.metric(
            "Avg Rating",
            f"{avg_rating:.2f} ⭐",
            delta=f"{delta_rating:+.1f}%" if delta_rating else None,
            help="Average customer rating"
        )
    
    with col5:
        delta_distance = calculate_delta(avg_distance, prev_avg_distance)
        st.metric(
            "Avg Distance",
            f"{avg_distance:.1f} km",
            delta=f"{delta_distance:+.1f}%" if delta_distance else None,
            help="Average ride distance"
        )
    
    st.markdown("---")
    
    # Row 2: Trends and Revenue by Vehicle
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Daily Rides Trend")
        with st.spinner("Loading trend data..."):
            trends = get_revenue_trends("daily")
        
        if trends:
            df_trends = pd.DataFrame(trends)
            # Rename API columns to match expected names
            df_trends = df_trends.rename(columns={
                'period_label': 'date',
                'ride_count': 'volume'
            })
            if "date" in df_trends.columns and "volume" in df_trends.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_trends["date"],
                    y=df_trends["volume"],
                    mode="lines+markers",
                    name="Rides",
                    line=dict(color="#FF5A36", width=2),
                    fill="tozeroy",
                ))
                fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Trend data format not available")
        else:
            st.info("No trend data available")
    
    with col2:
        st.markdown("### 🚗 Revenue by Vehicle Type")
        # Get vehicle breakdown from cancellation funnel and estimate revenue
        funnel_data = get_cancellation_funnel()
        if funnel_data and "by_vehicle" in funnel_data:
            vehicles_data = funnel_data["by_vehicle"]
            if vehicles_data:
                # Compute estimated revenue from booking counts and average booking value
                avg_value = metrics.get("avg_booking_value", 500)
                vehicle_names = [v["label"] for v in vehicles_data]
                # Estimate revenue: total_bookings * avg_booking_value * (1 - cancellation_rate)
                vehicle_revenues = [
                    (v["total_bookings"] * avg_value * (1 - v.get("cancellation_rate", 0.25)))
                    for v in vehicles_data
                ]
                
                fig = px.bar(
                    x=vehicle_names,
                    y=vehicle_revenues,
                    labels={"x": "Vehicle Type", "y": "Est. Revenue (₹)"},
                    color_discrete_sequence=["#FF5A36"]
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Vehicle revenue breakdown not available")
        else:
            st.info("Vehicle revenue breakdown not available")
    
    st.markdown("---")
    
    # Row 3: Status and Payment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Booking Status Distribution")
        cancelled = metrics.get("cancelled_rides", 0)
        statuses = {
            "Completed": completed_rides,
            "Cancelled": cancelled,
        }
        fig = go.Figure(data=[go.Pie(
            labels=list(statuses.keys()),
            values=list(statuses.values()),
            hole=0.3,
            marker=dict(colors=["#388e3c", "#d32f2f"])
        )])
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💳 Payment Method Distribution")
        funnel_data = get_cancellation_funnel()
        if funnel_data and funnel_data.get("by_payment"):
            payment_points = funnel_data.get("by_payment", [])
            payment_labels = [p.get("label", "Unknown") for p in payment_points]
            payment_totals = [p.get("total_bookings", 0) for p in payment_points]
            fig = px.bar(
                x=payment_labels,
                y=payment_totals,
                labels={"x": "Payment Method", "y": "Bookings"},
                color_discrete_sequence=["#FF5A36"]
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif metrics.get("bookings_by_payment_method"):
            payment_methods = metrics["bookings_by_payment_method"]
            fig = px.bar(
                x=list(payment_methods.keys()),
                y=list(payment_methods.values()),
                labels={"x": "Payment Method", "y": "Bookings"},
                color_discrete_sequence=["#FF5A36"]
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method breakdown not available")


# ============================================================================
# TAB 2: CUSTOMERS
# ============================================================================

def render_customers_tab():
    """Render customers tab with RFM, cohort, and churn analysis."""
    st.header("👥 Customers")
    
    # Row 1: RFM and Segment Summary
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📊 RFM Segments")
        with st.spinner("Loading RFM data..."):
            rfm_segments = get_rfm_segments()
        
        if rfm_segments:
            df_rfm = pd.DataFrame(rfm_segments)
            
            # Donut chart
            fig = go.Figure(data=[go.Pie(
                labels=df_rfm.get("segment", []),
                values=df_rfm.get("count", []),
                hole=0.4,
                hoverinfo="label+percent+value",
            )])
            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("RFM segment data not available")
    
    with col2:
        st.markdown("### 📈 Segment Summary")
        if rfm_segments:
            df_rfm = pd.DataFrame(rfm_segments)
            display_cols = ["segment", "count", "avg_revenue"]
            if all(col in df_rfm.columns for col in display_cols):
                df_display = df_rfm[display_cols].copy()
                df_display.columns = ["Segment", "Count", "Avg Revenue (₹)"]
                df_display["Avg Revenue (₹)"] = df_display["Avg Revenue (₹)"].apply(format_inr)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_rfm, use_container_width=True, hide_index=True)
        else:
            st.info("No segment data available")
    
    st.markdown("---")
    
    # Row 2: Cohort Retention Heatmap
    st.markdown("### 📅 Cohort Retention Analysis")
    with st.spinner("Loading cohort retention..."):
        cohort_data = get_cohort_retention()
    
    if cohort_data and "retention_matrix" in cohort_data:
        matrix = cohort_data["retention_matrix"]
        cohort_labels = cohort_data.get("cohort_labels", list(range(len(matrix))))
        index_labels = cohort_data.get("index_labels", list(range(len(matrix[0]) if matrix else 0)))
        
        # Create heatmap with annotations
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=index_labels,
            y=cohort_labels,
            colorscale="RdYlGn",
            text=[[f"{v:.1%}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Retention<br>Rate"),
            hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.2%}<extra></extra>"
        ))
        fig.update_layout(
            title="Monthly Cohort Retention Rates",
            xaxis_title="Months Since Cohort",
            yaxis_title="Cohort Period",
            height=500,
            margin=dict(t=60, b=40, l=120, r=100)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        if cohort_data.get("summary"):
            summary = cohort_data["summary"]
            
            # Calculate best and worst cohort retention values
            if cohort_data.get("retention_matrix") and cohort_data.get("cohort_labels"):
                matrix = cohort_data["retention_matrix"]
                # Exclude first column (month 0) to get meaningful month 1+ retention
                comparable_cols = list(range(1, len(matrix[0]))) if len(matrix[0]) > 1 else list(range(len(matrix[0])))
                if comparable_cols:
                    cohort_means = [sum([row[i] for i in comparable_cols if i < len(row)]) / len(comparable_cols) if comparable_cols else 0.0 for row in matrix]
                    best_retention = max(cohort_means) if cohort_means else 0.0
                    worst_retention = min(cohort_means) if cohort_means else 0.0
                else:
                    best_retention = summary.get('avg_month1_retention', 0)
                    worst_retention = summary.get('avg_month1_retention', 0)
            else:
                best_retention = 0.0
                worst_retention = 0.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Best Cohort",
                    f"{best_retention:.1%}",
                    help=f"Best performing: {summary.get('best_cohort', 'N/A')}"
                )
            with col2:
                st.metric(
                    "Worst Cohort",
                    f"{worst_retention:.1%}",
                    help=f"Worst performing: {summary.get('worst_cohort', 'N/A')}"
                )
            with col3:
                st.metric(
                    "Overall Avg",
                    f"{summary.get('overall_avg_retention', 0):.1%}"
                )
    else:
        st.info("Cohort retention data not available")
    
    st.markdown("---")
    
    # Row 3: Churn Risk Table
    st.markdown("### ⚠️ Customer Churn Risk")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_churn = st.checkbox("Show churn risk analysis", value=True)
    with col2:
        top_n = st.number_input("Top N Customers", 10, 100, 25)
    
    if show_churn:
        with st.spinner(f"Loading top {top_n} at-risk customers..."):
            churn_data = get_churn_risk(top_n)
            effective_top_n = top_n
            if not churn_data and top_n > 25:
                fallback_top_n = 25
                st.warning(
                    f"Top {top_n} request timed out. Showing Top {fallback_top_n} for faster results."
                )
                churn_data = get_churn_risk(fallback_top_n)
                effective_top_n = fallback_top_n
        
        if churn_data:
            df_churn = pd.DataFrame(churn_data)
            
            # Build display dataframe with formatted columns
            df_display = pd.DataFrame({
                "Customer ID": df_churn["customer_id"] if "customer_id" in df_churn.columns else [],
                "Churn Probability": df_churn["churn_probability"] if "churn_probability" in df_churn.columns else [],
                "Risk Level": df_churn["risk_level"] if "risk_level" in df_churn.columns else [],
                "Days Since Ride": df_churn["days_since_last_ride"] if "days_since_last_ride" in df_churn.columns else [],
                "Total Rides": df_churn["total_rides"] if "total_rides" in df_churn.columns else [],
            })
            
            # Format columns for display
            if "Churn Probability" in df_display.columns:
                df_display["Churn Probability"] = df_display["Churn Probability"].apply(lambda x: f"{x:.1%}")
            
            # Display with conditional coloring
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
            )
            
            # CSV export button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=f"churn_risk_top_{effective_top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = sum(df_churn["risk_level"] == "high") if "risk_level" in df_churn.columns else 0
                st.metric("High Risk Customers", high_risk)
            with col2:
                medium_risk = sum(df_churn["risk_level"] == "medium") if "risk_level" in df_churn.columns else 0
                st.metric("Medium Risk Customers", medium_risk)
            with col3:
                avg_probability = df_churn["churn_probability"].mean() if "churn_probability" in df_churn.columns else 0
                st.metric("Average Churn Probability", f"{avg_probability:.1%}")
        else:
            st.info("Churn risk data not available. Try reducing Top N or retrying in a few seconds.")


# ============================================================================
# TAB 3: LIVE PREDICTOR
# ============================================================================

def render_predictor_tab():
    """Render live prediction tab with cancellation risk calculator."""
    st.header("🤖 Real-time Cancellation Risk Predictor")
    
    col1, col2 = st.columns(2)
    
    # ========== LEFT COLUMN: INPUT FORM ==========
    with col1:
        st.markdown("### Input Booking Details")
        
        with st.form("cancellation_form"):
            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=["Auto", "Bike", "Go Mini", "Go Sedan", "Premier Sedan", "Uber XL", "eBike"],
                help="Type of vehicle for this booking"
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                options=["Cash", "UPI", "Credit Card", "Debit Card", "Uber Wallet"],
                help="Customer's payment preference"
            )
            
            hour_of_day = st.slider(
                "Pickup Hour",
                min_value=0,
                max_value=23,
                value=14,
                help="Hour of day (0-23)"
            )
            
            day_name = st.selectbox(
                "Day of Week",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=2
            )
            day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
            
            ride_distance = st.number_input(
                "Ride Distance (km)",
                min_value=0.5,
                max_value=100.0,
                value=8.5,
                step=0.1,
                help="Estimated ride distance"
            )
            
            booking_value = st.number_input(
                "Booking Value (₹)",
                min_value=50,
                max_value=5000,
                value=300,
                step=10,
                help="Estimated booking amount"
            )
            
            customer_ride_count = st.number_input(
                "Customer Ride History",
                min_value=0,
                max_value=1000,
                value=5,
                step=1,
                help="Total rides by this customer"
            )
            
            predict_button = st.form_submit_button(
                "🎯 Predict Cancellation Risk",
                use_container_width=True,
                type="primary"
            )
    
    # ========== RIGHT COLUMN: RESULTS ==========
    with col2:
        st.markdown("### Prediction Results")
        
        if predict_button:
            booking_data = {
                "vehicle_type": vehicle_type,
                "payment_method": payment_method,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "ride_distance_km": ride_distance,
                "booking_value_inr": booking_value,
                "customer_ride_count": customer_ride_count,
            }
            
            with st.spinner("🔄 Analyzing booking..."):
                prediction = predict_cancellation(booking_data)
            
            if prediction:
                # Extract values
                cancel_prob = prediction.get("probability_cancelled", 0)
                complete_prob = prediction.get("probability_completed", 0)
                risk_level = prediction.get("risk_level", "unknown")
                top_factors = prediction.get("top_risk_factors", "")
                booking_id = prediction.get("booking_id", "N/A")
                timestamp = prediction.get("timestamp", "")
                
                # Gauge chart for cancellation probability
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=cancel_prob * 100,
                    title={"text": "Cancellation Risk %"},
                    number={"suffix": "%", "valueformat": ".4f"},
                    delta={"reference": 30},  # Industry average
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#FF5A36"},
                        "steps": [
                            {"range": [0, 30], "color": "#f0f0f0"},
                            {"range": [30, 60], "color": "#fff3e0"},
                            {"range": [60, 100], "color": "#ffebee"},
                        ],
                        "threshold": {
                            "line": {"color": "#d32f2f", "width": 3},
                            "thickness": 0.75,
                            "value": 60,
                        }
                    }
                )])
                fig.update_layout(
                    height=300,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level badge
                st.markdown("#### Risk Level")
                st.markdown(risk_badge(risk_level), unsafe_allow_html=True)
                
                # Completion probability
                st.markdown("#### Completion Probability")
                st.markdown(f"### **{complete_prob:.2%}**", help="Likelihood of ride completion")

                # Explicit percentage text to avoid perceived zero for very small non-zero values
                st.caption(f"Cancellation probability: {cancel_prob:.4%}")
                
                # Risk factors
                st.markdown("#### Top Risk Factors")
                if top_factors:
                    # Handle both list and string formats
                    if isinstance(top_factors, list):
                        factors = top_factors
                    else:
                        factors = [f.strip() for f in top_factors.split(",")]
                    for i, factor in enumerate(factors[:3], 1):
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("• No specific risk factors identified")
                
                # Booking details
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"**Booking ID:** {booking_id}")
                with col2:
                    st.caption(f"**Time:** {timestamp}")
                
                # Historical comparison
                st.markdown("#### Historical Comparison")
                avg_cancel_rate = 0.25  # Assume 25% average
                diff = (cancel_prob - avg_cancel_rate) / avg_cancel_rate * 100
                comparison_text = f"This booking's risk is **{abs(diff):.0f}% {'higher' if diff > 0 else 'lower'}** than average"
                if diff > 0:
                    st.warning(f"⚠️ {comparison_text}")
                else:
                    st.success(f"✓ {comparison_text}")
            else:
                st.error("Failed to get prediction. Please check API connection and try again.")
        else:
            st.info("👈 Fill in the booking details and click 'Predict Cancellation Risk' to see results")


# ============================================================================
# TAB 4: DEMAND FORECAST
# ============================================================================

def render_forecast_tab():
    """Render demand forecast tab with Prophet forecasts."""
    st.header("📈 Demand Forecast")
    
    with st.spinner("Loading demand forecast..."):
        forecast_data = get_demand_forecast(days=30)
    
    if not forecast_data or ("forecast" not in forecast_data and "forecast_days" not in forecast_data):
        st.error("Demand forecast not available. Please check API connection.")
        return
    
    # Parse forecast data (support both current and legacy API shapes)
    forecast_rows = forecast_data.get("forecast", forecast_data.get("forecast_days", []))
    df_forecast = pd.DataFrame(forecast_rows)
    if df_forecast.empty:
        st.error("Demand forecast returned empty data.")
        return

    # Normalize column names used by charts/tables
    df_forecast = df_forecast.rename(
        columns={
            "yhat": "forecast",
            "yhat_lower": "confidence_lower",
            "yhat_upper": "confidence_upper",
        }
    )
    
    # Full-width forecast chart
    st.markdown("### 30-Day Ride Demand Forecast")
    
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=df_forecast.get("date", []),
        y=df_forecast.get("forecast", []),
        mode="lines",
        name="Forecast",
        line=dict(color="#FF5A36", width=3),
    ))
    
    # Add confidence band
    if "confidence_upper" in df_forecast.columns and "confidence_lower" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast["date"],
            y=df_forecast["confidence_upper"],
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast["date"],
            y=df_forecast["confidence_lower"],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% Confidence Interval",
            fillcolor="rgba(255, 90, 54, 0.2)",
        ))
    
    fig.update_layout(
        title="Expected Daily Ride Demand",
        xaxis_title="Date",
        yaxis_title="Estimated Rides",
        height=500,
        hovermode="x unified",
        margin=dict(t=40, b=40, l=60, r=20),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Peak Days and Decomposition
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔝 Peak Forecast Days")
        
        # Get top 5 peak days
        top_days = df_forecast.nlargest(5, "forecast")[["date", "forecast"]]
        if not top_days.empty:
            top_days_display = top_days.copy()
            top_days_display.columns = ["Date", "Forecast Rides"]
            top_days_display["Date"] = pd.to_datetime(top_days_display["Date"], errors="coerce")
            top_days_display["Day"] = top_days_display["Date"].dt.day_name()
            top_days_display["Date"] = top_days_display["Date"].dt.date
            top_days_display["Forecast Rides"] = pd.to_numeric(
                top_days_display["Forecast Rides"], errors="coerce"
            ).round(1)
            top_days_display = top_days_display[["Date", "Day", "Forecast Rides"]]
            st.dataframe(top_days_display, use_container_width=True, hide_index=True)
            
            # Summary stats
            st.markdown("#### Forecast Summary")
            total_forecast = df_forecast["forecast"].sum()
            avg_forecast = df_forecast["forecast"].mean()
            max_forecast = df_forecast["forecast"].max()
            min_forecast = df_forecast["forecast"].min()
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total", f"{total_forecast:,.0f} rides")
            with col_b:
                st.metric("Average", f"{avg_forecast:,.0f} rides/day")
            with col_c:
                st.metric("Peak Day", f"{max_forecast:,.0f} rides")
        else:
            st.info("No peak days data available")
    
    with col2:
        st.markdown("### 📊 Forecast Components")
        
        # Decomposition description
        with st.expander("📈 Trend Component", expanded=False):
            st.markdown("""
            **Overall demand trend** over the forecast period.
            Shows whether ride demand is generally increasing, decreasing, or stable.
            """)
        
        with st.expander("📅 Weekly Seasonality", expanded=False):
            st.markdown("""
            **Weekly patterns** in ride demand.
            - Peak: Typically Friday-Saturday nights
            - Low: Early morning hours (2-5 AM)
            - Mid-week surge: Tuesday-Thursday commutes
            """)
        
        with st.expander("📋 Yearly Seasonality", expanded=False):
            st.markdown("""
            **Seasonal variations** throughout the year.
            - Higher demand during winter months
            - Holiday spikes (Diwali, New Year, etc.)
            - Summer season variations
            """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Render sidebar and get filters
    filters = render_sidebar()
    
    # Main content area - Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "👥 Customers",
        "🤖 Predictor",
        "📈 Forecast"
    ])
    
    with tab1:
        render_overview_tab()
    with tab2:
        render_customers_tab()
    with tab3:
        render_predictor_tab()
    with tab4:
        render_forecast_tab()


if __name__ == "__main__":
    main()
