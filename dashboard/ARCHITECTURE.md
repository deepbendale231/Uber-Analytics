# Professional Streamlit Dashboard - Feature Documentation

## Overview

A production-ready, 4-tab interactive analytics dashboard for Uber NCR ride data with real-time ML predictions and business metrics.

**Page Configuration:**
- Title: "Uber NCR Analytics"
- Layout: Wide (optimized for analytics)
- Icon: 🚕
- Sidebar: Expanded by default

---

## Sidebar Components

### Filters Section
- **Date Range**: Start and end date inputs (default: 90 days back to today)
- **Vehicle Types**: Multi-select filters (Auto, Bike, Sedan, SUV, Prime Sedan)

### Settings Section
- **Auto-Refresh Toggle**: Enable/disable automatic refresh every 60 seconds

### API Status Section
- **Status Indicator**: 🟢 Green if API online, 🔴 Red if offline
- **Connection URL**: Displays current API endpoint
- **Row Count**: Shows number of loaded records
- **Error State**: Displays troubleshooting steps if API unavailable

### Footer
- Last update timestamp
- Version number and copyright

---

## Tab 1: Overview (📊)

### Row 1: Key Performance Indicators (5 cards)

Each metric card displays:
- **Current value** with formatting
- **Delta percentage** compared to previous period
- **Help tooltip** describing the metric

Metrics:
1. **Total Rides** - Count of all bookings (formatted with thousands separator)
2. **Completion Rate** - Percentage of successful rides
3. **Total Revenue** - ₹ formatted with K/M suffixes
4. **Avg Rating** - Star rating (⭐) with precision to 2 decimals
5. **Avg Distance** - Kilometers

### Row 2: Growth and Revenue Analysis

**Left Column - Daily Rides Trend**
- Plotly line chart with markers
- Fills area below trend line
- Uber brand color (#FF5A36)
- Interactive hover shows exact values

**Right Column - Revenue by Vehicle Type**
- Horizontal bar chart
- All vehicles in orange (#FF5A36)
- Shows breakdown of revenue per vehicle type

### Row 3: Distribution Charts

**Left Column - Booking Status Donut**
- Completed (green): #388e3c
- Cancelled (red): #d32f2f
- Donut chart with 30% hole
- Interactive hover with percentages

**Right Column - Payment Method Distribution**
- Bar chart showing booking counts by payment method
- Payment methods: Cash, UPI, Card, Wallet
- Orange bars (#FF5A36)

---

## Tab 2: Customers (👥)

### Row 1: RFM Segmentation (2 columns)

**Left - RFM Segments Donut Chart**
- Programmatic color assignment
- 40% hole for donut style
- Labels: segment names
- Interactive hover with segment distribution

**Right - Segment Summary Table**
- Columns: Segment, Count, Avg Revenue (₹)
- Revenue formatted with format_inr() (K/M suffixes)
- Professional styling with borders

### Row 2: Cohort Retention Analysis (Full Width)

**Cohort Retention Heatmap**
- Plotly imshow with "RdYlGn" colorscale (Red-Yellow-Green)
- Annotated with percentage values in each cell
- X-axis: Months (0, 1, 2, ... 12+)
- Y-axis: Cohort labels (month names or dates)
- Hover template shows cohort, month, retention %
- Zmid=0.5 for centered colorscale

**Summary Statistics**
- Best Cohort Retention: Highest performing cohort
- Worst Cohort Retention: Lowest performing cohort
- Overall Average: Mean of all cohort retentions

### Row 3: Customer Churn Risk (Full Width)

**Checkbox Toggle** to show/hide churn table
**Top N Selector** (10-100 customers, default 25)

**Churn Risk Table**
- Columns:
  - Customer ID
  - Churn Probability (formatted as %)
  - Risk Level (Low/Medium/High)
  - Days Since Last Ride
  - Total Rides
- Row coloring: Red (high), Yellow (medium), Green (low)

**Export Button**
- Downloads CSV with timestamp: `churn_risk_YYYYMMDD_HHMMSS.csv`
- Full dataset in results table

**Summary Statistics** (3 columns)
- High Risk Count
- Medium Risk Count
- Average Churn Probability

---

## Tab 3: Live Predictor (🤖)

### Layout: 2 Equal Columns

### Left Column: Input Form

**Streamlit Form** with fields:

1. **Vehicle Type** (Selectbox)
   - Options: Auto, Bike, Sedan, SUV, Prime Sedan
   - Help: "Type of vehicle for this booking"

2. **Payment Method** (Selectbox)
   - Options: Cash, UPI, Card, Wallet
   - Help: "Customer's payment preference"

3. **Pickup Hour** (Slider)
   - Range: 0-23 (0=midnight, 23=11 PM)
   - Default: 14 (2 PM)
   - Help: "Hour of day"

4. **Day of Week** (Selectbox)
   - Options: Monday through Sunday
   - Default: Wednesday (logical mid-week)
   - Maps to 0-6 numeric value

5. **Ride Distance** (Number Input)
   - Min: 0.5 km
   - Max: 100 km
   - Default: 8.5 km
   - Step: 0.1 km
   - Help: "Estimated ride distance"

6. **Booking Value** (Number Input)
   - Min: ₹50
   - Max: ₹5,000
   - Default: ₹300
   - Step: ₹10

7. **Customer Ride History** (Number Input)
   - Min: 0 rides
   - Max: 1,000 rides
   - Default: 5 rides
   - Step: 1 ride

**Submit Button**
- Label: "🎯 Predict Cancellation Risk"
- Type: Primary (blue)
- Full width

### Right Column: Prediction Results

**Gauge Chart** (Plotly Indicator)
- Shows cancellation probability as 0-100%
- Gauge color: #FF5A36 (Uber orange)
- Background zones:
  - 0-30%: Light gray (safe zone)
  - 30-60%: Light orange (warning zone)
  - 60-100%: Light red (danger zone)
- Threshold line at 60% (red)
- Delta reference: 30% (industry average)

**Risk Level Badge**
- HTML styled span with color:
  - High: #d32f2f (red)
  - Medium: #f57c00 (orange)
  - Low: #388e3c (green)
- Uppercase text with padding

**Completion Probability**
- Formatted as percentage
- Styled as metric heading

**Top Risk Factors**
- Up to 3 factors from SHAP analysis
- Bullet list format
- Fallback text if none available

**Booking Details**
- Booking ID (UUID from API)
- Timestamp of prediction

**Historical Comparison**
- Calculates % difference vs 25% average
- Green success box if below average
- Orange warning box if above average
- Shows actual percentage difference

---

## Tab 4: Demand Forecast (📈)

### Row 1: Full-Width Forecast Chart

**Plotly Line Chart**
- X-axis: Dates (next 30 days)
- Y-axis: Estimated rides
- Forecast line: Orange (#FF5A36), width 3
- 95% Confidence band: Shaded orange (0.2 alpha)
- Title: "Expected Daily Ride Demand"
- Height: 500px
- Interactive hover: "x unified" mode

---

### Row 2: Analytics Section (2 columns)

**Left - Peak Forecast Days Table**
- Top 5 days by forecast volume
- Columns: Date, Forecast Rides (integer format)
- Professional table styling

**Forecast Summary Metrics** (3 columns)
- Total forecast rides (sum of 30 days)
- Average rides per day
- Peak day forecast

**Right - Forecast Decomposition** (Expanders)

Three expandable sections explaining components:

1. **📈 Trend Component**
   - Description of overall demand trend
   - Whether rising/falling/stable

2. **📅 Weekly Seasonality**
   - Peak times (Friday-Saturday nights)
   - Low times (2-5 AM)
   - Mid-week patterns

3. **📋 Yearly Seasonality**
   - Winter vs summer patterns
   - Holiday effects
   - Event impacts

---

## Key Features & Design Patterns

### API Communication
- **fetch_api()** function: Generic HTTP client with error handling
  - Supports GET and POST
  - 10-second timeout
  - Connection/timeout/HTTP errors handled gracefully
  - Returns None on failure

### Data Caching
- **@st.cache_data(ttl=300)** on all GET endpoints
- 5-minute TTL for balance between freshness and performance
- Cache cleared on sidebar filter changes

### Error Handling
- Graceful degradation when API is unavailable
- St.error() messages with troubleshooting steps
- St.spinner() on all API calls
- Fallback messages when data missing

### Formatting Utilities
- **format_inr()**: Indian rupee formatting with K/M suffixes
- **risk_badge()**: HTML badge styling by risk level
- **status_indicator()**: API status visual indicator
- **calculate_delta()**: Percentage change calculation

### Professional Styling
- Custom CSS for badges, metrics, status indicators
- Uber brand color (#FF5A36) throughout
- Consistent spacing and margins
- Material Design principles

### Responsive Layout
- Sidebar and main content auto-responsive
- 2-column layouts for efficiency
- Full-width sections for charts
- Mobile-friendly where applicable

---

## How to Run

### 1. Ensure API is Running
```bash
# Terminal 1
make run-api
# API at http://localhost:8000
```

### 2. Configure Secrets (Optional)
```bash
# Copy template
cp dashboard/.streamlit/secrets.example.toml ~/.streamlit/secrets.toml

# Edit to set custom API_BASE_URL if needed
```

### 3. Launch Dashboard
```bash
# Terminal 2
source .venv/bin/activate
make run-dashboard
# Dashboard at http://localhost:8501
```

### 4. Explore Features
- Browse 4 tabs
- Use sidebar filters
- Test predictions
- Export churn data
- Refresh automatically

---

## Technical Stack

- **Streamlit**: 1.42.0 (web framework)
- **Plotly**: 5.24.1 (interactive charts)
- **Pandas**: 2.2.3 (data manipulation)
- **NumPy**: (numerical operations)
- **Requests**: (HTTP client for FastAPI)

---

## Production Ready Features

✅ Robust error handling  
✅ Data caching for performance  
✅ Professional UI/UX  
✅ Responsive design  
✅ Brand consistency  
✅ Export functionality  
✅ Real-time status indicators  
✅ Comprehensive documentation  
✅ Type hints throughout  
✅ Logging for debugging  

---

## Future Enhancements

- [ ] User authentication
- [ ] Custom date range filtering
- [ ] Advanced filtering (vehicle type, payment method)
- [ ] PDF report generation
- [ ] Email alerts for anomalies
- [ ] Bookmarking/favorites
- [ ] Custom dashboards builder
- [ ] API rate limiting display
- [ ] Real-time notifications
- [ ] Performance optimizations
