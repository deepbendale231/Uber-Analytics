# Streamlit Dashboard

Interactive web dashboard for Uber NCR ride analytics.

## Quick Start

### 1. Ensure API is Running

```bash
# Terminal 1: Start the FastAPI backend
make run-api
# API will be available at http://localhost:8000
```

### 2. Configure Streamlit Secrets

```bash
# Copy secrets template
cp dashboard/.streamlit/secrets.example.toml ~/.streamlit/secrets.toml

# Edit ~/.streamlit/secrets.toml to set:
# API_BASE_URL = "http://localhost:8000"
```

### 3. Launch Dashboard

```bash
# Terminal 2: Start Streamlit app
streamlit run dashboard/app.py
# Dashboard will open at http://localhost:8501
```

## Dashboard Features

### 📊 Overview Tab
- Total rides and revenue
- Completion rates and average ratings
- Key performance indicators

### 👥 Segmentation Tab
- RFM customer segments
- Segment distribution
- Revenue per segment

### 🤖 Predictions Tab

#### Cancellation Risk
- Input booking parameters
- Get cancellation risk prediction
- View top risk factors via SHAP

#### Churn Risk
- View top N customers at churn risk
- Churn probability distribution
- Risk factor analysis

#### Demand Forecast
- 7-30 day ride demand forecasting
- Confidence intervals
- Peak demand identification

### 📈 Retention Tab
- Cohort retention heatmap
- Retention metrics by cohort
- Month-over-month retention rates

## Configuration

### Theme Customization
Edit `dashboard/.streamlit/config.toml`:
- Colors: primaryColor, backgroundColor, textColor
- Font settings
- UI behavior

### API Connection
Set `API_BASE_URL` in secrets for custom API endpoint:
- Local: `http://localhost:8000`
- Remote: `https://api.example.com`

## Troubleshooting

### API Connection Issues
- Verify FastAPI is running: `curl http://localhost:8000/health`
- Check network connectivity
- Verify API_BASE_URL in secrets

### Streamlit Not Starting
- Clear cache: `streamlit cache clear`
- Remove old logs: `rm -rf ~/.streamlit/logs`
- Check Python environment

### Data Not Loading
- Check API logs for errors
- Verify data is loaded in API startup
- Restart both API and dashboard

## Architecture

```
dashboard/
├── app.py                      # Main Streamlit app
├── .streamlit/
│   ├── config.toml            # Theme & UI configuration
│   └── secrets.example.toml   # Template for secrets
└── README.md                  # This file
```

## Related Files

- **API**: [api/main.py](../api/main.py)
- **Requirements**: [requirements.txt](../requirements.txt)
- **Makefile**: [Makefile](../Makefile)

## Next Steps

- [ ] Add user authentication
- [ ] Implement data export (CSV/PDF)
- [ ] Add custom date range filters
- [ ] Implement alerts and notifications
- [ ] Add real-time data refresh
