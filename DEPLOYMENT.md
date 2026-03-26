# Deployment & Setup Guide

Complete guide for running the Uber Analytics platform (API + Dashboard).

## 📋 Prerequisites

- Python 3.14+
- Git
- PostgreSQL or MySQL (optional, for production database)
- Unix terminal (Mac/Linux) or WSL2 (Windows)

## 🚀 Quick Start (Local Development)

### Step 1: Clone & Setup Environment

```bash
# Clone or extract project
cd Uber-Analytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install
```

### Step 2: Configure Environment

```bash
# Create .env file from example
cp .env.example .env

# Edit .env with your settings (if needed)
nano .env
```

### Step 3: Run Tests (Optional)

```bash
# Verify everything is working
make test
# Expected: 28 passed
```

### Step 4: Start API Server

```bash
# Terminal 1: Start FastAPI backend
make run-api
# ✓ Server running at http://localhost:8000
# ✓ API docs at http://localhost:8000/docs
```

### Step 5: Start Dashboard

```bash
# Terminal 2: Start Streamlit dashboard (in new terminal with same venv)
source .venv/bin/activate
make run-dashboard
# ✓ Dashboard running at http://localhost:8501
```

## 📂 Directory Structure After Setup

```
Uber-Analytics/
├── DATA/                      # Input data
├── src/                       # Production code
│   ├── ingestion/
│   ├── analytics/
│   └── ml/
├── api/                       # FastAPI service
├── dashboard/                 # Streamlit app
├── tests/                     # Unit tests
├── models/                    # Trained ML models (generated)
├── outputs/                   # Generated reports (generated)
├── logs/                      # Application logs (generated)
├── requirements.txt
├── Makefile
├── README.md
└── DEPLOYMENT.md
```

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
# Summary KPIs
curl http://localhost:8000/metrics/summary

# Customer segments
curl http://localhost:8000/metrics/rfm-segments

# Cohort retention
curl http://localhost:8000/metrics/cohort-retention

# Cancellation funnel
curl http://localhost:8000/metrics/cancellation-funnel

# Revenue trends
curl http://localhost:8000/metrics/revenue-trends?period=daily
```

### Predictions
```bash
# Cancellation risk (POST)
curl -X POST http://localhost:8000/predict/cancellation \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_type": "Auto",
    "payment_method": "UPI",
    "hour_of_day": 14,
    "day_of_week": 2,
    "ride_distance_km": 8.5,
    "booking_value_inr": 250,
    "customer_ride_count": 15
  }'

# Churn risk
curl http://localhost:8000/predict/churn-risk?top_n=20

# Demand forecast
curl http://localhost:8000/predict/demand-forecast?days=30
```

## 🐳 Docker Deployment (Optional)

### Build Image
```bash
docker build -t uber-analytics:latest .
```

### Run Containers
```bash
# Using docker-compose
docker compose up -d

# Or individually
docker run -p 8000:8000 uber-analytics:latest python -m uvicorn api.main:app --host 0.0.0.0

docker run -p 8501:8501 uber-analytics:latest streamlit run dashboard/app.py \ 
  --server.port 8501 --server.address 0.0.0.0
```

## 📊 Data Loading

The API loads data at startup from `DATA/ncr_ride_bookings.csv`:

```python
# This happens automatically in api/main.py
from src.ingestion.loader import DataLoader

loader = DataLoader()
rides_df = loader.load_all_rides()
```

## 🔐 Production Checklist

- [ ] Set DEBUG=false in production
- [ ] Use environment-specific .env file
- [ ] Configure proper database (not CSV)
- [ ] Setup SSL/TLS certificates
- [ ] Enable CORS only for trusted domains
- [ ] Setup monitoring and alerting
- [ ] Configure log aggregation
- [ ] Backup ML models regularly
- [ ] Implement API rate limiting
- [ ] Setup automated model retraining

## 🛠 Useful Commands

```bash
# Clean cache and builds
make clean

# Run linter
make lint

# Run all tests
make test

# Run specific test file
pytest tests/test_cancellation.py -v

# Stop API (if stuck)
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Stop Streamlit
lsof -i :8501 | grep LISTEN | awk '{print $2}' | xargs kill -9

# View logs
tail -f logs/*.log
```

## 📈 Model Performance

### Training Models (Automatic on API startup)

Models are trained during API initialization:
- **Cancellation Predictor**: ~2-3 seconds
- **Churn Predictor**: ~2-3 seconds  
- **Demand Forecaster**: ~5-10 seconds

### Model Artifacts

All trained models are saved to `models/`:
```
models/
├── cancellation_model/
│   ├── model.pkl
│   └── metadata.json
├── churn_model/
│   ├── model.pkl
│   └── metadata.json
└── demand_model/
    ├── components.pkl
    └── metadata.json
```

## 🔄 Workflow

### Development
```bash
# 1. Make code changes in src/ or api/
# 2. Run tests
make test
# 3. Test API
curl http://localhost:8000/health
# 4. Check dashboard
# Open http://localhost:8501
```

### Deployment
```bash
# 1. Commit changes
git add -A
git commit -m "Feature: Add new prediction endpoint"
# 2. Build Docker image
docker build -t uber-analytics:latest .
# 3. Deploy
docker compose up -d
# 4. Verify
curl https://production-api.example.com/health
```

## 📞 Troubleshooting

### Port Already in Use
```bash
# Kill process using port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data Not Loading
```bash
# Check if CSV file exists
ls -la DATA/ncr_ride_bookings.csv

# Validate CSV
python -c "import pandas as pd; df = pd.read_csv('DATA/ncr_ride_bookings.csv'); print(df.shape)"
```

### Dashboard Not Connecting
1. Verify API is running: `curl http://localhost:8000/health`
2. Check secrets: `cat ~/.streamlit/secrets.toml`
3. Verify API_BASE_URL is correct
4. Check firewall rules

## 📚 Additional Resources

- [API Documentation](api/README.md) - API implementation details
- [Dashboard README](dashboard/README.md) - Dashboard setup
- [Main README](README.md) - Project overview
- [Makefile](Makefile) - Available commands

## 🚀 Next Steps

1. Explore API: http://localhost:8000/docs
2. Browse Dashboard: http://localhost:8501
3. Check [README.md](README.md) for architecture details
4. Review [config/settings.py](config/settings.py) for configuration
5. Read [dashboard/README.md](dashboard/README.md) for dashboard features

---

For questions or issues, refer to the logs in `logs/` directory or check API output.
