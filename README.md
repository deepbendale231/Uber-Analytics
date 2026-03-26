

# Uber NCR Ride Analytics (Final Version)

This project provides a local analytics workflow for Uber NCR ride data, including:
- Data ingestion and validation scripts
- Core analytics (RFM, cohort, descriptive, timeseries)
- Machine learning scripts for cancellation, churn, and demand
- A FastAPI backend for serving predictions and metrics
- An interactive Streamlit dashboard for visualization and exploration

**Note:**
- There is no CI/CD, Docker, or cloud deployment in this repo.
- The project is intended for local use only.
- All features described below are implemented and working as of the final commit.

---

## Project Structure

```
Uber-Analytics/
├── DATA/           # Input data (CSV)
├── src/            # Analytics, ingestion, ML code
├── api/            # FastAPI backend
├── dashboard/      # Streamlit dashboard
├── outputs/        # Generated outputs
├── models/         # Trained model artifacts
├── config/         # Settings
├── tests/          # Unit tests
└── Makefile        # Dev commands
```

---

## Getting Started

1. Clone the repo and set up the environment:
	```bash
	git clone https://github.com/deepbendale231/Uber-Analytics.git
	cd Uber-Analytics
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

2. Run tests (optional):
	```bash
	pytest tests/
	```

3. Start the FastAPI backend:
	```bash
	uvicorn api.main:app --reload
	# Docs: http://localhost:8000/docs
	```

4. Launch the dashboard:
	```bash
	cd dashboard
	streamlit run app.py
	# Dashboard: http://localhost:8501
	```

---

## Features

- Data ingestion and validation (CSV loader, schema checks)
- Analytics modules:
  - RFM segmentation
  - Cohort retention analysis
  - Descriptive stats and KPIs
  - Timeseries analysis
- ML scripts:
  - Cancellation prediction
  - Churn scoring
  - Demand forecasting
- FastAPI endpoints for predictions and metrics
- Streamlit dashboard with:
  - Overview tab (KPIs, trends)
  - Segmentation tab (RFM, revenue)
  - Predictions tab (cancellation, churn, demand)

---

## Configuration

Edit `config/settings.py` for any custom settings (e.g., database, MLflow, etc.).

---

## License

Internal use only. Not for production or public deployment.

---

## Contact

For questions, contact the project owner.
