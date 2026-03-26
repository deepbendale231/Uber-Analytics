.PHONY: install lint test run-api run-dashboard run-reports clean docker-up docker-down

install:
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest -q

run-api:
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	.venv/bin/streamlit run dashboard/app.py

run-reports:
	python -m reports.generator

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache mlruns

docker-up:
	docker compose up -d

docker-down:
	docker compose down
