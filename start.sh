#!/usr/bin/env bash
set -euo pipefail

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

cleanup() {
  kill "$api_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

exec streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"