#!/bin/bash
set -e

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Start FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
cd ui
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
