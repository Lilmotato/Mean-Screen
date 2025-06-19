#!/bin/bash
# set -e

# if [ -f .env ]; then
#   while IFS='=' read -r key value; do
#     if [[ "$key" != \#* && -n "$key" ]]; then
#       export "$key"="$(echo "$value" | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
#     fi
#   done < .env
# fi



# Start FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
cd ui
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
