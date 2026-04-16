#!/usr/bin/env bash
# Project Sonar — one-command demo (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"
echo "Installing dependencies..."
python -m pip install -r requirements.txt
echo "Starting Streamlit — open http://localhost:8501"
exec python -m streamlit run app.py --server.headless true
