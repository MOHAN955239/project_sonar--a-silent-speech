# Project Sonar — one-command demo (Windows)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt
Write-Host "Starting Streamlit — open http://localhost:8501" -ForegroundColor Green
python -m streamlit run app.py --server.headless true
