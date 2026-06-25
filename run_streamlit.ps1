# Construction Delay Risk Prediction System — Streamlit Launcher
# Run from project root: .\run_streamlit.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
$APP = Join-Path $ROOT "CDRPS\notebooks\app.py"

if (-not (Test-Path $PYTHON)) {
    Write-Error "Python interpreter not found at: $PYTHON"
    Write-Host "Create the venv first: python -m venv .venv"
    exit 1
}

if (-not (Test-Path $APP)) {
    Write-Error "App file not found at: $APP"
    exit 1
}

Write-Host ""
Write-Host "Starting Construction Delay Risk Prediction System..."
Write-Host "Open your browser at: http://localhost:8501"
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

& $PYTHON -m streamlit run $APP
