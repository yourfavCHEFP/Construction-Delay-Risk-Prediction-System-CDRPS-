@echo off
REM Construction Delay Risk Prediction System — Streamlit Launcher
REM Double-click this file to start the app, then visit http://localhost:8501

cd /d "%~dp0"

echo.
echo Starting Construction Delay Risk Prediction System...
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" -m streamlit run "app.py"

pause
