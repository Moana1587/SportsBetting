@echo off
echo Starting Sports Dashboard...

REM Change to the script directory
cd /d "%~dp0"

REM Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=production

REM Start the Flask application
echo Starting Flask application on http://0.0.0.0:5000
echo Press Ctrl+C to stop the application
echo.
python app.py

pause
