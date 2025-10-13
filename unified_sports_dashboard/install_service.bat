@echo off
echo Installing Sports Dashboard Windows Service...

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator - proceeding with installation
) else (
    echo This script must be run as administrator
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements_production.txt
pip install pywin32

REM Install the service
echo Installing Windows service...
python sports_dashboard_service.py install

REM Start the service
echo Starting service...
python sports_dashboard_service.py start

echo.
echo Sports Dashboard service installed and started!
echo The application should be available at: http://localhost:5000
echo.
echo To manage the service:
echo   Start:   python sports_dashboard_service.py start
echo   Stop:    python sports_dashboard_service.py stop
echo   Remove:  python sports_dashboard_service.py remove
echo.
pause
