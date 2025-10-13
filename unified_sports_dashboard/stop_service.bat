@echo off
echo Stopping Sports Dashboard Service...

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator - proceeding with stop
) else (
    echo This script must be run as administrator
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Stop the service
python sports_dashboard_service.py stop

echo Service stopped!
pause
