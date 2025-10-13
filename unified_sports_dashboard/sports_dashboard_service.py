#!/usr/bin/env python3
"""
Windows Service for Sports Dashboard
Install with: python sports_dashboard_service.py install
Start with: python sports_dashboard_service.py start
Stop with: python sports_dashboard_service.py stop
Remove with: python sports_dashboard_service.py remove
"""

import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import time
import subprocess
import threading

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SportsDashboardService(win32serviceutil.ServiceFramework):
    _svc_name_ = "SportsDashboard"
    _svc_display_name_ = "Sports Dashboard Flask Service"
    _svc_description_ = "Runs the Sports Dashboard Flask application as a Windows service"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                            servicemanager.PYS_SERVICE_STARTED,
                            (self._svc_name_, ''))
        
        # Change to the application directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Start the Flask application in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask_app)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        # Wait for stop signal
        while self.is_running:
            rc = win32event.WaitForSingleObject(self.hWaitStop, 1000)
            if rc == win32event.WAIT_OBJECT_0:
                break

    def run_flask_app(self):
        try:
            # Import and run the Flask app
            from app import app
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            servicemanager.LogErrorMsg(f"Error starting Flask app: {e}")

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(SportsDashboardService)
