# Windows VPS Deployment Guide for Sports Dashboard

This guide will help you deploy the Sports Dashboard Flask application on a Windows VPS so others can access it from different computers.

## Prerequisites

- Windows Server 2016+ or Windows 10+ VPS
- Python 3.8+ installed
- Administrator access
- All sports betting projects (CFB, MLB, NBA, NFL, NHL) in the parent directory

## Quick Deployment (Recommended)

### Method 1: PowerShell Script (Easiest)

1. **Upload your project to the VPS:**
   ```cmd
   # Copy the entire unified_sports_dashboard folder to your VPS
   # You can use RDP, file sharing, or any file transfer method
   ```

2. **Run PowerShell as Administrator:**
   ```powershell
   # Right-click PowerShell and select "Run as Administrator"
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   cd C:\path\to\unified_sports_dashboard
   .\deploy_windows.ps1
   ```

### Method 2: Batch Files

1. **Install dependencies:**
   ```cmd
   pip install -r requirements_production.txt
   ```

2. **Install as Windows Service:**
   ```cmd
   # Right-click Command Prompt and select "Run as Administrator"
   install_service.bat
   ```

3. **Or run manually:**
   ```cmd
   start_app.bat
   ```

## Manual Deployment Steps

### 1. Install Python Dependencies
```cmd
pip install -r requirements_production.txt
pip install pywin32
```

### 2. Configure Windows Firewall
```cmd
# Open port 5000
netsh advfirewall firewall add rule name="Sports Dashboard" dir=in action=allow protocol=TCP localport=5000
```

### 3. Install as Windows Service
```cmd
# Run as Administrator
python sports_dashboard_service.py install
python sports_dashboard_service.py start
```

### 4. Or Run Manually
```cmd
python app.py
```

## Accessing Your Application

- **Local access:** `http://localhost:5000`
- **External access:** `http://your-vps-ip:5000`
- **Health check:** `http://your-vps-ip:5000/health`

## Management Commands

### Service Management
```cmd
# Start service
python sports_dashboard_service.py start

# Stop service
python sports_dashboard_service.py stop

# Remove service
python sports_dashboard_service.py remove

# Check service status
sc query SportsDashboard
```

### Manual Management
```cmd
# Start application manually
start_app.bat

# Stop application
# Press Ctrl+C in the command window
```

## Windows Service Management

### Using Services Console
1. Press `Win + R`, type `services.msc`
2. Find "Sports Dashboard Flask Service"
3. Right-click to start/stop/restart

### Using Command Line
```cmd
# Start
net start SportsDashboard

# Stop
net stop SportsDashboard

# Check status
sc query SportsDashboard
```

## Troubleshooting

### Service Won't Start
```cmd
# Check Windows Event Log
eventvwr.msc
# Look in Windows Logs > Application

# Check service status
sc query SportsDashboard

# Test manually
python app.py
```

### Port Access Issues
```cmd
# Check if port is listening
netstat -an | findstr :5000

# Check firewall rules
netsh advfirewall firewall show rule name="Sports Dashboard"

# Test local access
curl http://localhost:5000/health
```

### Permission Issues
```cmd
# Run Command Prompt as Administrator
# Right-click Command Prompt > "Run as administrator"
```

### Python Path Issues
```cmd
# Check Python installation
python --version
pip --version

# Install missing packages
pip install -r requirements_production.txt
```

## Security Considerations

### Windows Firewall
```cmd
# Allow only specific IPs (optional)
netsh advfirewall firewall add rule name="Sports Dashboard Restricted" dir=in action=allow protocol=TCP localport=5000 remoteip=192.168.1.0/24
```

### User Account
- Run the service under a dedicated user account
- Avoid running as Administrator for security

### SSL/HTTPS (Optional)
For production use, consider setting up IIS with SSL certificates or using a reverse proxy.

## Performance Optimization

### Windows Service Settings
1. Open `services.msc`
2. Find "Sports Dashboard Flask Service"
3. Right-click > Properties
4. Set "Recovery" tab to restart on failure
5. Set "Dependencies" if needed

### Resource Monitoring
- Use Task Manager to monitor CPU and memory usage
- Use Resource Monitor for detailed analysis
- Set up Windows Performance Monitor for logging

## Backup and Recovery

### Backup Application
```cmd
# Create backup
xcopy C:\SportsDashboard C:\SportsDashboard_Backup /E /I /H /Y
```

### Restore Application
```cmd
# Stop service
net stop SportsDashboard

# Restore from backup
xcopy C:\SportsDashboard_Backup C:\SportsDashboard /E /I /H /Y

# Start service
net start SportsDashboard
```

## Log Files

### Application Logs
- Check Windows Event Log: `eventvwr.msc`
- Look in Windows Logs > Application
- Filter by source "Sports Dashboard"

### Service Logs
```cmd
# View service status
sc query SportsDashboard

# View detailed service info
sc qc SportsDashboard
```

## Remote Access Setup

### Enable Remote Desktop (if needed)
1. System Properties > Remote
2. Enable Remote Desktop
3. Add users who can connect

### File Sharing (for updates)
1. Right-click project folder
2. Properties > Sharing
3. Share with specific users

## Updates and Maintenance

### Updating the Application
1. Stop the service: `net stop SportsDashboard`
2. Replace application files
3. Install new dependencies: `pip install -r requirements_production.txt`
4. Start the service: `net start SportsDashboard`

### Regular Maintenance
- Monitor disk space
- Check Windows Updates
- Review security logs
- Backup configuration files

Your Sports Dashboard should now be accessible from any computer on the internet!
