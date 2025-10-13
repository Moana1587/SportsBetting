# Sports Dashboard Windows Deployment Script
# Run this script as Administrator on your Windows VPS

param(
    [string]$InstallPath = "C:\SportsDashboard",
    [switch]$InstallService = $true,
    [switch]$StartService = $true
)

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script must be run as Administrator!"
    exit 1
}

Write-Host "🚀 Starting Sports Dashboard Windows deployment..." -ForegroundColor Green

# Create application directory
Write-Host "📁 Creating application directory at $InstallPath..." -ForegroundColor Yellow
if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force
}

# Copy application files
Write-Host "📋 Copying application files..." -ForegroundColor Yellow
Copy-Item -Path ".\*" -Destination $InstallPath -Recurse -Force

# Change to application directory
Set-Location $InstallPath

# Install Python dependencies
Write-Host "🐍 Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements_production.txt

# Install dependencies for sports projects
Write-Host "⚽ Installing sports project dependencies..." -ForegroundColor Yellow
$sportsProjects = @("CFB-ML-Betting", "MLB-ML-Betting", "NBA-ML-Betting", "NFL-ML-Betting", "NHL-ML-Betting")
foreach ($sport in $sportsProjects) {
    $sportPath = "..\$sport"
    if (Test-Path $sportPath) {
        Write-Host "Installing dependencies for $sport..." -ForegroundColor Cyan
        if (Test-Path "$sportPath\requirements.txt") {
            pip install -r "$sportPath\requirements.txt"
        }
    }
}

# Configure Windows Firewall
Write-Host "🔥 Configuring Windows Firewall..." -ForegroundColor Yellow
New-NetFirewallRule -DisplayName "Sports Dashboard HTTP" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -ErrorAction SilentlyContinue

# Install as Windows Service
if ($InstallService) {
    Write-Host "⚙️ Installing Windows Service..." -ForegroundColor Yellow
    python sports_dashboard_service.py install
    
    if ($StartService) {
        Write-Host "🔄 Starting Windows Service..." -ForegroundColor Yellow
        python sports_dashboard_service.py start
    }
}

# Test the application
Write-Host "🧪 Testing application..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Application is running successfully!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Could not test application - it may still be starting up" -ForegroundColor Yellow
}

Write-Host "🎉 Deployment complete!" -ForegroundColor Green
Write-Host "Your Sports Dashboard should be available at: http://your-vps-ip:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Management commands:" -ForegroundColor Yellow
Write-Host "  Start service:   python sports_dashboard_service.py start" -ForegroundColor White
Write-Host "  Stop service:    python sports_dashboard_service.py stop" -ForegroundColor White
Write-Host "  Remove service:  python sports_dashboard_service.py remove" -ForegroundColor White
Write-Host "  Manual start:    start_app.bat" -ForegroundColor White

# Open Windows Firewall for the port
Write-Host "🔓 Opening port 5000 in Windows Firewall..." -ForegroundColor Yellow
netsh advfirewall firewall add rule name="Sports Dashboard" dir=in action=allow protocol=TCP localport=5000

Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
