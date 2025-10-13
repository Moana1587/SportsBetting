#!/bin/bash

# Sports Dashboard Deployment Script
# Run this script on your VPS to deploy the application

set -e

echo "🚀 Starting Sports Dashboard deployment..."

# Configuration
APP_DIR="/var/www/sports_dashboard"
VENV_DIR="/var/www/sports_dashboard/venv"
SERVICE_NAME="sports_dashboard"
USER="www-data"

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy application files
echo "📋 Copying application files..."
sudo cp -r . $APP_DIR/
sudo chown -R $USER:$USER $APP_DIR

# Create virtual environment
echo "🐍 Creating virtual environment..."
cd $APP_DIR
sudo -u $USER python3 -m venv $VENV_DIR

# Install dependencies
echo "📦 Installing dependencies..."
sudo -u $USER $VENV_DIR/bin/pip install --upgrade pip
sudo -u $USER $VENV_DIR/bin/pip install -r requirements_production.txt

# Install system dependencies for sports projects
echo "⚽ Installing sports project dependencies..."
for sport in CFB-ML-Betting MLB-ML-Betting NBA-ML-Betting NFL-ML-Betting NHL-ML-Betting; do
    if [ -d "../$sport" ]; then
        echo "Installing dependencies for $sport..."
        sudo -u $USER $VENV_DIR/bin/pip install -r ../$sport/requirements.txt
    fi
done

# Update service file with correct paths
echo "⚙️ Configuring systemd service..."
sudo cp sports_dashboard.service /etc/systemd/system/
sudo sed -i "s|/path/to/your/unified_sports_dashboard|$APP_DIR|g" /etc/systemd/system/sports_dashboard.service
sudo sed -i "s|/path/to/your/venv|$VENV_DIR|g" /etc/systemd/system/sports_dashboard.service

# Enable and start service
echo "🔄 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

# Check service status
echo "✅ Checking service status..."
sudo systemctl status $SERVICE_NAME --no-pager

echo "🎉 Deployment complete!"
echo "Your Sports Dashboard should be available at: http://your-vps-ip:5000"
echo "To check logs: sudo journalctl -u $SERVICE_NAME -f"
echo "To restart: sudo systemctl restart $SERVICE_NAME"
