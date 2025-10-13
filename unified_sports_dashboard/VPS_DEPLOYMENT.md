# VPS Deployment Guide for Sports Dashboard

This guide will help you deploy the Sports Dashboard Flask application on a VPS so others can access it from different computers.

## Prerequisites

- Ubuntu 20.04+ VPS with root access
- Python 3.8+ installed
- All sports betting projects (CFB, MLB, NBA, NFL, NHL) in the parent directory

## Quick Deployment

1. **Upload your project to the VPS:**
   ```bash
   # From your local machine
   scp -r unified_sports_dashboard/ user@your-vps-ip:/home/user/
   ```

2. **SSH into your VPS:**
   ```bash
   ssh user@your-vps-ip
   ```

3. **Run the deployment script:**
   ```bash
   cd unified_sports_dashboard
   sudo ./deploy.sh
   ```

## Manual Deployment Steps

If you prefer to deploy manually:

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx ufw -y
```

### 2. Set Up Application Directory
```bash
sudo mkdir -p /var/www/sports_dashboard
sudo chown www-data:www-data /var/www/sports_dashboard
sudo cp -r . /var/www/sports_dashboard/
```

### 3. Create Virtual Environment
```bash
cd /var/www/sports_dashboard
sudo -u www-data python3 -m venv venv
sudo -u www-data venv/bin/pip install -r requirements_production.txt
```

### 4. Install Sports Project Dependencies
```bash
# Install dependencies for each sports project
for sport in CFB-ML-Betting MLB-ML-Betting NBA-ML-Betting NFL-ML-Betting NHL-ML-Betting; do
    if [ -d "../$sport" ]; then
        sudo -u www-data venv/bin/pip install -r ../$sport/requirements.txt
    fi
done
```

### 5. Configure Systemd Service
```bash
sudo cp sports_dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sports_dashboard
sudo systemctl start sports_dashboard
```

### 6. Configure Firewall
```bash
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS (if using SSL)
sudo ufw allow 5000  # Direct Flask access (optional)
sudo ufw --force enable
```

### 7. Configure Nginx (Optional but Recommended)
```bash
sudo cp nginx_config.conf /etc/nginx/sites-available/sports_dashboard
sudo ln -s /etc/nginx/sites-available/sports_dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Accessing Your Application

- **Direct Flask access:** `http://your-vps-ip:5000`
- **Through Nginx:** `http://your-vps-ip` (if configured)

## Management Commands

```bash
# Check service status
sudo systemctl status sports_dashboard

# View logs
sudo journalctl -u sports_dashboard -f

# Restart service
sudo systemctl restart sports_dashboard

# Stop service
sudo systemctl stop sports_dashboard

# Start service
sudo systemctl start sports_dashboard
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs for errors
sudo journalctl -u sports_dashboard -n 50

# Check if port 5000 is in use
sudo netstat -tlnp | grep :5000

# Test the application manually
cd /var/www/sports_dashboard
sudo -u www-data venv/bin/python app.py
```

### Permission Issues
```bash
# Fix ownership
sudo chown -R www-data:www-data /var/www/sports_dashboard

# Fix permissions
sudo chmod -R 755 /var/www/sports_dashboard
```

### Port Access Issues
```bash
# Check firewall status
sudo ufw status

# Check if port is listening
sudo netstat -tlnp | grep :5000

# Test local access
curl http://localhost:5000/health
```

## Security Considerations

1. **Use HTTPS:** Set up SSL certificates with Let's Encrypt
2. **Firewall:** Only open necessary ports
3. **Updates:** Keep your system and dependencies updated
4. **Monitoring:** Set up log monitoring and alerts
5. **Backups:** Regular backups of your application and data

## SSL Setup (Optional)

To enable HTTPS:

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Performance Optimization

1. **Increase worker processes** in `gunicorn_config.py`
2. **Enable caching** for static content
3. **Use a CDN** for better global performance
4. **Monitor resource usage** with `htop` or similar tools

## Monitoring

Set up monitoring to track:
- Service status
- Resource usage (CPU, memory, disk)
- Response times
- Error rates
- Log file sizes

Your Sports Dashboard should now be accessible from any computer on the internet!
