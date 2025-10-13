# Unified Sports Betting Dashboard

A comprehensive Flask web application that aggregates predictions from all your sports betting machine learning projects (CFB, MLB, NBA, NFL, NHL) into a single, unified dashboard.

## Features

- **Multi-Sport Support**: Displays predictions for College Football, MLB, NBA, NFL, and NHL
- **Real-time Data**: Fetches live predictions from each project's main.py
- **Comprehensive Predictions**: Shows Moneyline, Spread, and Over/Under predictions
- **Expected Values**: Displays EV calculations for betting decisions
- **Responsive Design**: Modern, mobile-friendly interface using Bootstrap
- **API Endpoints**: RESTful API for programmatic access
- **CSV Export**: Export all predictions to CSV format
- **Auto-refresh**: Automatic page refresh every 10 minutes

## Project Structure

```
unified_sports_dashboard/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Dashboard HTML template
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Prerequisites

Make sure you have the following projects in the same parent directory:
- `CFB-ML-Betting/`
- `MLB-ML-Betting/`
- `NBA-ML-Betting/`
- `NFL-ML-Betting/`
- `NHL-ML-Betting/`

Each project should have a working `main.py` file that can generate predictions.

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Project Structure**:
   Ensure all sports betting projects are in the same parent directory as this dashboard.

## Usage

### Running the Dashboard

```bash
python app.py
```

The dashboard will be available at: `http://127.0.0.1:5000`

### API Endpoints

- `GET /` - Main dashboard page
- `GET /api/predictions` - All predictions as JSON
- `GET /api/predictions/<sport>` - Predictions for specific sport (CFB, MLB, NBA, NFL, NHL)
- `GET /export/csv` - Export all predictions as CSV
- `GET /health` - Health check endpoint

### Example API Usage

```bash
# Get all predictions
curl http://127.0.0.1:5000/api/predictions

# Get NFL predictions only
curl http://127.0.0.1:5000/api/predictions/NFL

# Export to CSV
curl http://127.0.0.1:5000/export/csv -o predictions.csv
```

## Dashboard Features

### Overview Statistics
- Total games across all sports
- Individual sport game counts
- Real-time updates

### Game Information
- Team matchups
- Game times and venues
- Prediction confidence levels
- Expected values (EV)
- Betting odds
- Recommendations

### Prediction Types
- **Moneyline**: Win/loss predictions
- **Spread**: Point spread predictions
- **Over/Under**: Total score predictions

### Visual Indicators
- Color-coded confidence levels
- Positive/negative EV highlighting
- Sport-specific icons
- Responsive card layout

## Command Parameters

The unified dashboard uses the following commands for each sport:

- **CFB**: `python main.py -all` - Gets all prediction types (ML, Spread, O/U)
- **MLB**: `python main.py -xgb` - Uses XGBoost model for predictions
- **NBA**: `python main.py -xgb` - Uses XGBoost model for predictions  
- **NFL**: `python main.py -xgb` - Uses XGBoost model for predictions
- **NHL**: `python main.py --fallback` - Uses fallback prediction method

## Troubleshooting

### Common Issues

1. **"Error fetching predictions"**:
   - Ensure each project's `main.py` is working correctly
   - Check that all required dependencies are installed in each project
   - Verify file paths are correct

2. **"No games scheduled"**:
   - This is normal when there are no games for the current day
   - Check individual project outputs to verify they're generating predictions

3. **Port already in use**:
   - Change the port in `app.py` if port 5000 is occupied
   - Or kill the process using port 5000

### Debug Mode

To enable debug mode, modify the last lines in `app.py`:

```python
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
```

## Performance

- Predictions are cached for 10 minutes to improve performance
- Each sport's predictions are fetched independently
- Failed predictions don't affect other sports

## Contributing

To add support for additional sports:

1. Add the project path to `PROJECT_PATHS`
2. Create a fetch function following the existing pattern
3. Create a parse function for the specific output format
4. Add the sport to the main data aggregation

## License

This project is part of your sports betting machine learning suite.
