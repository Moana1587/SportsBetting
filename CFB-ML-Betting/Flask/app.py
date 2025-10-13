from datetime import date
import json
from flask import Flask, render_template, jsonify, Response
from functools import lru_cache
import subprocess, requests, re, time
import os
import csv
import io


@lru_cache()
def fetch_cfb_predictions(ttl_hash=None):
    del ttl_hash
    return fetch_cfb_game_data()

def fetch_cfb_game_data():
    """Fetch CFB predictions using the updated main.py"""
    try:
        # Change to parent directory and run main.py with all predictions
        cmd = ["python", "main.py", "-all"]
        stdout = subprocess.check_output(cmd, cwd="../", stderr=subprocess.STDOUT).decode()
        
        # Parse the output to extract predictions
        games = parse_cfb_predictions(stdout)
        return games
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        return {}
    except Exception as e:
        print(f"Error in fetch_cfb_game_data: {e}")
        return {}

def parse_cfb_predictions(stdout):
    """Parse the output from main.py to extract ML, Spread, and OU predictions"""
    games = {}
    
    # Split output by prediction type sections
    sections = stdout.split("---------------XGBoost")
    
    for section in sections:
        if "ML Model Predictions" in section:
            games.update(parse_ml_predictions(section))
        elif "Spread Model Predictions" in section:
            games.update(parse_spread_predictions(section))
        elif "UO Model Predictions" in section:
            games.update(parse_uo_predictions(section))
    
    return games

def parse_ml_predictions(section):
    """Parse Moneyline predictions"""
    games = {}
    
    # Enhanced pattern to capture kickoff and venue
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)\n.*?ML Value: ([+-]?\d+)'
    
    for match in re.finditer(pattern, section, re.MULTILINE | re.DOTALL):
        away_team = match.group(1).strip()
        home_team = match.group(2).strip()
        kickoff = match.group(3).strip()
        venue = match.group(4).strip()
        prediction = match.group(5).strip()
        confidence = match.group(6).strip()
        ml_value = match.group(7).strip()
        
        game_key = f"{away_team}:{home_team}"
        if game_key not in games:
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'kickoff': kickoff,
                'venue': venue
            }
        
        games[game_key]['ml_prediction'] = prediction
        games[game_key]['ml_confidence'] = confidence
        games[game_key]['ml_value'] = ml_value
    
    return games

def parse_spread_predictions(section):
    """Parse Spread predictions"""
    games = {}
    
    # Enhanced pattern to capture kickoff and venue
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)\n.*?Spread Value: ([\d.]+)% confidence'
    
    for match in re.finditer(pattern, section, re.MULTILINE | re.DOTALL):
        away_team = match.group(1).strip()
        home_team = match.group(2).strip()
        kickoff = match.group(3).strip()
        venue = match.group(4).strip()
        prediction = match.group(5).strip()
        confidence = match.group(6).strip()
        spread_value = match.group(7).strip()
        
        game_key = f"{away_team}:{home_team}"
        if game_key not in games:
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'kickoff': kickoff,
                'venue': venue
            }
        
        games[game_key]['spread_prediction'] = prediction
        games[game_key]['spread_confidence'] = confidence
        games[game_key]['spread_value'] = spread_value
    
    return games

def parse_uo_predictions(section):
    """Parse Over/Under predictions"""
    games = {}
    # Pattern to match UO predictions with kickoff and venue
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)'
    
    for match in re.finditer(pattern, section, re.MULTILINE | re.DOTALL):
        away_team = match.group(1).strip()
        home_team = match.group(2).strip()
        kickoff = match.group(3).strip()
        venue = match.group(4).strip()
        prediction = match.group(5).strip()
        confidence = match.group(6).strip()
        
        game_key = f"{away_team}:{home_team}"
        if game_key not in games:
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'kickoff': kickoff,
                'venue': venue
            }
        
        games[game_key]['ou_prediction'] = prediction
        games[game_key]['ou_confidence'] = confidence
    
    return games


def get_ttl_hash(seconds=600):
    """Return the same value within `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    cfb_predictions = fetch_cfb_predictions(ttl_hash=get_ttl_hash())
    
    return render_template('index.html', today=date.today(), data={"cfb_predictions": cfb_predictions})


@app.route("/api/predictions")
def api_predictions():
    """API endpoint to get CFB predictions as JSON"""
    predictions = fetch_cfb_predictions(ttl_hash=get_ttl_hash())
    return jsonify(predictions)


@app.route("/export/csv")
def export_csv():
    """Export all predictions to CSV"""
    predictions = fetch_cfb_predictions(ttl_hash=get_ttl_hash())
    
    if not predictions:
        return "No predictions available", 404
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Away Team', 'Home Team', 'Kickoff', 'Venue',
        'ML Prediction', 'ML Confidence', 'ML Value',
        'Spread Prediction', 'Spread Confidence', 'Spread Value',
        'OU Prediction', 'OU Confidence'
    ])
    
    # Write data rows
    for game_key, game_data in predictions.items():
        writer.writerow([
            game_data.get('away_team', ''),
            game_data.get('home_team', ''),
            game_data.get('kickoff', ''),
            game_data.get('venue', ''),
            game_data.get('ml_prediction', ''),
            game_data.get('ml_confidence', ''),
            game_data.get('ml_value', ''),
            game_data.get('spread_prediction', ''),
            game_data.get('spread_confidence', ''),
            game_data.get('spread_value', ''),
            game_data.get('ou_prediction', ''),
            game_data.get('ou_confidence', '')
        ])
    
    # Prepare response
    output.seek(0)
    response = Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=cfb_predictions.csv'}
    )
    
    return response


if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)