#!/usr/bin/env python3
"""
Standalone script to export CFB predictions to CSV
This script runs main.py and exports the results to a CSV file
"""

import subprocess
import csv
import re
import os
from datetime import datetime

def run_predictions():
    """Run main.py to get predictions"""
    try:
        print("Running predictions...")
        result = subprocess.run(
            ["python", "main.py", "-all"], 
            capture_output=True, 
            text=True, 
            cwd="."
        )
        
        if result.returncode != 0:
            print(f"Error running main.py: {result.stderr}")
            return None
            
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_predictions(stdout):
    """Parse the output from main.py to extract predictions"""
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
    
    # Pattern to match ML predictions with kickoff and venue
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
    
    # Pattern to match Spread predictions with kickoff and venue
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

def export_to_csv(predictions, filename=None):
    """Export predictions to CSV file"""
    if not predictions:
        print("No predictions to export")
        return False
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cfb_predictions_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
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
        
        print(f"✅ Predictions exported to: {filename}")
        print(f"📊 Total games exported: {len(predictions)}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")
        return False

def main():
    print("College Football Predictions CSV Export")
    print("=" * 50)
    
    # Run predictions
    stdout = run_predictions()
    if not stdout:
        print("Failed to get predictions")
        return
    
    # Parse predictions
    print("Parsing predictions...")
    predictions = parse_predictions(stdout)
    
    if not predictions:
        print("No predictions found in output")
        return
    
    print(f"Found {len(predictions)} games with predictions")
    
    # Export to CSV
    print("Exporting to CSV...")
    success = export_to_csv(predictions)
    
    if success:
        print("\n🎉 CSV export completed successfully!")
        print("\nCSV file contains:")
        print("- Game details (teams, kickoff, venue)")
        print("- Moneyline predictions with confidence and ML values")
        print("- Spread predictions with confidence and spread values")
        print("- Over/Under predictions with confidence")
    else:
        print("\n❌ CSV export failed")

if __name__ == "__main__":
    main()
