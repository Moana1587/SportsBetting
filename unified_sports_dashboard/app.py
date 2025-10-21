from datetime import date, datetime
import json
import os
import sys
import subprocess
import re
import time
from functools import lru_cache
from flask import Flask, render_template, jsonify, Response
import csv
import io

# Add the project directories to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CFB-ML-Betting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLB-ML-Betting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NBA-ML-Betting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NFL-ML-Betting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NHL-ML-Betting'))

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')

# Project paths
PROJECT_PATHS = {
    'CFB': os.path.join(os.path.dirname(__file__), '..', 'CFB-ML-Betting'),
    'MLB': os.path.join(os.path.dirname(__file__), '..', 'MLB-ML-Betting'),
    'NBA': os.path.join(os.path.dirname(__file__), '..', 'NBA-ML-Betting'),
    'NFL': os.path.join(os.path.dirname(__file__), '..', 'NFL-ML-Betting'),
    'NHL': os.path.join(os.path.dirname(__file__), '..', 'NHL-ML-Betting')
}

@lru_cache()
def fetch_cfb_predictions(ttl_hash=None):
    """Fetch CFB predictions"""
    del ttl_hash
    try:
        # Try primary command first
        cmd = ["python", "main.py", "-all"]
        result = subprocess.run(cmd, cwd=PROJECT_PATHS['CFB'], 
                              capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            log_debug(f"CFB primary command failed with return code {result.returncode}")
            log_debug(f"CFB stderr: {result.stderr}")
            
            # Try fallback command
            log_debug("Trying CFB fallback command...")
            cmd_fallback = ["python", "main.py", "-all"]
            result = subprocess.run(cmd_fallback, cwd=PROJECT_PATHS['CFB'], 
                                  capture_output=True, text=True, timeout=20)
            
            if result.returncode != 0:
                log_debug(f"CFB fallback command also failed with return code {result.returncode}")
                log_debug(f"CFB fallback stderr: {result.stderr}")
                return {}
        # Check if there are no games scheduled
        if "No games found for today" in result.stdout or "No games found" in result.stdout:
            log_debug("CFB: No games scheduled for today")
            return {}
        
        # Check if stdout is empty or only contains debug messages
        if not result.stdout.strip() or len(result.stdout.strip()) < 50:
            log_debug("CFB: No meaningful output, likely no games scheduled")
            return {}
        
        return parse_cfb_predictions(result.stdout)
    except subprocess.TimeoutExpired:
        print("CFB command timed out")
        return {}
    except Exception as e:
        print(f"Error fetching CFB predictions: {e}")
        return {}

def parse_cfb_predictions(stdout):
    """Parse CFB predictions from main.py output"""
    games = {}
    
    # Check if there are no games in the output
    if "No games found for today" in stdout or "No games found" in stdout:
        return {}
    
    # Pattern for integrated recommended bet format with OU Value
    # Format: "Team vs Team, recommended bet: Team, Spread:value(confidence%), ML:value(confidence%), OU:Over/Under value(confidence%)"
    integrated_pattern = re.compile(
        r'(?P<home_team>[\w .]+) vs (?P<away_team>[\w .]+), recommended bet: (?P<recommended_team>[\w .]+), Spread:(?P<spread_value>[\d.+-]+)\((?P<spread_confidence>[\d.]+)%\), ML:(?P<ml_value>[\d+-]+)\((?P<ml_confidence>[\d.]+)%\), OU:(?P<ou_pick>\w+) (?P<ou_value>[\d.]+)\((?P<ou_confidence>[\d.]+)%\)',
        re.MULTILINE
    )
    
    # Try integrated pattern for new format with OU Value
    matches = list(integrated_pattern.finditer(stdout))
    for match in matches:
        home_team = match.group('home_team').strip()
        away_team = match.group('away_team').strip()
        recommended_team = match.group('recommended_team').strip()
        spread_value = match.group('spread_value').strip()
        spread_confidence = match.group('spread_confidence').strip()
        ml_value = match.group('ml_value').strip()
        ml_confidence = match.group('ml_confidence').strip()
        ou_pick = match.group('ou_pick').strip()
        ou_value = match.group('ou_value').strip()
        ou_confidence = match.group('ou_confidence').strip()
        
        game_key = f"{away_team}:{home_team}"
        
        # Create the raw prediction in the integrated format
        raw_prediction = f"{home_team} vs {away_team}, recommended bet: {recommended_team}, Spread:{spread_value}({spread_confidence}%), ML:{ml_value}({ml_confidence}%), OU:{ou_pick} {ou_value}({ou_confidence}%)"
        
        games[game_key] = {
            'away_team': away_team,
            'home_team': home_team,
            'recommended_team': recommended_team,
            'spread_value': spread_value,
            'spread_confidence': spread_confidence,
            'ml_value': ml_value,
            'ml_confidence': ml_confidence,
            'ou_pick': ou_pick,
            'ou_value': ou_value,
            'ou_confidence': ou_confidence,
            'raw_prediction': raw_prediction,
            'sport': 'CFB'
        }
    
    # If no games were parsed, return empty dict
    if not games:
        log_debug("CFB: No games parsed from output")
    
    return games

def parse_cfb_ml_predictions(section):
    """Parse CFB Moneyline predictions"""
    games = {}
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)\n.*?ML Value: ([+-]?\d+)(?:\n.*?Betting Lines: Spread ([^,]+), O/U ([^\n]+))?(?:\n.*?ML Lines: Home ([^,]+), Away ([^\n]+))?(?:\n.*?([^@\n]+) vs ([^@\n]+), recommended bet: ([^(]+)\(([^)]+)\), confidence: ([^\n]+))?'
    
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
                'venue': venue,
                'sport': 'CFB'
            }
        
        games[game_key]['ml_prediction'] = prediction
        games[game_key]['ml_confidence'] = confidence
        games[game_key]['ml_value'] = ml_value
        
        # Extract betting lines if available
        if match.group(8):  # Spread line
            games[game_key]['spread_line'] = match.group(8).strip()
        if match.group(9):  # O/U line
            games[game_key]['over_under_line'] = match.group(9).strip()
        if match.group(10):  # ML Home line
            games[game_key]['ml_home_line'] = match.group(10).strip()
        if match.group(11):  # ML Away line
            games[game_key]['ml_away_line'] = match.group(11).strip()
        
        # Extract recommended bet format if available
        if match.group(12) and match.group(13) and match.group(14) and match.group(15) and match.group(16):
            recommended_away = match.group(12).strip()
            recommended_home = match.group(13).strip()
            recommended_team = match.group(14).strip()
            spread_value = match.group(15).strip()
            recommended_confidence = match.group(16).strip()
            
            # Create the recommended bet format
            games[game_key]['raw_prediction'] = f"{recommended_away} vs {recommended_home}, recommended bet: {recommended_team} {spread_value}, confidence: {recommended_confidence}"
            games[game_key]['recommended_team'] = recommended_team
            games[game_key]['spread_value'] = spread_value
            games[game_key]['confidence'] = recommended_confidence
    
    return games

def parse_cfb_spread_predictions(section):
    """Parse CFB Spread predictions"""
    games = {}
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)\n.*?Spread Value: ([\d.]+)% confidence(?:\n.*?Betting Lines: Spread ([^,]+), O/U ([^\n]+))?(?:\n.*?ML Lines: Home ([^,]+), Away ([^\n]+))?(?:\n.*?([^@\n]+) vs ([^@\n]+), recommended bet: ([^(]+)\(([^)]+)\), confidence: ([^\n]+))?'
    
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
                'venue': venue,
                'sport': 'CFB'
            }
        
        games[game_key]['spread_prediction'] = prediction
        games[game_key]['spread_confidence'] = confidence
        games[game_key]['spread_value'] = spread_value
        
        # Extract betting lines if available
        if match.group(8):  # Spread line
            games[game_key]['spread_line'] = match.group(8).strip()
        if match.group(9):  # O/U line
            games[game_key]['over_under_line'] = match.group(9).strip()
        if match.group(10):  # ML Home line
            games[game_key]['ml_home_line'] = match.group(10).strip()
        if match.group(11):  # ML Away line
            games[game_key]['ml_away_line'] = match.group(11).strip()
        
        # Extract recommended bet format if available
        if match.group(12) and match.group(13) and match.group(14) and match.group(15) and match.group(16):
            recommended_away = match.group(12).strip()
            recommended_home = match.group(13).strip()
            recommended_team = match.group(14).strip()
            spread_value = match.group(15).strip()
            recommended_confidence = match.group(16).strip()
            
            # Create the recommended bet format
            games[game_key]['raw_prediction'] = f"{recommended_away} vs {recommended_home}, recommended bet: {recommended_team} {spread_value}, confidence: {recommended_confidence}"
            games[game_key]['recommended_team'] = recommended_team
            games[game_key]['spread_value'] = spread_value
            games[game_key]['confidence'] = recommended_confidence
    
    return games

def parse_cfb_uo_predictions(section):
    """Parse CFB Over/Under predictions"""
    games = {}
    pattern = r'([^@\n]+) @ ([^@\n]+)\n.*?Kickoff: ([^\n]+)\n.*?Venue: ([^\n]+)\n.*?Prediction: ([^(]+) \(([^)]+) confidence\)(?:\n.*?Betting Lines: Spread ([^,]+), O/U ([^\n]+))?(?:\n.*?ML Lines: Home ([^,]+), Away ([^\n]+))?(?:\n.*?([^@\n]+) vs ([^@\n]+), recommended bet: ([^(]+)\(([^)]+)\), confidence: ([^\n]+))?'
    
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
                'venue': venue,
                'sport': 'CFB'
            }
        
        games[game_key]['ou_prediction'] = prediction
        games[game_key]['ou_confidence'] = confidence
        
        # Extract betting lines if available
        if match.group(7):  # Spread line
            games[game_key]['spread_line'] = match.group(7).strip()
        if match.group(8):  # O/U line
            games[game_key]['over_under_line'] = match.group(8).strip()
        if match.group(9):  # ML Home line
            games[game_key]['ml_home_line'] = match.group(9).strip()
        if match.group(10):  # ML Away line
            games[game_key]['ml_away_line'] = match.group(10).strip()
        
        # Extract recommended bet format if available
        if match.group(11) and match.group(12) and match.group(13) and match.group(14) and match.group(15):
            recommended_away = match.group(11).strip()
            recommended_home = match.group(12).strip()
            recommended_team = match.group(13).strip()
            spread_value = match.group(14).strip()
            recommended_confidence = match.group(15).strip()
            
            # Create the recommended bet format
            games[game_key]['raw_prediction'] = f"{recommended_away} vs {recommended_home}, recommended bet: {recommended_team} {spread_value}, confidence: {recommended_confidence}"
            games[game_key]['recommended_team'] = recommended_team
            games[game_key]['spread_value'] = spread_value
            games[game_key]['confidence'] = recommended_confidence
    
    return games

@lru_cache()
def fetch_mlb_predictions(ttl_hash=None):
    """Fetch MLB predictions"""
    del ttl_hash
    try:
        cmd = ["python", "main.py", "-xgb"]
        result = subprocess.run(cmd, cwd=PROJECT_PATHS['MLB'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"MLB command failed with return code {result.returncode}")
            print(f"MLB stderr: {result.stderr}")
            print(f"MLB stdout: {result.stdout}")
            return {}
        
        # Check if there are no games scheduled
        if "No games found for today" in result.stdout or "No games found" in result.stdout or "No games entered" in result.stdout:
            log_debug("MLB: No games scheduled for today")
            return {}
        
        # Check if stdout is empty or only contains debug messages
        if not result.stdout.strip() or len(result.stdout.strip()) < 50:
            log_debug("MLB: No meaningful output, likely no games scheduled")
            return {}
        
        return parse_mlb_predictions(result.stdout)
    except subprocess.TimeoutExpired:
        print("MLB command timed out")
        return {}
    except Exception as e:
        print(f"Error fetching MLB predictions: {e}")
        return {}

def parse_mlb_predictions(stdout):
    """Parse MLB predictions from main.py output (handles multiple formats)"""
    games = {}
    
    # Check if there are no games in the output
    if "No games found for today" in stdout or "No games found" in stdout or "No games entered" in stdout:
        return {}
    
    # 1. Look for the new format with confidence values: "Team A vs Team B, recommended bet: Team C, Spread:X.X(confidence%), ML:Y(confidence%), OU:Z.Z(confidence%)"
    new_format_pattern = re.compile(r'(?P<away_team>[\w ]+) vs (?P<home_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+), Spread:(?P<spread_value>[\d.+-]+)\((?P<spread_confidence>[\d.]+)%\), ML:(?P<ml_value>[\d+-]+)\((?P<ml_confidence>[\d.]+)%\), OU:(?P<ou_value>[\d.]+)\((?P<ou_confidence>[\d.]+)%\)', re.MULTILINE)
    
    for match in new_format_pattern.finditer(stdout):
        away_team = match.group('away_team').strip()
        home_team = match.group('home_team').strip()
        recommended_team = match.group('recommended_team').strip()
        spread_value = match.group('spread_value').strip()
        spread_confidence = match.group('spread_confidence').strip()
        ml_value = match.group('ml_value').strip()
        ml_confidence = match.group('ml_confidence').strip()
        ou_value = match.group('ou_value').strip()
        ou_confidence = match.group('ou_confidence').strip()
        
        game_key = f"{away_team}:{home_team}"
        
        # Round values for better readability
        spread_confidence_rounded = round(float(spread_confidence), 1)
        ml_confidence_rounded = round(float(ml_confidence), 1)
        ou_confidence_rounded = round(float(ou_confidence), 1)
        spread_rounded = round(float(spread_value), 1)
        ou_rounded = round(float(ou_value), 1)
        
        # Format the spread with proper sign
        # Determine if recommended team is home or away to assign correct sign
        if recommended_team == home_team:
            formatted_spread = f"-{spread_rounded}"
        else:  # away team
            formatted_spread = f"+{spread_rounded}"
        
        # Use ML confidence as the primary confidence
        primary_confidence = ml_confidence_rounded
        
        # Determine OU prediction based on the OU value and recommended team
        # This is a simplified approach - in a real scenario, you'd need the actual OU prediction from the model
        ou_prediction = "Over"  # Default to Over, could be enhanced with actual model prediction
        
        games[game_key] = {
            'away_team': away_team,
            'home_team': home_team,
            'recommended_team': recommended_team,
            'recommended_bet': recommended_team,
            'confidence': f"{primary_confidence}%",
            'bet_category': "moneyline",
            'sport': 'MLB',
            'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {recommended_team}, Spread:{formatted_spread}({spread_confidence_rounded}%), ML:{ml_value}({ml_confidence_rounded}%), OU:{ou_prediction} {ou_rounded}({ou_confidence_rounded}%)",
            'spread_prediction': f"{recommended_team} {formatted_spread}",
            'spread_line': formatted_spread,
            'spread_confidence': f"{spread_confidence_rounded}%",
            'ml_value': ml_value,
            'ml_confidence': f"{ml_confidence_rounded}%",
            'ou_value': str(ou_rounded),
            'ou_prediction': f"{ou_prediction} {ou_rounded}",
            'ou_confidence': f"{ou_confidence_rounded}%"
        }
    
    # 2. Look for the old TEXT_OUTPUT format from XGBoost_Runner.py (fallback)
    if not games:
        text_pattern = re.compile(r'(?P<home_team>[\w ]+) vs (?P<away_team>[\w ]+), recommended bet: (?P<bet_description>[^,]+), confidence: (?P<confidence>[\d.]+)%', re.MULTILINE)
        
        for match in text_pattern.finditer(stdout):
            home_team = match.group('home_team').strip()
            away_team = match.group('away_team').strip()
            bet_description = match.group('bet_description').strip()
            confidence = match.group('confidence').strip()
            
            game_key = f"{away_team}:{home_team}"
            
            # Determine bet category based on bet description
            if 'ML' in bet_description:
                bet_category = "moneyline"
            elif '+' in bet_description or '-' in bet_description:
                bet_category = "spread"
            elif 'OVER' in bet_description or 'UNDER' in bet_description:
                bet_category = "over_under"
            else:
                bet_category = "other"
            
            # Round confidence for better readability
            confidence_rounded = round(float(confidence), 1)
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'recommended_bet': bet_description,
                'confidence': f"{confidence_rounded}%",
                'bet_category': bet_category,
                'sport': 'MLB',
                'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {bet_description}, confidence: {confidence_rounded}%",
                'spread_prediction': bet_description if bet_category == "spread" else None,
                'spread_line': bet_description.split()[-1] if bet_category == "spread" and '+' in bet_description or '-' in bet_description else None,
                'spread_confidence': f"{confidence_rounded}%" if bet_category == "spread" else None
            }
    
    # 3. Look for JSON format from XGBoost_Runner.py
    if not games:
        json_start = stdout.find("===JSON_OUTPUT_START===")
        json_end = stdout.find("===JSON_OUTPUT_END===")
        
        if json_start != -1 and json_end != -1:
            json_content = stdout[json_start:json_end]
            json_content = json_content.replace("===JSON_OUTPUT_START===", "").strip()
            
            try:
                import json
                predictions_data = json.loads(json_content)
                
                for pred in predictions_data:
                    home_team = pred['home_team']
                    away_team = pred['away_team']
                    game_key = f"{away_team}:{home_team}"
                    
                    recommended_team = pred['recommended_bet']['team']
                    bet_type = pred['recommended_bet']['type']
                    spread = pred['recommended_bet']['spread']
                    confidence = pred['recommended_bet']['confidence']
                    
                    # Round values for better readability
                    confidence_rounded = round(float(confidence), 1)
                    ou_confidence_rounded = round(float(pred['over_under']['confidence']), 1)
                    ou_value_rounded = round(float(pred['over_under']['line']), 1)
                    spread_rounded = round(float(spread), 1) if spread is not None else None
                    
                    # Create bet description
                    if bet_type == "ML":
                        bet_description = f"{recommended_team} ML"
                    else:
                        bet_description = f"{recommended_team} {bet_type} {spread_rounded}"
                    
                    # Format the recommended bet with only spread value
                    if spread_rounded is not None:
                        formatted_bet = f"{recommended_team} {spread_rounded}"
                    else:
                        formatted_bet = f"{recommended_team}"
                    
                    # Format OU prediction with value
                    ou_prediction = pred['over_under']['prediction']
                    ou_formatted = f"{ou_prediction} {ou_value_rounded}" if ou_prediction and ou_value_rounded else str(ou_value_rounded)
                    
                    games[game_key] = {
                        'away_team': away_team,
                        'home_team': home_team,
                        'recommended_bet': bet_description,
                        'confidence': f"{confidence_rounded}%",
                        'bet_category': bet_type.lower(),
                        'sport': 'MLB',
                        'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {formatted_bet}, confidence: {confidence_rounded}%",
                        'spread_prediction': f"{recommended_team} {bet_type} {spread_rounded}" if bet_type != "ML" else None,
                        'spread_line': str(spread_rounded) if bet_type != "ML" else None,
                        'spread_confidence': f"{confidence_rounded}%" if bet_type != "ML" else None,
                        'ou_prediction': ou_formatted,
                        'ou_value': str(ou_value_rounded),
                        'ou_confidence': f"{ou_confidence_rounded}%"
                    }
            except Exception as e:
                print(f"Error parsing MLB JSON output: {e}")
    
    # 4. Look for the simple format: "Team A vs Team B, recommended bet: Team A +1.5, confidence: 51.6%"
    if not games:
        simple_pattern = re.compile(r'(?P<away_team>[\w ]+) vs (?P<home_team>[\w ]+), recommended bet: (?P<bet_description>[^,]+), confidence: (?P<confidence>[\d.]+)%', re.MULTILINE)
        
        for match in simple_pattern.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            bet_description = match.group('bet_description').strip()
            confidence = match.group('confidence').strip()
            
            game_key = f"{away_team}:{home_team}"
            
            # Determine bet category based on bet description
            if 'ML' in bet_description:
                bet_category = "moneyline"
            elif '+' in bet_description or '-' in bet_description:
                bet_category = "spread"
            elif 'OVER' in bet_description or 'UNDER' in bet_description:
                bet_category = "over_under"
            else:
                bet_category = "other"
            
            # Round confidence for better readability
            confidence_rounded = round(float(confidence), 1)
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'recommended_bet': bet_description,
                'confidence': f"{confidence_rounded}%",
                'bet_category': bet_category,
                'sport': 'MLB',
                'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {bet_description}, confidence: {confidence_rounded}%",
                'spread_prediction': bet_description if bet_category == "spread" else None,
                'spread_line': bet_description.split()[-1] if bet_category == "spread" and ('+' in bet_description or '-' in bet_description) else None,
                'spread_confidence': f"{confidence_rounded}%" if bet_category == "spread" else None
            }
    
    # 5. If no simple format found, try the original complex parsing and create summary
    if not games:
        data_re = re.compile(r'\n(?P<home_team>[\w ]+)(\((?P<home_confidence>[\d+\.]+)%\))? vs (?P<away_team>[\w ]+)(\((?P<away_confidence>[\d+\.]+)%\))?: (?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d+\.]+) (\((?P<ou_confidence>[\d+\.]+)%\))?', re.MULTILINE)
        ev_re = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d+\.]+)', re.MULTILINE)
        
        for match in data_re.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            game_key = f"{away_team}:{home_team}"
            
            game_dict = {
                'away_team': away_team,
                'home_team': home_team,
                'away_confidence': match.group('away_confidence'),
                'home_confidence': match.group('home_confidence'),
                'ou_pick': match.group('ou_pick'),
                'ou_value': match.group('ou_value'),
                'ou_confidence': match.group('ou_confidence'),
                'away_team_ev': None,
                'home_team_ev': None,
                'sport': 'MLB'
            }
            
            # Extract EV data
            for ev_match in ev_re.finditer(stdout):
                if ev_match.group('team') == game_dict['away_team']:
                    game_dict['away_team_ev'] = ev_match.group('ev')
                if ev_match.group('team') == game_dict['home_team']:
                    game_dict['home_team_ev'] = ev_match.group('ev')

            # Create a comprehensive recommended bet based on all available data
            recommendations = []
            
            # Add ML recommendation if confidence is high
            if game_dict['away_confidence'] and float(game_dict['away_confidence']) > 55:
                recommendations.append(f"{away_team} ML ({game_dict['away_confidence']}%)")
            elif game_dict['home_confidence'] and float(game_dict['home_confidence']) > 55:
                recommendations.append(f"{home_team} ML ({game_dict['home_confidence']}%)")
            
            # Add OU recommendation
            if game_dict['ou_pick'] and game_dict['ou_confidence']:
                recommendations.append(f"{game_dict['ou_pick']} {game_dict['ou_value']} ({game_dict['ou_confidence']}%)")
            
            # Add EV-based recommendation
            if game_dict['away_team_ev'] and float(game_dict['away_team_ev']) > 0.05:
                recommendations.append(f"{away_team} (EV: {game_dict['away_team_ev']})")
            elif game_dict['home_team_ev'] and float(game_dict['home_team_ev']) > 0.05:
                recommendations.append(f"{home_team} (EV: {game_dict['home_team_ev']})")
            
            # Create the main recommended bet (use the highest confidence or best EV)
            main_recommendation = ""
            main_confidence = "0"
            
            if recommendations:
                # Use the first recommendation as the main one
                main_recommendation = recommendations[0]
                # Extract confidence from the recommendation or use OU confidence
                if game_dict['ou_confidence']:
                    main_confidence = game_dict['ou_confidence']
                elif game_dict['away_confidence']:
                    main_confidence = game_dict['away_confidence']
                elif game_dict['home_confidence']:
                    main_confidence = game_dict['home_confidence']
            
            # Round confidence values for better readability
            if game_dict.get('ou_confidence'):
                try:
                    game_dict['ou_confidence'] = f"{round(float(game_dict['ou_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
            
            if game_dict.get('away_confidence'):
                try:
                    game_dict['away_confidence'] = f"{round(float(game_dict['away_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
                    
            if game_dict.get('home_confidence'):
                try:
                    game_dict['home_confidence'] = f"{round(float(game_dict['home_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
            
            # Round main confidence
            try:
                main_confidence_rounded = round(float(main_confidence), 1)
            except (ValueError, TypeError):
                main_confidence_rounded = main_confidence
            
            # Round OU value for better readability
            if game_dict.get('ou_value'):
                try:
                    game_dict['ou_value'] = str(round(float(game_dict['ou_value']), 1))
                except (ValueError, TypeError):
                    pass
            
            # Add the comprehensive summary
            game_dict['raw_prediction'] = f"{away_team} vs {home_team}, recommended bet: {main_recommendation}, confidence: {main_confidence_rounded}%"
            game_dict['recommended_bet'] = main_recommendation
            game_dict['confidence'] = f"{main_confidence_rounded}%"
            game_dict['all_recommendations'] = recommendations

            games[game_key] = game_dict
    
    # If no games were parsed, return empty dict
    if not games:
        log_debug("MLB: No games parsed from output")
    
    return games

@lru_cache()
def fetch_nba_predictions(ttl_hash=None):
    """Fetch NBA predictions"""
    del ttl_hash
    try:
        cmd = ["python", "main.py", "-xgb"]
        result = subprocess.run(cmd, cwd=PROJECT_PATHS['NBA'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"NBA command failed with return code {result.returncode}")
            print(f"NBA stderr: {result.stderr}")
            print(f"NBA stdout: {result.stdout}")
            return {}
        
        # Check if there are no games scheduled
        if "No games found" in result.stdout or "No games available" in result.stdout:
            log_debug("NBA: No games scheduled for today")
            return {}
        
        # Check if stdout is empty or only contains debug messages
        if not result.stdout.strip() or len(result.stdout.strip()) < 50:
            log_debug("NBA: No meaningful output, likely no games scheduled")
            return {}
        
        return parse_nba_predictions(result.stdout)
    except subprocess.TimeoutExpired:
        print("NBA command timed out after 20 seconds")
        return {}
    except Exception as e:
        print(f"Error fetching NBA predictions: {e}")
        return {}

def parse_nba_predictions(stdout):
    """Parse NBA predictions from main.py output with new format"""
    games = {}
    
    # Check if there are no games in the output
    if "No games found" in stdout or "No games available" in stdout:
        return {}
    
    # New regex pattern to match the updated format:
    # "Oklahoma City Thunder vs Houston Rockets, recommended bet: Oklahoma City Thunder, Spread:+6.2(36.4%), ML:-175(63.6%), OU:OVER 220.0(47.6%)"
    prediction_re = re.compile(
        r'(?P<home_team>[\w ]+) vs (?P<away_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+)(?:, )?(?P<bet_details>.*?)(?:\n|$)',
        re.MULTILINE
    )
    
    # Regex patterns for individual bet components
    spread_re = re.compile(r'Spread:([+-]?[\d+\.]+)\(([\d+\.]+)%\)')
    ml_re = re.compile(r'ML:(-?\d+)\(([\d+\.]+)%\)')
    ou_re = re.compile(r'OU:(OVER|UNDER) ([\d+\.]+)\(([\d+\.]+)%\)')
    
    # EV data extraction
    ev_re = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d+\.]+)', re.MULTILINE)
    
    for match in prediction_re.finditer(stdout):
        home_team = match.group('home_team').strip()
        away_team = match.group('away_team').strip()
        recommended_team = match.group('recommended_team').strip()
        bet_details = match.group('bet_details').strip()
        
        game_key = f"{away_team}:{home_team}"
        
        # Initialize game dictionary
        game_dict = {
            'away_team': away_team,
            'home_team': home_team,
            'recommended_team': recommended_team,
            'away_team_ev': None,
            'home_team_ev': None,
            'away_team_odds': '100',
            'home_team_odds': '100',
            'sport': 'NBA',
            'spread_value': None,
            'spread_confidence': None,
            'ml_odds': None,
            'ml_confidence': None,
            'ou_pick': None,
            'ou_value': None,
            'ou_confidence': None,
            'all_recommendations': []
        }
        
        # Parse spread information
        spread_match = spread_re.search(bet_details)
        if spread_match:
            game_dict['spread_value'] = spread_match.group(1)
            game_dict['spread_confidence'] = spread_match.group(2)
            game_dict['all_recommendations'].append(f"Spread:{spread_match.group(1)}({spread_match.group(2)}%)")
        
        # Parse ML information
        ml_match = ml_re.search(bet_details)
        if ml_match:
            game_dict['ml_odds'] = ml_match.group(1)
            game_dict['ml_confidence'] = ml_match.group(2)
            game_dict['all_recommendations'].append(f"ML:{ml_match.group(1)}({ml_match.group(2)}%)")
        
        # Parse OU information
        ou_match = ou_re.search(bet_details)
        if ou_match:
            game_dict['ou_pick'] = ou_match.group(1)
            game_dict['ou_value'] = ou_match.group(2)
            game_dict['ou_confidence'] = ou_match.group(3)
            game_dict['all_recommendations'].append(f"OU:{ou_match.group(1)} {ou_match.group(2)}({ou_match.group(3)}%)")
        
        # Extract EV data
        for ev_match in ev_re.finditer(stdout):
            if ev_match.group('team') == game_dict['away_team']:
                game_dict['away_team_ev'] = ev_match.group('ev')
            if ev_match.group('team') == game_dict['home_team']:
                game_dict['home_team_ev'] = ev_match.group('ev')
        
        # Create comprehensive recommended bet
        recommendations = []
        
        # Add spread recommendation if available
        if game_dict['spread_value'] and game_dict['spread_confidence']:
            recommendations.append(f"Spread:{game_dict['spread_value']}({game_dict['spread_confidence']}%)")
        
        # Add ML recommendation if available
        if game_dict['ml_odds'] and game_dict['ml_confidence']:
            recommendations.append(f"ML:{game_dict['ml_odds']}({game_dict['ml_confidence']}%)")
        
        # Add OU recommendation if available
        if game_dict['ou_pick'] and game_dict['ou_value'] and game_dict['ou_confidence']:
            recommendations.append(f"OU:{game_dict['ou_pick']} {game_dict['ou_value']}({game_dict['ou_confidence']}%)")
        
        # Add EV-based recommendation
        if game_dict['away_team_ev'] and float(game_dict['away_team_ev']) > 0.05:
            recommendations.append(f"{away_team} (EV: {game_dict['away_team_ev']})")
        elif game_dict['home_team_ev'] and float(game_dict['home_team_ev']) > 0.05:
            recommendations.append(f"{home_team} (EV: {game_dict['home_team_ev']})")
        
        # Create the main recommended bet
        main_recommendation = ""
        main_confidence = "0"
        
        if recommendations:
            main_recommendation = ", ".join(recommendations)
            # Use the highest confidence available
            confidences = []
            if game_dict['spread_confidence']:
                confidences.append(float(game_dict['spread_confidence']))
            if game_dict['ml_confidence']:
                confidences.append(float(game_dict['ml_confidence']))
            if game_dict['ou_confidence']:
                confidences.append(float(game_dict['ou_confidence']))
            
            if confidences:
                main_confidence = str(max(confidences))
        
        # Format the recommended bet in the requested format
        if main_recommendation:
            formatted_bet = f"{recommended_team}, {main_recommendation}"
        else:
            formatted_bet = recommended_team
        
        # Add comprehensive summary
        game_dict['raw_prediction'] = f"{home_team} vs {away_team}, recommended bet: {formatted_bet}"
        game_dict['recommended_bet'] = formatted_bet
        game_dict['confidence'] = f"{main_confidence}%"
        
        # Add individual prediction components
        if game_dict.get('ou_pick') and game_dict.get('ou_value'):
            game_dict['ou_prediction'] = f"{game_dict['ou_pick']} {game_dict['ou_value']}"
            game_dict['ou_confidence'] = f"{game_dict['ou_confidence']}%" if game_dict.get('ou_confidence') and not game_dict['ou_confidence'].endswith('%') else game_dict.get('ou_confidence')

        games[game_key] = game_dict
    
    # If no games were parsed, return empty dict
    if not games:
        log_debug("NBA: No games parsed from output")
    
    return games

@lru_cache()
def fetch_nfl_predictions(ttl_hash=None):
    """Fetch NFL predictions"""
    del ttl_hash
    try:
        cmd = ["python", "main.py", "-xgb"]
        result = subprocess.run(cmd, cwd=PROJECT_PATHS['NFL'], 
                              capture_output=True, text=True, timeout=20)
        
        if result.returncode != 0:
            print(f"NFL command failed with return code {result.returncode}")
            print(f"NFL stderr: {result.stderr}")
            print(f"NFL stdout: {result.stdout}")
            return {}
        
        # Check if there are no games scheduled
        if "No games found" in result.stdout or "No games available" in result.stdout:
            log_debug("NFL: No games scheduled for today")
            return {}
        
        # Check if stdout is empty or only contains debug messages
        if not result.stdout.strip() or len(result.stdout.strip()) < 50:
            log_debug("NFL: No meaningful output, likely no games scheduled")
            return {}
        
        return parse_nfl_predictions(result.stdout)
    except subprocess.TimeoutExpired:
        print("NFL command timed out after 120 seconds")
        return {}
    except Exception as e:
        print(f"Error fetching NFL predictions: {e}")
        return {}

def parse_nfl_predictions(stdout):
    """Parse NFL predictions from main.py output (handles multiple formats)"""
    games = {}
    
    # Check if there are no games in the output
    if "No games found" in stdout or "No games available" in stdout:
        return {}
    
    # 1. Look for the new clean integrated format
    # Format: "Home Team vs Away Team, recommended bet: Recommended Team, Spread:value(confidence%), ML:odds(confidence%), OU:prediction line(confidence%)"
    integrated_pattern = re.compile(r'(?P<home_team>[\w ]+) vs (?P<away_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+), Spread:(?P<spread_value>[+-]?[\d.]+)\((?P<spread_confidence>[\d.]+)%\), ML:(?P<ml_odds>-?\d+)\((?P<ml_confidence>[\d.]+)%\), OU:(?P<ou_prediction>OVER|UNDER) (?P<ou_line>[\d.]+)\((?P<ou_confidence>[\d.]+)%\)', re.MULTILINE)
    
    for match in integrated_pattern.finditer(stdout):
        home_team = match.group('home_team').strip()
        away_team = match.group('away_team').strip()
        recommended_team = match.group('recommended_team').strip()
        spread_value = match.group('spread_value').strip()
        spread_confidence = match.group('spread_confidence').strip()
        ml_odds = match.group('ml_odds').strip()
        ml_confidence = match.group('ml_confidence').strip()
        ou_prediction = match.group('ou_prediction').strip()
        ou_line = match.group('ou_line').strip()
        ou_confidence = match.group('ou_confidence').strip()
        
        game_key = f"{away_team}:{home_team}"
        
        # Create bet description - typically ML since that's the recommended bet
        bet_description = f"{recommended_team} ML"
        
        # Round the values for better readability
        spread_value_rounded = round(float(spread_value), 2)
        spread_confidence_rounded = round(float(spread_confidence), 1)
        ml_confidence_rounded = round(float(ml_confidence), 1)
        ou_line_rounded = round(float(ou_line), 1)
        ou_confidence_rounded = round(float(ou_confidence), 1)
        
        games[game_key] = {
            'away_team': away_team,
            'home_team': home_team,
            'recommended_bet': bet_description,
            'confidence': f"{ml_confidence_rounded}%",
            'bet_category': 'moneyline',
            'sport': 'NFL',
            'raw_prediction': f"{home_team} vs {away_team}, recommended bet: {recommended_team}, Spread:{spread_value_rounded}({spread_confidence_rounded}%), ML:{ml_odds}({ml_confidence_rounded}%), OU:{ou_prediction} {ou_line_rounded}({ou_confidence_rounded}%)",
            'spread_prediction': f"{recommended_team} {spread_value_rounded}",
            'spread_line': str(spread_value_rounded),
            'spread_confidence': f"{spread_confidence_rounded}%",
            'ou_prediction': f"{ou_prediction} {ou_line_rounded}",
            'ou_value': str(ou_line_rounded),
            'ou_confidence': f"{ou_confidence_rounded}%",
            'ml_odds': ml_odds,
            'ml_confidence': f"{ml_confidence_rounded}%"
        }
    
    # 2. Look for the old TEXT_OUTPUT format (fallback)
    if not games:
        text_pattern = re.compile(r'(?P<home_team>[\w ]+) vs (?P<away_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+) (?P<ou_value>[\d.]+),(?P<spread_value>[+-]?[\d.]+), confidence: (?P<confidence>[\d.]+)%', re.MULTILINE)
        
        for match in text_pattern.finditer(stdout):
            home_team = match.group('home_team').strip()
            away_team = match.group('away_team').strip()
            recommended_team = match.group('recommended_team').strip()
            ou_value = match.group('ou_value').strip()
            spread_value = match.group('spread_value').strip()
            confidence = match.group('confidence').strip()
            
            game_key = f"{away_team}:{home_team}"
            
            # Create bet description - this is typically ML since the format shows ou_value,spread_value
            bet_description = f"{recommended_team} ML"
            
            # Round the values for better readability
            spread_value_rounded = round(float(spread_value), 2)
            confidence_rounded = round(float(confidence), 1)
            ou_value_rounded = round(float(ou_value), 1)
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'recommended_bet': bet_description,
                'confidence': f"{confidence_rounded}%",
                'bet_category': 'moneyline',
                'sport': 'NFL',
                'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {recommended_team} {spread_value_rounded}, confidence: {confidence_rounded}%",
                'spread_prediction': f"{recommended_team} {spread_value_rounded}",
                'spread_line': str(spread_value_rounded),
                'spread_confidence': f"{confidence_rounded}%",
                'ou_prediction': f"OVER {ou_value_rounded}",
                'ou_value': str(ou_value_rounded),
                'ou_confidence': f"{confidence_rounded}%"
            }
    
    # 3. Look for JSON format from XGBoost_Runner.py
    if not games:
        json_start = stdout.find("===JSON_OUTPUT_START===")
        json_end = stdout.find("===JSON_OUTPUT_END===")
        
        if json_start != -1 and json_end != -1:
            json_content = stdout[json_start:json_end]
            json_content = json_content.replace("===JSON_OUTPUT_START===", "").strip()
            
            try:
                import json
                predictions_data = json.loads(json_content)
                
                for pred in predictions_data:
                    home_team = pred['home_team']
                    away_team = pred['away_team']
                    game_key = f"{away_team}:{home_team}"
                    
                    recommended_team = pred['recommended_bet']['team']
                    bet_type = pred['recommended_bet']['type']
                    spread = pred['recommended_bet']['spread']
                    confidence = pred['recommended_bet']['confidence']
                    
                    # Create bet description
                    if bet_type == "ML":
                        bet_description = f"{recommended_team} ML"
                    else:
                        bet_description = f"{recommended_team} {bet_type} {spread}"
                    
                    # Format the recommended bet with ou value,spread value style
                    ou_value = pred['over_under']['line']
                    spread_value = spread if bet_type != "ML" else None
                    
                    # Format with only spread_value (no ou_value)
                    if spread_value is not None:
                        formatted_bet = f"{recommended_team} {spread_value}"
                    else:
                        # If no spread value, use the predicted spread from spread_analysis
                        predicted_spread = pred.get('spread_analysis', {}).get('predicted', 0)
                        formatted_bet = f"{recommended_team} {predicted_spread}"
                    
                    # Extract spread analysis data
                    spread_analysis = pred.get('spread_analysis', {})
                    predicted_spread = spread_analysis.get('predicted', 0)
                    actual_spread = spread_analysis.get('actual', 0)
                    spread_confidence = spread_analysis.get('confidence', 0)
                    
                    # Round the values for better readability
                    confidence_rounded = round(float(confidence), 1)
                    predicted_spread_rounded = round(float(predicted_spread), 2) if predicted_spread != 0 else 0
                    spread_confidence_rounded = round(float(spread_confidence), 1) if spread_confidence != 0 else 0
                    ou_value_rounded = round(float(pred['over_under']['line']), 1)
                    ou_confidence_rounded = round(float(pred['over_under']['confidence']), 1)
                    
                    games[game_key] = {
                        'away_team': away_team,
                        'home_team': home_team,
                        'recommended_bet': bet_description,
                        'confidence': f"{confidence_rounded}%",
                        'bet_category': bet_type.lower(),
                        'sport': 'NFL',
                        'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {formatted_bet}, confidence: {confidence_rounded}%",
                        'spread_prediction': f"{recommended_team} {predicted_spread_rounded}" if predicted_spread_rounded != 0 else None,
                        'spread_line': str(predicted_spread_rounded) if predicted_spread_rounded != 0 else None,
                        'spread_confidence': f"{spread_confidence_rounded}%" if spread_confidence_rounded != 0 else None,
                        'ou_prediction': pred['over_under']['prediction'],
                        'ou_value': str(ou_value_rounded),
                        'ou_confidence': f"{ou_confidence_rounded}%"
                    }
            except Exception as e:
                print(f"Error parsing NFL JSON output: {e}")
    
    # 4. Look for the simple format: "Team A vs Team B, recommended bet: Team A +1.5, confidence: 51.6%"
    if not games:
        simple_pattern = re.compile(r'(?P<away_team>[\w ]+) vs (?P<home_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+) (?P<bet_type>[+-]?[\d.]+), confidence: (?P<confidence>[\d.]+)%', re.MULTILINE)
        
        for match in simple_pattern.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            recommended_team = match.group('recommended_team').strip()
            bet_type = match.group('bet_type').strip()
            confidence = match.group('confidence').strip()
            
            game_key = f"{away_team}:{home_team}"
            
            # Determine if it's a spread bet or other type
            if '+' in bet_type or '-' in bet_type:
                bet_description = f"{recommended_team} {bet_type}"
                bet_category = "spread"
            else:
                bet_description = f"{recommended_team} {bet_type}"
                bet_category = "other"
            
            # Round the values for better readability
            confidence_rounded = round(float(confidence), 1)
            bet_type_rounded = round(float(bet_type), 2) if '+' in bet_type or '-' in bet_type else bet_type
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'recommended_bet': bet_description,
                'confidence': f"{confidence_rounded}%",
                'bet_category': bet_category,
                'sport': 'NFL',
                'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {bet_description}, confidence: {confidence_rounded}%",
                'spread_prediction': bet_description if bet_category == "spread" else None,
                'spread_line': str(bet_type_rounded) if bet_category == "spread" else None,
                'spread_confidence': f"{confidence_rounded}%" if bet_category == "spread" else None
            }
    
    # 5. If no simple format found, try the original NFL parsing
    if not games:
        data_re = re.compile(r'(?P<home_team>[\w\s]+)\s+\((?P<home_confidence>[\d\.]+)%\)\s+vs\s+(?P<away_team>[\w\s]+):\s+(?P<ou_pick>OVER|UNDER)\s+(?P<ou_value>[\d\.]+)\s+\((?P<ou_confidence>[\d\.]+)%\)', re.MULTILINE)
        ev_re = re.compile(r'(?P<team>[\w\s]+)\s+EV:\s+(?P<ev>[-\d\.]+)', re.MULTILINE)
        odds_re = re.compile(r'(?P<away_team>[\w\s]+)\s+\((?P<away_team_odds>-?\d+)\)\s+@\s+(?P<home_team>[\w\s]+)\s+\((?P<home_team_odds>-?\d+)\)', re.MULTILINE)
        recommendation_re = re.compile(r'Recommendation:\s+(?P<recommendation>Bet on [\w\s]+|No positive EV bets)', re.MULTILINE)
        
        for match in data_re.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            game_key = f"{away_team}:{home_team}"
            
            game_dict = {
                'away_team': away_team,
                'home_team': home_team,
                'home_confidence': match.group('home_confidence'),
                'ou_pick': match.group('ou_pick'),
                'ou_value': match.group('ou_value'),
                'ou_confidence': match.group('ou_confidence'),
                'away_confidence': None,
                'away_team_ev': None,
                'home_team_ev': None,
                'away_team_odds': None,
                'home_team_odds': None,
                'recommendation': None,
                'sport': 'NFL'
            }
            
            # Calculate away team confidence
            if game_dict['home_confidence']:
                try:
                    home_conf = float(game_dict['home_confidence'])
                    away_conf = round(100 - home_conf, 1)
                    game_dict['away_confidence'] = str(away_conf)
                except ValueError:
                    pass
            
            # Extract Expected Values
            for ev_match in ev_re.finditer(stdout):
                team_name = ev_match.group('team').strip()
                ev_value = ev_match.group('ev')
                
                if team_name == game_dict['away_team']:
                    game_dict['away_team_ev'] = ev_value
                elif team_name == game_dict['home_team']:
                    game_dict['home_team_ev'] = ev_value
            
            # Extract odds data
            for odds_match in odds_re.finditer(stdout):
                away_team_odds = odds_match.group('away_team').strip()
                home_team_odds = odds_match.group('home_team').strip()
                
                if (away_team_odds == game_dict['away_team'] and 
                    home_team_odds == game_dict['home_team']):
                    game_dict['away_team_odds'] = odds_match.group('away_team_odds')
                    game_dict['home_team_odds'] = odds_match.group('home_team_odds')
                    break
            
            # Extract betting recommendation
            for rec_match in recommendation_re.finditer(stdout):
                recommendation = rec_match.group('recommendation')
                if (game_dict['away_team'] in recommendation or 
                    game_dict['home_team'] in recommendation):
                    game_dict['recommendation'] = recommendation
                    break

            # Create comprehensive recommended bet
            recommendations = []
            
            # Add ML recommendation if confidence is high
            if game_dict['away_confidence'] and float(game_dict['away_confidence']) > 55:
                recommendations.append(f"{away_team} ML ({game_dict['away_confidence']}%)")
            elif game_dict['home_confidence'] and float(game_dict['home_confidence']) > 55:
                recommendations.append(f"{home_team} ML ({game_dict['home_confidence']}%)")
            
            # Add OU recommendation
            if game_dict['ou_pick'] and game_dict['ou_confidence']:
                recommendations.append(f"{game_dict['ou_pick']} {game_dict['ou_value']} ({game_dict['ou_confidence']}%)")
            
            # Add EV-based recommendation
            if game_dict['away_team_ev'] and float(game_dict['away_team_ev']) > 0.05:
                recommendations.append(f"{away_team} (EV: {game_dict['away_team_ev']})")
            elif game_dict['home_team_ev'] and float(game_dict['home_team_ev']) > 0.05:
                recommendations.append(f"{home_team} (EV: {game_dict['home_team_ev']})")
            
            # Add system recommendation if available
            if game_dict['recommendation'] and game_dict['recommendation'] != 'No positive EV bets':
                recommendations.append(game_dict['recommendation'])
            
            # Create the main recommended bet
            main_recommendation = ""
            main_confidence = "0"
            
            if recommendations:
                main_recommendation = recommendations[0]
                if game_dict['ou_confidence']:
                    main_confidence = game_dict['ou_confidence']
                elif game_dict['away_confidence']:
                    main_confidence = game_dict['away_confidence']
                elif game_dict['home_confidence']:
                    main_confidence = game_dict['home_confidence']
            
            # Round confidence values for better readability
            if game_dict.get('ou_confidence'):
                try:
                    game_dict['ou_confidence'] = f"{round(float(game_dict['ou_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
            
            if game_dict.get('away_confidence'):
                try:
                    game_dict['away_confidence'] = f"{round(float(game_dict['away_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
                    
            if game_dict.get('home_confidence'):
                try:
                    game_dict['home_confidence'] = f"{round(float(game_dict['home_confidence']), 1)}%"
                except (ValueError, TypeError):
                    pass
            
            # Round main confidence
            try:
                main_confidence_rounded = round(float(main_confidence), 1)
            except (ValueError, TypeError):
                main_confidence_rounded = main_confidence
            
            # Round OU value for better readability
            if game_dict.get('ou_value'):
                try:
                    game_dict['ou_value'] = str(round(float(game_dict['ou_value']), 1))
                except (ValueError, TypeError):
                    pass
            
            # Add comprehensive summary
            game_dict['raw_prediction'] = f"{away_team} vs {home_team}, recommended bet: {main_recommendation}, confidence: {main_confidence_rounded}%"
            game_dict['recommended_bet'] = main_recommendation
            game_dict['confidence'] = f"{main_confidence_rounded}%"
            game_dict['all_recommendations'] = recommendations

            games[game_key] = game_dict
    
    # If no games were parsed, return empty dict
    if not games:
        log_debug("NFL: No games parsed from output")
    
    return games

@lru_cache()
def fetch_nhl_predictions(ttl_hash=None):
    """Fetch NHL predictions"""
    del ttl_hash
    try:
        cmd = ["python", "main.py", "-xgb"]
        result = subprocess.run(cmd, cwd=PROJECT_PATHS['NHL'], 
                              capture_output=True, text=True, timeout=20,
                              encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            print(f"NHL command failed with return code {result.returncode}")
            print(f"NHL stderr: {result.stderr}")
            print(f"NHL stdout: {result.stdout}")
            return {}
        
        # Check if there are no games scheduled
        if "No games available" in result.stdout or "No upcoming games found" in result.stdout or "No games found for today" in result.stdout:
            log_debug("NHL: No games scheduled for today")
            return {}
        
        # Check if stdout is empty or only contains debug messages
        if not result.stdout.strip() or len(result.stdout.strip()) < 50:
            log_debug("NHL: No meaningful output, likely no games scheduled")
            return {}
        
        return parse_nhl_predictions(result.stdout)
    except subprocess.TimeoutExpired:
        print("NHL command timed out")
        return {}
    except Exception as e:
        print(f"Error fetching NHL predictions: {e}")
        return {}

def parse_nhl_predictions(stdout):
    """Parse NHL predictions from main.py output"""
    games = {}
    
    # Check if there are no games in the output
    if "No games available" in stdout or "No upcoming games found" in stdout or "No games found for today" in stdout:
        return {}
    
    # First, try to extract Over/Under information from the console output
    console_ou_pattern = re.compile(r'(?P<home_team>[\w .]+) vs (?P<away_team>[\w .]+) \([^)]+\): (?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d.]+) \([^)]+%\)', re.MULTILINE)
    ou_predictions = {}
    
    for match in console_ou_pattern.finditer(stdout):
        home_team = match.group('home_team').strip()
        away_team = match.group('away_team').strip()
        ou_pick = match.group('ou_pick').strip()
        ou_value = match.group('ou_value').strip()
        game_key = f"{away_team}:{home_team}"
        ou_predictions[game_key] = {'pick': ou_pick, 'value': ou_value}
    
    # Look for the new TXT Export format with integrated OU field: "Team A vs Team B, recommended bet: Team C, Spread:X.X(confidence%), ML:Y(confidence%), OU:Pick Value(confidence%)"
    txt_pattern = re.compile(r'(?P<home_team>[\w .]+) vs (?P<away_team>[\w .]+), recommended bet: (?P<recommended_team>[\w .]+), Spread:(?P<spread_value>[\d.+-]+)\((?P<spread_confidence>[\d.]+)%\), ML:(?P<ml_value>[\d+-]+)\((?P<ml_confidence>[\d.]+)%\), OU:(?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d.]+)\((?P<ou_confidence>[\d.]+)%\)', re.MULTILINE)
    
    for match in txt_pattern.finditer(stdout):
        home_team = match.group('home_team').strip()
        away_team = match.group('away_team').strip()
        recommended_team = match.group('recommended_team').strip()
        spread_value = match.group('spread_value').strip()
        spread_confidence = match.group('spread_confidence').strip()
        ml_value = match.group('ml_value').strip()
        ml_confidence = match.group('ml_confidence').strip()
        ou_pick = match.group('ou_pick').strip()
        ou_value = match.group('ou_value').strip()
        ou_confidence = match.group('ou_confidence').strip()
        
        game_key = f"{away_team}:{home_team}"
        
        # Round the values for better readability
        spread_value_rounded = round(float(spread_value), 1)
        ou_value_rounded = round(float(ou_value), 1)
        spread_confidence_rounded = round(float(spread_confidence), 1)
        ml_confidence_rounded = round(float(ml_confidence), 1)
        ou_confidence_rounded = round(float(ou_confidence), 1)
        
        # Create bet description - this is typically ML since the format shows ML prediction
        bet_description = f"{recommended_team} ML"
        
        # Format the spread value with proper sign based on recommended team
        # Determine if recommended team is home or away to assign correct sign
        if recommended_team == home_team:
            formatted_spread = f"-{spread_value_rounded}"
        else:  # away team
            formatted_spread = f"+{spread_value_rounded}"
        formatted_bet = f"{recommended_team} {formatted_spread}"
        
        # Use the OU prediction from the TXT format
        ou_prediction = ou_pick
        
        # Create the raw prediction in the requested format with individual confidence values
        raw_prediction = f"{home_team} vs {away_team}, recommended bet: {recommended_team}, Spread:{formatted_spread}({spread_confidence_rounded}%), ML:{ml_value}({ml_confidence_rounded}%), OU:{ou_prediction} {ou_value_rounded}({ou_confidence_rounded}%)"
        
        games[game_key] = {
            'away_team': away_team,
            'home_team': home_team,
            'recommended_team': recommended_team,
            'recommended_bet': bet_description,
            'confidence': f"{ml_confidence_rounded}%",  # Use ML confidence as primary confidence
            'bet_category': 'moneyline',
            'sport': 'NHL',
            'raw_prediction': raw_prediction,
            'spread_prediction': f"{recommended_team} {formatted_spread}",
            'spread_line': formatted_spread,
            'spread_confidence': f"{spread_confidence_rounded}%",
            'ou_prediction': f"{ou_prediction} {ou_value_rounded}",
            'ou_value': str(ou_value_rounded),
            'ou_confidence': f"{ou_confidence_rounded}%",
            'ml_value': ml_value,
            'ml_confidence': f"{ml_confidence_rounded}%"
        }
    
    # If no TXT format found, look for the old simple format: "Team A vs Team B, recommended bet: Team A OVER 5.5, confidence: 51.3%"
    if not games:
        simple_pattern = re.compile(r'(?P<away_team>[\w ]+) vs (?P<home_team>[\w ]+), recommended bet: (?P<recommended_team>[\w ]+) (?P<bet_type>OVER|UNDER) (?P<bet_value>[\d.]+), confidence: (?P<confidence>[\d.]+)%', re.MULTILINE)
        
        for match in simple_pattern.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            recommended_team = match.group('recommended_team').strip()
            bet_type = match.group('bet_type').strip()
            bet_value = match.group('bet_value').strip()
            confidence = match.group('confidence').strip()
            
            game_key = f"{away_team}:{home_team}"
            
            # Round the values for better readability
            bet_value_rounded = round(float(bet_value), 1)
            confidence_rounded = round(float(confidence), 1)
            
            bet_description = f"{recommended_team} {bet_type} {bet_value_rounded}"
            
            # Format the recommended bet with only spread value (add sign for spread bets)
            if bet_type in ['OVER', 'UNDER']:
                # For over/under bets, don't add spread sign
                formatted_bet = f"{recommended_team} {bet_type} {bet_value_rounded}"
            else:
                # For spread bets, determine if recommended team is home or away
                if recommended_team == home_team:
                    formatted_spread = f"-{bet_value_rounded}"
                else:  # away team
                    formatted_spread = f"+{bet_value_rounded}"
                formatted_bet = f"{recommended_team} {formatted_spread}"
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'recommended_bet': bet_description,
                'confidence': f"{confidence_rounded}%",
                'bet_category': "over_under",
                'sport': 'NHL',
                'raw_prediction': f"{away_team} vs {home_team}, recommended bet: {formatted_bet}, confidence: {confidence_rounded}%",
                'ou_prediction': f"{bet_type} {bet_value_rounded}",
                'ou_value': str(bet_value_rounded),
                'ou_confidence': f"{confidence_rounded}%"
            }
    
    # If no simple format found, try the original complex parsing and create summary
    if not games:
        data_re = re.compile(r'(?P<home_team>[\w ]+)(\s+\((?P<home_confidence>[\d+\.]+)%\))?\s+vs\s+(?P<away_team>[\w ]+)(\s+\((?P<away_confidence>[\d+\.]+)%\))?:\s+(?P<ou_pick>OVER|UNDER)\s+(?P<ou_value>[\d+\.]+)\s+\((?P<ou_confidence>[\d+\.]+)%\)', re.MULTILINE)
        ev_re = re.compile(r'(?P<team>[\w ]+)\s+EV:\s+(?P<ev>[-\d+\.]+)', re.MULTILINE)
        spread_re = re.compile(r'\|?\s*Recommended bet:\s+(?P<spread_team>[\w ]+)\s+(?P<spread_value>[+-]?[\d+\.]+)\s+\((?P<spread_confidence>[\d+\.]+)%\)', re.MULTILINE)
        
        for match in data_re.finditer(stdout):
            away_team = match.group('away_team').strip()
            home_team = match.group('home_team').strip()
            game_key = f"{away_team}:{home_team}"
            
            away_conf = round(float(match.group('away_confidence')) if match.group('away_confidence') else 0, 1)
            home_conf = round(float(match.group('home_confidence')) if match.group('home_confidence') else 0, 1)
            
            game_dict = {
                'away_team': away_team,
                'home_team': home_team,
                'away_confidence': away_conf,
                'home_confidence': home_conf,
                'ou_pick': match.group('ou_pick'),
                'ou_value': round(float(match.group('ou_value')), 1),
                'ou_confidence': round(float(match.group('ou_confidence')), 1),
                'away_team_odds': f"{away_conf}%",
                'home_team_odds': f"{home_conf}%",
                'ou_line': round(float(match.group('ou_value')), 1),
                'home_spread': 0,
                'sport': 'NHL'
            }
            
            # Add EV data
            for ev_match in ev_re.finditer(stdout):
                if ev_match.group('team') == game_dict['away_team']:
                    game_dict['away_team_ev'] = round(float(ev_match.group('ev')), 2)
                if ev_match.group('team') == game_dict['home_team']:
                    game_dict['home_team_ev'] = round(float(ev_match.group('ev')), 2)
            
            # Add spread data
            for spread_match in spread_re.finditer(stdout):
                spread_team = spread_match.group('spread_team').strip()
                spread_value = spread_match.group('spread_value')
                spread_confidence = round(float(spread_match.group('spread_confidence')), 1)
                
                # Check if this spread recommendation is for this game
                if spread_team in [game_dict['home_team'], game_dict['away_team']]:
                    # Format spread value with proper sign
                    clean_spread_value = spread_value.lstrip('+-')
                    if spread_team == game_dict['home_team']:
                        formatted_spread_value = f"-{clean_spread_value}"
                    else:  # away team
                        formatted_spread_value = f"+{clean_spread_value}"
                    game_dict['spread_pick'] = f"{spread_team} {formatted_spread_value}"
                    game_dict['spread_confidence'] = spread_confidence

            # Create comprehensive recommended bet
            recommendations = []
            
            # Add ML recommendation if confidence is high
            if game_dict['away_confidence'] > 55:
                recommendations.append(f"{away_team} ML ({game_dict['away_confidence']}%)")
            elif game_dict['home_confidence'] > 55:
                recommendations.append(f"{home_team} ML ({game_dict['home_confidence']}%)")
            
            # Add OU recommendation
            if game_dict['ou_pick'] and game_dict['ou_confidence']:
                recommendations.append(f"{game_dict['ou_pick']} {game_dict['ou_value']} ({game_dict['ou_confidence']}%)")
                # Also update the ou_prediction field
                game_dict['ou_prediction'] = f"{game_dict['ou_pick']} {game_dict['ou_value']}"
            
            # Add spread recommendation
            if 'spread_pick' in game_dict and game_dict['spread_confidence']:
                recommendations.append(f"{game_dict['spread_pick']} ({game_dict['spread_confidence']}%)")
            
            # Add EV-based recommendation
            if game_dict.get('away_team_ev') and game_dict['away_team_ev'] > 0.05:
                recommendations.append(f"{away_team} (EV: {game_dict['away_team_ev']})")
            elif game_dict.get('home_team_ev') and game_dict['home_team_ev'] > 0.05:
                recommendations.append(f"{home_team} (EV: {game_dict['home_team_ev']})")
            
            # Create the main recommended bet
            main_recommendation = ""
            main_confidence = "0"
            
            if recommendations:
                main_recommendation = recommendations[0]
                if game_dict['ou_confidence']:
                    main_confidence = str(game_dict['ou_confidence'])
                elif game_dict['away_confidence']:
                    main_confidence = str(game_dict['away_confidence'])
                elif game_dict['home_confidence']:
                    main_confidence = str(game_dict['home_confidence'])
            
            # Add comprehensive summary
            game_dict['raw_prediction'] = f"{away_team} vs {home_team}, recommended bet: {main_recommendation}, confidence: {main_confidence}%"
            game_dict['recommended_bet'] = main_recommendation
            game_dict['confidence'] = f"{main_confidence}%"
            game_dict['all_recommendations'] = recommendations

            games[game_key] = game_dict
    
    # If no games were parsed, return empty dict
    if not games:
        log_debug("NHL: No games parsed from output")
    
    return games

def get_mock_predictions(sport):
    """Return mock predictions when real data is unavailable"""
    mock_data = {
        'CFB': {
            'Alabama:Georgia': {
                'away_team': 'Alabama',
                'home_team': 'Georgia',
                'kickoff': '7:00 PM EST',
                'venue': 'Sanford Stadium',
                'ml_prediction': 'Georgia',
                'ml_confidence': '65% confidence',
                'ml_value': '+120',
                'spread_prediction': 'Georgia Covers',
                'spread_confidence': '58% confidence',
                'spread_value': '52% confidence',
                'spread_line': '-3.5',
                'ou_prediction': 'OVER 45.5',
                'ou_confidence': '62% confidence',
                'raw_prediction': 'Alabama vs Georgia, recommended bet: Georgia -3.5, confidence: 58%',
                'recommended_bet': 'Georgia -3.5',
                'confidence': '58%',
                'sport': 'CFB'
            }
        },
        'MLB': {
            'Yankees:Red Sox': {
                'away_team': 'Yankees',
                'home_team': 'Red Sox',
                'away_confidence': '55%',
                'home_confidence': '45%',
                'ou_pick': 'OVER',
                'ou_value': '8.5',
                'ou_confidence': '60%',
                'away_team_ev': '0.05',
                'home_team_ev': '-0.02',
                'recommended_bet': 'Yankees ML',
                'confidence': '55%',
                'raw_prediction': 'Yankees vs Red Sox, recommended bet: Yankees ML, confidence: 55%',
                'sport': 'MLB'
            }
        },
        'NBA': {
            'Lakers:Warriors': {
                'away_team': 'Lakers',
                'home_team': 'Warriors',
                'away_confidence': '48%',
                'home_confidence': '52%',
                'ou_pick': 'OVER',
                'ou_value': '225.5',
                'ou_confidence': '58%',
                'away_team_ev': '0.03',
                'home_team_ev': '0.01',
                'away_team_odds': '+110',
                'home_team_odds': '-130',
                'sport': 'NBA'
            }
        },
        'NFL': {
            'Chiefs:Bills': {
                'away_team': 'Chiefs',
                'home_team': 'Bills',
                'home_confidence': '55%',
                'ou_pick': 'OVER',
                'ou_value': '48.5',
                'ou_confidence': '62%',
                'away_confidence': '45%',
                'away_team_ev': '0.08',
                'home_team_ev': '0.02',
                'away_team_odds': '+105',
                'home_team_odds': '-125',
                'recommendation': 'Bet on Chiefs',
                'recommended_bet': 'Chiefs ML',
                'confidence': '55%',
                'raw_prediction': 'Chiefs vs Bills, recommended bet: Chiefs ML, confidence: 55%',
                'sport': 'NFL'
            }
        },
        'NHL': {
            'Maple Leafs:Bruins': {
                'away_team': 'Maple Leafs',
                'home_team': 'Bruins',
                'away_confidence': 52.0,
                'home_confidence': 48.0,
                'ou_pick': 'OVER',
                'ou_value': 5.5,
                'ou_confidence': 65.0,
                'away_team_odds': '52%',
                'home_team_odds': '48%',
                'ou_line': 5.5,
                'home_spread': 0,
                'away_team_ev': 0.05,
                'home_team_ev': -0.01,
                'spread_pick': 'Maple Leafs +1.5',
                'spread_confidence': 70.0,
                'recommended_bet': 'Maple Leafs OVER 5.5',
                'confidence': '51.3%',
                'raw_prediction': 'Maple Leafs vs Bruins, recommended bet: Maple Leafs OVER 5.5, confidence: 51.3%',
                'sport': 'NHL'
            }
        }
    }
    return mock_data.get(sport, {})

# Debug mode - set to True for detailed error logging
DEBUG_MODE = True

def log_debug(message):
    """Log debug messages if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def get_ttl_hash(seconds=600):
    """Return the same value within `seconds` time period"""
    return round(time.time() / seconds)

@app.route("/")
def index():
    """Main dashboard page showing all sports predictions"""
    try:
        # Fetch predictions from all sports - no mock data fallback
        cfb_predictions = fetch_cfb_predictions(ttl_hash=get_ttl_hash())
        if not cfb_predictions:
            print("No CFB games found")
            cfb_predictions = {}
            
        mlb_predictions = fetch_mlb_predictions(ttl_hash=get_ttl_hash())
        if not mlb_predictions:
            print("No MLB games found")
            mlb_predictions = {}
            
        nba_predictions = fetch_nba_predictions(ttl_hash=get_ttl_hash())
        if not nba_predictions:
            print("No NBA games found")
            nba_predictions = {}
            
        nfl_predictions = fetch_nfl_predictions(ttl_hash=get_ttl_hash())
        if not nfl_predictions:
            print("No NFL games found")
            nfl_predictions = {}
            
        nhl_predictions = fetch_nhl_predictions(ttl_hash=get_ttl_hash())
        if not nhl_predictions:
            print("No NHL games found")
            nhl_predictions = {}
        
        # Combine all predictions
        all_predictions = {
            'CFB': cfb_predictions,
            'MLB': mlb_predictions,
            'NBA': nba_predictions,
            'NFL': nfl_predictions,
            'NHL': nhl_predictions
        }
        
        return render_template('index.html', 
                            today=date.today(), 
                            data=all_predictions)
    except Exception as e:
        print(f"Error in index route: {e}")
        return f"Error loading predictions: {str(e)}", 500

@app.route("/api/predictions")
def api_predictions():
    """API endpoint to get all predictions as JSON"""
    try:
        predictions = {
            'CFB': fetch_cfb_predictions(ttl_hash=get_ttl_hash()) or {},
            'MLB': fetch_mlb_predictions(ttl_hash=get_ttl_hash()) or {},
            'NBA': fetch_nba_predictions(ttl_hash=get_ttl_hash()) or {},
            'NFL': fetch_nfl_predictions(ttl_hash=get_ttl_hash()) or {},
            'NHL': fetch_nhl_predictions(ttl_hash=get_ttl_hash()) or {}
        }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/predictions/<sport>")
def api_sport_predictions(sport):
    """API endpoint to get predictions for a specific sport"""
    sport = sport.upper()
    try:
        if sport == 'CFB':
            predictions = fetch_cfb_predictions(ttl_hash=get_ttl_hash()) or {}
        elif sport == 'MLB':
            predictions = fetch_mlb_predictions(ttl_hash=get_ttl_hash()) or {}
        elif sport == 'NBA':
            predictions = fetch_nba_predictions(ttl_hash=get_ttl_hash()) or {}
        elif sport == 'NFL':
            predictions = fetch_nfl_predictions(ttl_hash=get_ttl_hash()) or {}
        elif sport == 'NHL':
            predictions = fetch_nhl_predictions(ttl_hash=get_ttl_hash()) or {}
        else:
            return jsonify({'error': f'Unknown sport: {sport}'}), 400
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/export/csv")
def export_csv():
    """Export all predictions to CSV"""
    try:
        predictions = {
            'CFB': fetch_cfb_predictions(ttl_hash=get_ttl_hash()) or {},
            'MLB': fetch_mlb_predictions(ttl_hash=get_ttl_hash()) or {},
            'NBA': fetch_nba_predictions(ttl_hash=get_ttl_hash()) or {},
            'NFL': fetch_nfl_predictions(ttl_hash=get_ttl_hash()) or {},
            'NHL': fetch_nhl_predictions(ttl_hash=get_ttl_hash()) or {}
        }
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Sport', 'Away Team', 'Home Team', 'Kickoff/Time', 'Venue',
            'ML Prediction', 'ML Confidence', 'ML Value',
            'Spread Prediction', 'Spread Confidence', 'Spread Value',
            'OU Prediction', 'OU Confidence', 'OU Value',
            'Away EV', 'Home EV', 'Away Odds', 'Home Odds', 'Recommendation'
        ])
        
        # Write data rows for each sport
        for sport, sport_predictions in predictions.items():
            for game_key, game_data in sport_predictions.items():
                writer.writerow([
                    sport,
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
                    game_data.get('ou_confidence', ''),
                    game_data.get('ou_value', ''),
                    game_data.get('away_team_ev', ''),
                    game_data.get('home_team_ev', ''),
                    game_data.get('away_team_odds', ''),
                    game_data.get('home_team_odds', ''),
                    game_data.get('recommendation', '')
                ])
        
        # Prepare response
        output.seek(0)
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=sports_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )
        
        return response
    except Exception as e:
        return f"Error exporting CSV: {str(e)}", 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sports': list(PROJECT_PATHS.keys())
    })

if __name__ == "__main__":
    # Production configuration for VPS
    if sys.platform.startswith('win'):
        # Windows VPS production
        print("Starting Sports Dashboard on Windows...")
        print("Access at: http://0.0.0.0:5000")
        print("External access: http://your-vps-ip:5000")
        app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
    else:
        # Linux VPS production
        app.run(debug=False, host='0.0.0.0', port=5000)
