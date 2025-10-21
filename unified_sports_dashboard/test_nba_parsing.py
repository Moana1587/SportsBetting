#!/usr/bin/env python3
"""
Test script to verify NBA prediction parsing works with the new format.
"""

import re

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
            game_dict['ou_confidence'] = f"{game_dict['ou_confidence']}%" if game_dict.get('ou_confidence') else None

        games[game_key] = game_dict
    
    # If no games were parsed, return empty dict
    if not games:
        print("NBA: No games parsed from output")
    
    return games

def test_parsing():
    """Test the parsing function with sample NBA output"""
    
    # Sample NBA output from the terminal
    sample_output = """2025-10-21 14:12:58.761640: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-10-21 14:13:00.170263: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Loaded spread model: XGBoost_Spread_Value_6.774_MSE.json
Loaded ML classification model: XGBoost_70.3%_ML_Classification.json
Using modern NBA API approach with parameters
Attempting to fetch NBA games from ESPN API (attempt 1/3)...
Successfully fetched 2 NBA games from ESPN API
Fetching team statistics using nba_api...
Attempting to fetch team stats using nba_api (attempt 1/3)...
Successfully fetched team stats using nba_api: 76 teams
Team statistics data cached to Data/cached_team_stats.pkl
No odds provided for Oklahoma City Thunder vs Houston Rockets. Using default values.
No odds provided for Los Angeles Lakers vs Golden State Warriors. Using default values.
---------------XGBoost Model Predictions---------------
================================================================================
                    XGBOOST MODEL PREDICTIONS
================================================================================
Oklahoma City Thunder vs Houston Rockets, recommended bet: Oklahoma City Thunder, Spread:+6.2(36.4%), ML:-175(63.6%), OU:OVER 220.0(47.6%)
Los Angeles Lakers vs Golden State Warriors, recommended bet: Los Angeles Lakers, Spread:+6.9(42.2%), ML:-137(57.8%), OU:OVER 220.0(78.2%)

============================================================
EXPECTED VALUE
============================================================

Oklahoma City Thunder vs Houston Rockets:
  HOME Oklahoma City Thunder EV: 21.370
  AWAY Houston Rockets EV: -30.460

Los Angeles Lakers vs Golden State Warriors:
  HOME Los Angeles Lakers EV: 10.400
  AWAY Golden State Warriors EV: -19.490

================================================================================
-------------------------------------------------------"""
    
    print("Testing NBA Prediction Parsing")
    print("=" * 50)
    
    # Parse the sample output
    games = parse_nba_predictions(sample_output)
    
    print(f"Parsed {len(games)} games:")
    print()
    
    for game_key, game_data in games.items():
        print(f"Game: {game_key}")
        print(f"  Home Team: {game_data['home_team']}")
        print(f"  Away Team: {game_data['away_team']}")
        print(f"  Recommended Team: {game_data['recommended_team']}")
        print(f"  Spread: {game_data['spread_value']} ({game_data['spread_confidence']}%)")
        print(f"  ML: {game_data['ml_odds']} ({game_data['ml_confidence']}%)")
        print(f"  OU: {game_data['ou_pick']} {game_data['ou_value']} ({game_data['ou_confidence']}%)")
        print(f"  Raw Prediction: {game_data['raw_prediction']}")
        print(f"  All Recommendations: {game_data['all_recommendations']}")
        print()

if __name__ == "__main__":
    test_parsing()
