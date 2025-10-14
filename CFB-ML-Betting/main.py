import argparse
import sqlite3
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from colorama import Fore, Style
import os

def cfb_schedule_today_espn(tz="America/Los_Angeles", groups="80"):
    """Get today's college football schedule from ESPN API."""
    today = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d")
    url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
    js = requests.get(url, params={"dates": today, "groups": groups}, timeout=30).json()

    rows = []
    for e in js.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) < 2: 
            continue
        home = next((c for c in comps if c.get("homeAway")=="home"), comps[0])
        away = next((c for c in comps if c.get("homeAway")=="away"), comps[1])

        iso = comp.get("date") or e.get("date")
        # convert ISO time to your timezone (kickoff)
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))

        rows.append({
            "eventId": e.get("id"),
            "kickoff_local": dt.strftime("%Y-%m-%d %H:%M"),
            "status": e.get("status", {}).get("type", {}).get("description"),
            "away": away.get("team", {}).get("displayName"),
            "home": home.get("team", {}).get("displayName"),
            "venue": comp.get("venue", {}).get("fullName"),
            "tv": ",".join([b.get("name") for b in comp.get("broadcasts", []) if b.get("name")]),
        })

    df = pd.DataFrame(rows).sort_values(["kickoff_local","home","away"])
    return df


def clean_team_name(team_name):
    """Clean team name to match database format"""
    # Handle specific team name mappings first
    team_mappings = {
        'Middle Tennessee Blue Raiders': 'Middle Tennessee',
        'Missouri State Bears': 'Missouri State',
        'Liberty Flames': 'Liberty',
        'UTEP Miners': 'UTEP',
        'Miami (OH)': 'Miami (OH)',
        'San José State': 'San José State',
        'UL Monroe': 'UL Monroe',
        'South Alabama Jaguars': 'South Alabama',
        'Western Kentucky Hilltoppers': 'Western Kentucky',
        'Arkansas State Red Wolves': 'Arkansas State',
        'Florida International Panthers': 'Florida International',
        'New Mexico State Aggies': 'New Mexico State',
        'Air Force Falcons': 'Air Force',
        'Akron Zips': 'Akron',
        'Alabama Crimson Tide': 'Alabama',
        'App State Mountaineers': 'App State',
        'Arizona Wildcats': 'Arizona',
        'Arizona State Sun Devils': 'Arizona State',
        'Arkansas Razorbacks': 'Arkansas',
        'Army Black Knights': 'Army',
        'Auburn Tigers': 'Auburn',
        'Ball State Cardinals': 'Ball State',
        'Baylor Bears': 'Baylor',
        'Boise State Broncos': 'Boise State',
        'Boston College Eagles': 'Boston College',
        'Bowling Green Falcons': 'Bowling Green',
        'Buffalo Bulls': 'Buffalo',
        'BYU Cougars': 'BYU',
        'California Golden Bears': 'California',
        'Central Michigan Chippewas': 'Central Michigan',
        'Charlotte 49ers': 'Charlotte',
        'Cincinnati Bearcats': 'Cincinnati',
        'Clemson Tigers': 'Clemson',
        'Coastal Carolina Chanticleers': 'Coastal Carolina',
        'Colorado Buffaloes': 'Colorado',
        'Colorado State Rams': 'Colorado State',
        'Delaware Fightin\' Blue Hens': 'Delaware',
        'Duke Blue Devils': 'Duke',
        'East Carolina Pirates': 'East Carolina',
        'Eastern Michigan Eagles': 'Eastern Michigan',
        'Florida Gators': 'Florida',
        'Florida Atlantic Owls': 'Florida Atlantic',
        'Florida State Seminoles': 'Florida State',
        'Fresno State Bulldogs': 'Fresno State',
        'Georgia Bulldogs': 'Georgia',
        'Georgia Southern Eagles': 'Georgia Southern',
        'Georgia State Panthers': 'Georgia State',
        'Georgia Tech Yellow Jackets': 'Georgia Tech',
        'Hawai\'i Rainbow Warriors': 'Hawai\'i',
        'Houston Cougars': 'Houston',
        'Illinois Fighting Illini': 'Illinois',
        'Indiana Hoosiers': 'Indiana',
        'Iowa Hawkeyes': 'Iowa',
        'Iowa State Cyclones': 'Iowa State',
        'Jacksonville State Gamecocks': 'Jacksonville State',
        'James Madison Dukes': 'James Madison',
        'Kansas Jayhawks': 'Kansas',
        'Kansas State Wildcats': 'Kansas State',
        'Kennesaw State Owls': 'Kennesaw State',
        'Kent State Golden Flashes': 'Kent State',
        'Kentucky Wildcats': 'Kentucky',
        'Louisiana Ragin\' Cajuns': 'Louisiana',
        'Louisiana Tech Bulldogs': 'Louisiana Tech',
        'Louisville Cardinals': 'Louisville',
        'LSU Tigers': 'LSU',
        'Marshall Thundering Herd': 'Marshall',
        'Maryland Terrapins': 'Maryland',
        'Massachusetts Minutemen': 'Massachusetts',
        'Memphis Tigers': 'Memphis',
        'Miami Hurricanes': 'Miami',
        'Michigan Wolverines': 'Michigan',
        'Michigan State Spartans': 'Michigan State',
        'Minnesota Golden Gophers': 'Minnesota',
        'Mississippi State Bulldogs': 'Mississippi State',
        'Missouri Tigers': 'Missouri',
        'Navy Midshipmen': 'Navy',
        'NC State Wolfpack': 'NC State',
        'Nebraska Cornhuskers': 'Nebraska',
        'Nevada Wolf Pack': 'Nevada',
        'New Mexico Lobos': 'New Mexico',
        'North Carolina Tar Heels': 'North Carolina',
        'Northern Illinois Huskies': 'Northern Illinois',
        'North Texas Mean Green': 'North Texas',
        'Northwestern Wildcats': 'Northwestern',
        'Notre Dame Fighting Irish': 'Notre Dame',
        'Ohio Bobcats': 'Ohio',
        'Ohio State Buckeyes': 'Ohio State',
        'Oklahoma Sooners': 'Oklahoma',
        'Oklahoma State Cowboys': 'Oklahoma State',
        'Old Dominion Monarchs': 'Old Dominion',
        'Ole Miss Rebels': 'Ole Miss',
        'Oregon Ducks': 'Oregon',
        'Oregon State Beavers': 'Oregon State',
        'Penn State Nittany Lions': 'Penn State',
        'Pittsburgh Panthers': 'Pittsburgh',
        'Purdue Boilermakers': 'Purdue',
        'Rice Owls': 'Rice',
        'Rutgers Scarlet Knights': 'Rutgers',
        'Sam Houston Bearkats': 'Sam Houston',
        'San Diego State Aztecs': 'San Diego State',
        'SMU Mustangs': 'SMU',
        'South Carolina Gamecocks': 'South Carolina',
        'Southern Miss Golden Eagles': 'Southern Miss',
        'South Florida Bulls': 'South Florida',
        'Stanford Cardinal': 'Stanford',
        'Syracuse Orange': 'Syracuse',
        'TCU Horned Frogs': 'TCU',
        'Temple Owls': 'Temple',
        'Tennessee Volunteers': 'Tennessee',
        'Texas Longhorns': 'Texas',
        'Texas A&M Aggies': 'Texas A&M',
        'Texas State Bobcats': 'Texas State',
        'Texas Tech Red Raiders': 'Texas Tech',
        'Toledo Rockets': 'Toledo',
        'Troy Trojans': 'Troy',
        'Tulane Green Wave': 'Tulane',
        'Tulsa Golden Hurricane': 'Tulsa',
        'UAB Blazers': 'UAB',
        'UCF Knights': 'UCF',
        'UCLA Bruins': 'UCLA',
        'UConn Huskies': 'UConn',
        'UNLV Rebels': 'UNLV',
        'USC Trojans': 'USC',
        'Utah Utes': 'Utah',
        'Utah State Aggies': 'Utah State',
        'UTSA Roadrunners': 'UTSA',
        'Vanderbilt Commodores': 'Vanderbilt',
        'Virginia Cavaliers': 'Virginia',
        'Virginia Tech Hokies': 'Virginia Tech',
        'Wake Forest Demon Deacons': 'Wake Forest',
        'Washington Huskies': 'Washington',
        'Washington State Cougars': 'Washington State',
        'Western Michigan Broncos': 'Western Michigan',
        'West Virginia Mountaineers': 'West Virginia',
        'Wisconsin Badgers': 'Wisconsin',
        'Wyoming Cowboys': 'Wyoming'
    }
    
    # Check for exact matches first
    if team_name in team_mappings:
        return team_mappings[team_name]
    
    # Remove common suffixes and clean up the name
    suffixes_to_remove = [
        ' Bears', ' Raiders', ' Flames', ' Miners', ' Tigers', ' Bulldogs', 
        ' Eagles', ' Falcons', ' Hornets', ' Lions', ' Panthers', ' Rams',
        ' Rebels', ' Rockets', ' Spartans', ' Wildcats', ' Wolfpack',
        ' Aggies', ' Aztecs', ' Bobcats', ' Broncos', ' Buffaloes',
        ' Cardinals', ' Chippewas', ' Colonels', ' Cougars', ' Cowboys',
        ' Demon Deacons', ' Dukes', ' Fighting Irish', ' Golden Eagles',
        ' Golden Flashes', ' Golden Gophers', ' Golden Hurricane',
        ' Green Wave', ' Hawkeyes', ' Hoosiers', ' Hurricanes',
        ' Huskers', ' Huskies', ' Jayhawks', ' Knights', ' Lobos',
        ' Longhorns', ' Midshipmen', ' Mountaineers', ' Musketeers',
        ' Nittany Lions', ' Owls', ' Pirates', ' Rainbow Warriors',
        ' Red Raiders', ' Red Wolves', ' Runnin\' Utes', ' Scarlet Knights',
        ' Seminoles', ' Sooners', ' Sun Devils', ' Tar Heels', ' Terrapins',
        ' Thundering Herd', ' Trojans', ' Utes', ' Vandals', ' Volunteers',
        ' Warhawks', ' Warriors', ' Wolf Pack', ' Zips', ' Jaguars',
        ' Hilltoppers', ' Chanticleers', ' Fightin\' Blue Hens', ' Blue Devils',
        ' Gamecocks', ' Dukes', ' Golden Flashes', ' Ragin\' Cajuns',
        ' Midshipmen', ' Wolfpack', ' Cornhuskers', ' Wolf Pack',
        ' Lobos', ' Tar Heels', ' Huskies', ' Mean Green', ' Wildcats',
        ' Fighting Irish', ' Bobcats', ' Buckeyes', ' Sooners', ' Cowboys',
        ' Monarchs', ' Rebels', ' Ducks', ' Beavers', ' Nittany Lions',
        ' Panthers', ' Boilermakers', ' Owls', ' Scarlet Knights',
        ' Bearkats', ' Aztecs', ' Mustangs', ' Gamecocks', ' Golden Eagles',
        ' Bulls', ' Cardinal', ' Orange', ' Horned Frogs', ' Owls',
        ' Volunteers', ' Longhorns', ' Aggies', ' Bobcats', ' Red Raiders',
        ' Rockets', ' Trojans', ' Green Wave', ' Golden Hurricane',
        ' Blazers', ' Knights', ' Bruins', ' Huskies', ' Rebels',
        ' Trojans', ' Utes', ' Aggies', ' Roadrunners', ' Commodores',
        ' Cavaliers', ' Hokies', ' Demon Deacons', ' Huskies',
        ' Cougars', ' Broncos', ' Mountaineers', ' Badgers', ' Cowboys'
    ]
    
    clean_name = team_name
    for suffix in suffixes_to_remove:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break
    
    return clean_name.strip()

def get_team_stats_for_prediction(teams_con, team_name, season="2024-25"):
    """Get team stats for prediction from the database."""
    try:
        # Clean team name to match database format
        clean_name = clean_team_name(team_name)
        
        team_df = pd.read_sql_query(
            f"SELECT * FROM \"{season}_advanced_stats\" WHERE team = ?", 
            teams_con, 
            params=[clean_name]
        )
        
        if not team_df.empty:
            return team_df.iloc[0]
        else:
            print(f"  Warning: No stats found for {team_name} (cleaned: {clean_name}) in {season}")
            return None
    except Exception as e:
        print(f"  Error getting stats for {team_name} in {season}: {e}")
        return None

def get_betting_lines_for_game(odds_con, home_team, away_team, kickoff_date):
    """Get betting lines for a specific game from the odds database."""
    try:
        # Convert kickoff date to match database format
        kickoff_date_obj = datetime.strptime(kickoff_date, "%Y-%m-%d %H:%M")
        date_str = kickoff_date_obj.strftime("%Y-%m-%d")
        
        # Query for the most recent odds data for this game
        query = """
        SELECT OU, Spread, ML_Home, ML_Away 
        FROM odds_data 
        WHERE Home = ? AND Away = ? AND Date LIKE ?
        ORDER BY Date DESC
        LIMIT 1
        """
        
        result = pd.read_sql_query(query, odds_con, params=[home_team, away_team, f"{date_str}%"])
        
        if not result.empty:
            return {
                'spread': result.iloc[0]['Spread'],
                'over_under': result.iloc[0]['OU'],
                'ml_home': result.iloc[0]['ML_Home'],
                'ml_away': result.iloc[0]['ML_Away']
            }
        else:
            print(f"    Warning: No betting lines found for {away_team} @ {home_team}")
            return None
    except Exception as e:
        print(f"    Error getting betting lines for {away_team} @ {home_team}: {e}")
        return None

def createTodaysCFBGames(games_df, teams_con, odds_con, season="2024-25"):
    """Create today's CFB games for prediction using team stats and betting lines."""
    match_data = []
    game_info = []
    
    print(f"Processing {len(games_df)} games...")
    
    for idx, game in games_df.iterrows():
        home_team = game['home']
        away_team = game['away']
        kickoff = game['kickoff_local']
        
        print(f"  Processing: {away_team} @ {home_team} at {kickoff}")
        
        # Get team stats
        home_stats = get_team_stats_for_prediction(teams_con, home_team, season)
        away_stats = get_team_stats_for_prediction(teams_con, away_team, season)
        
        # Get betting lines
        betting_lines = get_betting_lines_for_game(odds_con, home_team, away_team, kickoff)
        
        if home_stats is not None and away_stats is not None:
            # Create game record
            game_record = {}
            
            # Add home team stats (prefix with 'home_')
            for col in home_stats.index:
                if col not in ['team', 'season', 'year']:
                    game_record[f'home_{col}'] = home_stats[col]
            
            # Add away team stats (prefix with 'away_')
            for col in away_stats.index:
                if col not in ['team', 'season', 'year']:
                    game_record[f'away_{col}'] = away_stats[col]
            
            # Add game info
            game_record['Date'] = kickoff
            game_record['Home_Team'] = home_team
            game_record['Away_Team'] = away_team
            
            match_data.append(game_record)
            game_info.append({
                'home': home_team,
                'away': away_team,
                'kickoff': kickoff,
                'venue': game.get('venue', ''),
                'tv': game.get('tv', ''),
                'betting_lines': betting_lines  # Include betting lines in game_info
            })
            
            print(f"    + Added to prediction data")
        else:
            print(f"    - Skipped - missing team stats")
    
    if match_data:
        games_df = pd.DataFrame(match_data)
        
        # Convert numeric columns to appropriate types
        numeric_columns = games_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Date', 'Home_Team', 'Away_Team']:
                games_df[col] = pd.to_numeric(games_df[col], errors='coerce')
        
        # Fill any NaN values with 0
        games_df = games_df.fillna(0)
        
        # Prepare features for prediction
        exclude_columns = ['Date', 'Home_Team', 'Away_Team']
        feature_columns = [col for col in games_df.columns if col not in exclude_columns]
        X = games_df[feature_columns].select_dtypes(include=[np.number])
        X = X.values.astype(float)
        
        return X, games_df, game_info
    else:
        print("No games could be processed for prediction.")
        return None, None, None


def calculate_spread_prediction(home_stats, away_stats, betting_lines=None):
    """Calculate spread prediction based on team stats and betting lines."""
    try:
        # Calculate basic spread based on team strength differential
        # This is a simplified calculation - in practice, you'd use more sophisticated metrics
        
        # Get key stats for spread calculation
        home_off_eff = home_stats.get('offensive_efficiency', 0) if home_stats is not None else 0
        home_def_eff = home_stats.get('defensive_efficiency', 0) if home_stats is not None else 0
        away_off_eff = away_stats.get('offensive_efficiency', 0) if away_stats is not None else 0
        away_def_eff = away_stats.get('defensive_efficiency', 0) if away_stats is not None else 0
        
        # Calculate net efficiency (offensive - defensive)
        home_net_eff = home_off_eff - home_def_eff
        away_net_eff = away_off_eff - away_def_eff
        
        # Calculate predicted point differential
        predicted_diff = home_net_eff - away_net_eff
        
        # If we have betting lines, compare our prediction to the line
        if betting_lines and betting_lines.get('spread'):
            spread_line = betting_lines.get('spread', 0)
            # Calculate how much our prediction differs from the line
            spread_value = abs(predicted_diff - spread_line)
            confidence = min(spread_value / 7.0, 1.0)  # Normalize confidence based on 7-point spread
        else:
            spread_value = abs(predicted_diff)
            confidence = min(spread_value / 14.0, 1.0)  # Normalize confidence based on 14-point spread
        
        return {
            'predicted_diff': predicted_diff,
            'spread_value': spread_value,
            'confidence': confidence,
            'home_net_eff': home_net_eff,
            'away_net_eff': away_net_eff
        }
    except Exception as e:
        print(f"Error calculating spread prediction: {e}")
        return {
            'predicted_diff': 0,
            'spread_value': 0,
            'confidence': 0,
            'home_net_eff': 0,
            'away_net_eff': 0
        }

def run_xgboost_spread_value_predictions(X, game_info, prediction_results=None):
    """Run XGBoost spread value regression predictions for today's games."""
    print(f"---------------XGBoost Spread Value Regression Predictions---------------")
    
    # Connect to database to load models
    try:
        # Find the best spread value model file
        import os
        import glob
        
        # Use absolute path for model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "Models")
        
        # Check if Models directory exists
        if not os.path.exists(models_dir):
            print(f"Models directory not found at: {models_dir}")
            print("Please train the spread value model first using XGBoost_Model_Spread.py")
            return None
        
        model_files = glob.glob(os.path.join(models_dir, "XGBoost_Spread_Value_*_MSE.json"))
        if not model_files:
            print(f"No spread value model found in {models_dir}")
            print("Please train the spread value model first using XGBoost_Model_Spread.py")
            return None
        
        # Get the model with lowest MSE
        best_model_file = min(model_files, key=lambda x: float(x.split('_')[-2]))
        print(f"Loading spread value model: {best_model_file}")
        
        # Load model
        model = xgb.Booster()
        model.load_model(best_model_file)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X)
        spread_predictions = model.predict(dmatrix)
        
        # Display results and collect for CSV export
        print(f"\nSpread Value Predictions for Today's Games:")
        print("-" * 80)
        
        for i, (game, pred_spread) in enumerate(zip(game_info, spread_predictions)):
            home = game['home']
            away = game['away']
            kickoff = game['kickoff']
            venue = game.get('venue', 'TBD')
            
            # Initialize game in prediction_results if not exists
            game_key = f"{away}:{home}"
            if game_key not in prediction_results:
                prediction_results[game_key] = {
                    'away_team': away,
                    'home_team': home,
                    'kickoff': kickoff,
                    'venue': venue
                }
                
                # Add betting lines if available
                betting_lines = game.get('betting_lines')
                if betting_lines:
                    prediction_results[game_key]['spread_line'] = betting_lines.get('spread', 'N/A')
                    prediction_results[game_key]['over_under_line'] = betting_lines.get('over_under', 'N/A')
                    prediction_results[game_key]['ml_home_line'] = betting_lines.get('ml_home', 'N/A')
                    prediction_results[game_key]['ml_away_line'] = betting_lines.get('ml_away', 'N/A')
                else:
                    prediction_results[game_key]['spread_line'] = 'N/A'
                    prediction_results[game_key]['over_under_line'] = 'N/A'
                    prediction_results[game_key]['ml_home_line'] = 'N/A'
                    prediction_results[game_key]['ml_away_line'] = 'N/A'
            
            # Calculate spread analysis
            betting_lines = game.get('betting_lines')
            spread_line = betting_lines.get('spread', 0) if betting_lines else 0
            
            # Determine prediction
            if pred_spread > 0:
                prediction = f"Home wins by {pred_spread:.1f}"
                recommended_team = home
                spread_direction = f"Home -{abs(pred_spread):.1f}"
            else:
                prediction = f"Away wins by {abs(pred_spread):.1f}"
                recommended_team = away
                spread_direction = f"Away +{abs(pred_spread):.1f}"
            
            # Calculate spread value vs betting line
            if spread_line != 0:
                spread_value = abs(pred_spread - spread_line)
                edge = abs(pred_spread) - abs(spread_line)
                edge_direction = "favorable" if edge > 0 else "unfavorable"
            else:
                spread_value = abs(pred_spread)
                edge = 0
                edge_direction = "no line"
            
            print(f"{away} @ {home}")
            print(f"  Kickoff: {kickoff}")
            print(f"  Venue: {venue}")
            print(f"  Predicted Margin: {pred_spread:+.1f} points")
            print(f"  Prediction: {prediction}")
            print(f"  Recommended Bet: {recommended_team} {spread_direction}")
            print(f"  Spread Value: {spread_value:.1f} points")
            print(f"  Edge vs Line: {edge:+.1f} points ({edge_direction})")
            
            # Show betting lines if available
            if betting_lines:
                print(f"  Betting Lines: Spread {betting_lines.get('spread', 'N/A')}, O/U {betting_lines.get('over_under', 'N/A')}")
                print(f"  ML Lines: Home {betting_lines.get('ml_home', 'N/A')}, Away {betting_lines.get('ml_away', 'N/A')}")
            else:
                print(f"  Betting Lines: Not available")
            
            # Store results for CSV
            prediction_results[game_key]['spread_value_prediction'] = f"{pred_spread:+.1f} points"
            prediction_results[game_key]['spread_value_margin'] = f"{pred_spread:+.1f}"
            prediction_results[game_key]['spread_value_recommendation'] = f"{recommended_team} {spread_direction}"
            prediction_results[game_key]['spread_value_value'] = f"{spread_value:.1f} points"
            prediction_results[game_key]['spread_value_edge'] = f"{edge:+.1f} points ({edge_direction})"
            print()
        
        print("=" * 80)
        return prediction_results
        
    except Exception as e:
        print(f"Error running spread value predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_xgboost_predictions(X, game_info, model_type="ML", prediction_results=None):
    """Run XGBoost predictions for today's games."""
    print(f"---------------XGBoost {model_type} Model Predictions---------------")
    
    # Connect to database to load models
    try:
        # Find the best model file
        import os
        import glob
        
        # Use absolute path for model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "Models")
        
        # Check if Models directory exists
        if not os.path.exists(models_dir):
            print(f"Models directory not found at: {models_dir}")
            print("Please train the models first using the XGBoost training scripts.")
            return None
        
        model_files = glob.glob(os.path.join(models_dir, f"XGBoost_*%_{model_type}-*.json"))
        if not model_files:
            print(f"No {model_type} model found in {models_dir}")
            print("Please train the model first using the XGBoost training scripts.")
            return None
        
        # Get the model with highest accuracy
        best_model_file = max(model_files, key=lambda x: float(x.split('_')[1].replace('%', '')))
        print(f"Loading model: {best_model_file}")
        
        # Load model
        model = xgb.Booster()
        model.load_model(best_model_file)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X)
        predictions_proba = model.predict(dmatrix)
        
        if model_type == "ML":
            predictions = (predictions_proba > 0.5).astype(int)
            prediction_labels = ['Away Win', 'Home Win']
        elif model_type == "Spread":
            predictions = (predictions_proba > 0.5).astype(int)
            prediction_labels = ['Away Covers', 'Home Covers']
        elif model_type == "UO":
            predictions = np.argmax(predictions_proba, axis=1)
            prediction_labels = ['Under', 'Over', 'Push']
        
        # Display results and collect for CSV export
        print(f"\n{model_type} Predictions for Today's Games:")
        print("-" * 60)
        
        for i, (game, pred) in enumerate(zip(game_info, predictions)):
            home = game['home']
            away = game['away']
            kickoff = game['kickoff']
            venue = game.get('venue', 'TBD')
            
            # Initialize game in prediction_results if not exists
            game_key = f"{away}:{home}"
            if game_key not in prediction_results:
                prediction_results[game_key] = {
                    'away_team': away,
                    'home_team': home,
                    'kickoff': kickoff,
                    'venue': venue
                }
                
                # Add betting lines if available
                betting_lines = game.get('betting_lines')
                if betting_lines:
                    prediction_results[game_key]['spread_line'] = betting_lines.get('spread', 'N/A')
                    prediction_results[game_key]['over_under_line'] = betting_lines.get('over_under', 'N/A')
                    prediction_results[game_key]['ml_home_line'] = betting_lines.get('ml_home', 'N/A')
                    prediction_results[game_key]['ml_away_line'] = betting_lines.get('ml_away', 'N/A')
                else:
                    prediction_results[game_key]['spread_line'] = 'N/A'
                    prediction_results[game_key]['over_under_line'] = 'N/A'
                    prediction_results[game_key]['ml_home_line'] = 'N/A'
                    prediction_results[game_key]['ml_away_line'] = 'N/A'
            
            if model_type == "ML":
                prob = predictions_proba[i]
                confidence = max(prob, 1-prob)
                # Convert probability to ML odds
                if prob > 0.5:
                    ml_odds = int(-100 * prob / (1 - prob))
                else:
                    ml_odds = int(100 * (1 - prob) / prob)
                
                print(f"{away} @ {home}")
                print(f"  Kickoff: {kickoff}")
                print(f"  Venue: {venue}")
                print(f"  Prediction: {prediction_labels[pred]} ({confidence:.1%} confidence)")
                print(f"  ML Value: {ml_odds:+d}")
                
                # Show betting lines if available
                betting_lines = game.get('betting_lines')
                if betting_lines:
                    print(f"  Betting Lines: Spread {betting_lines.get('spread', 'N/A')}, O/U {betting_lines.get('over_under', 'N/A')}")
                    print(f"  ML Lines: Home {betting_lines.get('ml_home', 'N/A')}, Away {betting_lines.get('ml_away', 'N/A')}")
                
                # Store results for CSV
                prediction_results[game_key]['ml_prediction'] = prediction_labels[pred]
                prediction_results[game_key]['ml_confidence'] = f"{confidence:.1%} confidence"
                prediction_results[game_key]['ml_value'] = f"{ml_odds:+d}"
                
                # Show recommended bet format
                if betting_lines:
                    spread_value = betting_lines.get('spread', 'N/A')
                    if prediction_labels[pred] == 'Home Win':
                        recommended_team = home
                    elif prediction_labels[pred] == 'Away Win':
                        recommended_team = away
                    else:
                        recommended_team = prediction_labels[pred]
                    
                    print(f"{away} vs {home}, recommended bet: {recommended_team}({spread_value}), confidence: {confidence:.1%}")
                print()
                
            elif model_type == "Spread":
                prob = predictions_proba[i]
                model_confidence = max(prob, 1-prob)
                
                # Get betting lines for spread calculation
                betting_lines = game.get('betting_lines')
                spread_line = betting_lines.get('spread', 0) if betting_lines else 0
                
                # Calculate detailed spread prediction using team stats
                # We need to get the team stats for this calculation
                # For now, we'll use the model prediction and betting lines
                
                # Determine which team covers based on model prediction
                if pred == 1:  # Home covers
                    covering_team = home
                    spread_direction = f"Home -{abs(spread_line)}" if spread_line != 0 else "Home favored"
                    recommended_bet = f"{home} -{abs(spread_line)}" if spread_line != 0 else f"{home}"
                else:  # Away covers
                    covering_team = away
                    spread_direction = f"Away +{abs(spread_line)}" if spread_line != 0 else "Away favored"
                    recommended_bet = f"{away} +{abs(spread_line)}" if spread_line != 0 else f"{away}"
                
                # Calculate spread value based on model confidence
                if spread_line != 0:
                    # Spread value is the confidence multiplied by the spread line
                    spread_value = abs(spread_line) * model_confidence
                else:
                    spread_value = 0
                
                # Calculate edge (how much better our prediction is vs the line)
                edge = model_confidence - 0.5  # Edge over 50% baseline
                edge_percentage = edge * 100
                
                print(f"{away} @ {home}")
                print(f"  Kickoff: {kickoff}")
                print(f"  Venue: {venue}")
                print(f"  Model Prediction: {prediction_labels[pred]} ({model_confidence:.1%} confidence)")
                print(f"  Recommended Bet: {recommended_bet}")
                print(f"  Spread Value: {spread_value:.1f} points")
                print(f"  Edge: {edge_percentage:+.1f}% over baseline")
                
                # Show betting lines if available
                if betting_lines:
                    print(f"  Betting Lines: Spread {betting_lines.get('spread', 'N/A')}, O/U {betting_lines.get('over_under', 'N/A')}")
                    print(f"  ML Lines: Home {betting_lines.get('ml_home', 'N/A')}, Away {betting_lines.get('ml_away', 'N/A')}")
                else:
                    print(f"  Betting Lines: Not available")
                
                # Store results for CSV
                prediction_results[game_key]['spread_prediction'] = prediction_labels[pred]
                prediction_results[game_key]['spread_confidence'] = f"{model_confidence:.1%}"
                prediction_results[game_key]['spread_value'] = f"{spread_value:.1f} points"
                prediction_results[game_key]['predicted_spread'] = spread_direction
                prediction_results[game_key]['recommended_bet'] = recommended_bet
                prediction_results[game_key]['edge'] = f"{edge_percentage:+.1f}%"
                
            else:  # UO
                prob = predictions_proba[i][pred]
                print(f"{away} @ {home}")
                print(f"  Kickoff: {kickoff}")
                print(f"  Venue: {venue}")
                print(f"  Prediction: {prediction_labels[pred]} ({prob:.1%} confidence)")
                
                # Show betting lines if available
                betting_lines = game.get('betting_lines')
                if betting_lines:
                    print(f"  Betting Lines: Spread {betting_lines.get('spread', 'N/A')}, O/U {betting_lines.get('over_under', 'N/A')}")
                    print(f"  ML Lines: Home {betting_lines.get('ml_home', 'N/A')}, Away {betting_lines.get('ml_away', 'N/A')}")
                
                # Store results for CSV
                prediction_results[game_key]['ou_prediction'] = prediction_labels[pred]
                prediction_results[game_key]['ou_confidence'] = f"{prob:.1%} confidence"
            print()
        
        print("-------------------------------------------------------")
        return prediction_results
        
    except Exception as e:
        print(f"Error running {model_type} predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_predictions_to_csv(prediction_results, filename=None):
    """Export prediction results to CSV file."""
    if not prediction_results:
        print("No prediction results to export.")
        return
    
    try:
        # Create DataFrame from prediction results
        rows = []
        for game_key, data in prediction_results.items():
            row = {
                'game_key': game_key,
                'away_team': data.get('away_team', ''),
                'home_team': data.get('home_team', ''),
                'kickoff': data.get('kickoff', ''),
                'venue': data.get('venue', ''),
                'spread_line': data.get('spread_line', 'N/A'),
                'over_under_line': data.get('over_under_line', 'N/A'),
                'ml_home_line': data.get('ml_home_line', 'N/A'),
                'ml_away_line': data.get('ml_away_line', 'N/A'),
                'ml_prediction': data.get('ml_prediction', ''),
                'ml_confidence': data.get('ml_confidence', ''),
                'ml_value': data.get('ml_value', ''),
                'spread_prediction': data.get('spread_prediction', ''),
                'spread_confidence': data.get('spread_confidence', ''),
                'spread_value': data.get('spread_value', ''),
                'predicted_spread': data.get('predicted_spread', ''),
                'recommended_bet': data.get('recommended_bet', ''),
                'edge': data.get('edge', ''),
                'spread_value_prediction': data.get('spread_value_prediction', ''),
                'spread_value_margin': data.get('spread_value_margin', ''),
                'spread_value_recommendation': data.get('spread_value_recommendation', ''),
                'spread_value_value': data.get('spread_value_value', ''),
                'spread_value_edge': data.get('spread_value_edge', ''),
                'ou_prediction': data.get('ou_prediction', ''),
                'ou_confidence': data.get('ou_confidence', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Generate filename if not provided
        if filename is None:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"cfb_predictions_{today}.csv"
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Predictions exported to: {filename}")
        
    except Exception as e:
        print(f"Error exporting predictions to CSV: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("College Football Betting Predictions")
    print("=" * 50)
    
    # Get today's games from ESPN
    try:
        games_df = cfb_schedule_today_espn()
        if games_df.empty:
            print("No games found for today.")
            return
        
        print(f"Found {len(games_df)} games today:")
        for _, game in games_df.iterrows():
            print(f"  {game['away']} @ {game['home']} at {game['kickoff_local']}")
        print()
        
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return
    
    # Connect to team stats database
    try:
        # Use absolute path or check if we're in the right directory
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        teams_db_path = os.path.join(current_dir, "Data", "TeamData.sqlite")
        odds_db_path = os.path.join(current_dir, "Data", "OddsData.sqlite")
        
        # Check if databases exist
        if not os.path.exists(teams_db_path):
            print(f"Team stats database not found at: {teams_db_path}")
            print("Please ensure you have run Get_Data.py to create the team stats database.")
            return
            
        if not os.path.exists(odds_db_path):
            print(f"Odds database not found at: {odds_db_path}")
            print("Please ensure you have run Get_Odds_Data.py to create the odds database.")
            return
        
        teams_con = sqlite3.connect(teams_db_path)
        odds_con = sqlite3.connect(odds_db_path)
        print(f"Connected to team stats database: {teams_db_path}")
        print(f"Connected to odds database: {odds_db_path}")
    except Exception as e:
        print(f"Error connecting to databases: {e}")
        return
    
    try:
        # Create prediction data
        X, games_df, game_info = createTodaysCFBGames(games_df, teams_con, odds_con)
        
        if X is None:
            print("No games could be processed for prediction.")
            return
        
        print(f"\nPrepared {len(game_info)} games for prediction")
        
        # Initialize dictionary to collect all prediction results
        prediction_results = {}
        
        # Run predictions based on arguments
        if args.ml:
            prediction_results = run_xgboost_predictions(X, game_info, "ML", prediction_results)
        
        if args.spread:
            prediction_results = run_xgboost_predictions(X, game_info, "Spread", prediction_results)
        
        if args.spread_value:
            prediction_results = run_xgboost_spread_value_predictions(X, game_info, prediction_results)
        
        if args.uo:
            prediction_results = run_xgboost_predictions(X, game_info, "UO", prediction_results)
        
        if args.all:
            prediction_results = run_xgboost_predictions(X, game_info, "ML", prediction_results)
            prediction_results = run_xgboost_predictions(X, game_info, "Spread", prediction_results)
            prediction_results = run_xgboost_spread_value_predictions(X, game_info, prediction_results)
            prediction_results = run_xgboost_predictions(X, game_info, "UO", prediction_results)
        
        if not any([args.ml, args.spread, args.spread_value, args.uo, args.all]):
            print("No prediction type specified. Use -ml, -spread, -spread-value, -uo, or -all")
            return
        
        # Export results to CSV if we have any predictions and export is requested
        if prediction_results and (args.export_csv or args.csv_filename):
            filename = args.csv_filename if args.csv_filename else None
            export_predictions_to_csv(prediction_results, filename)
    
    finally:
        teams_con.close()
        odds_con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='College Football Betting Predictions')
    parser.add_argument('-ml', action='store_true', help='Run Moneyline predictions')
    parser.add_argument('-spread', action='store_true', help='Run Spread predictions')
    parser.add_argument('-spread-value', action='store_true', help='Run Spread Value regression predictions')
    parser.add_argument('-uo', action='store_true', help='Run Over/Under predictions')
    parser.add_argument('-all', action='store_true', help='Run all prediction types')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV file')
    parser.add_argument('--csv-filename', type=str, help='Custom CSV filename for export')
    args = parser.parse_args()
    main()
