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
        'UL Monroe': 'UL Monroe'
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
        ' Warhawks', ' Warriors', ' Wolf Pack', ' Zips'
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
                confidence = max(prob, 1-prob)
                # Convert probability to spread confidence
                spread_confidence = confidence * 100
                
                print(f"{away} @ {home}")
                print(f"  Kickoff: {kickoff}")
                print(f"  Venue: {venue}")
                print(f"  Prediction: {prediction_labels[pred]} ({confidence:.1%} confidence)")
                print(f"  Spread Value: {spread_confidence:.1f}% confidence")
                
                # Show betting lines if available
                betting_lines = game.get('betting_lines')
                if betting_lines:
                    print(f"  Betting Lines: Spread {betting_lines.get('spread', 'N/A')}, O/U {betting_lines.get('over_under', 'N/A')}")
                    print(f"  ML Lines: Home {betting_lines.get('ml_home', 'N/A')}, Away {betting_lines.get('ml_away', 'N/A')}")
                
                # Store results for CSV
                prediction_results[game_key]['spread_prediction'] = prediction_labels[pred]
                prediction_results[game_key]['spread_confidence'] = f"{confidence:.1%} confidence"
                prediction_results[game_key]['spread_value'] = f"{spread_confidence:.1f}% confidence"
                
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
        
        if args.uo:
            prediction_results = run_xgboost_predictions(X, game_info, "UO", prediction_results)
        
        if args.all:
            prediction_results = run_xgboost_predictions(X, game_info, "ML", prediction_results)
            prediction_results = run_xgboost_predictions(X, game_info, "Spread", prediction_results)
            prediction_results = run_xgboost_predictions(X, game_info, "UO", prediction_results)
        
        if not any([args.ml, args.spread, args.uo, args.all]):
            print("No prediction type specified. Use -ml, -spread, -uo, or -all")
            return
        
    
    finally:
        teams_con.close()
        odds_con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='College Football Betting Predictions')
    parser.add_argument('-ml', action='store_true', help='Run Moneyline predictions')
    parser.add_argument('-spread', action='store_true', help='Run Spread predictions')
    parser.add_argument('-uo', action='store_true', help='Run Over/Under predictions')
    parser.add_argument('-all', action='store_true', help='Run all prediction types')
    args = parser.parse_args()
    main()
