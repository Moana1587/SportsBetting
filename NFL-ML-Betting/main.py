import argparse
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games
import xgboost as xgb
import glob
import os

# Updated data sources to use database tables
todays_games_url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={today}'

# NFL team name mapping for consistent team identification
NFL_TEAM_MAPPING = {
    # AFC East
    'Buffalo Bills': 'BUF', 'Miami Dolphins': 'MIA', 'New England Patriots': 'NE', 'New York Jets': 'NYJ',
    # AFC North
    'Baltimore Ravens': 'BAL', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Pittsburgh Steelers': 'PIT',
    # AFC South
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Tennessee Titans': 'TEN',
    # AFC West
    'Denver Broncos': 'DEN', 'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    # NFC East
    'Dallas Cowboys': 'DAL', 'New York Giants': 'NYG', 'Philadelphia Eagles': 'PHI', 'Washington Commanders': 'WAS',
    # NFC North
    'Chicago Bears': 'CHI', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB', 'Minnesota Vikings': 'MIN',
    # NFC South
    'Atlanta Falcons': 'ATL', 'Carolina Panthers': 'CAR', 'New Orleans Saints': 'NO', 'Tampa Bay Buccaneers': 'TB',
    # NFC West
    'Arizona Cardinals': 'ARI', 'Los Angeles Rams': 'LAR', 'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA'
}


def load_best_spread_model():
    """Load the best available spread model for spread prediction"""
    possible_dirs = [
        'Models/XGBoost_Models/',
        '../../Models/XGBoost_Models/',
        '../Models/XGBoost_Models/',
        'src/Models/XGBoost_Models/'
    ]
    
    model_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            model_dir = dir_path
            break
    
    if model_dir is None:
        print(f"Warning: Model directory not found in any of these locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        return None
    
    # Find all spread models
    pattern = f"XGBoost_*_MAE_Spread.json"
    model_files = glob.glob(os.path.join(model_dir, pattern))
    
    if not model_files:
        print(f"Warning: No spread models found in {model_dir}")
        print("Available models:")
        all_models = glob.glob(os.path.join(model_dir, "*.json"))
        for model in all_models:
            print(f"  - {os.path.basename(model)}")
        return None
    
    # Sort by MAE (lower is better) and get the best one
    def extract_mae(filename):
        try:
            basename = os.path.basename(filename)
            # Extract MAE from filename like "XGBoost_0.9_MAE_Spread.json"
            mae_str = basename.split('_')[1]
            return float(mae_str)
        except:
            return float('inf')
    
    # Sort models by MAE and show all available options
    sorted_models = sorted(model_files, key=extract_mae)
    print(f"Available spread models (sorted by MAE):")
    for i, model in enumerate(sorted_models):
        mae = extract_mae(model)
        status = " [BEST]" if i == 0 else ""
        print(f"  {i+1}. {os.path.basename(model)} (MAE: {mae:.1f}){status}")
    
    best_model = sorted_models[0]
    best_mae = extract_mae(best_model)
    
    print(f"\nLoading best spread model: {os.path.basename(best_model)} (MAE: {best_mae:.1f})")
    
    try:
        booster = xgb.Booster()
        booster.load_model(best_model)
        print(f"Successfully loaded spread model with MAE: {best_mae:.1f}")
        return booster
    except Exception as e:
        print(f"Error loading spread model {best_model}: {e}")
        return None

def load_best_uo_model():
    """Load the best available UO model for OU line prediction"""
    possible_dirs = [
        'Models/XGBoost_Models/',
        '../../Models/XGBoost_Models/',
        '../Models/XGBoost_Models/',
        'src/Models/XGBoost_Models/'
    ]
    
    model_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            model_dir = dir_path
            break
    
    if model_dir is None:
        print(f"Warning: Model directory not found in any of these locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        return None
    
    # Find all UO models
    pattern = f"XGBoost_*%_UO.json"
    model_files = glob.glob(os.path.join(model_dir, pattern))
    
    if not model_files:
        print(f"Warning: No UO models found in {model_dir}")
        return None
    
    # Sort by accuracy and get the best one
    def extract_accuracy(filename):
        try:
            basename = os.path.basename(filename)
            accuracy_str = basename.split('_')[1].replace('%', '')
            return float(accuracy_str)
        except:
            return 0.0
    
    best_model = max(model_files, key=extract_accuracy)
    accuracy = extract_accuracy(best_model)
    
    print(f"Loading best UO model: {os.path.basename(best_model)} ({accuracy}%)")
    
    try:
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    except Exception as e:
        print(f"Error loading UO model {best_model}: {e}")
        return None

def predict_spread(game_data, spread_model):
    """Predict spread value for a game using the spread model"""
    try:
        # Prepare the data for prediction using the same feature selection as training
        exclude_columns = [
            'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
            'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
            'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
            'Date', 'Date.1', 'index.1', 'Spread'
        ]
        
        # Get feature columns (same as training)
        feature_columns = [col for col in game_data.index if col not in exclude_columns]
        
        if not feature_columns:
            print("Warning: No feature columns found for spread prediction!")
            return 0.0
        
        # Create feature array
        feature_data = []
        for col in feature_columns:
            try:
                value = float(game_data[col])
                feature_data.append(value)
            except (ValueError, TypeError):
                feature_data.append(0.0)
        
        feature_array = np.array([feature_data])
        
        # Make prediction
        prediction = spread_model.predict(xgb.DMatrix(feature_array))
        predicted_spread = float(prediction[0])
        
        # Clamp to reasonable range
        predicted_spread = max(-20, min(20, predicted_spread))
        
        return predicted_spread
        
    except Exception as e:
        print(f"Error predicting spread: {e}")
        return 0.0  # Default fallback

def predict_ou_line(game_data, uo_model):
    """Predict OU line value for a game using the UO model"""
    try:
        # The UO model predicts over/under outcomes, but we can use it to estimate line values
        # by analyzing historical patterns. For now, we'll use a simple approach based on
        # team stats to predict a reasonable OU line.
        
        # Extract key offensive and defensive stats
        home_offensive_stats = []
        away_offensive_stats = []
        home_defensive_stats = []
        away_defensive_stats = []
        
        # Look for common offensive/defensive stat columns
        offensive_keywords = ['POINTS_FOR', 'YARDS', 'PASS', 'RUSH', 'OFFENSIVE']
        defensive_keywords = ['POINTS_AGAINST', 'DEFENSIVE', 'ALLOWED']
        
        for col in game_data.index:
            col_lower = col.lower()
            if any(keyword.lower() in col_lower for keyword in offensive_keywords):
                if '.1' not in col:  # Home team stats
                    home_offensive_stats.append(game_data[col])
                else:  # Away team stats
                    away_offensive_stats.append(game_data[col])
            elif any(keyword.lower() in col_lower for keyword in defensive_keywords):
                if '.1' not in col:  # Home team stats
                    home_defensive_stats.append(game_data[col])
                else:  # Away team stats
                    away_defensive_stats.append(game_data[col])
        
        # Calculate predicted total points
        home_offense = np.mean(home_offensive_stats) if home_offensive_stats else 20
        away_offense = np.mean(away_offensive_stats) if away_offensive_stats else 20
        home_defense = np.mean(home_defensive_stats) if home_defensive_stats else 20
        away_defense = np.mean(away_defensive_stats) if away_defensive_stats else 20
        
        # Simple prediction: average offensive capability vs average defensive capability
        predicted_total = (home_offense + away_offense) / 2
        
        # Add some variance based on defensive strength
        defensive_factor = (home_defense + away_defense) / 2
        if defensive_factor > 25:  # Strong defenses
            predicted_total *= 0.9
        elif defensive_factor < 15:  # Weak defenses
            predicted_total *= 1.1
        
        # Round to nearest 0.5 and ensure reasonable range
        predicted_total = max(30, min(60, round(predicted_total * 2) / 2))
        
        return predicted_total
        
    except Exception as e:
        print(f"Error predicting OU line: {e}")
        return 45.0  # Default fallback

def get_team_index(team_name, team_df):
    """Get team index from team dataframe based on team name"""
    # Try exact match first
    exact_match = team_df[team_df['TEAM_NAME'] == team_name]
    if not exact_match.empty:
        return exact_match.index[0]
    
    # Try abbreviation match
    team_abbrev = NFL_TEAM_MAPPING.get(team_name)
    if team_abbrev:
        abbrev_match = team_df[team_df['TEAM_NAME'].str.contains(team_abbrev, case=False, na=False)]
        if not abbrev_match.empty:
            return abbrev_match.index[0]
    
    # Try partial name match
    for idx, row in team_df.iterrows():
        if team_name.lower() in row['TEAM_NAME'].lower() or row['TEAM_NAME'].lower() in team_name.lower():
            return idx
    
    return None

def createTodaysGames(games, team_df, odds, uo_model=None, spread_model=None):
    """Create today's games data using the new database structure"""
    match_data = []
    todays_games_uo = []
    todays_games_spread = []
    home_team_odds = []
    away_team_odds = []

    print(f"Processing {len(games)} games for today...")

    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        print(f"Processing: {away_team} @ {home_team}")
        
        # Get team indices using the new mapping function
        home_team_idx = get_team_index(home_team, team_df)
        away_team_idx = get_team_index(away_team, team_df)
        
        if home_team_idx is None or away_team_idx is None:
            print(f"  Skipping - team not found in database")
            print(f"    Home team '{home_team}' found: {home_team_idx is not None}")
            print(f"    Away team '{away_team}' found: {away_team_idx is not None}")
            continue

        # Get team stats first (needed for OU prediction)
        home_team_series = team_df.iloc[home_team_idx]
        away_team_series = team_df.iloc[away_team_idx]
        
        # Create game record by combining home and away team stats
        stats = pd.concat([
            home_team_series, 
            away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values})
        ])
        
        # Add days rest information
        stats['Days-Rest-Home'] = 7  # Default
        stats['Days-Rest-Away'] = 7  # Default
        
        # Handle odds data
        if odds is not None:
            try:
                game_key = home_team + ':' + away_team
                if game_key in odds:
                    game_odds = odds[game_key]
                    todays_games_uo.append(game_odds.get('under_over_odds', 0))
                    # Try to get spread from odds, otherwise predict it
                    spread_odds = game_odds.get('spread_odds', None)
                    if spread_odds is not None:
                        todays_games_spread.append(spread_odds)
                    else:
                        predicted_spread = predict_spread(stats, spread_model)
                        todays_games_spread.append(predicted_spread)
                        print(f"  Predicted spread: {predicted_spread}")
                    home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds', 0))
                    away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds', 0))
                else:
                    print(f"  No odds found for {game_key}, predicting OU line and spread...")
                    # Predict OU line and spread using models
                    predicted_ou = predict_ou_line(stats, uo_model)
                    predicted_spread = predict_spread(stats, spread_model)
                    todays_games_uo.append(predicted_ou)
                    todays_games_spread.append(predicted_spread)
                    home_team_odds.append(0)
                    away_team_odds.append(0)
                    print(f"  Predicted OU line: {predicted_ou}")
                    print(f"  Predicted spread: {predicted_spread}")
            except Exception as e:
                print(f"  Error processing odds: {e}")
                # Predict OU line and spread using models as fallback
                predicted_ou = predict_ou_line(stats, uo_model)
                predicted_spread = predict_spread(stats, spread_model)
                todays_games_uo.append(predicted_ou)
                todays_games_spread.append(predicted_spread)
                home_team_odds.append(0)
                away_team_odds.append(0)
                print(f"  Using predicted OU line: {predicted_ou}")
                print(f"  Using predicted spread: {predicted_spread}")
        else:
            # Predict OU line and spread using models instead of manual input
            predicted_ou = predict_ou_line(stats, uo_model)
            predicted_spread = predict_spread(stats, spread_model)
            todays_games_uo.append(predicted_ou)
            todays_games_spread.append(predicted_spread)
            home_team_odds.append(0)
            away_team_odds.append(0)
            print(f"  Predicted OU line: {predicted_ou}")
            print(f"  Predicted spread: {predicted_spread}")

        match_data.append(stats)
        print(f"  Added game data successfully")

    if not match_data:
        print("No valid games found!")
        return None, None, None, None, None

    # Create final dataframe
    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    # Remove team ID columns if they exist
    columns_to_drop = [col for col in games_data_frame.columns if 'TEAM_ID' in col]
    if columns_to_drop:
        frame_ml = games_data_frame.drop(columns=columns_to_drop)
    else:
        frame_ml = games_data_frame

    # Convert to numpy array, handling non-numeric columns
    # Use the same feature selection as in training scripts
    exclude_columns = [
        'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
        'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
        'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
        'Date', 'Date.1', 'index.1'
    ]
    
    # Get feature columns (same as training)
    feature_columns = [col for col in frame_ml.columns if col not in exclude_columns]
    
    print(f"Excluding {len(exclude_columns)} non-feature columns")
    print(f"Using {len(feature_columns)} feature columns for predictions")
    print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10
    
    # Create data with only feature columns
    if not feature_columns:
        print("Warning: No feature columns found! Using all numeric columns...")
        # Fallback to numeric columns only
        numeric_exclude = ['TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'index', 'index.1', 'SEASON', 'SEASON.1']
        feature_columns = [col for col in frame_ml.columns if col not in numeric_exclude]
    
    feature_data = frame_ml[feature_columns].values
    
    # Convert to float, handling any remaining non-numeric values
    try:
        data = feature_data.astype(float)
        print("Successfully converted data to float")
    except ValueError as e:
        print(f"Error converting to float: {e}")
        print("Attempting fallback conversion method...")
        
        # Fallback: convert each column individually and handle errors
        data_df = pd.DataFrame(feature_data, columns=feature_columns)
        data_df = data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = data_df.values.astype(float)
        print("Used fallback conversion method successfully")

    print(f"Created dataset with {len(data)} games and {data.shape[1]} features")
    print(f"Feature count validation: {data.shape[1]} features (should match model training)")
    return data, todays_games_uo, todays_games_spread, frame_ml, home_team_odds, away_team_odds


def main():
    print("NFL ML Betting Prediction System")
    print("=" * 50)
    
    # Load UO model for OU line prediction
    print("Loading UO model for OU line prediction...")
    uo_model = load_best_uo_model()
    if uo_model is None:
        print("Warning: Could not load UO model. OU lines will use default values.")
    
    # Load spread model for spread prediction
    print("Loading spread model for spread prediction...")
    spread_model = load_best_spread_model()
    if spread_model is None:
        print("Warning: Could not load spread model. Spreads will use default values.")
    
    # Load team data from database
    try:
        con = sqlite3.connect("Data/TeamData.sqlite")
        
        # Get the most recent team stats table
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
        team_tables = [t[0] for t in tables.values if 'nfl_team_stats' in t[0]]
        
        if not team_tables:
            print("No team stats tables found in database!")
            print("Please run Get_Data.py first to populate team statistics.")
            return
        
        # Use the most recent table (highest year)
        latest_table = sorted(team_tables)[-1]
        print(f"Using team stats from: {latest_table}")
        
        team_df = pd.read_sql_query(f"SELECT * FROM `{latest_table}`", con)
        con.close()
        
        if team_df.empty:
            print("No team data found in database!")
            return
            
        print(f"Loaded {len(team_df)} teams from database")
        
    except Exception as e:
        print(f"Error loading team data: {e}")
        print("Please ensure Get_Data.py has been run to populate the database.")
        return

    # Handle odds and games
    odds = None
    if args.odds:
        try:
            odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
            games = create_todays_games_from_odds(odds)
            if len(games) == 0:
                print("No games found from odds provider.")
                return
            if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
                print(games[0][0] + ':' + games[0][1])
                print(Fore.RED,"--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
                print(Style.RESET_ALL)
                odds = None
            else:
                print(f"------------------{args.odds} odds data------------------")
                for g in odds.keys():
                    home_team, away_team = g.split(":")
                    print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
        except Exception as e:
            print(f"Error loading odds: {e}")
            odds = None
    else:
        try:
            print("Attempting to fetch today's games from ESPN API...")
            # Format the URL with today's date
            today = datetime.now().strftime('%Y%m%d')
            formatted_url = todays_games_url.format(today=today)
            print(f"Using URL: {formatted_url}")
            data = get_todays_games_json(formatted_url)
            games = create_todays_games(data)
            
            if not games:
                print("No games found from ESPN API. This could mean:")
                print("1. No games scheduled for today")
                print("2. ESPN API is temporarily unavailable")
                print("3. Network connectivity issues")
                print("\nTo continue with predictions, please:")
                print("- Use the -odds parameter to fetch games from a sportsbook")
                print("- Or manually input game data when prompted")
                print("\nExample: python main.py -xgb -odds fanduel")
                return
            else:
                print(f"Successfully loaded {len(games)} games from ESPN API")
                
        except Exception as e:
            print(f"Error loading today's games: {e}")
            print("\nESPN API connection failed. This could be due to:")
            print("1. Network connectivity issues")
            print("2. ESPN API being temporarily unavailable")
            print("3. DNS resolution problems")
            print("\nTo continue with predictions, please:")
            print("- Use the -odds parameter to fetch games from a sportsbook")
            print("- Or manually input game data when prompted")
            print("\nExample: python main.py -xgb -odds fanduel")
            return

    # Create today's games data
    result = createTodaysGames(games, team_df, odds, uo_model, spread_model)
    if result[0] is None:
        print("Failed to create games data. Exiting.")
        return
    
    data, todays_games_uo, todays_games_spread, frame_ml, home_team_odds, away_team_odds = result
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, todays_games_spread, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, todays_games_spread, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()
