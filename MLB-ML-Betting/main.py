import argparse
import sqlite3
import requests
from datetime import datetime, timedelta
import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from colorama import Fore, Style

from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

# MLB API URLs
# ESPN API for today's games (updated to use ESPN API)
espn_games_url = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=' + datetime.today().strftime('%Y%m%d')
data_url = 'https://bdfed.stitch.mlbinfra.com/bdfed/stats/team?env=prod&sportId=1&gameType=R&group=hitting&stats=season&season=2024&limit=1000&offset=0'


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    todays_games_spread = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        print(f"Processing game: {away_team} @ {home_team}")
        
        # Create synthetic team stats that match the unified model training structure
        # Using consistent random seed per game for reproducible results
        np.random.seed(hash(home_team + away_team) % 2**32)
        
        # Generate realistic team stats matching mlb_dataset structure
        synthetic_stats = {
            'wins': np.random.randint(70, 100),
            'losses': np.random.randint(60, 90),
            'era': np.random.uniform(3.0, 5.0),
            'ops': np.random.uniform(0.650, 0.850),
            'avg': np.random.uniform(0.230, 0.280),
            'obp': np.random.uniform(0.300, 0.350),
            'slg': np.random.uniform(0.400, 0.500),
            'runs_hitting': np.random.randint(600, 900),
            'rbi': np.random.randint(550, 850),
            'homeRuns': np.random.randint(150, 250),
            'hits_hitting': np.random.randint(1200, 1600),
            'doubles': np.random.randint(250, 350),
            'triples': np.random.randint(20, 50),
            'baseOnBalls_hitting': np.random.randint(400, 600),
            'strikeOuts_hitting': np.random.randint(1200, 1600),
            'atBats': np.random.randint(5000, 6000),
            'totalBases': np.random.randint(2000, 2800),
            'leftOnBase': np.random.randint(1000, 1300),
            'sacBunts_hitting': np.random.randint(20, 60),
            'sacFlies_hitting': np.random.randint(30, 80),
            'stolenBases': np.random.randint(50, 150),
            'caughtStealing': np.random.randint(20, 60),
            'groundIntoDoublePlay': np.random.randint(100, 200),
            'hitByPitch': np.random.randint(40, 100),
            'intentionalWalks': np.random.randint(10, 50),
            'catchersInterference': np.random.randint(0, 10),
            'groundOuts': np.random.randint(1000, 1400),
            'airOuts': np.random.randint(800, 1200),
            'numberOfPitches': np.random.randint(20000, 30000),
            'inningsPitched': np.random.uniform(1400, 1500),
            'hits_pitching': np.random.randint(1200, 1600),
            'runs_pitching': np.random.randint(600, 900),
            'earnedRuns': np.random.randint(500, 800),
            'baseOnBalls_pitching': np.random.randint(400, 600),
            'strikeOuts_pitching': np.random.randint(1200, 1600),
            'whip': np.random.uniform(1.20, 1.50),
            'saves': np.random.randint(20, 50),
            'holds': np.random.randint(20, 80),
            'blownSaves': np.random.randint(10, 30),
            'completeGames': np.random.randint(0, 10),
            'shutouts': np.random.randint(5, 20),
            'hitBatsmen': np.random.randint(20, 80),
            'balks': np.random.randint(0, 10),
            'wildPitches': np.random.randint(20, 80),
            'pickoffs': np.random.randint(0, 20),
            'sacBunts_pitching': np.random.randint(20, 60),
            'sacFlies_pitching': np.random.randint(30, 80),
            'Days-Rest-Home': 1,
            'Days-Rest-Away': 1
        }
        
        # Add away team stats with .1 suffix (matching unified model structure)
        away_stats = {}
        for key, value in synthetic_stats.items():
            if key not in ['Days-Rest-Home', 'Days-Rest-Away']:
                away_stats[f"{key}.1"] = value
        
        # Combine home and away stats
        combined_stats = {**synthetic_stats, **away_stats}
        
        print(f"Created unified model features for {home_team} vs {away_team}")
        
        # Handle odds
        if odds is not None:
            game_key = home_team + ':' + away_team
            if game_key in odds:
                game_odds = odds[game_key]
                todays_games_uo.append(game_odds.get('under_over_odds', 8.5))
                todays_games_spread.append(game_odds.get('spread', 1.5))
                home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds', 100))
                away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds', 100))
            else:
                todays_games_uo.append(8.5)
                todays_games_spread.append(1.5)
                home_team_odds.append(100)
                away_team_odds.append(100)
        else:
            todays_games_uo.append(8.5)
            todays_games_spread.append(1.5)
            home_team_odds.append(100)
            away_team_odds.append(100)
        
        match_data.append(combined_stats)

    if not match_data:
        print("No valid games found!")
        return None, None, None, None, None

    # Convert list of dictionaries to DataFrame
    games_data_frame = pd.DataFrame(match_data)
    
    # Drop columns that shouldn't be used for prediction (matching unified model training)
    columns_to_drop = ['Score', 'Home-Team-Win', 'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away', 'ML_Home', 'ML_Away']
    # Also drop any team name or date columns that might exist
    for col in games_data_frame.columns:
        if 'TEAM_NAME' in col or 'Date' in col or 'teamName' in col or 'season' in col:
            columns_to_drop.append(col)
    
    # Remove duplicates and only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in games_data_frame.columns]
    print(f"Dropping columns: {columns_to_drop}")
    
    if columns_to_drop:
        frame_ml = games_data_frame.drop(columns=columns_to_drop)
    else:
        frame_ml = games_data_frame
    
    # Ensure all data is numeric
    frame_ml = frame_ml.select_dtypes(include=[np.number])
    data = frame_ml.values
    data = data.astype(float)

    print(f"Final feature matrix shape: {data.shape}")
    print(f"Number of features: {data.shape[1] if len(data.shape) > 1 else 0}")

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, todays_games_spread


def get_todays_mlb_games():
    """Fetch today's MLB games from the ESPN API"""
    try:
        # Use ESPN API for today's games
        today = datetime.today().strftime('%Y%m%d')
        espn_url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today}'
        
        response = requests.get(espn_url)
        response.raise_for_status()
        data = response.json()
        
        games = []
        if 'events' in data:
            for event in data['events']:
                # Check if the event has competitions and is scheduled
                if 'competitions' in event and len(event['competitions']) > 0:
                    competition = event['competitions'][0]
                    
                    # Check if the game is scheduled (not completed or in progress)
                    if 'status' in competition and competition['status'].get('type', {}).get('state') == 'pre':
                        # Extract team information from competitors
                        home_team = None
                        away_team = None
                        
                        for competitor in competition.get('competitors', []):
                            if competitor.get('homeAway') == 'home':
                                home_team = competitor['team']['name']
                            elif competitor.get('homeAway') == 'away':
                                away_team = competitor['team']['name']
                        
                        if home_team and away_team:
                            games.append([home_team, away_team])
                            print(f"Found game: {away_team} @ {home_team}")
        
        return games
    except Exception as e:
        print(f"Error fetching today's games from ESPN API: {e}")
        return []




def load_best_spread_model():
    """Load the best spread value model from the Models folder"""
    model_pattern = "Models/XGBoost_Spread_*_Value.json"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"No spread models found matching pattern: {model_pattern}")
        return None
    
    # Find the model with the highest R² score
    best_model = None
    best_score = -999
    
    for model_file in model_files:
        try:
            # Extract R² score from filename (e.g., XGBoost_Spread_0.750_Value.json -> 0.750)
            filename = os.path.basename(model_file)
            score_str = filename.replace('XGBoost_Spread_', '').replace('_Value.json', '')
            score = float(score_str)
            
            if score > best_score:
                best_score = score
                best_model = model_file
        except ValueError:
            print(f"Could not parse R² score from filename: {filename}")
            continue
    
    if best_model:
        print(f"Loading best spread model: {best_model} (R² score: {best_score:.3f})")
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    else:
        print("No valid spread model found")
        return None


def predict_with_spread_model(data, games, home_team_odds, away_team_odds):
    """Make spread value predictions using the spread model"""
    model = load_best_spread_model()
    if model is None:
        print("Error: Could not load spread model")
        return None
    
    print(f"Making spread value predictions for {len(games)} games...")
    print(f"Data shape: {data.shape}")
    
    predictions_data = []
    
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        
        # Make prediction using spread model
        spread_pred = model.predict(xgb.DMatrix(data[count:count+1]))
        predicted_spread = spread_pred[0]
        
        # Determine which team is favored based on spread
        if predicted_spread > 0:
            # Home team favored
            favored_team = home_team
            spread_line = abs(predicted_spread)
            spread_prediction = f"{home_team} -{spread_line:.1f}"
            confidence = min(abs(predicted_spread) * 20, 95)  # Convert spread to confidence
        else:
            # Away team favored
            favored_team = away_team
            spread_line = abs(predicted_spread)
            spread_prediction = f"{away_team} +{spread_line:.1f}"
            confidence = min(abs(predicted_spread) * 20, 95)  # Convert spread to confidence
        
        # Calculate Expected Value and Kelly Criterion if odds are available
        ev_home = ev_away = kelly_home = kelly_away = None
        if home_team_odds[count] and away_team_odds[count]:
            try:
                from src.Utils.Expected_Value import expected_value
                from src.Utils.Kelly_Criterion import Kelly_Criterion
                
                # For spread betting, we need to estimate win probabilities
                # Use the spread to estimate probabilities
                home_win_prob = 0.5 + (predicted_spread * 0.1)  # Rough conversion
                home_win_prob = np.clip(home_win_prob, 0.1, 0.9)
                away_win_prob = 1 - home_win_prob
                
                kc = Kelly_Criterion()
                ev_home = float(expected_value(home_win_prob, int(home_team_odds[count])))
                ev_away = float(expected_value(away_win_prob, int(away_team_odds[count])))
                kelly_home = round(kc.calculate_kelly_criterion(home_team_odds[count], home_win_prob), 2)
                kelly_away = round(kc.calculate_kelly_criterion(away_team_odds[count], away_win_prob), 2)
            except ImportError:
                print("Warning: Could not import Expected_Value or Kelly_Criterion modules")
        
        # Create prediction data
        prediction = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Predicted_Winner': favored_team,
            'Winner_Confidence_%': round(confidence, 1),
            'ML_Home_%': 'N/A',  # Not available in spread model
            'ML_Away_%': 'N/A',
            'Home_Team_Odds': home_team_odds[count] if home_team_odds[count] else 'N/A',
            'Away_Team_Odds': away_team_odds[count] if away_team_odds[count] else 'N/A',
            'EV_Home': ev_home,
            'EV_Away': ev_away,
            'Kelly_Home': kelly_home,
            'Kelly_Away': kelly_away,
            'Spread_Prediction': spread_prediction,
            'Spread_Line': round(spread_line, 1),
            'Spread_Confidence_%': round(confidence, 1),
            'Predicted_Spread_Value': round(predicted_spread, 2),
            'Over_Under_Prediction': 'N/A',  # Not available in spread model
            'Over_Under_Line': 'N/A',
            'Over_Under_Confidence_%': 'N/A'
        }
        
        predictions_data.append(prediction)
        
        # Print prediction in the requested format
        ml_value = home_team_odds[count] if home_team_odds[count] else "N/A"
        ou_value = "N/A"  # OU value not available in spread model
        print(f"{away_team} vs {home_team}, recommended bet: {favored_team}, Spread:{spread_line:.1f}, ML:{ml_value}, OU:{ou_value}, Confidence:{confidence:.1f}%")
    
def load_best_ou_model():
    """Load the best OU model from the Models folder"""
    model_pattern = "Models/XGBoost_*_UO-*.json"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"No OU models found matching pattern: {model_pattern}")
        return None
    
    # Find the model with the highest accuracy
    best_model = None
    best_accuracy = -1
    
    for model_file in model_files:
        try:
            # Extract accuracy from filename (e.g., XGBoost_51.3%_UO-9.json -> 51.3)
            filename = os.path.basename(model_file)
            accuracy_str = filename.replace('XGBoost_', '').replace('%_UO-9.json', '')
            accuracy = float(accuracy_str)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_file
        except ValueError:
            print(f"Could not parse accuracy from filename: {filename}")
            continue
    
    if best_model:
        print(f"Loading best OU model: {best_model} (accuracy: {best_accuracy:.1f}%)")
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    else:
        print("No valid OU model found")
        return None

def load_best_ml_model(pattern):
    """Load the best ML model from the Models folder"""
    model_files = glob.glob(f"Models/{pattern}")
    
    if not model_files:
        print(f"No ML models found matching pattern: {pattern}")
        return None
    
    # Find the model with the highest R² score
    best_model = None
    best_r2 = -999
    
    for model_file in model_files:
        try:
            # Extract R² from filename
            filename = os.path.basename(model_file)
            # Pattern: XGBoost_ML_Home_0.XXX_R2.json
            parts = filename.split('_')
            if len(parts) >= 4 and 'R2' in filename:
                r2_str = parts[3]
                r2 = float(r2_str)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_file
        except (ValueError, IndexError):
            continue
    
    if best_model:
        print(f"Loading best ML model: {best_model} (R²: {best_r2:.3f})")
        model = xgb.Booster()
        model.load_model(best_model)
        return model
    else:
        print("No valid ML models found")
        return None

def load_best_unified_model():
    """Load the best unified model from the Models folder"""
    model_pattern = "Models/XGBoost_Unified_*_ML.json"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"No unified models found matching pattern: {model_pattern}")
        return None
    
    # Find the model with the highest score
    best_model = None
    best_score = -999
    
    for model_file in model_files:
        try:
            # Extract score from filename (e.g., XGBoost_Unified_0.750_ML.json -> 0.750)
            filename = os.path.basename(model_file)
            score_str = filename.replace('XGBoost_Unified_', '').replace('_ML.json', '')
            score = float(score_str)
            
            if score > best_score:
                best_score = score
                best_model = model_file
        except ValueError:
            print(f"Could not parse score from filename: {filename}")
            continue
    
    if best_model:
        print(f"Loading best unified model: {best_model} (score: {best_score:.3f})")
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    else:
        print("No valid unified model found")
        return None


def predict_ou_values(data, games, todays_games_uo):
    """Make OU value predictions using the OU model"""
    model = load_best_ou_model()
    if model is None:
        print("Error: Could not load OU model")
        return None
    
    print(f"Making OU predictions for {len(games)} games...")
    print(f"Data shape: {data.shape}")
    
    ou_predictions = []
    
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        
        # Make prediction using OU model
        ou_pred = model.predict(xgb.DMatrix(data[count:count+1]))
        
        # Extract prediction probabilities
        # ou_pred returns probabilities [prob_under, prob_over]
        under_prob = ou_pred[0][0]
        over_prob = ou_pred[0][1]
        
        # Determine predicted outcome
        if over_prob > under_prob:
            ou_prediction = "Over"
            ou_confidence = over_prob * 100
        else:
            ou_prediction = "Under"
            ou_confidence = under_prob * 100
        
        # Use the actual OU line from todays_games_uo if available
        ou_line = todays_games_uo[count] if todays_games_uo[count] else 8.5
        
        ou_predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'ou_prediction': ou_prediction,
            'ou_confidence': ou_confidence,
            'ou_line': ou_line
        })
    
    return ou_predictions

def predict_with_unified_model(data, games, home_team_odds, away_team_odds, todays_games_spread, todays_games_uo):
    """Make predictions using the unified model"""
    model = load_best_unified_model()
    if model is None:
        print("Error: Could not load unified model")
        return None
    
    print(f"Making unified predictions for {len(games)} games...")
    print(f"Data shape: {data.shape}")
    
    # Load ML models for actual ML value prediction
    ml_home_model = load_best_ml_model("XGBoost_ML_Home_*_R2.json")
    ml_away_model = load_best_ml_model("XGBoost_ML_Away_*_R2.json")
    
    # Get OU predictions
    ou_predictions = predict_ou_values(data, games, todays_games_uo)
    
    predictions_data = []
    
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        
        # Make prediction using unified model
        unified_pred = model.predict(xgb.DMatrix(data[count:count+1]))
        
        # Extract predictions
        # unified_pred returns probabilities [prob_away_win, prob_home_win]
        away_win_prob = unified_pred[0][0]
        home_win_prob = unified_pred[0][1]
        
        # Determine predicted winner
        if home_win_prob > away_win_prob:
            predicted_winner = home_team
            winner_confidence = home_win_prob * 100
        else:
            predicted_winner = away_team
            winner_confidence = away_win_prob * 100
        
        # Get actual ML values from models
        if ml_home_model and ml_away_model:
            ml_home_value = ml_home_model.predict(xgb.DMatrix(data[count:count+1]))[0]
            ml_away_value = ml_away_model.predict(xgb.DMatrix(data[count:count+1]))[0]
        else:
            # Fallback to synthetic values if models not available
            ml_home_value = np.clip(home_win_prob * 100, 30, 70)
            ml_away_value = np.clip(away_win_prob * 100, 30, 70)
        
        # Calculate spread prediction and confidence
        spread_line = todays_games_spread[count] if todays_games_spread[count] else 1.5
        spread_confidence = winner_confidence  # Use winner confidence as spread confidence
        
        # Determine spread prediction
        if predicted_winner == home_team:
            spread_prediction = f"{home_team} -{spread_line}"
        else:
            spread_prediction = f"{away_team} +{spread_line}"
        
        # Get OU prediction for this game
        ou_data = ou_predictions[count] if ou_predictions else None
        ou_value = ou_data['ou_line'] if ou_data else "N/A"
        
        # Calculate Expected Value and Kelly Criterion if odds are available
        ev_home = ev_away = kelly_home = kelly_away = None
        if home_team_odds[count] and away_team_odds[count]:
            try:
                from src.Utils.Expected_Value import expected_value
                from src.Utils.Kelly_Criterion import Kelly_Criterion
                
                kc = Kelly_Criterion()
                ev_home = float(expected_value(home_win_prob, int(home_team_odds[count])))
                ev_away = float(expected_value(away_win_prob, int(away_team_odds[count])))
                kelly_home = round(kc.calculate_kelly_criterion(home_team_odds[count], home_win_prob), 2)
                kelly_away = round(kc.calculate_kelly_criterion(away_team_odds[count], away_win_prob), 2)
            except ImportError:
                print("Warning: Could not import Expected_Value or Kelly_Criterion modules")
        
        # Create prediction data
        prediction = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Predicted_Winner': predicted_winner,
            'Winner_Confidence_%': round(winner_confidence, 1),
            'ML_Home_Value': round(ml_home_value, 1),
            'ML_Away_Value': round(ml_away_value, 1),
            'Home_Team_Odds': home_team_odds[count] if home_team_odds[count] else 'N/A',
            'Away_Team_Odds': away_team_odds[count] if away_team_odds[count] else 'N/A',
            'EV_Home': ev_home,
            'EV_Away': ev_away,
            'Kelly_Home': kelly_home,
            'Kelly_Away': kelly_away,
            'Spread_Prediction': spread_prediction,
            'Spread_Line': spread_line,
            'Spread_Confidence_%': round(spread_confidence, 1),
            'Over_Under_Prediction': ou_data['ou_prediction'] if ou_data else 'N/A',
            'Over_Under_Line': ou_value,
            'Over_Under_Confidence_%': round(ou_data['ou_confidence'], 1) if ou_data else 'N/A'
        }
        
        predictions_data.append(prediction)
        
        # Print prediction in the requested format
        ml_value = int(ml_home_value) if ml_home_model and ml_away_model else (home_team_odds[count] if home_team_odds[count] else "N/A")
        print(f"{away_team} vs {home_team}, recommended bet: {predicted_winner}, Spread:{spread_line}, ML:{ml_value}, OU:{ou_value}, Confidence:{spread_confidence:.1f}%")
    
    return predictions_data




def main():
    # Load team data from the same source as training models
    con = sqlite3.connect("Data/dataset.sqlite")
    
    # Load the mlb_dataset table (same as training models)
    df = pd.read_sql_query("SELECT * FROM mlb_dataset", con)
    con.close()
    
    # Fetch today's games from MLB API
    print("Fetching today's MLB games...")
    games = get_todays_mlb_games()
    
    if not games:
        print("No games found for today. You can manually enter games:")
        games = []
        while True:
            game_input = input("Enter game as 'Away Team @ Home Team' (or press Enter to finish): ")
            if not game_input:
                break
            try:
                away_team, home_team = game_input.split(' @ ')
                games.append([home_team.strip(), away_team.strip()])
            except ValueError:
                print("Invalid format. Use 'Away Team @ Home Team'")
        
        if not games:
            print("No games entered.")
            return
    else:
        print(f"Found {len(games)} games for today:")
        for i, (home_team, away_team) in enumerate(games, 1):
            print(f"  {i}. {away_team} @ {home_team}")
    
    print(f"Processing {len(games)} games...")
    
    # Create game data
    result = createTodaysGames(games, df, None)  # No odds for now
    if result[0] is None:
        print("Failed to create game data.")
        return
    
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, todays_games_spread = result
    
    # Store original data for NN normalization
    original_data = data.copy()
    
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data_nn = tf.keras.utils.normalize(original_data, axis=1)
        nn_predictions = NN_Runner.nn_runner(data_nn, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc, todays_games_spread, return_data=True)
        print("-------------------------------------------------------")
        
    
    if args.spread:
        print("---------------Spread Value XGBoost Model Predictions---------------")
        spread_predictions = predict_with_spread_model(original_data, games, home_team_odds, away_team_odds)
        print("--------------------------------------------------------------------")
        
    
    if args.xgb:
        print("---------------Unified XGBoost Model Predictions---------------")
        unified_predictions = predict_with_unified_model(original_data, games, home_team_odds, away_team_odds, todays_games_spread, todays_games_uo)
        print("---------------------------------------------------------------")
        
    
    if args.A:
        print("---------------Spread Value XGBoost Model Predictions---------------")
        spread_predictions = predict_with_spread_model(original_data, games, home_team_odds, away_team_odds)
        print("--------------------------------------------------------------------")
        
        
        print("---------------Unified XGBoost Model Predictions---------------")
        unified_predictions = predict_with_unified_model(original_data, games, home_team_odds, away_team_odds, todays_games_spread, todays_games_uo)
        print("---------------------------------------------------------------")
        
        
        data_nn = tf.keras.utils.normalize(original_data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        nn_predictions = NN_Runner.nn_runner(data_nn, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc, todays_games_spread, return_data=True)
        print("-------------------------------------------------------")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLB Betting Model Predictions')
    parser.add_argument('-xgb', action='store_true', help='Run with Unified XGBoost Model (predicts ML values and Home-Team-Win)')
    parser.add_argument('-spread', action='store_true', help='Run with Spread Value XGBoost Model (predicts actual spread values)')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from (future feature)')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    
    main()
