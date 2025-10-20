import copy
import glob
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

def load_best_model(model_type):
    """Load the best available model for the given type (ML or UO)"""
    # Try multiple possible model directory locations
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
        print("Please train models first by running:")
        print("  python src/Train-Models/XGBoost_Model_ML.py")
        print("  python src/Train-Models/XGBoost_Model_UO.py")
        return None
    
    # Find all models of the specified type
    pattern = f"XGBoost_*%_{model_type}.json"
    model_files = glob.glob(os.path.join(model_dir, pattern))
    
    if not model_files:
        print(f"Warning: No {model_type} models found in {model_dir}")
        return None
    
    # Sort by accuracy (extract from filename) and get the best one
    def extract_accuracy(filename):
        try:
            # Extract accuracy from filename like "XGBoost_68.7%_ML.json"
            basename = os.path.basename(filename)
            accuracy_str = basename.split('_')[1].replace('%', '')
            return float(accuracy_str)
        except:
            return 0.0
    
    best_model = max(model_files, key=extract_accuracy)
    accuracy = extract_accuracy(best_model)
    
    print(f"Loading best {model_type} model: {os.path.basename(best_model)} ({accuracy}%)")
    
    try:
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    except Exception as e:
        print(f"Error loading model {best_model}: {e}")
        return None

# Load the best available models
xgb_ml = load_best_model('ML')
xgb_uo = load_best_model('UO')

if xgb_ml is None or xgb_uo is None:
    print("Error: Could not load required models. Please train models first.")
    print("Run: python src/Train-Models/XGBoost_Model_ML.py")
    print("Run: python src/Train-Models/XGBoost_Model_UO.py")


def xgb_runner(data, todays_games_uo, todays_games_spread, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """Run XGBoost predictions for today's games"""
    
    if xgb_ml is None or xgb_uo is None:
        print("Error: Models not loaded. Cannot make predictions.")
        return
    
    if data is None or len(data) == 0:
        print("Error: No data provided for predictions.")
        return
    
    # Make ML predictions
    ml_predictions_array = []
    try:
        for i, row in enumerate(data):
            prediction = xgb_ml.predict(xgb.DMatrix(np.array([row])))
            ml_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making ML predictions: {e}")
        return

    # Make spread predictions using the spread model
    spread_predictions_array = []
    try:
        # Load the best spread model
        from main import load_best_spread_model
        spread_model = load_best_spread_model()
        
        if spread_model is not None:
            for i, row in enumerate(data):
                # Create feature array for spread prediction
                exclude_columns = [
                    'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
                    'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
                    'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
                    'Date', 'Date.1', 'index.1', 'Spread'
                ]
                
                # Get feature columns from frame_ml
                feature_columns = [col for col in frame_ml.columns if col not in exclude_columns]
                feature_data = frame_ml.iloc[i][feature_columns].values
                
                # Make spread prediction
                prediction = spread_model.predict(xgb.DMatrix(np.array([feature_data])))
                spread_predictions_array.append(float(prediction[0]))
        else:
            spread_predictions_array = [0.0] * len(games)
    except Exception as e:
        spread_predictions_array = [0.0] * len(games)

    # Prepare OU data with same feature selection as ML data
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    
    # Use the same feature selection logic as in training
    exclude_columns = [
        'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
        'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away',
        'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
        'Date', 'Date.1', 'index.1'
    ]
    
    # Get feature columns (same as ML prediction)
    feature_columns = [col for col in frame_uo.columns if col not in exclude_columns]
    
    # Create OU data with only feature columns
    ou_feature_data = frame_uo[feature_columns].values
    
    # Convert to float, handling any remaining non-numeric values
    try:
        ou_data = ou_feature_data.astype(float)
    except ValueError as e:
        # Fallback: convert each column individually and handle errors
        ou_data_df = pd.DataFrame(ou_feature_data, columns=feature_columns)
        ou_data_df = ou_data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        ou_data = ou_data_df.values.astype(float)

    # Make OU predictions
    ou_predictions_array = []
    try:
        for i, row in enumerate(ou_data):
            prediction = xgb_uo.predict(xgb.DMatrix(np.array([row])))
            ou_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making OU predictions: {e}")
        return

    # Prepare structured data for export
    import json
    
    predictions_data = []
    
    count = 0
    for game in games:
        if count >= len(ml_predictions_array) or count >= len(ou_predictions_array):
            break
            
        home_team = game[0]
        away_team = game[1]
        
        # Get predictions
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        
        # Get spread prediction
        predicted_spread = spread_predictions_array[count] if count < len(spread_predictions_array) else 0.0
        actual_spread = todays_games_spread[count] if count < len(todays_games_spread) else predicted_spread
        
        # Calculate spread confidence based on how close prediction is to actual spread
        spread_diff = abs(predicted_spread - actual_spread)
        spread_confidence = max(50, 100 - (spread_diff * 10))  # Higher confidence for smaller differences
        
        # Determine recommended bet based on ML prediction and spread
        if winner == 1:  # Home team wins
            winner_confidence_pct = round(winner_confidence[0][1] * 100, 1)
            recommended_team = home_team
            bet_type = "ML"
        else:  # Away team wins
            winner_confidence_pct = round(winner_confidence[0][0] * 100, 1)
            recommended_team = away_team
            bet_type = "ML"
        
        # Determine if we should recommend a spread bet instead of ML
        # If the spread prediction is significantly different from the actual spread,
        # and we have high confidence, recommend the spread bet
        spread_edge = abs(predicted_spread - actual_spread)
        if spread_edge > 1.0 and spread_confidence > 60:  # Significant edge and high confidence
            if predicted_spread > actual_spread:  # Home team is undervalued
                if winner == 1:  # Home team wins
                    recommended_team = home_team
                    bet_type = "Spread"
                    winner_confidence_pct = spread_confidence
            else:  # Away team is undervalued
                if winner == 0:  # Away team wins
                    recommended_team = away_team
                    bet_type = "Spread"
                    winner_confidence_pct = spread_confidence
        
        # Create structured data with proper type conversion for JSON serialization
        game_prediction = {
            "game": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "recommended_bet": {
                "team": recommended_team,
                "type": bet_type,
                "spread": float(actual_spread),
                "confidence": float(winner_confidence_pct)
            },
            "over_under": {
                "prediction": "UNDER" if under_over == 0 else "OVER",
                "line": float(todays_games_uo[count]),
                "confidence": float(round(ou_predictions_array[count][0][under_over] * 100, 1))
            },
            "spread_analysis": {
                "predicted": float(predicted_spread),
                "actual": float(actual_spread),
                "confidence": float(spread_confidence)
            }
        }
        
        predictions_data.append(game_prediction)
        count += 1
    
    # Print only the integrated recommended bet format
    for pred in predictions_data:
        # Extract game teams
        home_team = pred['home_team']
        away_team = pred['away_team']
        
        # Get recommended team
        recommended_team = pred['recommended_bet']['team']
        
        # Format spread with sign and confidence
        spread_value = pred['recommended_bet']['spread']
        spread_confidence = pred['recommended_bet']['confidence']
        
        # Format ML odds (convert confidence to ML odds format)
        ml_confidence = pred['recommended_bet']['confidence']
        if ml_confidence > 50:
            # Convert confidence to ML odds (simplified)
            ml_odds = int(-100 * ml_confidence / (100 - ml_confidence))
        else:
            ml_odds = int(100 * (100 - ml_confidence) / ml_confidence)
        
        # Format OU prediction
        ou_prediction = pred['over_under']['prediction']
        ou_line = pred['over_under']['line']
        ou_confidence = pred['over_under']['confidence']
        
        # Create the integrated line in the requested format
        integrated_line = f"{home_team} vs {away_team}, recommended bet: {recommended_team}, Spread:{spread_value}({spread_confidence:.1f}%), ML:{ml_odds}({ml_confidence:.1f}%), OU:{ou_prediction} {ou_line}({ou_confidence:.1f}%)"
        print(integrated_line)
    deinit()
