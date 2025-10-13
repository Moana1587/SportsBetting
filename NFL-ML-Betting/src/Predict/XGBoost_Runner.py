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
    
    print(f"Making predictions for {len(games)} games...")
    print(f"Data shape: {data.shape} (rows: {data.shape[0]}, features: {data.shape[1]})")
    
    # Make ML predictions
    ml_predictions_array = []
    try:
        for i, row in enumerate(data):
            if i == 0:  # Debug first row
                print(f"First row shape: {row.shape}, sample values: {row[:5]}")
            prediction = xgb_ml.predict(xgb.DMatrix(np.array([row])))
            ml_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making ML predictions: {e}")
        print(f"Data shape: {data.shape}")
        print(f"First row shape: {data[0].shape if len(data) > 0 else 'No data'}")
        print(f"Expected features: Check model training logs")
        return

    # Make spread predictions using the spread model
    spread_predictions_array = []
    try:
        # Load the best spread model
        from main import load_best_spread_model
        spread_model = load_best_spread_model()
        
        if spread_model is not None:
            print("Making spread predictions...")
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
                print(f"  Game {i+1}: Predicted spread = {prediction[0]:.2f}")
        else:
            print("Warning: No spread model available, using default spread values")
            spread_predictions_array = [0.0] * len(games)
    except Exception as e:
        print(f"Error making spread predictions: {e}")
        print("Using default spread values as fallback")
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
    
    print(f"OU data - Using {len(feature_columns)} feature columns")
    print(f"OU feature columns: {feature_columns[:10]}...")
    
    # Create OU data with only feature columns
    ou_feature_data = frame_uo[feature_columns].values
    
    # Convert to float, handling any remaining non-numeric values
    try:
        ou_data = ou_feature_data.astype(float)
        print("Successfully converted OU data to float")
    except ValueError as e:
        print(f"Error converting OU data to float: {e}")
        print("Attempting fallback conversion method...")
        
        # Fallback: convert each column individually and handle errors
        ou_data_df = pd.DataFrame(ou_feature_data, columns=feature_columns)
        ou_data_df = ou_data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        ou_data = ou_data_df.values.astype(float)
        print("Used fallback conversion method for OU data")

    # Make OU predictions
    print(f"OU data shape: {ou_data.shape} (rows: {ou_data.shape[0]}, features: {ou_data.shape[1]})")
    ou_predictions_array = []
    try:
        for i, row in enumerate(ou_data):
            if i == 0:  # Debug first row
                print(f"OU first row shape: {row.shape}, sample values: {row[:5]}")
            prediction = xgb_uo.predict(xgb.DMatrix(np.array([row])))
            ou_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making OU predictions: {e}")
        print(f"OU data shape: {ou_data.shape}")
        print(f"OU first row shape: {ou_data[0].shape if len(ou_data) > 0 else 'No data'}")
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
    
    # Output in multiple formats for different applications to catch
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # 1. JSON format (for applications that can parse JSON)
    print("===JSON_OUTPUT_START===")
    print(json.dumps(predictions_data, indent=2, cls=NumpyEncoder))
    print("===JSON_OUTPUT_END===")
    
    # 2. CSV format (for spreadsheet applications)
    print("===CSV_OUTPUT_START===")
    print("Game,Recommended_Team,Bet_Type,Spread,Confidence,OU_Prediction,OU_Line,OU_Confidence,Predicted_Spread,Actual_Spread,Spread_Confidence")
    for pred in predictions_data:
        print(f'"{pred["game"]}",{pred["recommended_bet"]["team"]},{pred["recommended_bet"]["type"]},{pred["recommended_bet"]["spread"]},{pred["recommended_bet"]["confidence"]},{pred["over_under"]["prediction"]},{pred["over_under"]["line"]},{pred["over_under"]["confidence"]},{pred["spread_analysis"]["predicted"]},{pred["spread_analysis"]["actual"]},{pred["spread_analysis"]["confidence"]}')
    print("===CSV_OUTPUT_END===")
    
    # 3. Simple text format (for basic text processing)
    print("===TEXT_OUTPUT_START===")
    for pred in predictions_data:
        # Format the recommended bet with ou value,spread value style
        ou_value = pred['over_under']['line']
        spread_value = pred['recommended_bet']['spread']
        
        if pred['recommended_bet']['type'] == "ML":
            formatted_bet = f"{pred['recommended_bet']['team']} {ou_value},{spread_value}"
            print(f"{pred['game']}, recommended bet: {formatted_bet}, confidence: {pred['recommended_bet']['confidence']}%")
        else:
            formatted_bet = f"{pred['recommended_bet']['team']} {pred['recommended_bet']['type']} {pred['recommended_bet']['spread']}"
            print(f"{pred['game']}, recommended bet: {formatted_bet}, confidence: {pred['recommended_bet']['confidence']}%")
        
        # Additional detailed information
        print(f"  OU Prediction: {pred['over_under']['prediction']} {pred['over_under']['line']} (confidence: {pred['over_under']['confidence']}%)")
        print(f"  Spread Analysis: Predicted {pred['spread_analysis']['predicted']:.2f}, Actual {pred['spread_analysis']['actual']:.2f} (confidence: {pred['spread_analysis']['confidence']:.1f}%)")
    print("===TEXT_OUTPUT_END===")
    
    # 4. XML format (for XML processing applications)
    print("===XML_OUTPUT_START===")
    print("<?xml version='1.0' encoding='UTF-8'?>")
    print("<predictions>")
    for pred in predictions_data:
        print(f"  <game home='{pred['home_team']}' away='{pred['away_team']}'>")
        print(f"    <recommended_bet team='{pred['recommended_bet']['team']}' type='{pred['recommended_bet']['type']}' spread='{pred['recommended_bet']['spread']}' confidence='{pred['recommended_bet']['confidence']}'/>")
        print(f"    <over_under prediction='{pred['over_under']['prediction']}' line='{pred['over_under']['line']}' confidence='{pred['over_under']['confidence']}'/>")
        print(f"    <spread_analysis predicted='{pred['spread_analysis']['predicted']}' actual='{pred['spread_analysis']['actual']}' confidence='{pred['spread_analysis']['confidence']}'/>")
        print("  </game>")
    print("</predictions>")
    print("===XML_OUTPUT_END===")

    print("\n" + "="*60)
    deinit()
