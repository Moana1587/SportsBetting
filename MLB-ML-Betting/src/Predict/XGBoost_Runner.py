import copy
import os
import glob

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

# Load the most recent XGBoost models
def load_latest_model(pattern, model_type):
    """Load the most recent model matching the pattern"""
    # Try multiple possible paths
    search_paths = [
        f"../../Models/{pattern}",
        f"Models/{pattern}",
        f"../Models/{pattern}",
        f"../../Models/XGBoost_Models/{pattern}",
        f"Models/XGBoost_Models/{pattern}"
    ]
    
    model_files = []
    for path in search_paths:
        files = glob.glob(path)
        if files:
            model_files.extend(files)
            print(f"Found {len(files)} {model_type} models in {path}")
    
    if not model_files:
        print(f"Warning: No {model_type} model found matching pattern {pattern}")
        print(f"Searched in: {search_paths}")
        # List available files for debugging
        for path in ["../../Models", "Models", "../Models"]:
            all_files = glob.glob(f"{path}/*.json")
            if all_files:
                print(f"Available model files in {path}: {all_files}")
                break
        return None
    
    # Sort by modification time and get the most recent
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading {model_type} model: {latest_model}")
    
    booster = xgb.Booster()
    booster.load_model(latest_model)
    return booster

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, todays_games_spread=None, return_data=False):
    # Load unified ML model and other models
    xgb_unified_ml = load_latest_model("XGBoost_Unified_*_ML.json", "Unified Moneyline")
    xgb_uo = load_latest_model("XGBoost_*_UO-9.json", "Over/Under")
    xgb_spread = load_latest_model("XGBoost_*_Spread-10.json", "Spread")

    if xgb_unified_ml is None or xgb_uo is None:
        print("Error: Could not load required models. Please train models first.")
        return None if return_data else None
    
    print(f"Making predictions for {len(games)} games...")
    print(f"Data shape: {data.shape}")
    
    ml_predictions_array = []

    for row in data:
        # Use unified ML model for predictions
        unified_pred = xgb_unified_ml.predict(xgb.DMatrix(np.array([row])))
        # Convert to expected format: [[prob_away, prob_home]]
        ml_pred = np.array([[unified_pred[0][0], unified_pred[0][1]]])
        ml_predictions_array.append(ml_pred)

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data_uo = frame_uo.values
    data_uo = data_uo.astype(float)

    ou_predictions_array = []

    for row in data_uo:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    # Spread predictions (if model is available)
    spread_predictions_array = []
    if xgb_spread is not None:
        # For spread, we need to add the spread line as a feature
        frame_spread = copy.deepcopy(frame_ml)
        # Use actual spread values if provided, otherwise default to 1.5
        if todays_games_spread is not None:
            frame_spread['Spread_Line'] = todays_games_spread
        else:
            frame_spread['Spread_Line'] = 1.5
        data_spread = frame_spread.values
        data_spread = data_spread.astype(float)

        for row in data_spread:
            spread_predictions_array.append(xgb_spread.predict(xgb.DMatrix(np.array([row]))))
    else:
        print("Warning: Spread model not available, skipping spread predictions")
        spread_predictions_array = [None] * len(games)

    # Prepare data for CSV export
    predictions_data = []
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        # Handle different prediction formats
        ml_pred = ml_predictions_array[count]
        ou_pred = ou_predictions_array[count]
        
        # Check if predictions are in the new format (multi:softprob) or old format
        if len(ml_pred.shape) == 1:
            # New format: single array of probabilities [prob_class_0, prob_class_1]
            winner = int(np.argmax(ml_pred))
            winner_confidence_pct = round(ml_pred[winner] * 100, 1)
            predicted_winner = home_team if winner == 1 else away_team
        else:
            # Old format: nested array [[prob_class_0, prob_class_1]]
            winner = int(np.argmax(ml_pred))
            winner_confidence_pct = round(ml_pred[0][winner] * 100, 1)
            predicted_winner = home_team if winner == 1 else away_team
        
        # Handle Over/Under predictions
        if len(ou_pred.shape) == 1:
            # New format
            under_over = int(np.argmax(ou_pred))
            un_confidence_pct = round(ou_pred[under_over] * 100, 1)
        else:
            # Old format
            under_over = int(np.argmax(ou_pred))
            un_confidence_pct = round(ou_pred[0][under_over] * 100, 1)
            
        if under_over == 0:
            ou_prediction = "UNDER"
        else:
            ou_prediction = "OVER"
        
        # Spread prediction
        spread_prediction = ""
        spread_winner = ""
        spread_confidence = 0
        if spread_predictions_array[count] is not None:
            spread_winner_idx = int(np.argmax(spread_predictions_array[count]))
            spread_confidence = round(spread_predictions_array[count][0][spread_winner_idx] * 100, 1)
            spread_value = round(todays_games_spread[count], 1) if todays_games_spread else 1.5
            if spread_winner_idx == 1:
                spread_winner = "HOME COVERS"
                spread_prediction = f"HOME COVERS {spread_value} ({spread_confidence:.1f}%)"
            else:
                spread_winner = "AWAY COVERS"
                spread_prediction = f"AWAY COVERS {spread_value} ({spread_confidence:.1f}%)"
        
        # Calculate expected values and Kelly criterion
        ev_home = ev_away = 0
        kelly_home = kelly_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            # Handle different prediction formats for EV and Kelly calculations
            if len(ml_pred.shape) == 1:
                # New format: [prob_class_0, prob_class_1]
                home_prob = ml_pred[1]  # Class 1 is home team win
                away_prob = ml_pred[0]  # Class 0 is away team win
            else:
                # Old format: [[prob_class_0, prob_class_1]]
                home_prob = ml_pred[0][1]
                away_prob = ml_pred[0][0]
            
            ev_home = float(Expected_Value.expected_value(home_prob, int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(away_prob, int(away_team_odds[count])))
            kelly_home = round(kc.calculate_kelly_criterion(home_team_odds[count], home_prob), 2)
            kelly_away = round(kc.calculate_kelly_criterion(away_team_odds[count], away_prob), 2)
        
        # Store prediction data
        prediction_row = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Predicted_Winner': predicted_winner,
            'Winner_Confidence_%': winner_confidence_pct,
            'Over_Under_Line': round(todays_games_uo[count], 1),
            'Over_Under_Prediction': ou_prediction,
            'Over_Under_Confidence_%': un_confidence_pct,
            'Spread_Line': round(todays_games_spread[count], 1) if todays_games_spread else 1.5,
            'Spread_Prediction': spread_winner,
            'Spread_Confidence_%': spread_confidence,
            'Home_Team_Odds': home_team_odds[count],
            'Away_Team_Odds': away_team_odds[count],
            'Home_Team_EV': round(ev_home, 2),
            'Away_Team_EV': round(ev_away, 2),
            'Home_Team_Kelly_%': kelly_home,
            'Away_Team_Kelly_%': kelly_away
        }
        predictions_data.append(prediction_row)
        
        # Display predictions (original functionality)
        spread_display = ""
        if spread_predictions_array[count] is not None:
            spread_display = f" | {Fore.GREEN if spread_winner == 'HOME COVERS' else Fore.RED}{spread_prediction}{Style.RESET_ALL}"
        
        if winner == 1:
            if under_over == 0:
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct:.1f}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        round(todays_games_uo[count], 1)) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct:.1f}%)" + Style.RESET_ALL + spread_display)
            else:
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct:.1f}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        round(todays_games_uo[count], 1)) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct:.1f}%)" + Style.RESET_ALL + spread_display)
        else:
            if under_over == 0:
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct:.1f}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        round(todays_games_uo[count], 1)) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct:.1f}%)" + Style.RESET_ALL + spread_display)
            else:
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct:.1f}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        round(todays_games_uo[count], 1)) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct:.1f}%)" + Style.RESET_ALL + spread_display)
        count += 1

    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                        'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(round(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1]), 2)) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(round(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0]), 2)) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(round(ev_home, 2)) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(round(ev_away, 2)) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    deinit()
    
    if return_data:
        return predictions_data
    return None
