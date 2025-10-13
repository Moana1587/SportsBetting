"""
XGBoost Prediction Runner

This script loads trained XGBoost models and makes predictions on new NHL games.
Updated to work with the new data structure from Get_Data.py, Get_Odds_Data.py, and Create_Games.py.
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit

# Add project root to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))

from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

# Load models (update paths as needed)
def load_models():
    """Load the trained XGBoost models"""
    models = {}
    
    # Create Models directory if it doesn't exist
    models_dir = "Models"
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(models_dir):
        print("Could not create Models directory.")
        return None
    
    print("Scanning for available models...")
    
    # Try to find the best ML model
    ml_models = [f for f in os.listdir(models_dir) if f.startswith("XGBoost_") and f.endswith("_ML_All-4.json")]
    if ml_models:
        # Sort by accuracy and get the best one
        ml_models.sort(key=lambda x: float(x.split('_')[1].replace('%', '')), reverse=True)
        best_ml_model = os.path.join(models_dir, ml_models[0])
        best_ml_accuracy = float(ml_models[0].split('_')[1].replace('%', ''))
        
        print(f"Found {len(ml_models)} ML models:")
        for i, model in enumerate(ml_models[:3]):  # Show top 3
            acc = float(model.split('_')[1].replace('%', ''))
            marker = "BEST" if i == 0 else "   "
            print(f"   {marker} {model} ({acc}%)")
        
        models['ml'] = xgb.Booster()
        models['ml'].load_model(best_ml_model)
        print(f"Loaded BEST ML model: {ml_models[0]} ({best_ml_accuracy}%)")
    else:
        print("No ML models found. Please train a model first.")
        return None
    
    # Try to find the best UO model
    uo_models = [f for f in os.listdir(models_dir) if f.startswith("XGBoost_") and f.endswith("_UO-9.json")]
    if uo_models:
        uo_models.sort(key=lambda x: float(x.split('_')[1].replace('%', '')), reverse=True)
        best_uo_model = os.path.join(models_dir, uo_models[0])
        best_uo_accuracy = float(uo_models[0].split('_')[1].replace('%', ''))
        
        print(f"Found {len(uo_models)} UO models:")
        for i, model in enumerate(uo_models[:3]):  # Show top 3
            acc = float(model.split('_')[1].replace('%', ''))
            marker = "BEST" if i == 0 else "   "
            print(f"   {marker} {model} ({acc}%)")
        
        models['uo'] = xgb.Booster()
        models['uo'].load_model(best_uo_model)
        print(f"Loaded BEST UO model: {uo_models[0]} ({best_uo_accuracy}%)")
    else:
        print("No UO models found. Please train a model first.")
        return None
    
    # Try to find the best Spread model
    spread_models = [f for f in os.listdir(models_dir) if f.startswith("XGBoost_") and f.endswith("_Spread-4.json")]
    if spread_models:
        spread_models.sort(key=lambda x: float(x.split('_')[1].replace('%', '')), reverse=True)
        best_spread_model = os.path.join(models_dir, spread_models[0])
        best_spread_accuracy = float(spread_models[0].split('_')[1].replace('%', ''))
        
        print(f"Found {len(spread_models)} Spread models:")
        for i, model in enumerate(spread_models[:3]):  # Show top 3
            acc = float(model.split('_')[1].replace('%', ''))
            marker = "BEST" if i == 0 else "   "
            print(f"   {marker} {model} ({acc}%)")
        
        models['spread'] = xgb.Booster()
        models['spread'].load_model(best_spread_model)
        print(f"Loaded BEST Spread model: {spread_models[0]} ({best_spread_accuracy}%)")
    else:
        print("No Spread models found. Please train a model first.")
        return None
    
    print(f"\nModel Selection Summary:")
    print(f"   ML: {best_ml_accuracy}% accuracy")
    print(f"   UO: {best_uo_accuracy}% accuracy") 
    print(f"   Spread: {best_spread_accuracy}% accuracy")
    
    return models

# Initialize models as None - will be loaded when needed
models = None

def prepare_prediction_data(team_stats_home, team_stats_away, game_data):
    """
    Prepare data for prediction in the new format
    
    Args:
        team_stats_home: Home team statistics (pandas Series)
        team_stats_away: Away team statistics (pandas Series) 
        game_data: Game-specific data (OU, rest days, etc.)
    
    Returns:
        Prepared data array for prediction
    """
    # Combine home and away team stats
    combined_stats = pd.concat([
        team_stats_home,
        team_stats_away.rename(index={col: f"{col}.1" for col in team_stats_away.index})
    ])
    
    # Add game-specific data
    for key, value in game_data.items():
        combined_stats[key] = value
    
    # Convert to array and ensure numeric
    data_array = combined_stats.values.astype(float)
    
    return data_array

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, todays_games_spread=None):
    """Run XGBoost predictions on game data"""
    global models
    
    # Load models if not already loaded
    if models is None:
        print("Loading XGBoost models...")
        models = load_models()
        if not models:
            print("Error: No models loaded. Please train models first.")
            print("   Run the following commands to train models:")
            print("   python src/Train-Models/XGBoost_Model_ML.py")
            print("   python src/Train-Models/XGBoost_Model_UO.py")
            print("   python src/Train-Models/XGBoost_Model_Spread.py")
            return
    
    ml_predictions_array = []
    ou_predictions_array = []
    spread_predictions_array = []

    # Make ML predictions - data is now a DataFrame with proper column names
    print(f"Making ML predictions on {len(data)} games...")
    for i in range(len(data)):
        row_data = data.iloc[i:i+1]  # Get single row as DataFrame
        ml_predictions_array.append(models['ml'].predict(xgb.DMatrix(row_data)))

    # Prepare UO data - UO model uses same features as ML model (no OU column)
    frame_uo = copy.deepcopy(frame_ml)
    
    # UO model uses the same features as ML model (team statistics only)
    # No need to reorder columns since frame_uo is a copy of frame_ml
    print(f"Making UO predictions on {len(frame_uo)} games...")

    # Make UO predictions
    for i in range(len(frame_uo)):
        row_data = frame_uo.iloc[i:i+1]  # Get single row as DataFrame
        ou_predictions_array.append(models['uo'].predict(xgb.DMatrix(row_data)))
    
    # Make Spread predictions (if spread data provided)
    if todays_games_spread is not None and 'spread' in models:
        frame_spread = copy.deepcopy(frame_ml)  # Use the same features as ML model
        print(f"Making Spread predictions on {len(frame_spread)} games...")
        
        for i in range(len(frame_spread)):
            row_data = frame_spread.iloc[i:i+1]  # Get single row as DataFrame
            spread_predictions_array.append(models['spread'].predict(xgb.DMatrix(row_data)))

    # Debug: Print array lengths
    print(f"\nDebug: Prediction array lengths:")
    print(f"  Games: {len(games)}")
    print(f"  ML predictions: {len(ml_predictions_array)}")
    print(f"  OU predictions: {len(ou_predictions_array)}")
    print(f"  Spread predictions: {len(spread_predictions_array)}")
    print(f"  OU values: {len(todays_games_uo)}")
    print(f"  Spread values: {len(todays_games_spread) if todays_games_spread else 0}")
    print()

    count = 0
    for game in games:
        try:
            home_team = game[0]
            away_team = game[1]
            
            # Check bounds before accessing prediction arrays
            if count >= len(ml_predictions_array):
                print(f"Warning: Game {count} ({home_team} vs {away_team}) - No ML prediction available")
                count += 1
                continue
            
            # ML predictions: Handle both old (single value) and new (3 values) formats
            ml_pred = ml_predictions_array[count]
            
            # Check if prediction is single value (old model) or 3 values (new model)
            # Handle both 1D and 2D arrays
            if ml_pred.ndim == 2:
                ml_pred = ml_pred[0]  # Flatten 2D array to 1D
            
            if len(ml_pred) == 1:
                # Old model format: single probability value
                winner_confidence = round(float(ml_pred[0]), 3)
                winner = int(winner_confidence > 0.5)
            else:
                # New model format: [ML_Home, ML_Away, Home-Team-Win]
                winner = int(ml_pred[2] > 0.5)  # Home-Team-Win > 0.5 means home team wins
                winner_confidence = round(ml_pred[2] if winner == 1 else (1 - ml_pred[2]), 3)
        
            # UO predictions: single regression value - compare to actual OU line
            if count >= len(ou_predictions_array):
                print(f"Warning: Game {count} ({home_team} vs {away_team}) - No OU prediction available")
                count += 1
                continue
                
            ou_pred = ou_predictions_array[count]
            # Ensure we get a scalar value (handle both single values and arrays)
            if hasattr(ou_pred, '__len__') and len(ou_pred) > 0:
                ou_pred = ou_pred[0]  # Take first element if it's an array
            else:
                ou_pred = float(ou_pred)  # Convert to float if it's a scalar
            
            actual_ou = todays_games_uo[count]
            # Ensure actual_ou is also a scalar
            actual_ou = float(actual_ou)
            under_over = 0 if ou_pred < actual_ou else 1  # 0=Under, 1=Over
            un_confidence = round(abs(ou_pred - actual_ou) / actual_ou, 3)  # Relative difference as confidence
        
            # Get spread prediction if available (single regression value)
            spread_value = None
            spread_confidence = None
            if spread_predictions_array and count < len(spread_predictions_array):
                spread_pred = spread_predictions_array[count]
                # Ensure we get a scalar value (handle both single values and arrays)
                if hasattr(spread_pred, '__len__') and len(spread_pred) > 0:
                    spread_value = spread_pred[0]  # Take first element if it's an array
                else:
                    spread_value = float(spread_pred)  # Convert to float if it's a scalar
                spread_confidence = 0.5  # Placeholder confidence for regression value
            # Build prediction display
            winner_confidence_pct = round(winner_confidence * 100, 1)
            if winner == 1:
                winner_display = Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL
            else:
                winner_display = Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_pct}%)" + Style.RESET_ALL
            
            # Build O/U display
            un_confidence_pct = round(un_confidence * 100, 1)
            if under_over == 0:
                ou_display = Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(actual_ou) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct}%)" + Style.RESET_ALL
            else:
                ou_display = Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(actual_ou) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_pct}%)" + Style.RESET_ALL
        
            # Build spread display with predicted spread values
            spread_display = ""
            if spread_value is not None and todays_games_spread:
                actual_spread = todays_games_spread[count]
                spread_conf_pct = round(spread_confidence * 100, 1)
                
                # Ensure actual_spread is also a scalar
                actual_spread = float(actual_spread)
                
                # Compare predicted spread to actual spread to determine recommendation
                if spread_value > actual_spread:
                    # Predicted spread is higher, recommend home team
                    if actual_spread > 0:
                        spread_display = f" | " + Fore.YELLOW + f"Recommended bet: {home_team} -{actual_spread} (Pred: {spread_value:.1f})" + Style.RESET_ALL
                    else:
                        spread_display = f" | " + Fore.YELLOW + f"Recommended bet: {home_team} +{abs(actual_spread)} (Pred: {spread_value:.1f})" + Style.RESET_ALL
                else:
                    # Predicted spread is lower, recommend away team
                    if actual_spread > 0:
                        spread_display = f" | " + Fore.YELLOW + f"Recommended bet: {away_team} +{actual_spread} (Pred: {spread_value:.1f})" + Style.RESET_ALL
                    else:
                        spread_display = f" | " + Fore.YELLOW + f"Recommended bet: {away_team} -{abs(actual_spread)} (Pred: {spread_value:.1f})" + Style.RESET_ALL
        
            # Print combined prediction
            print(winner_display + ': ' + ou_display + spread_display)
            count += 1
        except Exception as e:
            print(f"Error processing game {count}: {e}")
            print(f"  ML pred type: {type(ml_predictions_array[count])}, value: {ml_predictions_array[count]}")
            if count < len(ou_predictions_array):
                print(f"  OU pred type: {type(ou_predictions_array[count])}, value: {ou_predictions_array[count]}")
            if count < len(spread_predictions_array):
                print(f"  Spread pred type: {type(spread_predictions_array[count])}, value: {spread_predictions_array[count]}")
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
        
        # Check bounds before accessing prediction arrays
        if count >= len(ml_predictions_array):
            print(f"Warning: Game {count} ({home_team} vs {away_team}) - No ML prediction for EV calculation")
            count += 1
            continue
            
        if home_team_odds[count] and away_team_odds[count]:
            # ML predictions: Handle both old and new formats
            ml_pred = ml_predictions_array[count]
            
            # Handle both 1D and 2D arrays
            if ml_pred.ndim == 2:
                ml_pred = ml_pred[0]  # Flatten 2D array to 1D
            
            if len(ml_pred) == 1:
                # Old model format: single probability value
                home_win_prob = round(float(ml_pred[0]), 3)
                away_win_prob = round(1 - home_win_prob, 3)
            else:
                # New model format: [ML_Home, ML_Away, Home-Team-Win]
                home_win_prob = round(ml_pred[2], 3)  # Home-Team-Win probability
                away_win_prob = round(1 - home_win_prob, 3)  # Away team win probability
            
            ev_home = float(Expected_Value.expected_value(home_win_prob, int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(away_win_prob, int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                        'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], home_win_prob)) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], away_win_prob)) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    # Add recommended bets summary
    print("\n" + "="*60)
    print(Fore.CYAN + "RECOMMENDED BETS SUMMARY" + Style.RESET_ALL)
    print("="*60)
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        # Check bounds before accessing prediction arrays
        if count >= len(ml_predictions_array):
            print(f"Warning: Game {count} ({home_team} vs {away_team}) - No ML prediction for summary")
            count += 1
            continue
        
        # Moneyline recommendation
        ml_pred = ml_predictions_array[count]
        
        # Handle both 1D and 2D arrays
        if ml_pred.ndim == 2:
            ml_pred = ml_pred[0]  # Flatten 2D array to 1D
        
        if len(ml_pred) == 1:
            # Old model format: single probability value
            winner_confidence = round(float(ml_pred[0]), 3)
            winner = int(winner_confidence > 0.5)
            winner_confidence = round(winner_confidence * 100, 1) if winner == 1 else round((1 - winner_confidence) * 100, 1)
        else:
            # New model format: [ML_Home, ML_Away, Home-Team-Win]
            winner = int(ml_pred[2] > 0.5)  # Home-Team-Win > 0.5 means home team wins
            winner_confidence = round(ml_pred[2] * 100, 1) if winner == 1 else round((1 - ml_pred[2]) * 100, 1)
        
        ml_pick = home_team if winner == 1 else away_team
        
        # Over/Under recommendation
        if count >= len(ou_predictions_array):
            print(f"Warning: Game {count} ({home_team} vs {away_team}) - No OU prediction for summary")
            count += 1
            continue
            
        ou_pred = ou_predictions_array[count]
        # Ensure we get a scalar value (handle both single values and arrays)
        if hasattr(ou_pred, '__len__') and len(ou_pred) > 0:
            ou_pred = ou_pred[0]  # Take first element if it's an array
        else:
            ou_pred = float(ou_pred)  # Convert to float if it's a scalar
        
        actual_ou = todays_games_uo[count]
        # Ensure actual_ou is also a scalar
        actual_ou = float(actual_ou)
        under_over = 0 if ou_pred < actual_ou else 1  # 0=Under, 1=Over
        un_confidence = round(abs(ou_pred - actual_ou) / actual_ou * 100, 1)
        ou_pick = "OVER" if under_over == 1 else "UNDER"
        ou_value = actual_ou
        
        # Spread recommendation
        spread_pick = ""
        if spread_predictions_array and todays_games_spread and count < len(spread_predictions_array):
            predicted_spread = spread_predictions_array[count]
            # Ensure we get a scalar value (handle both single values and arrays)
            if hasattr(predicted_spread, '__len__') and len(predicted_spread) > 0:
                predicted_spread = predicted_spread[0]  # Take first element if it's an array
            else:
                predicted_spread = float(predicted_spread)  # Convert to float if it's a scalar
            actual_spread = todays_games_spread[count]
            # Ensure actual_spread is also a scalar
            actual_spread = float(actual_spread)
            
            if predicted_spread > actual_spread:
                # Predicted spread is higher, recommend home team
                if actual_spread > 0:
                    spread_pick = f"{home_team} -{actual_spread} (Pred: {predicted_spread:.1f})"
                else:
                    spread_pick = f"{home_team} +{abs(actual_spread)} (Pred: {predicted_spread:.1f})"
            else:
                # Predicted spread is lower, recommend away team
                if actual_spread > 0:
                    spread_pick = f"{away_team} +{actual_spread} (Pred: {predicted_spread:.1f})"
                else:
                    spread_pick = f"{away_team} -{abs(actual_spread)} (Pred: {predicted_spread:.1f})"
        
        print(f"\n{Style.BRIGHT}{home_team} vs {away_team}{Style.RESET_ALL}")
        print(f"  Moneyline: {Fore.GREEN}{ml_pick} ({winner_confidence}%){Style.RESET_ALL}")
        print(f"  Over/Under: {Fore.BLUE}{ou_pick} {ou_value} ({un_confidence}%){Style.RESET_ALL}")
        if spread_pick:
            print(f"  Spread: {Fore.YELLOW}{spread_pick}{Style.RESET_ALL}")
        
        count += 1
    
    print("\n" + "="*60)
    deinit()
    
    # Return prediction results for CSV export
    return {
        'ml_predictions': ml_predictions_array,
        'ou_predictions': ou_predictions_array,
        'spread_predictions': spread_predictions_array
    }
