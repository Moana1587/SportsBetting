import copy
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()

# Load available models
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.7%_ML-4.json')

xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_53.7%_UO-9.json')

# Load the best spread model
xgb_spread = None
spread_models = [f for f in os.listdir('Models') if f.startswith('XGBoost_Spread_Value_') and f.endswith('.json')]
if spread_models:
    # Sort by MSE (lowest first) and take the best one
    # Extract MSE value from filename like "XGBoost_Spread_Value_6.774_MSE.json"
    def extract_mse(filename):
        try:
            # Split by '_' and find the MSE value
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part == 'MSE':
                    return float(parts[i-1])
            return float('inf')  # If MSE not found, put at end
        except:
            return float('inf')
    
    best_spread_model = sorted(spread_models, key=extract_mse)[0]
    try:
        xgb_spread = xgb.Booster()
        xgb_spread.load_model(f'Models/{best_spread_model}')
        print(f"Loaded spread model: {best_spread_model}")
    except Exception as e:
        print(f"Warning: Could not load spread model: {e}")
        xgb_spread = None
else:
    print("No spread model found. Only ML and OU predictions will be shown.")

# Load the best ML classification model
xgb_ml_classification = None
ml_classification_models = [f for f in os.listdir('Models') if f.startswith('XGBoost_') and 'ML_Classification' in f and f.endswith('.json')]
if ml_classification_models:
    # Sort by accuracy (highest first) and take the best one
    best_ml_model = sorted(ml_classification_models, key=lambda x: float(x.split('%')[0].split('_')[-1]), reverse=True)[0]
    try:
        xgb_ml_classification = xgb.Booster()
        xgb_ml_classification.load_model(f'Models/{best_ml_model}')
        print(f"Loaded ML classification model: {best_ml_model}")
    except Exception as e:
        print(f"Warning: Could not load ML classification model: {e}")
        xgb_ml_classification = None
else:
    print("No ML classification model found. Using original ML model.")

# Load the best OU value regression model
xgb_ou_value = None
ou_value_models = [f for f in os.listdir('Models') if f.startswith('XGBoost_OU_Value_') and f.endswith('.json')]
if ou_value_models:
    # Sort by MSE (lowest first) and take the best one
    def extract_ou_mse(filename):
        try:
            # Split by '_' and find the MSE value
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part == 'MSE':
                    return float(parts[i-1])
            return float('inf')  # If MSE not found, put at end
        except:
            return float('inf')
    
    best_ou_model = sorted(ou_value_models, key=extract_ou_mse)[0]
    try:
        xgb_ou_value = xgb.Booster()
        xgb_ou_value.load_model(f'Models/{best_ou_model}')
        print(f"Loaded OU value model: {best_ou_model}")
    except Exception as e:
        print(f"Warning: Could not load OU value model: {e}")
        xgb_ou_value = None
else:
    print("No OU value model found. Only classification OU predictions will be shown.")


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    print("=" * 80)
    print("                    XGBOOST MODEL PREDICTIONS")
    print("=" * 80)
    
    # Get ML predictions (original model)
    ml_predictions_array = []
    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    # Get ML classification predictions (new model with odds features)
    ml_classification_predictions_array = []
    if xgb_ml_classification is not None:
        # Prepare data with ML odds features
        frame_ml_with_odds = copy.deepcopy(frame_ml)
        frame_ml_with_odds['ML_Home_Odds'] = home_team_odds
        frame_ml_with_odds['ML_Away_Odds'] = away_team_odds
        data_ml_with_odds = frame_ml_with_odds.values
        data_ml_with_odds = data_ml_with_odds.astype(float)
        
        for row in data_ml_with_odds:
            ml_classification_predictions_array.append(xgb_ml_classification.predict(xgb.DMatrix(np.array([row]))))

    # Get OU predictions
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data_uo = frame_uo.values
    data_uo = data_uo.astype(float)

    ou_predictions_array = []
    for row in data_uo:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    # Get spread predictions if model is available
    spread_predictions_array = []
    if xgb_spread is not None:
        for row in data:
            spread_predictions_array.append(xgb_spread.predict(xgb.DMatrix(np.array([row]))))

    # Get OU value predictions if model is available
    ou_value_predictions_array = []
    if xgb_ou_value is not None:
        for row in data:
            ou_value_predictions_array.append(xgb_ou_value.predict(xgb.DMatrix(np.array([row]))))

    # Display predictions for each game in the requested format
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        # Get prediction probabilities
        ml_winner = int(np.argmax(ml_predictions_array[count]))
        home_ml_prob = round(ml_predictions_array[count][0][1] * 100, 1)
        away_ml_prob = round(ml_predictions_array[count][0][0] * 100, 1)
        
        ou_winner = int(np.argmax(ou_predictions_array[count]))
        under_prob = round(ou_predictions_array[count][0][0] * 100, 1)
        over_prob = round(ou_predictions_array[count][0][1] * 100, 1)
        ou_line = todays_games_uo[count]
        
        # Determine recommended team and bets
        recommended_team = None
        recommended_bets = []
        
        # Determine which team to recommend based on ML prediction
        if ml_winner == 1 and home_ml_prob > 50:  # Home team wins with confidence
            recommended_team = home_team
        elif ml_winner == 0 and away_ml_prob > 50:  # Away team wins with confidence
            recommended_team = away_team
        
        # Spread recommendation (if available, always show)
        if xgb_spread is not None and len(spread_predictions_array) > count:
            spread_prediction = spread_predictions_array[count][0]
            if abs(spread_prediction) > 0.1:  # Show if spread is meaningful (>0.1)
                if spread_prediction > 0:
                    # Home team favored
                    recommended_bets.append(f"Spread:+{abs(spread_prediction):.1f}({away_ml_prob:.1f}%)")
                else:
                    # Away team favored
                    recommended_bets.append(f"Spread:-{abs(spread_prediction):.1f}({home_ml_prob:.1f}%)")
        
        # Moneyline recommendation
        if ml_winner == 1 and home_ml_prob > 50:  # Home team wins
            # Convert probability to ML odds format
            if home_ml_prob >= 50:
                # For favorites (probability >= 50%), calculate negative odds
                ml_odds = round(-100 * home_ml_prob / (100 - home_ml_prob)) if home_ml_prob < 100 else -1000
            else:
                # For underdogs (probability < 50%), calculate positive odds
                ml_odds = round(100 * (100 - home_ml_prob) / home_ml_prob) if home_ml_prob > 0 else 1000
            recommended_bets.append(f"ML:{ml_odds}({home_ml_prob:.1f}%)")
        elif ml_winner == 0 and away_ml_prob > 50:  # Away team wins
            # Convert probability to ML odds format
            if away_ml_prob >= 50:
                # For favorites (probability >= 50%), calculate negative odds
                ml_odds = round(-100 * away_ml_prob / (100 - away_ml_prob)) if away_ml_prob < 100 else -1000
            else:
                # For underdogs (probability < 50%), calculate positive odds
                ml_odds = round(100 * (100 - away_ml_prob) / away_ml_prob) if away_ml_prob > 0 else 1000
            recommended_bets.append(f"ML:{ml_odds}({away_ml_prob:.1f}%)")
        
        # Over/Under recommendation (always show, regardless of confidence)
        # Removed traditional OU classification - using OU value prediction instead
        
        # OU Value prediction (renamed to just "OU")
        if xgb_ou_value is not None and len(ou_value_predictions_array) > count:
            ou_value_prediction = ou_value_predictions_array[count][0]
            
            # Calculate confidence based on how far the prediction is from the line
            # The further the prediction is from the line, the higher the confidence
            difference = abs(ou_value_prediction - ou_line)
            
            # Convert difference to confidence percentage (0-100%)
            # Use a sigmoid-like function to map difference to confidence
            # Typical NBA OU lines range from ~180-250, so differences of 10+ points are significant
            max_difference = 20.0  # Maximum expected difference for 100% confidence
            confidence = min(95.0, max(50.0, 50.0 + (difference / max_difference) * 45.0))
            
            if ou_value_prediction > ou_line:
                recommended_bets.append(f"OU:OVER {ou_value_prediction:.1f}({confidence:.1f}%)")
            else:
                recommended_bets.append(f"OU:UNDER {ou_value_prediction:.1f}({confidence:.1f}%)")
        
        # Print the result in the requested format
        if recommended_team and recommended_bets:
            recommended_str = ", ".join(recommended_bets)
            print(f"{home_team} vs {away_team}, recommended bet: {recommended_team}, {recommended_str}")
        elif recommended_bets:
            recommended_str = ", ".join(recommended_bets)
            print(f"{home_team} vs {away_team}, recommended bet: {recommended_str}")
        else:
            print(f"{home_team} vs {away_team}, recommended bet: No strong recommendations (all probabilities < 50%)")
        
        count += 1

    # Expected Value and Kelly Criterion
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    if kelly_criterion:
        print(f"{Fore.YELLOW}EXPECTED VALUE & KELLY CRITERION{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}EXPECTED VALUE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        
        expected_value_colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }
        
        print(f"\n{Fore.YELLOW}{home_team} vs {away_team}:{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}HOME {home_team} EV: {expected_value_colors['home_color']}{ev_home:.3f}{Style.RESET_ALL}", end="")
        if kelly_criterion:
            kelly_home = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])
            print(f" | Kelly: {kelly_home:.2f}%")
        else:
            print()
            
        print(f"  {Fore.MAGENTA}AWAY {away_team} EV: {expected_value_colors['away_color']}{ev_away:.3f}{Style.RESET_ALL}", end="")
        if kelly_criterion:
            kelly_away = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])
            print(f" | Kelly: {kelly_away:.2f}%")
        else:
            print()
        
        count += 1

    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    deinit()
