import copy
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

_model = None
_ou_model = None

def _load_models():
    global _model, _ou_model
    if _model is None:
        _model = load_model('Models/NN_Models/Trained-Model-ML-1699315388.285516')
    if _ou_model is None:
        _ou_model = load_model("Models/NN_Models/Trained-Model-OU-1699315414.2268295")

def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, todays_games_spread=None):
    _load_models()
    
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(_model.predict(np.array([row])))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(_ou_model.predict(np.array([row])))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        
        # Determine recommended bet and confidence
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                recommended_bet = f"{home_team} UNDER {todays_games_uo[count]}"
                confidence = un_confidence
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                recommended_bet = f"{home_team} OVER {todays_games_uo[count]}"
                confidence = un_confidence
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                recommended_bet = f"{away_team} UNDER {todays_games_uo[count]}"
                confidence = un_confidence
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                recommended_bet = f"{away_team} OVER {todays_games_uo[count]}"
                confidence = un_confidence
        
        # Print in the specified format
        print(f"{away_team} vs {home_team}, recommended bet: {recommended_bet}, confidence: {confidence}%")
        count += 1
    deinit()
