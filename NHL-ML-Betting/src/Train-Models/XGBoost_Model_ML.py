"""
XGBoost Model for NHL Moneyline Odds and Win Prediction

This script trains XGBoost models to predict ML_Home, ML_Away (moneyline odds) 
and Home-Team-Win using team statistics and game data from the updated dataset structure.
"""

import sqlite3
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))

def load_dataset():
    """Load the dataset and prepare it for training"""
    dataset = "dataset_2012-24_new"
    db_path = "../../Data/dataset.sqlite"
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Dataset not found at {db_path}. Please run Create_Games.py first.")
    
    con = sqlite3.connect(db_path)
    data = pd.read_sql_query(f"SELECT * FROM \"{dataset}\"", con)
    con.close()

    return data

def prepare_training_data(data):
    """Prepare the data for training by selecting features and target"""
    # Target variables: ML_Home, ML_Away (regression) and Home-Team-Win (classification)
    target = data[['ML_Home', 'ML_Away', 'Home-Team-Win']].copy()
    
    # Ensure Home-Team-Win is properly formatted for binary classification
    # Convert to int and ensure it's 0 or 1
    target['Home-Team-Win'] = target['Home-Team-Win'].astype(int)
    target['Home-Team-Win'] = target['Home-Team-Win'].clip(0, 1)  # Ensure values are 0 or 1
    
    # Features: All team statistics and game data except target and non-predictive columns
    columns_to_drop = [
        'Score',           # Game outcome - not predictive
        'Home-Team-Win',   # Target variable
        'OU-Cover',        # Game outcome - not predictive
        'OU',              # Game outcome - not predictive
        'teamFullName',    # Team name - not predictive
        'teamFullName.1',  # Away team name - not predictive
        'Season',          # Season - not predictive for individual games
        'Season.1',        # Season - not predictive for individual games
        'NHL_SeasonId',    # Season ID - not predictive
        'NHL_SeasonId.1',  # Season ID - not predictive
        'teamId',          # Team ID - not predictive
        'teamId.1',        # Team ID - not predictive
        'ML_Home',         # Target variable
        'ML_Away',         # Target variable
        'Home-Score',      # Game outcome - not predictive
        'Away-Score',      # Game outcome - not predictive
        'Spread',          # Game outcome - not predictive
        'Days-Rest-Home',  # Game context - not predictive
        'Days-Rest-Away',  # Game context - not predictive
    ]
    
    # Drop columns that exist in the dataset
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns=existing_columns_to_drop)
    
    print(f"Features shape: {features.shape}")
    print(f"Dropped columns: {existing_columns_to_drop}")
    
    # Convert to numeric, handling any non-numeric values
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    
    # Fill any NaN values with 0
    features = features.fillna(0)
    
    # Debug: Check target data types and values
    print(f"Target data types: {target.dtypes}")
    print(f"Home-Team-Win unique values: {target['Home-Team-Win'].unique()}")
    print(f"ML_Home range: {target['ML_Home'].min()} to {target['ML_Home'].max()}")
    print(f"ML_Away range: {target['ML_Away'].min()} to {target['ML_Away'].max()}")
    
    return features, target

def train_xgboost_model(features, target, num_iterations=300):
    """Train single XGBoost model for ML_Home, ML_Away, and Home-Team-Win prediction"""
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    
    print(f"Training single XGBoost model for ML_Home, ML_Away, and Home-Team-Win prediction with {num_iterations} iterations...")
    
    results = []
    
    for x in tqdm(range(num_iterations), desc="Training"):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=x)
        
        # Ensure win targets are properly formatted for binary classification
        y_train_win = y_train.iloc[:, 2].astype(int)
        y_test_win = y_test.iloc[:, 2].astype(int)
        
        # Debug: Check target values
        if x == 0:  # Only print debug info for first iteration
            print(f"Training iteration {x} - Target shapes:")
            print(f"  ML_Home: {y_train.iloc[:, 0].shape}, range: {y_train.iloc[:, 0].min():.2f} to {y_train.iloc[:, 0].max():.2f}")
            print(f"  ML_Away: {y_train.iloc[:, 1].shape}, range: {y_train.iloc[:, 1].min():.2f} to {y_train.iloc[:, 1].max():.2f}")
            print(f"  Win: {y_train_win.shape}, unique values: {np.unique(y_train_win)}")
        
        # Create single XGBoost model for multi-output regression
        # We'll use regression for all outputs and handle the classification differently
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            eta=0.01,
            n_estimators=750,
            random_state=x,
            objective='reg:squarederror'
        )
        
        # Train the model on all three outputs
        xgb_model.fit(x_train, y_train)
        
        # Make predictions
        predictions = xgb_model.predict(x_test)
        
        # Calculate R² score for ML_Home and ML_Away (regression outputs)
        r2_home = r2_score(y_test.iloc[:, 0], predictions[:, 0])  # ML_Home
        r2_away = r2_score(y_test.iloc[:, 1], predictions[:, 1])  # ML_Away
        
        # For Home-Team-Win, convert regression output to classification
        # Round the prediction to 0 or 1 for binary classification
        win_pred_rounded = np.round(predictions[:, 2]).astype(int)
        win_pred_rounded = np.clip(win_pred_rounded, 0, 1)  # Ensure 0 or 1
        acc_win = accuracy_score(y_test_win, win_pred_rounded)
        
        # Calculate combined score (weighted average)
        combined_score = (r2_home + r2_away + acc_win) / 3
        combined_score_percent = round(combined_score * 100, 1)
        
        results.append({
            'r2_home': r2_home,
            'r2_away': r2_away,
            'acc_win': acc_win,
            'combined_score': combined_score,
            'model': xgb_model
        })
        
        # Save best model
        if combined_score == max([r['combined_score'] for r in results]):
            # Save the single model to one JSON file
            model_path = f'../../Models/XGBoost_{combined_score_percent}%_ML_All-4.json'
            xgb_model.save_model(model_path)
            print(f"New best combined score: {combined_score_percent}% (ML_Home: {r2_home*100:.1f}%, ML_Away: {r2_away*100:.1f}%, Win: {acc_win*100:.1f}%) - Model saved to {model_path}")
    
    return results

def main():
    """Main training function"""
    try:
        # Load dataset
        data = load_dataset()
        
        # Prepare training data
        features, target = prepare_training_data(data)
        
        # Train models
        results = train_xgboost_model(features, target)
        
        # Extract results
        home_r2_scores = [r['r2_home'] for r in results]
        away_r2_scores = [r['r2_away'] for r in results]
        win_acc_scores = [r['acc_win'] for r in results]
        combined_scores = [r['combined_score'] for r in results]
        
        # Print results
        print(f"\nTraining completed!")
        print(f"\nSingle Multi-Output Model Results:")
        print(f"Best Combined Score: {max(combined_scores)*100:.1f}%")
        print(f"Average Combined Score: {np.mean(combined_scores)*100:.1f}%")
        print(f"Combined Score std: {np.std(combined_scores)*100:.1f}%")
        
        print(f"\nML_Home (Output 1) Results:")
        print(f"Best R²: {max(home_r2_scores)*100:.1f}%")
        print(f"Average R²: {np.mean(home_r2_scores)*100:.1f}%")
        print(f"R² std: {np.std(home_r2_scores)*100:.1f}%")
        
        print(f"\nML_Away (Output 2) Results:")
        print(f"Best R²: {max(away_r2_scores)*100:.1f}%")
        print(f"Average R²: {np.mean(away_r2_scores)*100:.1f}%")
        print(f"R² std: {np.std(away_r2_scores)*100:.1f}%")
        
        print(f"\nHome-Team-Win (Output 3) Results:")
        print(f"Best Accuracy: {max(win_acc_scores)*100:.1f}%")
        print(f"Average Accuracy: {np.mean(win_acc_scores)*100:.1f}%")
        print(f"Accuracy std: {np.std(win_acc_scores)*100:.1f}%")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ XGBoost model training completed successfully!")
    else:
        print("\n❌ XGBoost model training failed!")
