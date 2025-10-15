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
    # Target variables: ML_Home, ML_Away (regression)
    target = data[['ML_Home', 'ML_Away']].copy()
    
    # Ensure ML values are properly formatted
    target['ML_Home'] = target['ML_Home'].astype(float)
    target['ML_Away'] = target['ML_Away'].astype(float)
    
    # Features: All team statistics and game data except target and non-predictive columns
    columns_to_drop = [
        'Score',           # Game outcome - not predictive
        'Home-Team-Win',   # Game outcome - not predictive
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
    print(f"ML_Home range: {target['ML_Home'].min()} to {target['ML_Home'].max()}")
    print(f"ML_Away range: {target['ML_Away'].min()} to {target['ML_Away'].max()}")
    print(f"ML_Home unique values sample: {sorted(target['ML_Home'].unique())[:10]}")
    print(f"ML_Away unique values sample: {sorted(target['ML_Away'].unique())[:10]}")
    
    return features, target

def train_xgboost_model(features, target, num_iterations=300):
    """Train XGBoost models specifically for ML_Home and ML_Away prediction"""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    print(f"Training XGBoost models for ML_Home and ML_Away prediction with {num_iterations} iterations...")
    
    results = []
    
    for x in tqdm(range(num_iterations), desc="Training"):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=x)
        
        # Separate ML targets
        y_train_ml_home = y_train.iloc[:, 0]  # ML_Home
        y_train_ml_away = y_train.iloc[:, 1]  # ML_Away
        y_test_ml_home = y_test.iloc[:, 0]
        y_test_ml_away = y_test.iloc[:, 1]
        
        # Debug: Check target values
        if x == 0:  # Only print debug info for first iteration
            print(f"Training iteration {x} - Target shapes:")
            print(f"  ML_Home: {y_train_ml_home.shape}, range: {y_train_ml_home.min():.2f} to {y_train_ml_home.max():.2f}")
            print(f"  ML_Away: {y_train_ml_away.shape}, range: {y_train_ml_away.min():.2f} to {y_train_ml_away.max():.2f}")
        
        # Create separate XGBoost models for ML_Home and ML_Away
        xgb_ml_home = xgb.XGBRegressor(
            max_depth=8,
            eta=0.01,
            n_estimators=1000,
            random_state=x,
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        xgb_ml_away = xgb.XGBRegressor(
            max_depth=8,
            eta=0.01,
            n_estimators=1000,
            random_state=x,
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        # Train models
        xgb_ml_home.fit(x_train, y_train_ml_home)
        xgb_ml_away.fit(x_train, y_train_ml_away)
        
        # Make predictions
        pred_ml_home = xgb_ml_home.predict(x_test)
        pred_ml_away = xgb_ml_away.predict(x_test)
        
        # Calculate metrics for ML_Home
        r2_home = r2_score(y_test_ml_home, pred_ml_home)
        mse_home = mean_squared_error(y_test_ml_home, pred_ml_home)
        mae_home = mean_absolute_error(y_test_ml_home, pred_ml_home)
        
        # Calculate metrics for ML_Away
        r2_away = r2_score(y_test_ml_away, pred_ml_away)
        mse_away = mean_squared_error(y_test_ml_away, pred_ml_away)
        mae_away = mean_absolute_error(y_test_ml_away, pred_ml_away)
        
        # Calculate combined score (average R²)
        combined_score = (r2_home + r2_away) / 2
        combined_score_percent = round(combined_score * 100, 1)
        
        results.append({
            'r2_home': r2_home,
            'r2_away': r2_away,
            'mse_home': mse_home,
            'mse_away': mse_away,
            'mae_home': mae_home,
            'mae_away': mae_away,
            'combined_score': combined_score,
            'model_home': xgb_ml_home,
            'model_away': xgb_ml_away
        })
        
        # Save best models
        if combined_score == max([r['combined_score'] for r in results]):
            # Save ML_Home model
            model_home_path = f'../../Models/XGBoost_ML_Home_{combined_score_percent}%_R2-{r2_home*100:.1f}.json'
            xgb_ml_home.save_model(model_home_path)
            
            # Save ML_Away model
            model_away_path = f'../../Models/XGBoost_ML_Away_{combined_score_percent}%_R2-{r2_away*100:.1f}.json'
            xgb_ml_away.save_model(model_away_path)
            
            print(f"New best combined score: {combined_score_percent}% (ML_Home R²: {r2_home*100:.1f}%, ML_Away R²: {r2_away*100:.1f}%)")
            print(f"  ML_Home model saved to: {model_home_path}")
            print(f"  ML_Away model saved to: {model_away_path}")
    
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
        home_mse_scores = [r['mse_home'] for r in results]
        away_mse_scores = [r['mse_away'] for r in results]
        home_mae_scores = [r['mae_home'] for r in results]
        away_mae_scores = [r['mae_away'] for r in results]
        combined_scores = [r['combined_score'] for r in results]
        
        # Print results
        print(f"\nTraining completed!")
        print(f"\nML Prediction Model Results:")
        print(f"Best Combined Score: {max(combined_scores)*100:.1f}%")
        print(f"Average Combined Score: {np.mean(combined_scores)*100:.1f}%")
        print(f"Combined Score std: {np.std(combined_scores)*100:.1f}%")
        
        print(f"\nML_Home Model Results:")
        print(f"Best R²: {max(home_r2_scores)*100:.1f}%")
        print(f"Average R²: {np.mean(home_r2_scores)*100:.1f}%")
        print(f"R² std: {np.std(home_r2_scores)*100:.1f}%")
        print(f"Best MSE: {min(home_mse_scores):.2f}")
        print(f"Average MSE: {np.mean(home_mse_scores):.2f}")
        print(f"Best MAE: {min(home_mae_scores):.2f}")
        print(f"Average MAE: {np.mean(home_mae_scores):.2f}")
        
        print(f"\nML_Away Model Results:")
        print(f"Best R²: {max(away_r2_scores)*100:.1f}%")
        print(f"Average R²: {np.mean(away_r2_scores)*100:.1f}%")
        print(f"R² std: {np.std(away_r2_scores)*100:.1f}%")
        print(f"Best MSE: {min(away_mse_scores):.2f}")
        print(f"Average MSE: {np.mean(away_mse_scores):.2f}")
        print(f"Best MAE: {min(away_mae_scores):.2f}")
        print(f"Average MAE: {np.mean(away_mae_scores):.2f}")
        
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
