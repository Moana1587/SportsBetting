"""
XGBoost Model for NHL Spread Value Prediction

This script trains an XGBoost model to predict NHL game spread values using team statistics
and game data from the updated dataset structure.
"""

import sqlite3
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
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
    
    print(f"Loaded dataset with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    return data

def validate_spread_data(data):
    """
    Validate and display spread data statistics
    
    Args:
        data: DataFrame with 'Spread' column
    
    Returns:
        Boolean indicating if spread data is valid
    """
    if 'Spread' not in data.columns:
        print("Warning: 'Spread' column not found in dataset.")
        return False
    
    spread_data = data['Spread'].dropna()
    
    if len(spread_data) == 0:
        print("Warning: No valid spread data found.")
        return False
    
    print(f"Spread data validation:")
    print(f"  Total games: {len(data)}")
    print(f"  Valid spread data: {len(spread_data)}")
    print(f"  Min spread: {spread_data.min():.2f}")
    print(f"  Max spread: {spread_data.max():.2f}")
    print(f"  Mean spread: {spread_data.mean():.2f}")
    print(f"  Std spread: {spread_data.std():.2f}")
    
    return True

def prepare_training_data(data):
    """Prepare the data for training by selecting features and target"""
    # Validate spread data
    if not validate_spread_data(data):
        raise ValueError("Invalid spread data. Cannot proceed with training.")
    
    # Target variable: Spread value
    target = data['Spread']
    
    # Features: All team statistics and game data except target and non-predictive columns
    columns_to_drop = [
        'Score',           # Game outcome - not predictive
        'Home-Score',      # Game outcome - not predictive
        'Away-Score',      # Game outcome - not predictive
        'Home-Team-Win',   # Game outcome - not predictive
        'OU-Cover',        # Game outcome - not predictive
        'OU',              # Game outcome - not predictive
        'Spread',          # Target variable
        'teamFullName',    # Team name - not predictive
        'teamFullName.1',  # Away team name - not predictive
        'Season',          # Season - not predictive for individual games
        'Season.1',        # Season - not predictive for individual games
        'NHL_SeasonId',    # Season ID - not predictive
        'NHL_SeasonId.1',  # Season ID - not predictive
        'teamId',          # Team ID - not predictive
        'teamId.1',        # Team ID - not predictive
        'ML_Home',         # Moneyline odds for home team - betting data, not team performance
        'ML_Away',         # Moneyline odds for away team - betting data, not team performance
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
    
    # Debug: Check target data
    print(f"Target (Spread) range: {target.min():.2f} to {target.max():.2f}")
    print(f"Target (Spread) mean: {target.mean():.2f}")
    
    return features, target

def train_xgboost_model(features, target, num_iterations=300):
    """Train XGBoost model for spread value prediction with multiple iterations"""
    r2_results = []
    
    print(f"Training XGBoost Spread value model with {num_iterations} iterations...")
    
    for x in tqdm(range(num_iterations)):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=x)
        
        # Create DMatrix for XGBoost
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test)
        
        # XGBoost parameters for regression
        param = {
            'max_depth': 6,
            'eta': 0.01,
            'objective': 'reg:squarederror',
            'random_state': x
        }
        epochs = 750
        
        # Train model
        model = xgb.train(param, train, epochs)
        
        # Make predictions
        predictions = model.predict(test)
        
        # Calculate R² score
        r2 = r2_score(y_test, predictions)
        r2_percent = round(r2 * 100, 1)
        r2_results.append(r2)
        
        # Save best model
        if r2 == max(r2_results):
            model_path = f'../../Models/XGBoost_{r2_percent}%_Spread-4.json'
            model.save_model(model_path)
            print(f"New best R²: {r2_percent}% - Model saved to {model_path}")
    
    return r2_results

def main():
    """Main training function"""
    try:
        # Load dataset
        data = load_dataset()
        
        # Prepare training data
        features, target = prepare_training_data(data)
        
        # Train model
        r2_results = train_xgboost_model(features, target)
        
        # Print results
        print(f"\nTraining completed!")
        print(f"Best R²: {max(r2_results)*100:.1f}%")
        print(f"Average R²: {np.mean(r2_results)*100:.1f}%")
        print(f"R² std: {np.std(r2_results)*100:.1f}%")
        
        # Print target statistics
        print(f"\nTarget (Spread) statistics:")
        print(f"Min: {target.min():.2f}")
        print(f"Max: {target.max():.2f}")
        print(f"Mean: {target.mean():.2f}")
        print(f"Std: {target.std():.2f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ XGBoost Spread value model training completed successfully!")
    else:
        print("\n❌ XGBoost Spread value model training failed!")
