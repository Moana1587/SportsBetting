"""
XGBoost Model for NHL Over/Under Value Prediction

This script trains an XGBoost model to predict NHL game Over/Under values using team statistics
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

def prepare_training_data(data):
    """Prepare the data for training by selecting features and target"""
    # Target variable: OU (Over/Under value)
    target = data['OU']
    
    # Features: All team statistics and game data except target and non-predictive columns
    columns_to_drop = [
        'Score',           # Game outcome - not predictive
        'Home-Team-Win',   # Game outcome - not predictive
        'OU',              # Target variable
        'OU-Cover',        # Game outcome - not predictive
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
    
    # Debug: Check target data
    print(f"Target (OU) range: {target.min():.2f} to {target.max():.2f}")
    print(f"Target (OU) mean: {target.mean():.2f}")
    
    return features, target

def train_xgboost_model(features, target, num_iterations=100):
    """Train XGBoost model for Over/Under value prediction with multiple iterations and confidence estimation"""
    r2_results = []
    mse_results = []
    mae_results = []
    best_model = None
    best_r2 = 0
    
    print(f"Training XGBoost Over/Under value model with {num_iterations} iterations...")
    
    for x in tqdm(range(num_iterations)):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=x)
        
        # XGBoost parameters for regression - optimized for OU value prediction
        model = xgb.XGBRegressor(
            max_depth=8,
            eta=0.01,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=x,
            objective='reg:squarederror'
        )
        
        # Train model
        model.fit(x_train, y_train)
        
        # Make predictions
        predictions = model.predict(x_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = np.mean(np.abs(y_test - predictions))
        
        r2_percent = round(r2 * 100, 1)
        r2_results.append(r2)
        mse_results.append(mse)
        mae_results.append(mae)
        
        # Save best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            model_path = f'../../Models/XGBoost_{r2_percent}%_UO-9.json'
            model.save_model(model_path)
            print(f"New best R²: {r2_percent}% - Model saved to {model_path}")
    
    return r2_results, mse_results, mae_results, best_model

def create_ensemble_models(features, target, num_models=10):
    """Create an ensemble of models for confidence estimation"""
    print(f"Creating ensemble of {num_models} models for confidence estimation...")
    
    ensemble_models = []
    
    for i in tqdm(range(num_models)):
        # Split data with different random states
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.1, random_state=i*42
        )
        
        # XGBoost parameters for regression
        model = xgb.XGBRegressor(
            max_depth=8,
            eta=0.01,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=i*42,
            objective='reg:squarederror'
        )
        
        # Train model
        model.fit(x_train, y_train)
        ensemble_models.append(model)
    
    return ensemble_models

def predict_with_confidence(model, ensemble_models, features):
    """Make predictions with confidence intervals using ensemble"""
    # Get prediction from main model
    main_prediction = model.predict(features)
    
    # Get predictions from ensemble
    ensemble_predictions = []
    for ensemble_model in ensemble_models:
        pred = ensemble_model.predict(features)
        ensemble_predictions.append(pred)
    
    # Calculate confidence based on prediction variance
    ensemble_predictions = np.array(ensemble_predictions)
    prediction_std = np.std(ensemble_predictions, axis=0)
    prediction_mean = np.mean(ensemble_predictions, axis=0)
    
    # Calculate confidence percentage (higher confidence = lower variance)
    # Normalize to 0-100% range
    max_std = np.max(prediction_std) if len(prediction_std) > 0 else 1.0
    confidence_percentage = np.clip((1 - prediction_std / max_std) * 100, 0, 100)
    
    return main_prediction, confidence_percentage

def main():
    """Main training function"""
    try:
        # Load dataset
        data = load_dataset()
        
        # Prepare training data
        features, target = prepare_training_data(data)
        
        # Train main model
        r2_results, mse_results, mae_results, best_model = train_xgboost_model(features, target)
        
        # Create ensemble for confidence estimation
        ensemble_models = create_ensemble_models(features, target, num_models=10)
        
        # Save ensemble models
        ensemble_dir = "../../Models/Ensemble_UO"
        os.makedirs(ensemble_dir, exist_ok=True)
        
        for i, model in enumerate(ensemble_models):
            model_path = os.path.join(ensemble_dir, f"ensemble_model_{i}.json")
            model.save_model(model_path)
        
        print(f"Saved {len(ensemble_models)} ensemble models to {ensemble_dir}")
        
        # Test confidence prediction on a sample
        if best_model is not None:
            sample_features = features.iloc[:5]  # Test on first 5 samples
            predictions, confidence = predict_with_confidence(best_model, ensemble_models, sample_features)
            
            print(f"\nSample predictions with confidence:")
            for i in range(len(sample_features)):
                print(f"  Sample {i+1}: OU={predictions[i]:.2f}, Confidence={confidence[i]:.1f}%")
        
        # Print results
        print(f"\nTraining completed!")
        print(f"Best R²: {max(r2_results)*100:.1f}%")
        print(f"Average R²: {np.mean(r2_results)*100:.1f}%")
        print(f"R² std: {np.std(r2_results)*100:.1f}%")
        print(f"Best MSE: {min(mse_results):.4f}")
        print(f"Average MSE: {np.mean(mse_results):.4f}")
        print(f"Best MAE: {min(mae_results):.4f}")
        print(f"Average MAE: {np.mean(mae_results):.4f}")
        
        # Print target statistics
        print(f"\nTarget (OU) statistics:")
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
        print("\nXGBoost Over/Under value model training completed successfully!")
    else:
        print("\nXGBoost Over/Under value model training failed!")
