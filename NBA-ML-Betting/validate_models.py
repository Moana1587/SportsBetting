#!/usr/bin/env python3
"""
Validate trained models by testing their performance on sample data.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

def validate_spread_model():
    """Validate the spread regression model."""
    print("Validating Spread Model...")
    print("-" * 40)
    
    # Load data
    dataset = "dataset_2012-24_new"
    con = sqlite3.connect("Data/dataset.sqlite")
    data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
    con.close()
    
    # Prepare features
    spread = data['Spread']
    data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU', 'ML_Home', 'ML_Away', 'Spread'],
              axis=1, inplace=True)
    
    X = data.values.astype(float)
    y = spread.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if model exists
    model_files = [f for f in os.listdir("Models") if f.startswith("XGBoost_Spread_Value_") and f.endswith(".json")]
    
    if not model_files:
        print("No spread model found. Please train the model first.")
        return False
    
    # Load the best model (lowest MSE)
    best_model_file = sorted(model_files)[0]
    print(f"Loading model: {best_model_file}")
    
    model = xgb.Booster()
    model.load_model(f"Models/{best_model_file}")
    
    # Make predictions
    test_dmatrix = xgb.DMatrix(X_test)
    predictions = model.predict(test_dmatrix)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 5):")
    for i in range(5):
        print(f"  Actual: {y_test[i]:.2f}, Predicted: {predictions[i]:.2f}")
    
    return True

def validate_ml_model():
    """Validate the ML classification model."""
    print("\nValidating ML Classification Model...")
    print("-" * 40)
    
    # Load data
    dataset = "dataset_2012-24_new"
    con = sqlite3.connect("Data/dataset.sqlite")
    data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
    con.close()
    
    # Prepare features
    target = data['Home-Team-Win']
    ml_home = data['ML_Home']
    ml_away = data['ML_Away']
    
    data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU', 'ML_Home', 'ML_Away', 'Spread'],
              axis=1, inplace=True)
    
    # Add ML odds as features
    data['ML_Home_Odds'] = ml_home
    data['ML_Away_Odds'] = ml_away
    
    X = data.values.astype(float)
    y = target.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if model exists
    model_files = [f for f in os.listdir("Models") if f.startswith("XGBoost_") and "ML_Classification" in f and f.endswith(".json")]
    
    if not model_files:
        print("No ML classification model found. Please train the model first.")
        return False
    
    # Load the best model (highest accuracy)
    best_model_file = sorted(model_files, key=lambda x: float(x.split('%')[0].split('_')[-1]), reverse=True)[0]
    print(f"Loading model: {best_model_file}")
    
    model = xgb.Booster()
    model.load_model(f"Models/{best_model_file}")
    
    # Make predictions
    test_dmatrix = xgb.DMatrix(X_test)
    predictions = model.predict(test_dmatrix)
    y_pred = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 5):")
    for i in range(5):
        print(f"  Actual: {y_test[i]}, Predicted: {y_pred[i]} (Prob: {predictions[i]:.3f})")
    
    return True

def main():
    """Main validation function."""
    print("NBA Betting Model Validation")
    print("=" * 50)
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Validate models
    spread_success = validate_spread_model()
    ml_success = validate_ml_model()
    
    print(f"\n{'='*50}")
    print("Validation Summary:")
    print(f"  Spread Model: {'PASS' if spread_success else 'FAIL'}")
    print(f"  ML Model: {'PASS' if ml_success else 'FAIL'}")
    print("=" * 50)

if __name__ == "__main__":
    main()
