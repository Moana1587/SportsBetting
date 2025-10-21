#!/usr/bin/env python3
"""
Quick test script to verify the spread model training works.
"""

import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def test_spread_model():
    """Test the spread model with a small sample."""
    print("Testing Spread Model Training...")
    print("=" * 40)
    
    # Load data
    dataset = "dataset_2012-24_new"
    con = sqlite3.connect("Data/dataset.sqlite")
    data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
    con.close()
    
    print(f"Loaded dataset with {len(data)} records")
    print(f"Columns: {list(data.columns)}")
    
    # Check if spread column exists
    if 'Spread' not in data.columns:
        print("❌ Error: Spread column not found in dataset!")
        return False
    
    # Use Spread as target variable
    spread = data['Spread']
    print(f"Spread statistics:")
    print(f"  Mean: {spread.mean():.3f}")
    print(f"  Std: {spread.std():.3f}")
    print(f"  Min: {spread.min():.3f}")
    print(f"  Max: {spread.max():.3f}")
    
    # Prepare features
    data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU', 'ML_Home', 'ML_Away', 'Spread'],
              axis=1, inplace=True)
    
    print(f"Features shape: {data.shape}")
    
    # Convert to numpy arrays
    X = data.values.astype(float)
    y = spread.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train a simple model
    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)
    
    param = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    model = xgb.train(param, train, 100, evals=[(test, 'eval')])
    predictions = model.predict(test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nModel Performance:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    
    # Show some predictions
    print(f"\nSample Predictions:")
    for i in range(5):
        print(f"  Actual: {y_test[i]:.2f}, Predicted: {predictions[i]:.2f}")
    
    print("\n[SUCCESS] Spread model test completed successfully!")
    return True

if __name__ == "__main__":
    test_spread_model()
