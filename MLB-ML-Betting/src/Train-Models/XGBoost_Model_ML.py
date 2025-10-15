import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "mlb_dataset"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con)
con.close()

print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
print(f"Columns: {list(data.columns)}")

# Use real ML data from the dataset
print(f"\nUsing real ML data from dataset...")

# Check if ML_Home and ML_Away columns exist and have data
if 'ML_Home' in data.columns and 'ML_Away' in data.columns:
    # Filter out rows with missing ML data
    ml_data_mask = data['ML_Home'].notna() & data['ML_Away'].notna()
    data = data[ml_data_mask].copy()
    
    print(f"Using real ML data:")
    print(f"  ML_Home: {data['ML_Home'].min():.1f} to {data['ML_Home'].max():.1f} (avg: {data['ML_Home'].mean():.1f})")
    print(f"  ML_Away: {data['ML_Away'].min():.1f} to {data['ML_Away'].max():.1f} (avg: {data['ML_Away'].mean():.1f})")
    print(f"  Total games with ML data: {len(data)}")
else:
    print("Error: ML_Home and ML_Away columns not found in dataset!")
    exit(1)

# Set up target variables
home_team_win = data['Home-Team-Win']
ml_home_target = data['ML_Home']
ml_away_target = data['ML_Away']

print(f"\nTarget variables:")
print(f"  Home-Team-Win distribution: {home_team_win.value_counts().to_dict()}")
print(f"  ML_Home range: {ml_home_target.min():.1f}% - {ml_home_target.max():.1f}%")
print(f"  ML_Away range: {ml_away_target.min():.1f}% - {ml_away_target.max():.1f}%")
# Updated column names to match new data structure
columns_to_drop = ['Score', 'Home-Team-Win', 'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away', 'ML_Home', 'ML_Away']
# Also drop any team name or date columns that might exist
for col in data.columns:
    if 'TEAM_NAME' in col or 'Date' in col or 'teamName' in col or 'season' in col:
        columns_to_drop.append(col)

# Remove duplicates and only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in data.columns]
print(f"Dropping columns: {columns_to_drop}")

data.drop(columns_to_drop, axis=1, inplace=True)

print(f"Final feature matrix shape: {data.shape}")
print(f"Features: {list(data.columns)}")

# Check for any remaining non-numeric columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Warning: Found non-numeric columns: {list(non_numeric_cols)}")
    data = data.select_dtypes(include=[np.number])

data = data.values

# Handle any NaN values
if np.isnan(data).any():
    print("Warning: Found NaN values in data, filling with 0")
    data = np.nan_to_num(data, nan=0.0)

data = data.astype(float)
print(f"Training data shape: {data.shape}")

# Training results storage
ml_home_results = []
ml_away_results = []
best_ml_home_score = -999
best_ml_away_score = -999

print(f"\nTraining separate XGBoost regression models for ML prediction...")
print(f"Targets: ML_Home (regression), ML_Away (regression)")
print(f"Approach: Separate regression models for each ML target")

# Train ML_Home model
print(f"\nTraining ML_Home regression model...")
for x in tqdm(range(100), desc="Training ML_Home model"):
    # Split data for training
    x_train, x_test, y_mlh_train, y_mlh_test = train_test_split(data, ml_home_target, test_size=.1, random_state=x)
    
    # Create DMatrix for XGBoost
    train_mlh = xgb.DMatrix(x_train, label=y_mlh_train)
    test_mlh = xgb.DMatrix(x_test)
    
    # Set parameters for regression
    param_mlh = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'reg:squarederror',
        'random_state': x
    }
    
    # Train the model
    model_mlh = xgb.train(param_mlh, train_mlh, 1000)
    
    # Get predictions
    predictions_mlh = model_mlh.predict(test_mlh)
    
    # Calculate R² score
    r2_mlh = r2_score(y_mlh_test, predictions_mlh)
    ml_home_results.append(r2_mlh)
    
    # Save best ML_Home model
    if r2_mlh > best_ml_home_score:
        best_ml_home_score = r2_mlh
        model_mlh.save_model('../../Models/XGBoost_ML_Home_{:.3f}_R2.json'.format(r2_mlh))
        print(f"New best ML_Home R²: {r2_mlh:.3f}")

# Train ML_Away model
print(f"\nTraining ML_Away regression model...")
for x in tqdm(range(100), desc="Training ML_Away model"):
    # Split data for training
    x_train, x_test, y_mla_train, y_mla_test = train_test_split(data, ml_away_target, test_size=.1, random_state=x)
    
    # Create DMatrix for XGBoost
    train_mla = xgb.DMatrix(x_train, label=y_mla_train)
    test_mla = xgb.DMatrix(x_test)
    
    # Set parameters for regression
    param_mla = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'reg:squarederror',
        'random_state': x
    }
    
    # Train the model
    model_mla = xgb.train(param_mla, train_mla, 1000)
    
    # Get predictions
    predictions_mla = model_mla.predict(test_mla)
    
    # Calculate R² score
    r2_mla = r2_score(y_mla_test, predictions_mla)
    ml_away_results.append(r2_mla)
    
    # Save best ML_Away model
    if r2_mla > best_ml_away_score:
        best_ml_away_score = r2_mla
        model_mla.save_model('../../Models/XGBoost_ML_Away_{:.3f}_R2.json'.format(r2_mla))
        print(f"New best ML_Away R²: {r2_mla:.3f}")

print(f"\nML Model Training completed!")
print(f"ML_Home - Best R²: {max(ml_home_results):.3f}, Average: {np.mean(ml_home_results):.3f}")
print(f"ML_Away - Best R²: {max(ml_away_results):.3f}, Average: {np.mean(ml_away_results):.3f}")

print(f"\nModels saved as:")
print(f"  - XGBoost_ML_Home_{max(ml_home_results):.3f}_R2.json")
print(f"  - XGBoost_ML_Away_{max(ml_away_results):.3f}_R2.json")
print(f"These models predict actual ML odds values based on team statistics.")
