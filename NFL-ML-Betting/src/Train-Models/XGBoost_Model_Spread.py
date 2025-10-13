import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Updated dataset name to match Create_Games.py output
dataset = "dataset_nfl_new"
con = sqlite3.connect("../../Data/dataset.sqlite")

try:
    data = pd.read_sql_query(f"SELECT * FROM `{dataset}`", con)
    print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Available tables:")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
    print(tables)
    con.close()
    exit(1)

con.close()

# Extract target variable - use actual spread data from OddsData.sqlite
# The Spread column contains the actual betting spread for each game
spread_data = data['Spread']

print(f"Using actual spread data from OddsData.sqlite")
print(f"Spread range: {spread_data.min():.1f} to {spread_data.max():.1f}")
print(f"Spread distribution:")
print(f"  Home favorites (positive): {sum(1 for s in spread_data if s > 0)} games")
print(f"  Away favorites (negative): {sum(1 for s in spread_data if s < 0)} games")
print(f"  Pick'em (zero): {sum(1 for s in spread_data if s == 0)} games")
print(f"  Missing spreads: {sum(1 for s in spread_data if pd.isna(s))} games")

# Define columns to exclude from features
exclude_columns = [
    'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
    'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
    'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
    'Spread'  # Exclude the target variable
]

# Remove non-feature columns
feature_columns = [col for col in data.columns if col not in exclude_columns]
data_features = data[feature_columns]

print(f"Using {len(feature_columns)} features for spread prediction")
print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10 features

# Convert features to numpy array and ensure proper data types
X = data_features.values
y = spread_data.values

# Handle any non-numeric values by converting to numeric and filling NaN
X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values
X = X.astype(float)

# Handle missing spread values
y = pd.Series(y).fillna(0).values  # Fill missing spreads with 0 (pick'em)
y = y.astype(float)

print(f"Data shape: {X.shape}")
print(f"Target (spread) statistics:")
print(f"  Mean: {np.mean(y):.2f}")
print(f"  Std: {np.std(y):.2f}")
print(f"  Min: {np.min(y):.2f}")
print(f"  Max: {np.max(y):.2f}")

# Check for any remaining issues
if np.isnan(X).any():
    print("Warning: NaN values found in features, replacing with 0")
    X = np.nan_to_num(X)

if np.isinf(X).any():
    print("Warning: Infinite values found in features, replacing with 0")
    X = np.nan_to_num(X, posinf=0, neginf=0)

if np.isnan(y).any():
    print("Warning: NaN values found in target, replacing with 0")
    y = np.nan_to_num(y)

mse_results = []
mae_results = []
best_mse = float('inf')

print("Starting XGBoost spread prediction training...")
for x in tqdm(range(200), desc="Training iterations"):
    try:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=x)
        
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        # Parameters optimized for regression (spread prediction)
        param = {
            'max_depth': 6,  # Deeper trees for regression
            'eta': 0.1,      # Learning rate
            'objective': 'reg:squarederror',  # Regression objective
            'eval_metric': 'rmse',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': x
        }
        epochs = 300

        model = xgb.train(param, train, epochs, verbose_eval=False)
        predictions = model.predict(test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # Only print every 25 iterations to reduce output
        if x % 25 == 0 or mse < best_mse:
            print(f"Iteration {x}: MSE={mse:.3f}, MAE={mae:.3f}")
        
        mse_results.append(mse)
        mae_results.append(mae)
        
        # Save model if it's the best so far (lowest MSE)
        if mse < best_mse:
            best_mse = mse
            # Ensure the directory exists
            import os
            model_dir = '../../Models/XGBoost_Models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/XGBoost_{mae:.1f}_MAE_Spread.json'
            model.save_model(model_path)
            print(f"New best spread model saved: {model_path} (MSE: {mse:.3f}, MAE: {mae:.3f})")
            
    except Exception as e:
        print(f"Iteration {x}: Error - {e}")
        continue

# Print final results
print(f"\nSpread prediction training completed!")
print(f"Best MSE: {min(mse_results):.3f}")
print(f"Best MAE: {min(mae_results):.3f}")
print(f"Average MSE: {np.mean(mse_results):.3f}")
print(f"Average MAE: {np.mean(mae_results):.3f}")
print(f"MSE range: {min(mse_results):.3f} - {max(mse_results):.3f}")
print(f"MAE range: {min(mae_results):.3f} - {max(mae_results):.3f}")

# Additional analysis
print(f"\nSpread prediction analysis:")
print(f"Model performance indicates average prediction error of {np.mean(mae_results):.2f} points")
print(f"This means the model typically predicts spreads within {np.mean(mae_results):.2f} points of actual")
print(f"For reference, NFL spreads typically range from -14 to +14 points")
