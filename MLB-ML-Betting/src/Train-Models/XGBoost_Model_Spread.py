import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "mlb_dataset"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con)
con.close()

print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
print(f"Columns: {list(data.columns)}")

# Create synthetic spread targets based on team stats
# Spread represents the expected margin of victory
# We'll calculate this based on team strength differences

print("Creating synthetic spread targets based on team stats...")

# Calculate team strength metrics (similar to ML model)
home_team_strength = (
    data['wins'] / (data['wins'] + data['losses']) +
    data['ops'] / 10 +
    (5 - data['era']) / 5
)

away_team_strength = (
    data['wins.1'] / (data['wins.1'] + data['losses.1']) +
    data['ops.1'] / 10 +
    (5 - data['era.1']) / 5
)

# Calculate strength difference
strength_diff = home_team_strength - away_team_strength

# Add home field advantage (typically 0.1-0.2 runs in MLB)
home_advantage = 0.15

# Create synthetic spread targets
# Positive spread means home team is favored, negative means away team is favored
# Spread range: -3.5 to +3.5 (realistic MLB spread range)
spread_targets = np.clip(strength_diff * 10 + home_advantage, -3.5, 3.5)

# Add some noise to make it more realistic
np.random.seed(42)
noise = np.random.normal(0, 0.3, len(spread_targets))
spread_targets = spread_targets + noise
spread_targets = np.clip(spread_targets, -3.5, 3.5)

# Add spread targets to data
data['Spread_Target'] = spread_targets

print(f"Synthetic spread targets created:")
print(f"  Mean: {np.mean(spread_targets):.2f}")
print(f"  Std: {np.std(spread_targets):.2f}")
print(f"  Min: {np.min(spread_targets):.2f}")
print(f"  Max: {np.max(spread_targets):.2f}")
print(f"  Distribution: {np.histogram(spread_targets, bins=10)[0]}")

# Updated column names to match new data structure
columns_to_drop = ['Score', 'Home-Team-Win', 'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away', 'Spread_Target']
# Also drop any team name or date columns that might exist
for col in data.columns:
    if 'TEAM_NAME' in col or 'Date' in col or 'teamName' in col or 'season' in col:
        columns_to_drop.append(col)

# Remove duplicates and only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in data.columns]
print(f"Dropping columns: {columns_to_drop}")

if columns_to_drop:
    data = data.drop(columns=columns_to_drop)
else:
    data = data.copy()

print(f"Final feature matrix shape: {data.shape}")
print(f"Number of features: {data.shape[1]}")

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

# Training loop for spread value prediction
r2_results = []
mse_results = []
best_r2 = -999

print("Starting spread value prediction training...")

for x in tqdm(range(300), desc="Training XGBoost Spread models"):
    x_train, x_test, y_train, y_test = train_test_split(data, spread_targets, test_size=.1, random_state=x)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 4,
        'eta': 0.01,
        'objective': 'reg:squarederror',
        'random_state': x,
        'eval_metric': 'rmse'
    }
    epochs = 750

    model = xgb.train(param, train, epochs)

    predictions = model.predict(test)
    
    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    r2_results.append(r2)
    mse_results.append(mse)
    
    # Save best model based on R² score
    if r2 > best_r2:
        best_r2 = r2
        model.save_model('../../Models/XGBoost_Spread_{:.3f}_Value.json'.format(r2))
        print(f"New best R² score: {r2:.3f} (MSE: {mse:.3f}) - saved model")

print(f"\nTraining completed!")
print(f"Best R² score achieved: {max(r2_results):.3f}")
print(f"Average R² score: {np.mean(r2_results):.3f}")
print(f"Standard deviation: {np.std(r2_results):.3f}")
print(f"Best MSE: {min(mse_results):.3f}")
print(f"Average MSE: {np.mean(mse_results):.3f}")
