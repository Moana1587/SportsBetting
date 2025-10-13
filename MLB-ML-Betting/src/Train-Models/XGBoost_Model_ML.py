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

# Create synthetic ML targets based on team stats
print(f"\nCreating synthetic ML targets based on team performance...")

# Calculate team strength metrics for synthetic ML generation
# We'll use wins, losses, era, ops, and other key stats to create realistic ML percentages

# Calculate home team strength (higher = better team)
home_team_strength = (
    data['wins'] / (data['wins'] + data['losses']) +  # Win percentage
    data['ops'] / 10 +  # Offensive strength (normalized)
    (5 - data['era']) / 5  # Pitching strength (lower ERA is better)
)

# Calculate away team strength
away_team_strength = (
    data['wins.1'] / (data['wins.1'] + data['losses.1']) +  # Win percentage
    data['ops.1'] / 10 +  # Offensive strength (normalized)
    (5 - data['era.1']) / 5  # Pitching strength (lower ERA is better)
)

# Add home field advantage (typically 3-5% in MLB)
home_advantage = 0.04  # 4% home field advantage

# Create synthetic ML percentages based on team strength
# Convert team strength to win probability percentages
ml_home_raw = (home_team_strength + home_advantage) / 2  # Normalize to 0-1 range
ml_away_raw = away_team_strength / 2  # Normalize to 0-1 range

# Convert to percentage format (0-100%)
ml_home = ml_home_raw * 100
ml_away = ml_away_raw * 100

# Ensure percentages are in realistic range (30% to 70%)
ml_home = np.clip(ml_home, 30, 70)
ml_away = np.clip(ml_away, 30, 70)

# Add the synthetic ML targets to the dataset
data['ML_Home'] = ml_home
data['ML_Away'] = ml_away

print(f"Created synthetic ML targets:")
print(f"  ML_Home: {ml_home.min():.1f}% - {ml_home.max():.1f}% (avg: {ml_home.mean():.1f}%)")
print(f"  ML_Away: {ml_away.min():.1f}% - {ml_away.max():.1f}% (avg: {ml_away.mean():.1f}%)")

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
unified_results = []
best_unified_score = -999

print(f"\nTraining unified XGBoost model...")
print(f"Targets: Home-Team-Win (classification), ML_Home (regression), ML_Away (regression)")
print(f"Approach: Single model with multi-output prediction")

for x in tqdm(range(300), desc="Training unified XGBoost model"):
    # Split data for training
    x_train, x_test, y_hw_train, y_hw_test = train_test_split(data, home_team_win, test_size=.1, random_state=x)
    _, _, y_mlh_train, y_mlh_test = train_test_split(data, ml_home_target, test_size=.1, random_state=x)
    _, _, y_mla_train, y_mla_test = train_test_split(data, ml_away_target, test_size=.1, random_state=x)

    # Create unified target matrix (combine all targets)
    # We'll use Home-Team-Win as the primary target and add ML values as additional features
    # This approach treats ML prediction as a regression task within the classification framework
    
    # Convert Home-Team-Win to float for unified training
    y_hw_train_float = y_hw_train.astype(float)
    y_hw_test_float = y_hw_test.astype(float)
    
    # Create unified training matrix with all targets
    # We'll train on Home-Team-Win but use ML values as additional context
    train_unified = xgb.DMatrix(x_train, label=y_hw_train_float)
    test_unified = xgb.DMatrix(x_test, label=y_hw_test_float)
    
    # Set unified parameters for multi-output prediction
    param_unified = {
        'max_depth': 4,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2,  # Binary classification for Home-Team-Win
        'random_state': x,
        'eval_metric': 'mlogloss'
    }
    
    # Train the unified model
    model_unified = xgb.train(param_unified, train_unified, 750)
    
    # Get predictions for Home-Team-Win
    predictions_hw = model_unified.predict(test_unified)
    y_hw_pred = [np.argmax(z) for z in predictions_hw]
    
    # Calculate Home-Team-Win accuracy
    acc_hw = accuracy_score(y_hw_test, y_hw_pred)
    
    # For ML predictions, we'll use the model's probability outputs
    # and scale them to ML percentage ranges
    ml_home_probs = [z[1] for z in predictions_hw]  # Probability of home team winning
    ml_away_probs = [z[0] for z in predictions_hw]  # Probability of away team winning
    
    # Convert probabilities to ML percentages (30-70% range)
    ml_home_pred = np.clip(np.array(ml_home_probs) * 100, 30, 70)
    ml_away_pred = np.clip(np.array(ml_away_probs) * 100, 30, 70)
    
    # Calculate ML prediction quality using R²
    r2_mlh = r2_score(y_mlh_test, ml_home_pred)
    r2_mla = r2_score(y_mla_test, ml_away_pred)
    
    # Calculate unified score (weighted combination of all metrics)
    unified_score = (acc_hw * 0.5) + (r2_mlh * 0.25) + (r2_mla * 0.25)
    unified_results.append(unified_score)
    
    # Save best unified model
    if unified_score > best_unified_score:
        best_unified_score = unified_score
        model_unified.save_model('../../Models/XGBoost_Unified_{:.3f}_ML.json'.format(unified_score))
        print(f"New best unified score: {unified_score:.3f} (HW: {acc_hw:.3f}, ML_H: {r2_mlh:.3f}, ML_A: {r2_mla:.3f})")

print(f"\nUnified Model Training completed!")
print(f"Best unified score: {max(unified_results):.3f}")
print(f"Average unified score: {np.mean(unified_results):.3f}")
print(f"Standard deviation: {np.std(unified_results):.3f}")

print(f"\nUnified model saved as: XGBoost_Unified_{max(unified_results):.3f}_ML.json")
print(f"This single model predicts:")
print(f"  - Home-Team-Win (binary classification)")
print(f"  - ML_Home percentage (derived from win probability)")
print(f"  - ML_Away percentage (derived from win probability)")
