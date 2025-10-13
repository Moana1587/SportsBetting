import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "mlb_dataset"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con)
con.close()

print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
print(f"Columns: {list(data.columns)}")

OU = data['OU-Cover']
total = data['OU']

# Updated column names to match new data structure
columns_to_drop = ['Score', 'Home-Team-Win', 'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away']
# Also drop any team name or date columns that might exist
for col in data.columns:
    if 'TEAM_NAME' in col or 'Date' in col or 'teamName' in col or 'season' in col:
        columns_to_drop.append(col)

# Remove duplicates and only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in data.columns]
print(f"Dropping columns: {columns_to_drop}")

data.drop(columns_to_drop, axis=1, inplace=True)

# Add OU total back as a feature
data['OU'] = np.asarray(total)

print(f"Final feature matrix shape: {data.shape}")
print(f"Target variable distribution: {OU.value_counts().to_dict()}")

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

acc_results = []
best_accuracy = 0

for x in tqdm(range(100), desc="Training XGBoost UO models"):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1, random_state=x)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test)

    param = {
        'max_depth': 20,
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': x
    }
    epochs = 750

    model = xgb.train(param, train, epochs)

    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y) * 100, 1)
    acc_results.append(acc)
    
    # only save results if they are the best so far
    if acc > best_accuracy:
        best_accuracy = acc
        model.save_model('../../Models/XGBoost_{}%_UO-9.json'.format(acc))
        print(f"New best accuracy: {acc}% (saved model)")

print(f"\nTraining completed!")
print(f"Best accuracy achieved: {max(acc_results)}%")
print(f"Average accuracy: {np.mean(acc_results):.1f}%")
print(f"Standard deviation: {np.std(acc_results):.1f}%")
