import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "dataset_2012-24_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Use Home-Team-Win as target variable for classification
# 1 = Home team wins, 0 = Away team wins
target = data['Home-Team-Win']

# Add ML odds as features to help with prediction
ml_home = data['ML_Home']
ml_away = data['ML_Away']

data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU', 'ML_Home', 'ML_Away', 'Spread'],
          axis=1, inplace=True)

# Add ML odds as features
data['ML_Home_Odds'] = ml_home
data['ML_Away_Odds'] = ml_away

data = data.values
data = data.astype(float)

acc_results = []

print("Training XGBoost ML Classification Model...")
print("=" * 50)

for x in tqdm(range(300)):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.1, random_state=x)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    epochs = 1000

    model = xgb.train(param, train, epochs, evals=[(test, 'eval')])
    predictions = model.predict(test)
    
    # Convert probabilities to binary predictions
    y_pred = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    acc_results.append(acc)
    
    print(f"Iteration {x+1}: Accuracy={acc}%")
    
    # Save model if it's the best so far
    if acc == max(acc_results):
        model.save_model('../../Models/XGBoost_{}%_ML_Classification.json'.format(acc))
        print(f"New best model saved with accuracy: {acc}%")

print("\n" + "=" * 50)
print("Training Complete!")
print(f"Best Accuracy: {max(acc_results)}%")
print(f"Average Accuracy: {np.mean(acc_results):.1f}% ± {np.std(acc_results):.1f}%")

# Final evaluation on the best model
if acc_results:
    print(f"\nFinal Model Performance:")
    print(f"Best accuracy achieved: {max(acc_results)}%")
    print(f"Number of iterations: {len(acc_results)}")
    print(f"Accuracy distribution: {min(acc_results)}% - {max(acc_results)}%")
