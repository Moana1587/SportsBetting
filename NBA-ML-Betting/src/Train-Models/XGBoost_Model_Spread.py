import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "dataset_2012-24_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Use Spread as target variable for regression
spread = data['Spread']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU', 'ML_Home', 'ML_Away', 'Spread'],
          axis=1, inplace=True)

data = data.values
data = data.astype(float)

mse_results = []
mae_results = []
r2_results = []

print("Training XGBoost Spread Regression Model...")
print("=" * 50)

for x in tqdm(range(200)):
    x_train, x_test, y_train, y_test = train_test_split(data, spread, test_size=.1, random_state=x)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    epochs = 1000

    model = xgb.train(param, train, epochs, evals=[(test, 'eval')])
    predictions = model.predict(test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    mse_results.append(mse)
    mae_results.append(mae)
    r2_results.append(r2)
    
    print(f"Iteration {x+1}: MSE={mse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
    
    # Save model if it's the best so far (lowest MSE)
    if mse == min(mse_results):
        model.save_model('../../Models/XGBoost_Spread_Value_{:.3f}_MSE.json'.format(mse))
        print(f"New best model saved with MSE: {mse:.3f}")

print("\n" + "=" * 50)
print("Training Complete!")
print(f"Best MSE: {min(mse_results):.3f}")
print(f"Best MAE: {min(mae_results):.3f}")
print(f"Best R²: {max(r2_results):.3f}")
print(f"Average MSE: {np.mean(mse_results):.3f} ± {np.std(mse_results):.3f}")
print(f"Average MAE: {np.mean(mae_results):.3f} ± {np.std(mae_results):.3f}")
print(f"Average R²: {np.mean(r2_results):.3f} ± {np.std(r2_results):.3f}")
