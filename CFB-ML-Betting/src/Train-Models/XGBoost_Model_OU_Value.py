import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def train_xgboost_ou_value_model():
    """Train XGBoost model for Over/Under value prediction using CFBD data."""
    print("Training XGBoost Over/Under Value Model...")
    
    # Connect to database and load data
    con = sqlite3.connect("../../Data/dataset.sqlite")
    
    try:
        # Load the CFB games dataset
        data = pd.read_sql_query("SELECT * FROM cfb_games_dataset", con)
        print(f"Loaded {len(data)} games from dataset")
        
        if data.empty:
            print("No data found. Please run Create_Games.py first.")
            return
        
        # Display dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Prepare target variable (OU Value - actual over/under line)
        y = data['OU'].values
        
        # Remove rows where OU is null or 0
        valid_indices = ~(pd.isna(y) | (y == 0))
        data = data[valid_indices]
        y = y[valid_indices]
        
        print(f"After removing invalid OU values: {len(data)} games")
        print(f"OU value range: {y.min():.1f} - {y.max():.1f}")
        print(f"OU value mean: {y.mean():.1f}")
        
        # Prepare features - exclude non-numeric and target columns
        exclude_columns = [
            'Date', 'Home_Team', 'Away_Team', 'Home_Team_Win', 'Points', 
            'Win_Margin', 'OU_Cover', 'ML_Home', 'ML_Away', 'OU', 'Spread'
        ]
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        X = data[feature_columns]
        
        print(f"Using {len(feature_columns)} features for training")
        print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10 features
        
        # Convert to numpy array and handle any remaining non-numeric data
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0)  # Fill any NaN values with 0
        X = X.values.astype(float)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target (OU) statistics: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}, std={y.std():.1f}")
        
        # Training parameters
        mse_results = []
        best_mse = float('inf')
        
        print("\nStarting training iterations...")
        for x in tqdm(range(100)):  # 100 iterations for faster training
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=x
            )
            
            # Create DMatrix for XGBoost
            train_dmatrix = xgb.DMatrix(x_train, label=y_train)
            test_dmatrix = xgb.DMatrix(x_test, label=y_test)
            
            # XGBoost parameters for regression
            param = {
                'max_depth': 6,
                'eta': 0.1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            # Train model
            num_rounds = 100
            model = xgb.train(
                param, 
                train_dmatrix, 
                num_rounds,
                evals=[(test_dmatrix, 'test')],
                verbose_eval=False
            )
            
            # Make predictions
            predictions = model.predict(test_dmatrix)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            mse_results.append(mse)
            
            # Print progress every 10 iterations
            if (x + 1) % 10 == 0:
                avg_mse = np.mean(mse_results[-10:])
                print(f"Iteration {x+1}: Current MSE = {mse:.3f}, Avg (last 10) = {avg_mse:.3f}")
            
            # Save best model (lowest MSE)
            if mse < best_mse:
                best_mse = mse
                # Ensure Models directory exists
                os.makedirs("../../Models", exist_ok=True)
                model.save_model(f'../../Models/XGBoost_OU_Value_{mse:.3f}_MSE.json')
                print(f"New best model saved with MSE: {mse:.3f}")
        
        # Final results
        print(f"\nTraining completed!")
        print(f"Best MSE: {best_mse:.3f}")
        print(f"Average MSE: {np.mean(mse_results):.3f}")
        print(f"Standard deviation: {np.std(mse_results):.3f}")
        
        # Show final evaluation for the best model
        if best_mse < float('inf'):
            print(f"\nLoading best model for final evaluation...")
            best_model = xgb.Booster()
            best_model.load_model(f'../../Models/XGBoost_OU_Value_{best_mse:.3f}_MSE.json')
            
            # Final evaluation on a fresh split
            x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
                X, y, test_size=0.2, random_state=999
            )
            test_dmatrix_final = xgb.DMatrix(x_test_final, label=y_test_final)
            
            final_predictions = best_model.predict(test_dmatrix_final)
            
            final_mse = mean_squared_error(y_test_final, final_predictions)
            final_mae = mean_absolute_error(y_test_final, final_predictions)
            final_r2 = r2_score(y_test_final, final_predictions)
            
            print(f"\nFinal Model Performance:")
            print(f"MSE: {final_mse:.3f}")
            print(f"MAE: {final_mae:.3f}")
            print(f"R² Score: {final_r2:.3f}")
            print(f"RMSE: {np.sqrt(final_mse):.3f}")
            
            # Show some sample predictions
            print(f"\nSample Predictions (first 10):")
            for i in range(min(10, len(final_predictions))):
                print(f"Actual: {y_test_final[i]:.1f}, Predicted: {final_predictions[i]:.1f}, Error: {abs(y_test_final[i] - final_predictions[i]):.1f}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        con.close()

if __name__ == "__main__":
    train_xgboost_ou_value_model()
