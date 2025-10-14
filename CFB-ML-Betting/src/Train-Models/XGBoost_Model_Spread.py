import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def train_xgboost_spread_model():
    """Train XGBoost model for spread value prediction using CFBD data."""
    print("Training XGBoost Spread Value Regression Model...")
    
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
        
        # Prepare target variable (Actual Spread Value)
        # The target is the actual point differential (Win_Margin)
        # This represents how much the home team won/lost by
        y = data['Win_Margin'].values
        
        print(f"Target variable (Win_Margin) statistics:")
        print(f"  Mean: {np.mean(y):.2f}")
        print(f"  Std: {np.std(y):.2f}")
        print(f"  Min: {np.min(y):.2f}")
        print(f"  Max: {np.max(y):.2f}")
        print(f"  Range: {np.max(y) - np.min(y):.2f}")
        
        # Prepare features - exclude non-numeric and target columns
        exclude_columns = [
            'Date', 'Home_Team', 'Away_Team', 'Home_Team_Win', 'Points', 
            'Win_Margin', 'OU_Cover', 'OU', 'ML_Home', 'ML_Away'
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
        print(f"Target variable shape: {y.shape}")
        print(f"Home team wins: {np.sum(y > 0)}")
        print(f"Away team wins: {np.sum(y < 0)}")
        print(f"Ties: {np.sum(y == 0)}")
        
        # Training parameters
        mse_results = []
        mae_results = []
        r2_results = []
        best_mse = float('inf')
        
        print("\nStarting training iterations...")
        for x in tqdm(range(100)):  # Reduced from 300 for faster testing
            # Split data (no stratification needed for regression)
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
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            mse_results.append(mse)
            mae_results.append(mae)
            r2_results.append(r2)
            
            # Print progress every 10 iterations
            if (x + 1) % 10 == 0:
                avg_mse = np.mean(mse_results[-10:])
                avg_mae = np.mean(mae_results[-10:])
                avg_r2 = np.mean(r2_results[-10:])
                print(f"Iteration {x+1}: MSE = {mse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")
                print(f"  Avg (last 10): MSE = {avg_mse:.3f}, MAE = {avg_mae:.3f}, R² = {avg_r2:.3f}")
            
            # Save best model (lowest MSE)
            if mse < best_mse:
                best_mse = mse
                # Ensure Models directory exists
                os.makedirs("../../Models", exist_ok=True)
                model.save_model(f'../../Models/XGBoost_Spread_Value_{mse:.3f}_MSE.json')
                print(f"New best model saved with MSE: {mse:.3f}")
        
        # Final results
        print(f"\nTraining completed!")
        print(f"Best MSE: {best_mse:.3f}")
        print(f"Average MSE: {np.mean(mse_results):.3f} ± {np.std(mse_results):.3f}")
        print(f"Average MAE: {np.mean(mae_results):.3f} ± {np.std(mae_results):.3f}")
        print(f"Average R²: {np.mean(r2_results):.3f} ± {np.std(r2_results):.3f}")
        
        # Show final regression evaluation for the best model
        if best_mse < float('inf'):
            print(f"\nLoading best model for final evaluation...")
            best_model = xgb.Booster()
            best_model.load_model(f'../../Models/XGBoost_Spread_Value_{best_mse:.3f}_MSE.json')
            
            # Final evaluation on a fresh split
            x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
                X, y, test_size=0.2, random_state=999
            )
            test_dmatrix_final = xgb.DMatrix(x_test_final, label=y_test_final)
            
            final_predictions = best_model.predict(test_dmatrix_final)
            
            final_mse = mean_squared_error(y_test_final, final_predictions)
            final_mae = mean_absolute_error(y_test_final, final_predictions)
            final_r2 = r2_score(y_test_final, final_predictions)
            
            print("\nFinal Regression Evaluation:")
            print(f"  MSE: {final_mse:.3f}")
            print(f"  MAE: {final_mae:.3f}")
            print(f"  R²: {final_r2:.3f}")
            print(f"  RMSE: {np.sqrt(final_mse):.3f}")
            
            # Show prediction vs actual comparison
            print(f"\nPrediction vs Actual (first 10 samples):")
            for i in range(min(10, len(final_predictions))):
                print(f"  Actual: {y_test_final[i]:6.2f}, Predicted: {final_predictions[i]:6.2f}, Diff: {abs(y_test_final[i] - final_predictions[i]):6.2f}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        con.close()

if __name__ == "__main__":
    train_xgboost_spread_model()
