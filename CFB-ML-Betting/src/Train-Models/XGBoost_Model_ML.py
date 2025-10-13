import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def train_xgboost_ml_model():
    """Train XGBoost model for moneyline (ML) prediction using CFBD data."""
    print("Training XGBoost Moneyline Model...")
    
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
        
        # Prepare target variable (Home Team Win)
        y = data['Home_Team_Win'].values
        
        # Prepare features - exclude non-numeric and target columns
        exclude_columns = [
            'Date', 'Home_Team', 'Away_Team', 'Home_Team_Win', 'Points', 
            'Win_Margin', 'OU_Cover', 'OU', 'Spread', 'ML_Home', 'ML_Away'
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
        print(f"Target distribution: {np.bincount(y)}")
        
        # Training parameters
        acc_results = []
        best_accuracy = 0
        
        print("\nStarting training iterations...")
        for x in tqdm(range(100)):  # Reduced from 300 for faster testing
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=x, stratify=y
            )
            
            # Create DMatrix for XGBoost
            train_dmatrix = xgb.DMatrix(x_train, label=y_train)
            test_dmatrix = xgb.DMatrix(x_test, label=y_test)
            
            # XGBoost parameters
            param = {
                'max_depth': 6,
                'eta': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
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
            predictions_proba = model.predict(test_dmatrix)
            predictions = (predictions_proba > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)
            acc_results.append(accuracy)
            
            # Print progress every 10 iterations
            if (x + 1) % 10 == 0:
                avg_acc = np.mean(acc_results[-10:])
                print(f"Iteration {x+1}: Current accuracy = {accuracy:.3f}, Avg (last 10) = {avg_acc:.3f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Ensure Models directory exists
                os.makedirs("../../Models", exist_ok=True)
                model.save_model(f'../../Models/XGBoost_{accuracy*100:.1f}%_ML-4.json')
                print(f"New best model saved with accuracy: {accuracy*100:.1f}%")
        
        # Final results
        print(f"\nTraining completed!")
        print(f"Best accuracy: {best_accuracy:.3f}")
        print(f"Average accuracy: {np.mean(acc_results):.3f}")
        print(f"Standard deviation: {np.std(acc_results):.3f}")
        
        # Show final classification report for the best model
        if best_accuracy > 0:
            print(f"\nLoading best model for final evaluation...")
            best_model = xgb.Booster()
            best_model.load_model(f'../../Models/XGBoost_{best_accuracy*100:.1f}%_ML-4.json')
            
            # Final evaluation on a fresh split
            x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
                X, y, test_size=0.2, random_state=999, stratify=y
            )
            test_dmatrix_final = xgb.DMatrix(x_test_final, label=y_test_final)
            
            final_predictions_proba = best_model.predict(test_dmatrix_final)
            final_predictions = (final_predictions_proba > 0.5).astype(int)
            
            print("\nFinal Classification Report:")
            print(classification_report(y_test_final, final_predictions))
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        con.close()

if __name__ == "__main__":
    train_xgboost_ml_model()
