"""
XGBoost Over/Under Classification Model Training

This script trains an XGBoost model to directly predict OVER/UNDER outcomes
instead of just predicting OU values.
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Add project root to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))

def load_dataset():
    """Load the NHL dataset for training"""
    try:
        # Load the processed games data from SQLite database
        import sqlite3
        import os
        
        # Get the absolute path to the Data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "..", "..", "Data")
        dataset_path = os.path.join(data_dir, "dataset.sqlite")
        
        # Check if database file exists
        if not os.path.exists(dataset_path):
            print(f"Error: dataset.sqlite not found at {dataset_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script location: {current_dir}")
            print(f"Looking for database in: {data_dir}")
            print("Please run Create_Games.py first to create the dataset.")
            return None
        
        con = sqlite3.connect(dataset_path)
        data = pd.read_sql_query("SELECT * FROM `dataset_2012-24_new`", con)
        con.close()
        
        print(f"Loaded dataset with {len(data)} games")
        print(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_training_data(data):
    """Prepare training data for OVER/UNDER classification"""
    print("Preparing training data for OVER/UNDER classification...")
    
    # Select features for training (exclude target columns and non-numeric columns)
    exclude_columns = [
        'Score', 'Home-Score', 'Away-Score', 'Home-Team-Win', 'OU-Cover', 'OU', 
        'Spread', 'ML_Home', 'ML_Away', 'Days-Rest-Home', 'Days-Rest-Away',
        'teamFullName', 'teamFullName.1', 'Season', 'Season.1'
    ]
    
    # Get only numeric columns and exclude target columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    print(f"Selected {len(feature_columns)} numeric features for training")
    print(f"Feature columns: {feature_columns}")
    
    features = data[feature_columns].copy()
    
    # Create OVER/UNDER target based on actual game scores vs OU line
    # 1 = OVER, 0 = UNDER
    total_score = data['Home-Score'] + data['Away-Score']  # Home + Away scores
    ou_line = data['OU']
    
    # Create binary classification target
    ou_target = (total_score > ou_line).astype(int)
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {ou_target.shape}")
    print(f"OVER games: {ou_target.sum()} ({ou_target.mean()*100:.1f}%)")
    print(f"UNDER games: {(1-ou_target).sum()} ({(1-ou_target.mean())*100:.1f}%)")
    
    # Remove any rows with missing values
    mask = features.notna().all(axis=1) & ou_target.notna()
    features = features[mask]
    ou_target = ou_target[mask]
    
    print(f"After removing missing values: {len(features)} games")
    
    return features, ou_target

def train_xgboost_classification_model(features, target, num_iterations=200):
    """Train XGBoost model for OVER/UNDER classification"""
    accuracy_results = []
    best_model = None
    best_accuracy = 0
    
    print(f"Training XGBoost OVER/UNDER classification model with {num_iterations} iterations...")
    
    for x in tqdm(range(num_iterations), desc="Training"):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=x, stratify=target)
        
        # XGBoost parameters for classification
        model = xgb.XGBClassifier(
            max_depth=6,
            eta=0.1,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=x,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        # Train model
        model.fit(x_train, y_train)
        
        # Make predictions
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]  # Probability of OVER
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_results.append(accuracy)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            accuracy_percent = round(accuracy * 100, 1)
            # Get absolute path for model saving
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "..", "..", "Models")
            model_path = os.path.join(models_dir, f'XGBoost_{accuracy_percent}%_OU_Classification.json')
            model.save_model(model_path)
            print(f"New best accuracy: {accuracy_percent}% - Model saved to {model_path}")
    
    return accuracy_results, best_model

def create_ensemble_classification_models(features, target, num_models=10):
    """Create an ensemble of classification models for confidence estimation"""
    print(f"Creating ensemble of {num_models} classification models...")
    
    ensemble_models = []
    
    for i in tqdm(range(num_models), desc="Creating ensemble"):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=i, stratify=target)
        
        # Create model with different parameters for diversity
        model = xgb.XGBClassifier(
            max_depth=np.random.randint(4, 8),
            eta=np.random.uniform(0.05, 0.2),
            n_estimators=np.random.randint(300, 700),
            subsample=np.random.uniform(0.7, 0.9),
            colsample_bytree=np.random.uniform(0.7, 0.9),
            random_state=i,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        # Train model
        model.fit(x_train, y_train)
        ensemble_models.append(model)
    
    return ensemble_models

def predict_with_classification_confidence(model, ensemble_models, features):
    """Make OVER/UNDER predictions with confidence using ensemble method"""
    if not ensemble_models:
        # Fallback: use simple confidence based on prediction probability
        probabilities = model.predict_proba(features)[:, 1]
        # Convert probabilities to confidence (closer to 0.5 = lower confidence)
        confidence = np.abs(probabilities - 0.5) * 200  # Scale to 0-100%
        return model.predict(features), probabilities, confidence
    
    # Get prediction from main model
    main_predictions = model.predict(features)
    main_probabilities = model.predict_proba(features)[:, 1]
    
    # Get predictions from ensemble
    ensemble_probabilities = []
    for ensemble_model in ensemble_models:
        prob = ensemble_model.predict_proba(features)[:, 1]
        ensemble_probabilities.append(prob)
    
    # Calculate confidence based on prediction variance
    ensemble_probabilities = np.array(ensemble_probabilities)
    prediction_std = np.std(ensemble_probabilities, axis=0)
    
    # Calculate confidence percentage (higher confidence = lower variance)
    # Normalize to 0-100% range
    max_std = np.max(prediction_std) if len(prediction_std) > 0 else 1.0
    confidence_percentage = np.clip((1 - prediction_std / max_std) * 100, 0, 100)
    
    return main_predictions, main_probabilities, confidence_percentage

def main():
    """Main training function"""
    try:
        # Load dataset
        data = load_dataset()
        if data is None:
            return
        
        # Prepare training data
        features, target = prepare_training_data(data)
        
        # Train main model
        accuracy_results, best_model = train_xgboost_classification_model(features, target)
        
        # Create ensemble for confidence estimation
        ensemble_models = create_ensemble_classification_models(features, target, num_models=10)
        
        # Save ensemble models
        # Get absolute path for ensemble models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "..", "..", "Models")
        ensemble_dir = os.path.join(models_dir, "Ensemble_OU_Classification")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        for i, model in enumerate(ensemble_models):
            model_path = os.path.join(ensemble_dir, f"ensemble_ou_classification_{i}.json")
            model.save_model(model_path)
        
        print(f"Saved {len(ensemble_models)} ensemble models to {ensemble_dir}")
        
        # Test confidence prediction on a sample
        if best_model is not None:
            sample_features = features.iloc[:5]  # Test on first 5 samples
            predictions, probabilities, confidence = predict_with_classification_confidence(best_model, ensemble_models, sample_features)
            
            print(f"\nSample predictions with confidence:")
            for i in range(len(sample_features)):
                prediction_text = "OVER" if predictions[i] == 1 else "UNDER"
                print(f"  Sample {i+1}: {prediction_text} (Prob: {probabilities[i]:.3f}, Confidence: {confidence[i]:.1f}%)")
        
        # Print results
        print(f"\nTraining completed!")
        print(f"Best accuracy: {max(accuracy_results)*100:.1f}%")
        print(f"Average accuracy: {np.mean(accuracy_results)*100:.1f}%")
        print(f"Accuracy std: {np.std(accuracy_results)*100:.1f}%")
        
        # Print target statistics
        print(f"\nTarget (OVER/UNDER) statistics:")
        print(f"OVER: {target.sum()} ({target.mean()*100:.1f}%)")
        print(f"UNDER: {(1-target).sum()} ({(1-target).mean()*100:.1f}%)")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
