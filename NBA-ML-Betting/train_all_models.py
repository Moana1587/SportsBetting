#!/usr/bin/env python3
"""
Comprehensive training script for all NBA betting models.
This script trains models for Moneyline, Over/Under, and Spread predictions.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_training_script(script_name, description):
    """Run a training script and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"Training: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd="src/Train-Models")
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description} completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print(f"[ERROR] {description} failed!")
            print("Error:")
            print(result.stderr)
            
    except Exception as e:
        print(f"[ERROR] Error running {script_name}: {e}")

def main():
    """Main training function."""
    print("NBA Betting Model Training Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Define training scripts and their descriptions
    training_scripts = [
        ("XGBoost_Model_ML.py", "XGBoost Moneyline Classification (Original)"),
        ("XGBoost_Model_UO.py", "XGBoost Over/Under Classification (Original)"),
        ("XGBoost_Model_Spread.py", "XGBoost Spread Regression (NEW)"),
        ("XGBoost_Model_ML_Classification.py", "XGBoost ML Classification with Odds (NEW)"),
        ("XGBoost_Model_ML_Regression.py", "XGBoost ML Difference Regression (NEW)"),
        ("NN_Model_ML.py", "Neural Network Moneyline Classification (Original)"),
        ("NN_Model_UO.py", "Neural Network Over/Under Classification (Original)"),
        ("NN_Model_Spread.py", "Neural Network Spread Regression (NEW)"),
        ("NN_Model_ML_Classification.py", "Neural Network ML Classification with Odds (NEW)"),
    ]
    
    # Run each training script
    for script, description in training_scripts:
        run_training_script(script, description)
    
    print(f"\n{'='*60}")
    print("All training completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # List all models in the Models directory
    print("\nGenerated Models:")
    try:
        models_dir = "Models"
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(('.json', '.pb')):
                        print(f"  - {os.path.join(root, file)}")
        else:
            print("  No Models directory found.")
    except Exception as e:
        print(f"  Error listing models: {e}")

if __name__ == "__main__":
    main()
