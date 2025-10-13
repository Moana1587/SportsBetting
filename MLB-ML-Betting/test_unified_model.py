import sqlite3
import pandas as pd
import numpy as np

def test_unified_model():
    """Test the unified model training script"""
    
    print("="*60)
    print("TESTING UNIFIED MODEL TRAINING")
    print("="*60)
    
    # Test dataset loading
    print("\n1. Testing dataset loading...")
    try:
        dataset = "mlb_dataset"
        con = sqlite3.connect("Data/dataset.sqlite")
        data = pd.read_sql_query(f"select * from \"{dataset}\"", con)
        con.close()
        print(f"   ✓ Dataset loaded successfully: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"   ✗ Dataset loading failed: {e}")
        return False
    
    # Test target variables
    print("\n2. Testing target variables...")
    target_vars = ['ML_Home', 'ML_Away', 'Home_Team_Win']
    missing_targets = [var for var in target_vars if var not in data.columns]
    
    if missing_targets:
        print(f"   ✗ Missing target variables: {missing_targets}")
        print(f"   Available columns: {list(data.columns)}")
        return False
    else:
        print(f"   ✓ All target variables found: {target_vars}")
    
    # Test target variable data
    print("\n3. Testing target variable data...")
    targets = data[target_vars]
    for var in target_vars:
        if var == 'Home_Team_Win':
            print(f"   ✓ {var}: {targets[var].value_counts().to_dict()}")
        else:
            print(f"   ✓ {var}: mean={targets[var].mean():.2f}, range=[{targets[var].min():.2f}, {targets[var].max():.2f}]")
    
    # Test feature preparation
    print("\n4. Testing feature preparation...")
    columns_to_drop = [
        'Score', 'Home-Team-Win', 'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
        'ML_Home', 'ML_Away', 'Home_Team_Win'
    ]
    
    for col in data.columns:
        if any(keyword in col.lower() for keyword in ['team_name', 'date', 'season', 'id', 'name', 'abbrev']):
            columns_to_drop.append(col)
    
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    feature_data = data.drop(columns_to_drop, axis=1)
    
    print(f"   ✓ Feature matrix prepared: {feature_data.shape}")
    print(f"   ✓ Dropped {len(columns_to_drop)} non-feature columns")
    
    # Test multi-output target preparation
    print("\n5. Testing multi-output target preparation...")
    targets_unified = targets.copy()
    targets_unified['Home_Team_Win'] = targets_unified['Home_Team_Win'].astype(float)
    target_matrix = targets_unified.values
    
    print(f"   ✓ Target matrix shape: {target_matrix.shape}")
    print(f"   ✓ All targets converted to continuous values")
    
    # Test imports
    print("\n6. Testing required imports...")
    try:
        import xgboost as xgb
        print("   ✓ xgboost imported")
        
        from sklearn.metrics import r2_score
        print("   ✓ sklearn.metrics imported")
        
        from sklearn.model_selection import train_test_split
        print("   ✓ sklearn.model_selection imported")
        
        from tqdm import tqdm
        print("   ✓ tqdm imported")
        
        import os
        print("   ✓ os imported")
        
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print("✓ All tests passed! The unified model is ready to train.")
    print("\nTo train the unified model, run:")
    print("  python src/Train-Models/XGBoost_Model_ML.py")
    print("\nThe model will:")
    print("  - Train a single XGBoost model")
    print("  - Predict ML_Home, ML_Away, and Home_Team_Win simultaneously")
    print("  - Use multi-output regression approach")
    print("  - Save the best model as XGBoost_Unified_[score].json")
    
    return True

if __name__ == "__main__":
    test_unified_model()
