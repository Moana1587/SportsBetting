#!/usr/bin/env python3
"""
Test script to demonstrate the updated prediction format with spread and ML values.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Predict.XGBoost_Runner import xgb_runner
import numpy as np
import pandas as pd

def test_predictions():
    """Test the prediction format with sample data."""
    print("Testing NBA Betting Predictions with Spread and ML Values")
    print("=" * 60)
    
    # Create sample data (this would normally come from the main script)
    # Sample team stats data (106 features)
    sample_data = np.random.rand(2, 106)  # 2 games, 106 features
    
    # Sample over/under lines
    todays_games_uo = [220.0, 215.5]
    
    # Sample frame_ml (team stats without some columns)
    sample_frame_ml = pd.DataFrame(np.random.rand(2, 104))  # 104 features after dropping some columns
    
    # Sample games
    games = [
        ["Oklahoma City Thunder", "Houston Rockets"],
        ["Los Angeles Lakers", "Golden State Warriors"]
    ]
    
    # Sample odds
    home_team_odds = [-110, -105]
    away_team_odds = [-110, -105]
    
    print("Sample Games:")
    for game in games:
        print(f"  {game[0]} vs {game[1]}")
    
    print(f"\nRunning predictions with XGBoost models...")
    print("-" * 60)
    
    # Run the predictions
    xgb_runner(sample_data, todays_games_uo, sample_frame_ml, games, home_team_odds, away_team_odds, False)
    
    print("\n" + "=" * 60)
    print("Test completed! The output shows the new format with:")
    print("  - Recommended team name")
    print("  - Spread predictions (e.g., Spread:+6.2(36.4%))")
    print("  - ML predictions (e.g., ML:-175(63.6%))")
    print("  - OU predictions (e.g., OU:OVER 220.0(78.2%))")

if __name__ == "__main__":
    test_predictions()
