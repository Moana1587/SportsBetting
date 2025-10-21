"""
NHL Games Data Creator

This script combines team statistics data with odds data to create a comprehensive
dataset for machine learning. It reads from both TeamData.sqlite and OddsData.sqlite
and creates a unified dataset with team stats and game outcomes.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

# Initialize data storage lists
scores = []
home_scores = []
away_scores = []
win_margin = []
OU = []
OU_Cover = []
spreads = []
games = []
days_rest_away = []
days_rest_home = []
ml_home_odds = []
ml_away_odds = []

# Connect to databases
import os

# Get the absolute path to the Data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "Data")
teams_db_path = os.path.join(data_dir, "TeamData.sqlite")
odds_db_path = os.path.join(data_dir, "OddsData.sqlite")

# Check if database files exist
if not os.path.exists(teams_db_path):
    print(f"Error: TeamData.sqlite not found at {teams_db_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {current_dir}")
    print(f"Looking for database in: {data_dir}")
    exit(1)

if not os.path.exists(odds_db_path):
    print(f"Error: OddsData.sqlite not found at {odds_db_path}")
    exit(1)

teams_con = sqlite3.connect(teams_db_path)
odds_con = sqlite3.connect(odds_db_path)

def process_season_games(season_key):
    """Process games for a specific season"""
    print(f"Processing season: {season_key}")
    
    # Convert season format (e.g., "2012-13" -> "2012_13")
    season_table = season_key.replace('-', '_')
    
    try:
        # Read odds data from the new schema
        odds_df = pd.read_sql_query(f"SELECT * FROM odds_{season_table}", odds_con)
        print(f"  Loaded {len(odds_df)} odds records for {season_key}")
        
        # Read team data for the season
        team_df = pd.read_sql_query(f"SELECT * FROM season_{season_table}", teams_con)
        print(f"  Loaded {len(team_df)} team records for {season_key}")
        
        if team_df.empty or odds_df.empty:
            print(f"  No data available for season {season_key}")
            return
        
        # Debug: Show team names in team data
        if 'teamFullName' in team_df.columns:
            team_names = team_df['teamFullName'].unique()
            print(f"  Available team names in team data: {list(team_names)[:5]}...")  # Show first 5
        
        # Debug: Show team names in odds data
        odds_teams = set(odds_df['Home'].unique()) | set(odds_df['Away'].unique())
        print(f"  Team names in odds data: {list(odds_teams)[:5]}...")  # Show first 5
        
        # Process each game
        for _, row in odds_df.iterrows():
            home_team = row['Home']
            away_team = row['Away']
            date = row['Date']
            points = row['Points']
            ou = row['OU']
            spread = row['Spread']
            win_margin_val = row['Win_Margin']
            days_rest_home_val = row['Days_Rest_Home']
            days_rest_away_val = row['Days_Rest_Away']
            ml_home = row.get('ML_Home', -110)  # Moneyline odds for home team
            ml_away = row.get('ML_Away', -110)  # Moneyline odds for away team
            
            # Calculate individual scores from total points and win margin
            # If win_margin > 0, home team won by that margin
            # If win_margin < 0, away team won by that margin
            if win_margin_val > 0:
                # Home team won
                home_final = (points + win_margin_val) // 2
                away_final = points - home_final
            elif win_margin_val < 0:
                # Away team won
                away_final = (points + abs(win_margin_val)) // 2
                home_final = points - away_final
            else:
                # Tie game (rare in NHL due to overtime)
                home_final = points // 2
                away_final = points - home_final
            
            # Get team statistics from season data first
            try:
                home_team_series = get_team_series(team_df, home_team, season_key)
                away_team_series = get_team_series(team_df, away_team, season_key)
                
                if home_team_series is not None and away_team_series is not None:
                    # Only store game data if we found both teams
                    scores.append(points)
                    home_scores.append(home_final)
                    away_scores.append(away_final)
                    OU.append(ou)
                    spreads.append(spread)
                    days_rest_home.append(days_rest_home_val)
                    days_rest_away.append(days_rest_away_val)
                    ml_home_odds.append(ml_home)
                    ml_away_odds.append(ml_away)
                    
                    # Determine win margin (1 for home win, 0 for away win)
                    if win_margin_val > 0:
                        win_margin.append(1)
                    else:
                        win_margin.append(0)
                    
                    # Determine OU cover
                    if points < ou:
                        OU_Cover.append(0)  # Under
                    elif points > ou:
                        OU_Cover.append(1)  # Over
                    else:
                        OU_Cover.append(2)  # Push
                    
                    # Combine home and away team data
                    game = pd.concat([
                        home_team_series, 
                        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values})
                    ])
                    games.append(game)
                else:
                    print(f"  Warning: Could not find team data for {home_team} vs {away_team}")
                    
            except Exception as e:
                print(f"  Error processing game {home_team} vs {away_team} on {date}: {e}")
                
    except Exception as e:
        print(f"  Error processing season {season_key}: {e}")

def get_team_series(team_data, team_name, season):
    """Get team statistics series for a specific team and season"""
    try:
        # Find the team in the data - try different column names
        team_row = None
        
        # Try different possible column names for team name
        for col in ['teamFullName', 'teamName', 'Team', 'TEAM', 'team_name']:
            if col in team_data.columns:
                team_row = team_data[team_data[col] == team_name]
                if not team_row.empty:
                    break
        
        # If still not found, try partial matching
        if team_row is None or team_row.empty:
            for col in ['teamFullName', 'teamName', 'Team', 'TEAM', 'team_name']:
                if col in team_data.columns:
                    # Try matching the last word of the team name
                    last_word = team_name.split()[-1]
                    team_row = team_data[team_data[col].str.contains(last_word, na=False)]
                    if not team_row.empty:
                        break
        
        if team_row is not None and not team_row.empty:
            return team_row.iloc[0]
        else:
            print(f"    Team not found: {team_name}")
            # Show available team names for debugging
            if 'teamFullName' in team_data.columns:
                available_teams = team_data['teamFullName'].unique()[:10]  # Show first 10 teams
                print(f"    Available teams (first 10): {list(available_teams)}")
            return None
            
    except Exception as e:
        print(f"    Error getting team series for {team_name}: {e}")
        return None

# Process each season
print("Starting game data processing...")

# Get available seasons from the database
try:
    # Get available odds tables
    odds_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'odds_%'", odds_con)
    odds_seasons = [table[0].replace('odds_', '').replace('_', '-') for table in odds_tables.values]
    
    # Get available team tables
    team_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'season_%'", teams_con)
    team_seasons = [table[0].replace('season_', '').replace('_', '-') for table in team_tables.values]
    
    # Find common seasons
    available_seasons = list(set(odds_seasons) & set(team_seasons))
    available_seasons.sort()
    
    print(f"Found {len(available_seasons)} seasons with both odds and team data:")
    for season in available_seasons:
        print(f"  - {season}")
    
    # Process each available season
    for season in available_seasons:
        process_season_games(season)
        
except Exception as e:
    print(f"Error getting available seasons: {e}")
    print("Trying to process seasons manually...")
    
    # Fallback: try common season formats
    common_seasons = ['2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
    
    for season in common_seasons:
        try:
            process_season_games(season)
        except Exception as e:
            print(f"  Season {season} not available: {e}")
            continue

# Close database connections
odds_con.close()
teams_con.close()

# Create final dataset
if games:
    print(f"\nCreating final dataset with {len(games)} games...")
    
    # Combine all game data
    season = pd.concat(games, ignore_index=True, axis=1)
    season = season.T
    
    # Remove team ID columns if they exist
    columns_to_drop = [col for col in season.columns if 'TEAM_ID' in col or 'teamId' in col]
    if columns_to_drop:
        season = season.drop(columns=columns_to_drop)
    
    # Add game outcome data
    season['Score'] = np.asarray(scores)
    season['Home-Score'] = np.asarray(home_scores)
    season['Away-Score'] = np.asarray(away_scores)
    season['Home-Team-Win'] = np.asarray(win_margin)
    season['OU'] = np.asarray(OU)
    season['OU-Cover'] = np.asarray(OU_Cover)
    season['Spread'] = np.asarray(spreads)
    season['Days-Rest-Home'] = np.asarray(days_rest_home)
    season['Days-Rest-Away'] = np.asarray(days_rest_away)
    season['ML_Home'] = np.asarray(ml_home_odds)
    season['ML_Away'] = np.asarray(ml_away_odds)
    
    # Fix data types
    for field in season.columns.values:
        if 'TEAM_' in field or 'Date' in field or field not in season:
            continue
        try:
            season[field] = season[field].astype(float)
        except (ValueError, TypeError):
            # Skip fields that can't be converted to float
            continue
    
    # Save to database
    dataset_db_path = os.path.join(data_dir, "dataset.sqlite")
    con = sqlite3.connect(dataset_db_path)
    season.to_sql("dataset_2012-24_new", con, if_exists="replace", index=False)
    con.close()
    
    print(f"✅ Dataset created successfully with {len(season)} records")
    print(f"   Features: {len(season.columns)}")
    print(f"   Games processed: {len(games)}")
    
    # Show sample of the dataset
    print(f"\nSample of created dataset:")
    print(f"Columns: {list(season.columns)}")
    print(f"Shape: {season.shape}")
    print(f"First few rows:")
    print(season.head())
    
else:
    print("❌ No games were processed. Check your data sources.")

print("Game data processing completed!")
