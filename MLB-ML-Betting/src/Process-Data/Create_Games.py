import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, \
    team_index_current

config = toml.load("../../config.toml")

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
spread = []
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")

for key, value in config['create-games'].items():
    print(f"Processing season: {key}")
    # Updated table name to match Get_Odds_Data.py output
    try:
        odds_df = pd.read_sql_query(f"select * from \"{key}\"", odds_con)
        print(f"Found {len(odds_df)} odds records for {key}")
    except Exception as e:
        print(f"Error reading odds data for {key}: {e}")
        continue
    
    team_table_str = key
    year_count = 0
    season = key

    for row in odds_df.itertuples():
        # Updated column indices to match new data structure
        # Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points, Win_Margin, Days_Rest_Home, Days_Rest_Away
        home_team = row[2]  # Home
        away_team = row[3]  # Away
        date = row[1]       # Date

        # Updated team table name to match Get_Data.py output
        try:
            team_df = pd.read_sql_query(f"select * from \"{key}_teams\"", teams_con)
            if len(team_df.index) != 30:
                print(f"Warning: Expected 30 teams, found {len(team_df.index)} for {key}")
                continue
        except Exception as e:
            print(f"Error reading team data for {key}: {e}")
            continue
        
        # Updated team lookup logic to work with new team data structure
        # The team data now uses 'teamName' field instead of relying on dictionaries
        try:
            # Find home team by matching teamName
            home_team_mask = team_df['teamName'] == home_team
            if not home_team_mask.any():
                # Try alternative matching if exact match fails
                home_team_mask = team_df['teamName'].str.contains(home_team, case=False, na=False)
            
            # Find away team by matching teamName
            away_team_mask = team_df['teamName'] == away_team
            if not away_team_mask.any():
                # Try alternative matching if exact match fails
                away_team_mask = team_df['teamName'].str.contains(away_team, case=False, na=False)
            
            if home_team_mask.any() and away_team_mask.any():
                home_team_series = team_df[home_team_mask].iloc[0]
                away_team_series = team_df[away_team_mask].iloc[0]
            else:
                print(f"Could not find teams: {home_team} or {away_team}")
                continue
                
        except Exception as e:
            print(f"Error finding teams {home_team} vs {away_team}: {e}")
            continue
        
        # Only add data to arrays if team lookup was successful
        # Updated column indices for new structure
        scores.append(row[8])        # Points
        OU.append(row[4])            # OU
        spread.append(row[5])        # Spread
        days_rest_home.append(row[10])  # Days_Rest_Home
        days_rest_away.append(row[11])  # Days_Rest_Away
        
        # Updated win margin logic
        if row[9] > 0:  # Win_Margin
            win_margin.append(1)
        else:
            win_margin.append(0)

        # Updated OU cover logic
        if row[8] < row[4]:  # Points < OU
            OU_Cover.append(0)
        elif row[8] > row[4]:  # Points > OU
            OU_Cover.append(1)
        elif row[8] == row[4]:  # Points == OU
            OU_Cover.append(2)
        
        game = pd.concat([home_team_series, away_team_series.rename(
            index={col: f"{col}.1" for col in team_df.columns.values}
        )])
        games.append(game)
odds_con.close()
teams_con.close()

print(f"Processing {len(games)} games...")

if len(games) == 0:
    print("No games found to process!")
else:
    # Validate that all arrays have the same length
    array_lengths = [len(games), len(scores), len(win_margin), len(OU), len(OU_Cover), len(days_rest_home), len(days_rest_away), len(spread)]
    if len(set(array_lengths)) > 1:
        print(f"Error: Array lengths don't match!")
        print(f"Games: {len(games)}, Scores: {len(scores)}, Win Margin: {len(win_margin)}")
        print(f"OU: {len(OU)}, OU Cover: {len(OU_Cover)}, Days Rest Home: {len(days_rest_home)}, Days Rest Away: {len(days_rest_away)}, Spread: {len(spread)}")
        # Truncate arrays to match the shortest length
        min_length = min(array_lengths)
        scores = scores[:min_length]
        win_margin = win_margin[:min_length]
        OU = OU[:min_length]
        OU_Cover = OU_Cover[:min_length]
        days_rest_home = days_rest_home[:min_length]
        days_rest_away = days_rest_away[:min_length]
        spread = spread[:min_length]
        games = games[:min_length]
        print(f"Truncated all arrays to length {min_length}")
    
    season = pd.concat(games, ignore_index=True, axis=1)
    season = season.T
    
    # Drop team ID columns if they exist
    columns_to_drop = [col for col in ['TEAM_ID', 'TEAM_ID.1'] if col in season.columns]
    if columns_to_drop:
        frame = season.drop(columns=columns_to_drop)
    else:
        frame = season
    
    # Add target variables
    frame['Score'] = np.asarray(scores)
    frame['Home-Team-Win'] = np.asarray(win_margin)
    frame['OU'] = np.asarray(OU)
    frame['OU-Cover'] = np.asarray(OU_Cover)
    frame['Spread'] = np.asarray(spread)
    frame['Days-Rest-Home'] = np.asarray(days_rest_home)
    frame['Days-Rest-Away'] = np.asarray(days_rest_away)
    
    # Fix data types
    for field in frame.columns.values:
        if 'TEAM_' in field or 'Date' in field or field not in frame:
            continue
        try:
            frame[field] = frame[field].astype(float)
        except (ValueError, TypeError):
            # Skip fields that can't be converted to float
            continue
    
    # Save to database
    con = sqlite3.connect("../../Data/dataset.sqlite")
    # Updated table name to be more descriptive
    frame.to_sql("mlb_dataset", con, if_exists="replace")
    con.close()
    
    print(f"Successfully created dataset with {len(frame)} games and {len(frame.columns)} features")
