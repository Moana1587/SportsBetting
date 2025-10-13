import os
import sqlite3
import sys
import numpy as np
import pandas as pd
import toml
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

def get_team_stats_for_game(teams_con, team_name, season, date):
    """Get team stats for a specific team and season."""
    try:
        # Try to find the team in the season stats table
        team_df = pd.read_sql_query(
            f"SELECT * FROM \"{season}_advanced_stats\" WHERE team = ?", 
            teams_con, 
            params=[team_name]
        )
        
        if not team_df.empty:
            return team_df.iloc[0]  # Return first match
        else:
            print(f"  Warning: No stats found for {team_name} in {season}")
            return None
    except Exception as e:
        print(f"  Error getting stats for {team_name} in {season}: {e}")
        return None

def create_games_dataset():
    """Create the games dataset by combining team stats with odds data."""
    print("Creating games dataset from CFBD data...")
    
    # Connect to databases
    teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
    odds_con = sqlite3.connect("../../Data/OddsData.sqlite")
    
    all_games = []
    scores = []
    win_margins = []
    OUs = []
    OU_covers = []
    
    # Get all available odds data
    try:
        odds_df = pd.read_sql_query("SELECT * FROM odds_data", odds_con)
        print(f"Found {len(odds_df)} odds records")
        
        if odds_df.empty:
            print("No odds data found. Please run Get_Odds_Data.py first.")
            return
        
        # Group by season to process each year
        for season in odds_df['Date'].str[:4].unique():
            season_odds = odds_df[odds_df['Date'].str[:4] == season]
            season_name = f"{season}-{str(int(season)+1)[2:]}"
            
            print(f"\nProcessing season: {season_name}")
            
            # Get team stats for this season
            try:
                team_stats_df = pd.read_sql_query(
                    f"SELECT * FROM \"{season_name}_advanced_stats\"", 
                    teams_con
                )
                print(f"  Found {len(team_stats_df)} team stat records for {season_name}")
            except Exception as e:
                print(f"  No team stats found for {season_name}: {e}")
                continue
            
            # Process each game
            for idx, game in season_odds.iterrows():
                home_team = game['Home']
                away_team = game['Away']
                date = game['Date']
                
                # Get team stats
                home_stats = get_team_stats_for_game(teams_con, home_team, season_name, date)
                away_stats = get_team_stats_for_game(teams_con, away_team, season_name, date)
                
                if home_stats is not None and away_stats is not None:
                    # Create game record
                    game_record = {}
                    
                    # Add home team stats (prefix with 'home_')
                    for col in home_stats.index:
                        if col not in ['team', 'season', 'year']:
                            game_record[f'home_{col}'] = home_stats[col]
                    
                    # Add away team stats (prefix with 'away_')
                    for col in away_stats.index:
                        if col not in ['team', 'season', 'year']:
                            game_record[f'away_{col}'] = away_stats[col]
                    
                    # Add game outcome data
                    game_record['Date'] = date
                    game_record['Home_Team'] = home_team
                    game_record['Away_Team'] = away_team
                    
                    # Add betting data
                    game_record['OU'] = game['OU']
                    game_record['Spread'] = game['Spread']
                    game_record['ML_Home'] = game['ML_Home']
                    game_record['ML_Away'] = game['ML_Away']
                    
                    # Add outcome data
                    points = game['Points']
                    win_margin = game['Win_Margin']
                    
                    game_record['Points'] = points
                    game_record['Win_Margin'] = win_margin
                    game_record['Home_Team_Win'] = 1 if win_margin > 0 else 0
                    
                    # Calculate OU cover
                    if points < game['OU']:
                        ou_cover = 0  # Under
                    elif points > game['OU']:
                        ou_cover = 1  # Over
                    else:
                        ou_cover = 2  # Push
                    
                    game_record['OU_Cover'] = ou_cover
                    
                    all_games.append(game_record)
                    scores.append(points)
                    win_margins.append(win_margin)
                    OUs.append(game['OU'])
                    OU_covers.append(ou_cover)
                    
                    print(f"  Processed: {away_team} @ {home_team} on {date}")
                else:
                    print(f"  Skipped: {away_team} @ {home_team} on {date} - missing team stats")
    
    except Exception as e:
        print(f"Error processing odds data: {e}")
    
    finally:
        teams_con.close()
        odds_con.close()
    
    if all_games:
        # Create final dataset
        print(f"\nCreating final dataset with {len(all_games)} games...")
        
        games_df = pd.DataFrame(all_games)
        
        # Convert numeric columns to appropriate types
        numeric_columns = games_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Date', 'Home_Team', 'Away_Team']:
                games_df[col] = pd.to_numeric(games_df[col], errors='coerce')
        
        # Remove any rows with NaN values in critical columns
        critical_columns = ['Points', 'Win_Margin', 'OU', 'Spread', 'ML_Home', 'ML_Away']
        initial_count = len(games_df)
        games_df = games_df.dropna(subset=critical_columns)
        final_count = len(games_df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} games with missing critical data")
        
        print(f"Final dataset shape: {games_df.shape}")
        print(f"Columns: {list(games_df.columns)}")
        
        # Save to database
        con = sqlite3.connect("../../Data/dataset.sqlite")
        games_df.to_sql("cfb_games_dataset", con, if_exists="replace", index=False)
        con.close()
        
        print(f"✓ Saved {len(games_df)} games to dataset.sqlite")
        
        # Show sample data
        print("\nSample data:")
        print(games_df[['Date', 'Away_Team', 'Home_Team', 'Points', 'Win_Margin', 'OU', 'Spread']].head())
        
    else:
        print("No games processed. Please ensure both team stats and odds data are available.")

if __name__ == "__main__":
    create_games_dataset()