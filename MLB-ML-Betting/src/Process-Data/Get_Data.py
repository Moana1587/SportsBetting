import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

# MLB API configuration
BASE_URL = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/team"

con = sqlite3.connect("../../Data/TeamData.sqlite")

for key, value in config['get-data'].items():
    start_year = int(value['start_year'])
    end_year = int(value['end_year'])
    
    print(f"Getting MLB team data for {start_year}-{end_year}")
    
    # MLB API parameters for team hitting stats
    params = {
        "env": "prod",
        "sportId": 1,
        "gameType": "R",
        "group": "hitting",
        "stats": "season",
        "season": start_year,
        "limit": 1000,
        "offset": 0,
    }
    
    try:
        # Get hitting stats
        print(f"Fetching hitting stats for {start_year}")
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        hitting_data = r.json()
        
        # Get pitching stats
        params["group"] = "pitching"
        print(f"Fetching pitching stats for {start_year}")
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        pitching_data = r.json()
        
        # Process hitting data - data is in 'stats' array, not 'data'
        hitting_rows = hitting_data.get("stats", [])
        hitting_df = pd.DataFrame(hitting_rows)
        print(f"Found {len(hitting_rows)} hitting records for {start_year}")
        
        # Process pitching data - data is in 'stats' array, not 'data'
        pitching_rows = pitching_data.get("stats", [])
        pitching_df = pd.DataFrame(pitching_rows)
        print(f"Found {len(pitching_rows)} pitching records for {start_year}")
        
        # Select relevant columns for hitting based on actual response structure
        hitting_cols = ["teamId", "teamName", "teamAbbrev", "teamShortName", "leagueAbbrev", "leagueName", 
                       "rank", "gamesPlayed", "plateAppearances", "hits", "doubles", "triples", "homeRuns", 
                       "runs", "rbi", "baseOnBalls", "strikeOuts", "avg", "obp", "slg", "ops", "atBats", 
                       "totalBases", "leftOnBase", "sacBunts", "sacFlies", "babip", "groundOutsToAirouts",
                       "atBatsPerHomeRun", "stolenBases", "caughtStealing", "stolenBasePercentage",
                       "groundIntoDoublePlay", "hitByPitch", "intentionalWalks", "catchersInterference",
                       "groundOuts", "airOuts", "numberOfPitches"]
        hitting_selected = hitting_df[[c for c in hitting_cols if c in hitting_df.columns]]
        
        # Select relevant columns for pitching based on actual response structure
        pitching_cols = ["teamId", "teamName", "teamAbbrev", "teamShortName", "leagueAbbrev", "leagueName", 
                        "rank", "gamesPlayed", "wins", "losses", "era", "inningsPitched", "hits", "runs", 
                        "earnedRuns", "baseOnBalls", "strikeOuts", "whip", "battingAvgAgainst", "saves", 
                        "holds", "blownSaves", "completeGames", "shutouts", "noHitters", "perfectGames", 
                        "hitBatsmen", "balks", "wildPitches", "pickoffs", "rbi", "sacBunts", "sacFlies"]
        pitching_selected = pitching_df[[c for c in pitching_cols if c in pitching_df.columns]]
        
        # Merge hitting and pitching data
        merge_keys = ['teamId', 'teamName', 'teamAbbrev', 'teamShortName', 'leagueAbbrev', 'leagueName', 'gamesPlayed']
        merged_df = pd.merge(hitting_selected, pitching_selected, on=merge_keys, 
                            how='outer', suffixes=('_hitting', '_pitching'))
        
        # Add season year
        merged_df['season'] = start_year
        
        # Store in database
        merged_df.to_sql(f"{key}_teams", con, if_exists="replace", index=False)
        
        print(f"Successfully stored {len(merged_df)} team records for {start_year}")
        
        # Add delay between requests
        time.sleep(random.randint(1, 3))
        
    except Exception as e:
        print(f"Error fetching data for {start_year}: {str(e)}")
        continue

con.close()
print("Data collection completed!")
