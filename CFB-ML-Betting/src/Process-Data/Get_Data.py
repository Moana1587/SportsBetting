import os
import time
import sqlite3
import sys
import requests
import pandas as pd
from datetime import datetime

import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

# CFBD API configuration
API = os.environ.get("CFBD_KEY") or "dTYsx4FGjqSNJdqnIylk32HsJ6dBGeTry5nqHg3R+rKnu/745bzicjQF6GDlSTQf"
H = {"Authorization": f"Bearer {API}"}

def cfbd_season_team_stats(year: int):
    """Advanced season stats by team for a given year."""
    url = "https://api.collegefootballdata.com/stats/season/advanced"
    r = requests.get(url, params={"year": year}, headers=H, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_season_rows(rows, year):
    """Flatten advanced season stats data from CFBD API into a structured format."""
    out = []
    for row in rows:
        # Base team information
        base = {
            "season": row.get("season", year),
            "team": row.get("team", ""),
            "conference": row.get("conference", ""),
            "year": year
        }
        
        # Flatten offense stats
        offense = row.get("offense", {})
        if offense:
            # Offense passing plays
            passing_plays = offense.get("passingPlays", {})
            for key, value in passing_plays.items():
                if value is not None:
                    base[f"offense_passingPlays_{key}"] = value
            
            # Offense rushing plays
            rushing_plays = offense.get("rushingPlays", {})
            for key, value in rushing_plays.items():
                if value is not None:
                    base[f"offense_rushingPlays_{key}"] = value
            
            # Offense passing downs
            passing_downs = offense.get("passingDowns", {})
            for key, value in passing_downs.items():
                if value is not None:
                    base[f"offense_passingDowns_{key}"] = value
            
            # Offense standard downs
            standard_downs = offense.get("standardDowns", {})
            for key, value in standard_downs.items():
                if value is not None:
                    base[f"offense_standardDowns_{key}"] = value
            
            # Offense havoc
            havoc = offense.get("havoc", {})
            for key, value in havoc.items():
                if value is not None:
                    base[f"offense_havoc_{key}"] = value
            
            # Offense field position
            field_position = offense.get("fieldPosition", {})
            for key, value in field_position.items():
                if value is not None:
                    base[f"offense_fieldPosition_{key}"] = value
            
            # Other offense stats
            offense_keys = [
                "pointsPerOpportunity", "totalOpportunies", "openFieldYardsTotal",
                "openFieldYards", "secondLevelYardsTotal", "secondLevelYards",
                "lineYardsTotal", "lineYards", "stuffRate", "powerSuccess",
                "explosiveness", "successRate", "totalPPA", "ppa", "drives", "plays"
            ]
            for key in offense_keys:
                if key in offense and offense[key] is not None:
                    base[f"offense_{key}"] = offense[key]
        
        # Flatten defense stats
        defense = row.get("defense", {})
        if defense:
            # Defense passing plays
            passing_plays = defense.get("passingPlays", {})
            for key, value in passing_plays.items():
                if value is not None:
                    base[f"defense_passingPlays_{key}"] = value
            
            # Defense rushing plays
            rushing_plays = defense.get("rushingPlays", {})
            for key, value in rushing_plays.items():
                if value is not None:
                    base[f"defense_rushingPlays_{key}"] = value
            
            # Defense passing downs
            passing_downs = defense.get("passingDowns", {})
            for key, value in passing_downs.items():
                if value is not None:
                    base[f"defense_passingDowns_{key}"] = value
            
            # Defense standard downs
            standard_downs = defense.get("standardDowns", {})
            for key, value in standard_downs.items():
                if value is not None:
                    base[f"defense_standardDowns_{key}"] = value
            
            # Defense havoc
            havoc = defense.get("havoc", {})
            for key, value in havoc.items():
                if value is not None:
                    base[f"defense_havoc_{key}"] = value
            
            # Defense field position
            field_position = defense.get("fieldPosition", {})
            for key, value in field_position.items():
                if value is not None:
                    base[f"defense_fieldPosition_{key}"] = value
            
            # Other defense stats
            defense_keys = [
                "pointsPerOpportunity", "totalOpportunies", "openFieldYardsTotal",
                "openFieldYards", "secondLevelYardsTotal", "secondLevelYards",
                "lineYardsTotal", "lineYards", "stuffRate", "powerSuccess",
                "explosiveness", "successRate", "totalPPA", "ppa", "drives", "plays"
            ]
            for key in defense_keys:
                if key in defense and defense[key] is not None:
                    base[f"defense_{key}"] = defense[key]
        
        out.append(base)
    return out

def harvest_season_stats(start_year=2004, end_year=2025, sleep=0.3):
    """Harvest season team statistics from CFBD API for specified year range."""
    all_rows = []
    for y in range(start_year, end_year+1):
        try:
            print(f"  Fetching season stats for year {y}...")
            data = cfbd_season_team_stats(y)
            print(f"  Raw API response for {y}: {len(data)} records")
            if data:
                print(f"  Sample raw record: {data[0] if data else 'No data'}")
            
            flattened_data = flatten_season_rows(data, y)
            all_rows += flattened_data
            print(f"  Found {len(flattened_data)} teams for {y}")
            
            if flattened_data:
                print(f"  Sample flattened record: {flattened_data[0] if flattened_data else 'No data'}")
                print(f"  Available stats columns: {list(flattened_data[0].keys()) if flattened_data else 'No data'}")
            
            time.sleep(sleep)  # be polite to free tier limits
        except requests.HTTPError as e:
            print(f"  [{y}] HTTP error: {e}")
        except Exception as e:
            print(f"  [{y}] Error: {e}")
    
    if all_rows:
        df = pd.DataFrame(all_rows).drop_duplicates()
        print(f"  Total unique records: {len(df)}")
        return df
    else:
        print("  No data collected")
        return pd.DataFrame()

# Connect to database
con = sqlite3.connect("../../Data/TeamData.sqlite")

# Get season stats data for each year range specified in config
for key, value in config['get-data'].items():
    start_year = int(value['start_year'])
    end_year = int(value['end_year'])
    
    print(f"\n=== Getting CFBD Advanced Season Stats for {key}: {start_year}-{end_year} ===")
    
    # Collect advanced season statistics
    df = harvest_season_stats(start_year=start_year, end_year=end_year, sleep=0.3)
    
    if not df.empty:
        # Show detailed data structure before storing
        print(f"\n  📊 Final DataFrame Structure:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample teams: {df['team'].head().tolist()}")
        print(f"\n  📋 Sample Data (first 3 rows):")
        print(df.head(3).to_string(index=False))
        
        # Store data in database with year range as table name
        table_name = f"{key}_advanced_stats"
        df.to_sql(table_name, con, if_exists="replace", index=False)
        print(f"\n✓ Stored {len(df)} advanced stat records in table '{table_name}'")
    else:
        print(f"✗ No data collected for {key}")

con.close()
print("\n🎉 Advanced season stats data collection complete!")
