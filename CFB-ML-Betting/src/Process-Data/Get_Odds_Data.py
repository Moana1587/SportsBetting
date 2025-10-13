import os
import sqlite3
import sys
import pandas as pd
import requests
import time
from datetime import datetime

import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

# CFBD API configuration
API = os.environ.get("CFBD_KEY") or "dTYsx4FGjqSNJdqnIylk32HsJ6dBGeTry5nqHg3R+rKnu/745bzicjQF6GDlSTQf"
H = {"Authorization": f"Bearer {API}"}

def create_odds_data_table(con):
    """Create the odds_data table with the required structure from the image"""
    cursor = con.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS odds_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            Home TEXT,
            Away TEXT,
            OU REAL,
            Spread REAL,
            ML_Home REAL,
            ML_Away REAL,
            Points INTEGER,
            Win_Margin INTEGER
        )
    ''')
    con.commit()

def cfbd_get_lines(year: int, season_type: str = "regular"):
    """Get betting lines from CFBD API for a given year and season type."""
    url = "https://api.collegefootballdata.com/lines"
    params = {"year": year, "seasonType": season_type}
    r = requests.get(url, params=params, headers=H, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_lines_data(games_data, year):
    """Flatten the lines data from CFBD API into a structured format."""
    all_rows = []
    
    for game in games_data:
        base_info = {
            "game_id": game.get("id"),
            "season": game.get("season", year),
            "season_type": game.get("seasonType"),
            "week": game.get("week"),
            "start_date": game.get("startDate"),
            "home_team_id": game.get("homeTeamId"),
            "home_team": game.get("homeTeam"),
            "home_conference": game.get("homeConference"),
            "home_classification": game.get("homeClassification"),
            "home_score": game.get("homeScore"),
            "away_team_id": game.get("awayTeamId"),
            "away_team": game.get("awayTeam"),
            "away_conference": game.get("awayConference"),
            "away_classification": game.get("awayClassification"),
            "away_score": game.get("awayScore")
        }
        
        # Calculate derived fields
        if base_info["home_score"] is not None and base_info["away_score"] is not None:
            base_info["total_points"] = base_info["home_score"] + base_info["away_score"]
            base_info["win_margin"] = base_info["home_score"] - base_info["away_score"]
        else:
            base_info["total_points"] = None
            base_info["win_margin"] = None
        
        # Process each line provider for this game
        lines = game.get("lines", [])
        if lines:
            for line in lines:
                row = base_info.copy()
                row.update({
                    "provider": line.get("provider"),
                    "spread": line.get("spread"),
                    "formatted_spread": line.get("formattedSpread"),
                    "spread_open": line.get("spreadOpen"),
                    "over_under": line.get("overUnder"),
                    "over_under_open": line.get("overUnderOpen"),
                    "home_moneyline": line.get("homeMoneyline"),
                    "away_moneyline": line.get("awayMoneyline")
                })
                
                # Only add row if no critical fields are null
                if is_valid_record(row):
                    all_rows.append(row)
                else:
                    print(f"  Skipping record with null values: Game {row.get('game_id')}, Provider: {row.get('provider')}")
        else:
            # If no lines data, skip this game entirely
            print(f"  Skipping game {base_info.get('game_id')} - no betting lines available")
    
    return all_rows

def is_valid_record(row):
    """Check if a record has all required fields non-null."""
    required_fields = [
        "game_id", "season", "season_type", "week", "start_date",
        "home_team_id", "home_team", "away_team_id", "away_team",
        "provider", "spread", "over_under", "home_moneyline", "away_moneyline"
    ]
    
    for field in required_fields:
        if row.get(field) is None:
            return False
    return True


def harvest_odds_data(start_year=2009, end_year=2025, season_type="regular", sleep=0.3):
    """Harvest odds data from CFBD API for specified year range."""
    all_rows = []
    
    for year in range(start_year, end_year + 1):
        try:
            print(f"  Fetching odds data for {year}...")
            data = cfbd_get_lines(year, season_type)
            print(f"  Raw API response for {year}: {len(data)} games")
            
            if data:
                print(f"  Sample raw game: {data[0] if data else 'No data'}")
            
            flattened_data = flatten_lines_data(data, year)
            all_rows += flattened_data
            print(f"  Found {len(flattened_data)} betting lines for {year}")
            
            if flattened_data:
                print(f"  Sample flattened record: {flattened_data[0] if flattened_data else 'No data'}")
            
            time.sleep(sleep)  # be polite to free tier limits
            
        except requests.HTTPError as e:
            print(f"  [{year}] HTTP error: {e}")
        except Exception as e:
            print(f"  [{year}] Error: {e}")
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"  Total records before validation: {len(df)}")
        
        # Additional validation - remove any rows with null values in critical fields
        initial_count = len(df)
        df = df.dropna(subset=[
            "game_id", "season", "season_type", "week", "start_date",
            "home_team_id", "home_team", "away_team_id", "away_team",
            "provider", "spread", "over_under", "home_moneyline", "away_moneyline"
        ])
        
        filtered_count = len(df)
        if initial_count != filtered_count:
            print(f"  Filtered out {initial_count - filtered_count} records with null values")
        
        print(f"  Total valid records: {len(df)}")
        return df
    else:
        print("  No data collected")
        return pd.DataFrame()

def load_odds_data_to_database():
    """Load odds data from CFBD API into OddsData.sqlite"""
    print("Loading odds data from CFBD API...")
    
    # Connect to SQLite database
    con = sqlite3.connect("../../Data/OddsData.sqlite")
    
    # Create table if it doesn't exist
    create_odds_data_table(con)
    
    try:
        # Get data for each year range specified in config
        for key, value in config['get-odds-data'].items():
            start_year = int(value['start_year'])
            end_year = int(value['end_year'])
            
            print(f"\n=== Getting CFBD Odds Data for {key}: {start_year}-{end_year} ===")
            
            # Collect odds data
            df = harvest_odds_data(start_year=start_year, end_year=end_year, season_type="regular", sleep=0.3)
            
            if not df.empty:
                # Final validation - ensure no null values in critical fields
                before_validation = len(df)
                df = df.dropna(subset=[
                    "game_id", "season", "season_type", "week", "start_date",
                    "home_team_id", "home_team", "away_team_id", "away_team",
                    "provider", "spread", "over_under", "home_moneyline", "away_moneyline",
                    "home_score", "away_score"
                ])
                after_validation = len(df)
                
                if before_validation != after_validation:
                    print(f"  Final validation: Removed {before_validation - after_validation} records with null values")
                
                if not df.empty:
                    # Transform DataFrame to match the image field names
                    transformed_df = pd.DataFrame()
                    transformed_df['Date'] = df['start_date']
                    transformed_df['Home'] = df['home_team']
                    transformed_df['Away'] = df['away_team']
                    transformed_df['OU'] = df['over_under']
                    transformed_df['Spread'] = df['spread']
                    transformed_df['ML_Home'] = df['home_moneyline']
                    transformed_df['ML_Away'] = df['away_moneyline']
                    transformed_df['Points'] = df['home_score'] + df['away_score']
                    transformed_df['Win_Margin'] = df['home_score'] - df['away_score']
                    
                    # Final validation on transformed data
                    initial_transformed_count = len(transformed_df)
                    transformed_df = transformed_df.dropna(subset=[
                        'Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away',
                        'Points', 'Win_Margin'
                    ])
                    filtered_transformed_count = len(transformed_df)
                    if initial_transformed_count != filtered_transformed_count:
                        print(f"  Filtered out {initial_transformed_count - filtered_transformed_count} records due to nulls in target fields")
                    
                    if not transformed_df.empty:
                        # Show detailed data structure before storing
                        print(f"\n  📊 Transformed DataFrame Structure:")
                        print(f"  Shape: {transformed_df.shape}")
                        print(f"  Columns: {list(transformed_df.columns)}")
                        print(f"  Sample games: {transformed_df['Home'].head().tolist()} vs {transformed_df['Away'].head().tolist()}")
                        print(f"\n  📋 Sample Data (first 3 rows):")
                        print(transformed_df.head(3).to_string(index=False))
                        
                        # Store data in database with year range as table name
                        table_name = f"{key}_odds_data"
                        transformed_df.to_sql(table_name, con, if_exists="replace", index=False)
                        print(f"\n✓ Stored {len(transformed_df)} valid odds records in table '{table_name}'")
                    else:
                        print(f"✗ No valid records remaining after transformation for {key}")
                else:
                    print(f"✗ No valid records remaining after validation for {key}")
            else:
                print(f"✗ No data collected for {key}")
        
        # Also store in main odds_data table
        print(f"\n=== Storing all odds data in main table ===")
        all_transformed_data = []
        for key, value in config['get-odds-data'].items():
            start_year = int(value['start_year'])
            end_year = int(value['end_year'])
            print(f"  Processing {key} for combined table...")
            df = harvest_odds_data(start_year=start_year, end_year=end_year, season_type="regular", sleep=0.3)
            if not df.empty:
                # Apply validation before adding to combined data
                df = df.dropna(subset=[
                    "game_id", "season", "season_type", "week", "start_date",
                    "home_team_id", "home_team", "away_team_id", "away_team",
                    "provider", "spread", "over_under", "home_moneyline", "away_moneyline",
                    "home_score", "away_score"
                ])
                if not df.empty:
                    # Transform DataFrame to match the image field names
                    transformed_df = pd.DataFrame()
                    transformed_df['Date'] = df['start_date']
                    transformed_df['Home'] = df['home_team']
                    transformed_df['Away'] = df['away_team']
                    transformed_df['OU'] = df['over_under']
                    transformed_df['Spread'] = df['spread']
                    transformed_df['ML_Home'] = df['home_moneyline']
                    transformed_df['ML_Away'] = df['away_moneyline']
                    transformed_df['Points'] = df['home_score'] + df['away_score']
                    transformed_df['Win_Margin'] = df['home_score'] - df['away_score']
                    
                    # Final validation on transformed data
                    transformed_df = transformed_df.dropna(subset=[
                        'Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away',
                        'Points', 'Win_Margin'
                    ])
                    
                    if not transformed_df.empty:
                        all_transformed_data.append(transformed_df)
        
        if all_transformed_data:
            combined_df = pd.concat(all_transformed_data, ignore_index=True)
            print(f"\n  📊 Combined DataFrame Structure:")
            print(f"  Shape: {combined_df.shape}")
            print(f"  Columns: {list(combined_df.columns)}")
            print(f"  Sample games: {combined_df['Home'].head().tolist()} vs {combined_df['Away'].head().tolist()}")
            print(f"\n  📋 Sample Combined Data (first 3 rows):")
            print(combined_df.head(3).to_string(index=False))
            
            combined_df.to_sql('odds_data', con, if_exists='replace', index=False)
            print(f"\n✓ Stored {len(combined_df)} total valid odds records in main 'odds_data' table")
        else:
            print("✗ No data collected for combined 'odds_data' table")
        
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    load_odds_data_to_database()
