import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import toml
import requests
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

# NHL API configuration
nhl_base_url = "https://api.nhle.com/stats/rest/en/team/summary"
nhl_headers = {
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_nhl_team_data(season_id):
    """Fetch NHL team summary data for a specific season"""
    params = {
        'isAggregate': 'false',
        'isGame': 'false',
        'start': '0',
        'limit': '200',
        'cayenneExp': f'seasonId={season_id}'
    }
    
    try:
        response = requests.get(nhl_base_url, headers=nhl_headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        print(f"Error fetching data for season {season_id}: {e}")
        return []

def nhl_data_to_dataframe(data):
    """Convert NHL API data to pandas DataFrame"""
    if not data:
        return pd.DataFrame()
    
    # Flatten the data and create DataFrame
    df = pd.DataFrame(data)
    
    # Map team names to full team names to match odds data format
    team_name_mapping = {
        'Anaheim Ducks': 'Anaheim Ducks',
        'Arizona Coyotes': 'Arizona Coyotes',
        'Boston Bruins': 'Boston Bruins',
        'Buffalo Sabres': 'Buffalo Sabres',
        'Calgary Flames': 'Calgary Flames',
        'Carolina Hurricanes': 'Carolina Hurricanes',
        'Chicago Blackhawks': 'Chicago Blackhawks',
        'Colorado Avalanche': 'Colorado Avalanche',
        'Columbus Blue Jackets': 'Columbus Blue Jackets',
        'Dallas Stars': 'Dallas Stars',
        'Detroit Red Wings': 'Detroit Red Wings',
        'Edmonton Oilers': 'Edmonton Oilers',
        'Florida Panthers': 'Florida Panthers',
        'Los Angeles Kings': 'Los Angeles Kings',
        'Minnesota Wild': 'Minnesota Wild',
        'Montreal Canadiens': 'Montréal Canadiens',  # Note: API uses "Montreal" but odds uses "Montréal"
        'Nashville Predators': 'Nashville Predators',
        'New Jersey Devils': 'New Jersey Devils',
        'New York Islanders': 'New York Islanders',
        'New York Rangers': 'New York Rangers',
        'Ottawa Senators': 'Ottawa Senators',
        'Philadelphia Flyers': 'Philadelphia Flyers',
        'Pittsburgh Penguins': 'Pittsburgh Penguins',
        'San Jose Sharks': 'San Jose Sharks',
        'Seattle Kraken': 'Seattle Kraken',
        'St. Louis Blues': 'St. Louis Blues',
        'Tampa Bay Lightning': 'Tampa Bay Lightning',
        'Toronto Maple Leafs': 'Toronto Maple Leafs',
        'Utah Hockey Club': 'Utah Hockey Club',
        'Vancouver Canucks': 'Vancouver Canucks',
        'Vegas Golden Knights': 'Vegas Golden Knights',
        'Washington Capitals': 'Washington Capitals',
        'Winnipeg Jets': 'Winnipeg Jets',
        'Atlanta Thrashers': 'Atlanta Thrashers',  # Historical team
        'Phoenix Coyotes': 'Arizona Coyotes'  # Historical name
    }
    
    # Update team names to match odds data format
    if 'teamFullName' in df.columns:
        print(f"  Original team names: {list(df['teamFullName'].unique())}")
        df['teamFullName'] = df['teamFullName'].map(team_name_mapping).fillna(df['teamFullName'])
        print(f"  Mapped team names: {list(df['teamFullName'].unique())}")
    
    return df

con = sqlite3.connect("../../Data/TeamData.sqlite")

for key, value in config['get-data'].items():
    # Extract season ID from the key (e.g., "2012-13" -> 20122013)
    season_parts = key.split('-')
    if len(season_parts) == 2:
        start_year = season_parts[0]
        end_year = season_parts[1]
        # For NHL API: season "2012-13" becomes seasonId "20122013"
        season_id = int(start_year + "20" + end_year)
    else:
        print(f"Invalid season format: {key}")
        continue
    
    print(f"Getting NHL team data for season: {key} (seasonId: {season_id})")
    
    # Get team data for the season
    team_data = get_nhl_team_data(season_id)
    
    if team_data:
        df = nhl_data_to_dataframe(team_data)
        
        if not df.empty:
            # Add season information (avoid duplicate seasonId column)
            df['Season'] = key
            # Rename the existing seasonId column to avoid confusion
            if 'seasonId' in df.columns:
                df = df.rename(columns={'seasonId': 'NHL_SeasonId'})
            
            # Drop the ties column since it's always null
            if 'ties' in df.columns:
                df = df.drop(columns=['ties'])
            
            # Save to database with season as table name
            table_name = f"season_{key.replace('-', '_')}"
            df.to_sql(table_name, con, if_exists="replace", index=False)
            print(f"Saved {len(df)} team records for season {key}")
        else:
            print(f"No data found for season {key}")
    else:
        print(f"Failed to fetch data for season {key}")
    
    # Add delay between requests
    time.sleep(random.randint(1, 3))

con.close()
print("Data collection completed!")
