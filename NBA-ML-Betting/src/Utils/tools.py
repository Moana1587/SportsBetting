import re
import time
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

games_header = {'Accept': 'application/json, text/plain, */*','Accept-Encoding': 'gzip, deflate, br',
          'Accept-Language': 'en-US,en;q=0.9','Connection': 'keep-alive','Host': 'stats.nba.com',
          'Referer': 'https://stats.nba.com/','Sec-Fetch-Mode': 'cors','Sec-Fetch-Site': 'same-origin',
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
          'x-nba-stats-origin': 'stats','x-nba-stats-token':'true',}

# data_headers = {
#     'Accept': 'application/json, text/plain, */*',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Host': 'stats.nba.com',
#     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
#     'Accept-Language': 'en-US,en;q=0.9',
#     'Referer': 'https://www.nba.com/',
#     'Connection': 'keep-alive'
# }


data_headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/stats/",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}


def get_json_data(url=None, params=None, max_retries=3, delay=2):
    """
    Fetch JSON data from NBA API with retry logic and proper error handling.
    
    Args:
        url (str): URL to fetch data from (optional, defaults to NBA stats API)
        params (dict): Parameters for the API request (optional)
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
    
    Returns:
        dict: JSON data or empty dict if all retries fail
    """
    # Default to NBA stats API if no URL provided
    if url is None:
        url = "https://stats.nba.com/stats/leaguedashteamstats"
    
    # Default parameters for team stats
    if params is None:
        params = {
            "Season": "2024-25",
            "SeasonType": "Regular Season",
            "PerMode": "PerGame",
            "MeasureType": "Base",
            "LeagueID": "00"
        }
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to fetch data from NBA API (attempt {attempt + 1}/{max_retries})...")
            
            # Use modern approach with params
            response = requests.get(url, params=params, headers=data_headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            json_data = response.json()
            print("Successfully fetched data from NBA API")
            return json_data.get('resultSets', {})
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("All connection attempts failed. NBA API may be temporarily unavailable.")
                return {}
                
        except requests.exceptions.Timeout as e:
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All timeout attempts failed.")
                return {}
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error on attempt {attempt + 1}: {e}")
            if e.response.status_code == 429:  # Rate limited
                if attempt < max_retries - 1:
                    print(f"Rate limited. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print("Rate limited and all retries exhausted.")
                    return {}
            else:
                print(f"HTTP error {e.response.status_code}: {e}")
                return {}
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All attempts failed due to unexpected error.")
                return {}
    
    return {}


def get_todays_games_json(url, max_retries=3, delay=2):
    """
    Fetch today's games JSON data with retry logic and proper error handling.
    
    Args:
        url (str): URL to fetch games data from
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
    
    Returns:
        list: Games data or empty list if all retries fail
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting to fetch games data (attempt {attempt + 1}/{max_retries})...")
            
            response = requests.get(url, headers=games_header, timeout=30)
            response.raise_for_status()
            
            json_data = response.json()
            print("Successfully fetched games data")
            return json_data.get('gs', {}).get('g', [])
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All connection attempts failed for games data.")
                return []
                
        except requests.exceptions.Timeout as e:
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All timeout attempts failed for games data.")
                return []
                
        except Exception as e:
            print(f"Error fetching games data on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All attempts failed for games data.")
                return []
    
    return []


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
