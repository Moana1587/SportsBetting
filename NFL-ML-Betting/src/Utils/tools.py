import re
import time
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'https://site.api.espn.com',
    'Referer': 'https://www.espn.com/'
}

data_headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Referer': 'https://gist.github.com/',
    'Sec-Ch-Ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36'
}


def get_json_data(url, timeout=30, max_retries=3):
    """Get JSON data from URL with timeout and retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting API call (attempt {attempt + 1}/{max_retries}): {url}")
            raw_data = requests.get(url, headers=data_headers, timeout=timeout)
            raw_data.raise_for_status()  # Raise an exception for bad status codes
            
            try:
                json_data = raw_data.json()
                print(f"Successfully retrieved data from API")
                return json_data
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return {}
                
        except requests.exceptions.Timeout:
            print(f"Timeout error on attempt {attempt + 1}/{max_retries} for URL: {url}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached. Skipping this request.")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached. Skipping this request.")
                return {}
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            return {}
    
    return {}


def get_todays_games_json(url, timeout=30, max_retries=3):
    """Get today's games JSON from ESPN API with proper error handling"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting ESPN API call (attempt {attempt + 1}/{max_retries}): {url}")
            raw_data = requests.get(url, headers=games_header, timeout=timeout)
            raw_data.raise_for_status()  # Raise an exception for bad status codes
            
            try:
                json_data = raw_data.json()
                print(f"Successfully retrieved games data from ESPN API")
                
                # Extract events from the response
                events = json_data.get('events', [])
                if not events:
                    print("No events found in API response")
                    return []
                
                print(f"Found {len(events)} events in API response")
                return events
                
            except Exception as e:
                print(f"Error parsing JSON from ESPN API: {e}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"Timeout error on attempt {attempt + 1}/{max_retries} for ESPN API: {url}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached for ESPN API. Returning empty games list.")
                return []
                
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
            if "Failed to resolve" in str(e) or "getaddrinfo failed" in str(e):
                print(f"DNS resolution failed for ESPN API. This may be a network connectivity issue.")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached for ESPN API. Returning empty games list.")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached for ESPN API. Returning empty games list.")
                return []
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            return []
    
    return []


def to_data_frame(data):
    try:
        # ESPN NFL teams API returns data in specific format
        if isinstance(data, dict) and 'sports' in data:
            teams_data = []
            for sport in data['sports']:
                if sport.get('name') == 'Football':
                    for league in sport.get('leagues', []):
                        if league.get('name') == 'NFL':
                            for team in league.get('teams', []):
                                team_info = team.get('team', {})
                                # Extract basic team stats - you may need to adjust based on actual API response
                                team_data = {
                                    'TEAM_ID': team_info.get('id', ''),
                                    'TEAM_NAME': team_info.get('displayName', ''),
                                    'WINS': 0,  # These would need to be fetched from stats API
                                    'LOSSES': 0,
                                    'TIES': 0,
                                    'PCT': 0.0,
                                    'POINTS_FOR': 0,
                                    'POINTS_AGAINST': 0,
                                    'POINT_DIFF': 0
                                }
                                teams_data.append(team_data)
            return pd.DataFrame(teams_data)
        elif isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data=data)
        else:
            return pd.DataFrame(data={})
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})


def create_todays_games(input_list):
    games = []
    for game in input_list:
        # Process all games regardless of status
        status = game.get('status', {})
        status_type = status.get('type', {})
        status_name = status_type.get('name', '')
        
        # Include all games regardless of status (scheduled, in progress, completed, etc.)
        competitions = game.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            if len(competitors) >= 2:
                # Get team information
                home_team_info = competitors[0].get('team', {})
                away_team_info = competitors[1].get('team', {})
                
                # Extract team names - try different possible fields
                home_team = (home_team_info.get('displayName', '') or 
                           home_team_info.get('name', '') or 
                           home_team_info.get('shortDisplayName', ''))
                
                away_team = (away_team_info.get('displayName', '') or 
                           away_team_info.get('name', '') or 
                           away_team_info.get('shortDisplayName', ''))
                
                if home_team and away_team:
                    games.append([home_team, away_team])
                    print(f"Found game: {away_team} @ {home_team} (Status: {status_name})")
    
    if not games:
        print("No games found for today")
    else:
        print(f"Successfully parsed {len(games)} games from ESPN API")
    
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
