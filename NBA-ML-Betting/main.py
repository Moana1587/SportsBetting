import argparse
from datetime import datetime, timedelta
import requests
import time
import toml

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style
from nba_api.stats.endpoints import leaguedashteamstats

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

def load_config():
    """Load configuration from config.toml file"""
    try:
        with open('config.toml', 'r') as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print("Warning: config.toml not found. Using default configuration.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading config.toml: {e}. Using default configuration.")
        return {}

def get_api_params(config):
    """Get API parameters from config or use defaults"""
    # Default parameters
    default_params = {
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "PerMode": "PerGame",
        "MeasureType": "Base",
        "LeagueID": "00"
    }
    
    # Check if config has API parameters
    if 'api_params' in config:
        # Merge config params with defaults
        params = default_params.copy()
        params.update(config['api_params'])
        return params
    
    return default_params

def get_team_stats_nba_api(config, max_retries=3, delay=2):
    """
    Fetch team statistics using the nba_api library with retry logic.
    
    Args:
        config (dict): Configuration dictionary
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
    
    Returns:
        pandas.DataFrame: Team statistics DataFrame or empty DataFrame if all retries fail
    """
    # Get API parameters from config
    api_params = get_api_params(config)
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to fetch team stats using nba_api (attempt {attempt + 1}/{max_retries})...")
            
            # Use nba_api to get team stats
            teams = leaguedashteamstats.LeagueDashTeamStats(
                season=api_params.get('Season', '2024-25'),
                season_type_all_star=api_params.get('SeasonType', 'Regular Season'),
                per_mode_detailed=api_params.get('PerMode', 'PerGame'),
                measure_type_detailed_defense=api_params.get('MeasureType', 'Base')
            )
            
            # Get the DataFrame from the API response
            df = teams.get_data_frames()[0]
            
            print(f"Successfully fetched team stats using nba_api: {len(df)} teams")
            return df
            
        except Exception as e:
            print(f"Error fetching team stats on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All attempts failed for nba_api.")
                return pd.DataFrame()
    
    return pd.DataFrame()

def get_todays_nba_games():
    """Fetch today's NBA games from ESPN API with retry logic"""
    today = datetime.now().strftime('%Y%m%d')
    espn_games_url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}'
    
    max_retries = 3
    delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to fetch NBA games from ESPN API (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(espn_games_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            games = []
            if 'events' in data:
                for event in data['events']:
                    # Extract team information from competitors
                    home_team = None
                    away_team = None
                    
                    for competitor in event.get('competitions', [{}])[0].get('competitors', []):
                        if competitor.get('homeAway') == 'home':
                            home_team = competitor['team']['displayName']
                        elif competitor.get('homeAway') == 'away':
                            away_team = competitor['team']['displayName']
                    
                    if home_team and away_team:
                        games.append([home_team, away_team])
            
            print(f"Successfully fetched {len(games)} NBA games from ESPN API")
            return games
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All connection attempts failed for ESPN API.")
                return []
                
        except requests.exceptions.Timeout as e:
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All timeout attempts failed for ESPN API.")
                return []
                
        except Exception as e:
            print(f"Error fetching NBA games from ESPN API on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("All attempts failed for ESPN API.")
                return []
    
    return []

todays_games_url = 'http://data.nba.net/v2015/json/mobile_teams/nba/2025/scores/00_todays_scores.json'#'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'

# Load configuration
config = load_config()

# Use modern NBA API approach with parameters
print("Using modern NBA API approach with parameters")


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds is not None:
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])

            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            # Use default values when no odds are provided and running non-interactively
            print(f"No odds provided for {home_team} vs {away_team}. Using default values.")
            todays_games_uo.append(220.0)  # Default over/under line
            home_team_odds.append(-110)    # Default home team odds
            away_team_odds.append(-110)    # Default away team odds

        # calculate days rest for both teams
        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + datetime.today() - last_home_date
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + datetime.today() - last_away_date
        else:
            away_days_off = timedelta(days=7)
        # print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")

        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return
        if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
            print(games[0][0] + ':' + games[0][1])
            print(Fore.RED,"--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
            print(Style.RESET_ALL)
            odds = None
        else:
            print(f"------------------{args.odds} odds data------------------")
            for g in odds.keys():
                home_team, away_team = g.split(":")
                print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
    else:
        games = get_todays_nba_games()
        if len(games) == 0:
            print("No NBA games found for today.")
            return
    # Get team statistics using nba_api
    print("Fetching team statistics using nba_api...")
    df = get_team_stats_nba_api(config)
    
    if df.empty:
        print("Failed to fetch NBA team statistics data from nba_api.")
        print("Attempting to use cached data if available...")
        
        # Try to load from a cached file if API fails
        try:
            import os
            import pickle
            cache_file = "Data/cached_team_stats.pkl"
            if os.path.exists(cache_file):
                print(f"Loading cached team statistics from {cache_file}")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                if not df.empty:
                    print("Successfully loaded cached team statistics data.")
                else:
                    print("Cached data is empty or invalid.")
                    return
            else:
                print("No cached data available. Cannot proceed with predictions.")
                return
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Cannot proceed with predictions.")
            return
    else:
        # Cache the successfully fetched data
        try:
            import os
            import pickle
            os.makedirs("Data", exist_ok=True)
            cache_file = "Data/cached_team_stats.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"Team statistics data cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")
    
    if df.empty:
        print("No team statistics data available. Cannot proceed with predictions.")
        return
        
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()
