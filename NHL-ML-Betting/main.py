import argparse
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from colorama import Fore, Style

from src.Predict import XGBoost_Runner

# Timezone configuration
TZ = ZoneInfo("America/Los_Angeles")

def scrape_todays_nhl_schedule():
    """Scrape today's NHL game schedule from the ESPN NHL scoreboard API"""
    try:
        today = datetime.now(TZ).date().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={today}"
        
        print(f"Scraping NHL schedule for {today}...")
        
        # Try with different timeout and retry settings
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9"
        })
        
        r = session.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        
        rows = []
        for event in js.get("events", []):
            # Parse game date and time
            game_date_str = event.get("date", "")
            if game_date_str:
                # ESPN uses ISO format with timezone
                t_utc = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                start_local = t_utc.astimezone(TZ).strftime("%Y-%m-%d %H:%M")
            else:
                start_local = "TBD"
            
            # Get team information
            competitions = event.get("competitions", [])
            if not competitions:
                continue
                
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                continue
                
            # Find home and away teams
            home_team = None
            away_team = None
            
            for competitor in competitors:
                if competitor.get("homeAway") == "home":
                    home_team = competitor.get("team", {}).get("displayName", "")
                elif competitor.get("homeAway") == "away":
                    away_team = competitor.get("team", {}).get("displayName", "")
            
            if not home_team or not away_team:
                continue
            
            # Get game status
            status = competition.get("status", {}).get("type", {}).get("name", "Scheduled")
            
            # Map ESPN status to our expected status
            status_mapping = {
                "STATUS_SCHEDULED": "Scheduled",
                "STATUS_IN_PROGRESS": "In Progress",
                "STATUS_HALFTIME": "In Progress",
                "STATUS_FINAL": "Final",
                "STATUS_POSTPONED": "Postponed",
                "STATUS_CANCELLED": "Cancelled"
            }
            status = status_mapping.get(status, status)
            
            rows.append({
                "gamePk": event.get("id", ""),
                "start_local": start_local,
                "away": map_espn_team_names(away_team),
                "home": map_espn_team_names(home_team),
                "venue": competition.get("venue", {}).get("fullName", ""),
                "status": status,
            })
        
        df = pd.DataFrame(rows).sort_values("start_local")
        
        if df.empty:
            print("No NHL games scheduled for today.")
            return None
        
        print(f"Found {len(df)} NHL games scheduled for today:")
        for _, game in df.iterrows():
            print(f"   {game['away']} @ {game['home']} - {game['start_local']} ({game['status']})")
        
        # Save to CSV for reference
        df.to_csv("nhl_today_schedule.csv", index=False)
        print(f"Schedule saved to nhl_today_schedule.csv")
        
        return df
        
    except requests.exceptions.ConnectionError as e:
        print(f"Network connection error: {e}")
        print("This could be due to:")
        print("   - Network connectivity issues")
        print("   - Firewall blocking the request")
        print("   - DNS resolution problems")
        print("   - ESPN API being temporarily unavailable")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Request timeout: {e}")
        print("The ESPN API is taking too long to respond")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error scraping NHL schedule: {e}")
        return None

def map_espn_team_names(team_name):
    """Map ESPN team names to full team names used in our database"""
    espn_to_full_name = {
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
        'Montreal Canadiens': 'Montréal Canadiens',  # ESPN uses "Montreal" but our DB uses "Montréal"
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
    
    return espn_to_full_name.get(team_name, team_name)


def load_nhl_team_data():
    """Load NHL team data from the database"""
    try:
        # Load the most recent team data
        con = sqlite3.connect("Data/TeamData.sqlite")
        
        # Get the most recent season
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
        season_tables = [t[0] for t in tables.values if t[0].startswith('season_')]
        
        if not season_tables:
            print("No team data found. Please run Get_Data.py first.")
            return None
        
        # Get the most recent season
        latest_season = sorted(season_tables)[-1]
        team_df = pd.read_sql_query(f"SELECT * FROM {latest_season}", con)
        con.close()
        
        print(f"Loaded team data from {latest_season}: {len(team_df)} teams")
        return team_df
        
    except Exception as e:
        print(f"Error loading team data: {e}")
        return None

def get_team_series(team_data, team_name):
    """Get team statistics series for a specific team"""
    try:
        # Find the team in the data
        team_row = team_data[team_data['teamFullName'] == team_name]
        
        if not team_row.empty:
            return team_row.iloc[0]
        else:
            print(f"    Team not found: {team_name}")
            return None
            
    except Exception as e:
        print(f"    Error getting team series for {team_name}: {e}")
        return None

def prepare_nhl_prediction_data(games, team_df, custom_ou=None, custom_spread=None, custom_home_odds=None, custom_away_odds=None):
    """Prepare NHL prediction data for XGBoost models"""
    match_data = []
    todays_games_uo = []
    todays_games_spread = []
    home_team_odds = []
    away_team_odds = []
    
    print(f"Preparing prediction data for {len(games)} games...")
    
    for i, (home_team, away_team) in enumerate(games):
        print(f"   Processing: {away_team} @ {home_team}")
        
        # Get team statistics
        home_team_series = get_team_series(team_df, home_team)
        away_team_series = get_team_series(team_df, away_team)
        
        if home_team_series is None or away_team_series is None:
            print(f"     Warning: Could not find team data for {home_team} vs {away_team}")
            continue
        
        # Combine team statistics - this creates the same structure as training data
        combined_stats = pd.concat([
            home_team_series,
            away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values})
        ])
        
        # Add NHL_SeasonId values for classification model compatibility
        # Use current season ID (2025-26 season = 20252026)
        current_season_id = 20252026
        combined_stats['NHL_SeasonId'] = current_season_id
        combined_stats['NHL_SeasonId.1'] = current_season_id
        
        # Note: Days-Rest columns are not used in training, so we don't add them here
        
        match_data.append(combined_stats)
        
        # Use custom odds if provided, otherwise use defaults
        # Updated default OU to be more realistic for current NHL games (typically 6.0-6.5)
        todays_games_uo.append(custom_ou[i] if custom_ou and i < len(custom_ou) else 6.0)
        todays_games_spread.append(custom_spread[i] if custom_spread and i < len(custom_spread) else 1.5)
        home_team_odds.append(custom_home_odds[i] if custom_home_odds and i < len(custom_home_odds) else -110)
        away_team_odds.append(custom_away_odds[i] if custom_away_odds and i < len(custom_away_odds) else -110)
    
    if not match_data:
        print("No valid games found.")
        return None, None, None, None, None, None
    
    # Create DataFrame
    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T
    
    # Remove non-predictive columns - but keep NHL_SeasonId for classification model
    columns_to_drop = [
        'Score',           # Game outcome - not predictive
        'Home-Score',      # Used to calculate target - not predictive
        'Away-Score',      # Used to calculate target - not predictive
        'Home-Team-Win',   # Game outcome - not predictive
        'OU-Cover',        # Game outcome - not predictive
        'OU',              # Game outcome - not predictive
        'Spread',          # Used to calculate target - not predictive
        'teamFullName',    # Team name - not predictive
        'teamFullName.1',  # Away team name - not predictive
        'Season',          # Season - not predictive for individual games
        'Season.1',        # Season - not predictive for individual games
        # Note: Keeping NHL_SeasonId and NHL_SeasonId.1 for classification model compatibility
        'teamId',          # Team ID - not predictive
        'teamId.1',        # Team ID - not predictive
    ]
    
    # Drop columns that exist in the dataset
    existing_columns_to_drop = [col for col in columns_to_drop if col in games_data_frame.columns]
    frame_ml = games_data_frame.drop(columns=existing_columns_to_drop)
    
    print(f"Features shape: {frame_ml.shape}")
    print(f"Dropped columns: {existing_columns_to_drop}")
    print(f"Remaining columns: {list(frame_ml.columns)}")
    
    # Convert to numeric
    for col in frame_ml.columns:
        frame_ml[col] = pd.to_numeric(frame_ml[col], errors='coerce')
    frame_ml = frame_ml.fillna(0)
    
    # Return the DataFrame with column names, not just values
    return frame_ml, todays_games_uo, todays_games_spread, frame_ml, home_team_odds, away_team_odds

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
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

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
        # Note: Days-Rest columns are not used in training, so we don't add them here
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds

def export_predictions_to_csv(games, prediction_results, todays_games_uo, todays_games_spread, home_team_odds, away_team_odds):
    """Export prediction results to CSV file"""
    try:
        # Create a list to store all prediction data
        csv_data = []
        
        # Get current date for filename
        current_date = datetime.now(TZ).strftime("%Y%m%d")
        filename = f"nhl_predictions_{current_date}.csv"
        
        # Process each game
        for i, (home_team, away_team) in enumerate(games):
            if i < len(todays_games_uo):
                # Get prediction data for this game
                ml_prediction = prediction_results.get('ml_predictions', [])[i] if i < len(prediction_results.get('ml_predictions', [])) else None
                ou_prediction = prediction_results.get('ou_predictions', [])[i] if i < len(prediction_results.get('ou_predictions', [])) else None
                spread_prediction = prediction_results.get('spread_predictions', [])[i] if i < len(prediction_results.get('spread_predictions', [])) else None
                
                # Determine ML winner and confidence
                if ml_prediction is not None:
                    ml_pred = ml_prediction
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ml_pred, '__len__') and len(ml_pred) > 0:
                        if ml_pred.ndim == 2:
                            ml_pred = ml_pred[0]  # Flatten 2D array to 1D
                        
                        if len(ml_pred) == 1:
                            # Old model format: single probability value
                            ml_confidence = round(float(ml_pred[0]), 3)
                            ml_winner = home_team if ml_confidence > 0.5 else away_team
                        else:
                            # New model format: [ML_Home, ML_Away, Home-Team-Win]
                            winner = int(ml_pred[2] > 0.5)  # Home-Team-Win > 0.5 means home team wins
                            ml_winner = home_team if winner == 1 else away_team
                            ml_confidence = round(ml_pred[2] if winner == 1 else (1 - ml_pred[2]), 3)
                    
                    # Convert to percentage and round
                    ml_confidence = round(ml_confidence * 100, 1)
                else:
                    ml_winner = "Unknown"
                    ml_confidence = 0.0
                
                # Determine OU prediction using classification model
                ou_pick = "Unknown"
                ou_confidence = 0.0
                ou_value = todays_games_uo[i] if i < len(todays_games_uo) else 6.0
                ou_value_confidence = 50.0
                
                # Try to get OU classification prediction first
                if 'ou_classification_predictions' in prediction_results and i < len(prediction_results['ou_classification_predictions']):
                    ou_class_pred = prediction_results['ou_classification_predictions'][i]
                    ou_class_confidence = prediction_results.get('ou_classification_confidence', [])[i] if i < len(prediction_results.get('ou_classification_confidence', [])) else 50.0
                    
                    # Classification model: 0 = UNDER, 1 = OVER
                    ou_pick = "OVER" if ou_class_pred == 1 else "UNDER"
                    ou_confidence = round(ou_class_confidence, 1)
                    
                    # Get OU value prediction for display
                    ou_value = prediction_results.get('ou_value_predictions', [])[i] if i < len(prediction_results.get('ou_value_predictions', [])) else (todays_games_uo[i] if i < len(todays_games_uo) else 6.0)
                    ou_value_confidence = prediction_results.get('ou_confidence', [])[i] if i < len(prediction_results.get('ou_confidence', [])) else 50.0
                
                # Fallback to old regression method if classification not available
                elif ou_prediction is not None:
                    ou_pred = ou_prediction
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ou_pred, '__len__') and len(ou_pred) > 0:
                        ou_pred = ou_pred[0] if hasattr(ou_pred, '__len__') else float(ou_pred)
                    else:
                        ou_pred = float(ou_pred)
                    
                    # Get OU value and confidence - use predicted value if available, otherwise use default
                    ou_value = prediction_results.get('ou_value_predictions', [])[i] if i < len(prediction_results.get('ou_value_predictions', [])) else (todays_games_uo[i] if i < len(todays_games_uo) else 6.0)
                    ou_value_confidence = prediction_results.get('ou_confidence', [])[i] if i < len(prediction_results.get('ou_confidence', [])) else 50.0
                    
                    # Determine if prediction is over or under using the predicted OU value as the line
                    if ou_pred > ou_value:
                        ou_pick = "OVER"
                        ou_confidence = round(abs(ou_pred - ou_value) / ou_value * 100, 1)
                    else:
                        ou_pick = "UNDER"
                        ou_confidence = round(abs(ou_pred - ou_value) / ou_value * 100, 1)
                
                # Determine spread prediction
                if spread_prediction is not None:
                    spread_pred = spread_prediction
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(spread_pred, '__len__') and len(spread_pred) > 0:
                        spread_pred = spread_pred[0] if hasattr(spread_pred, '__len__') else float(spread_pred)
                    else:
                        spread_pred = float(spread_pred)
                    
                    # Get spread value
                    spread_value = todays_games_spread[i] if i < len(todays_games_spread) else 0
                    
                    # Determine spread winner based on prediction vs actual spread
                    if spread_pred > spread_value:
                        spread_winner = home_team
                        spread_confidence = round(abs(spread_pred - spread_value) / abs(spread_value) * 100, 1) if spread_value != 0 else 50.0
                    else:
                        spread_winner = away_team
                        spread_confidence = round(abs(spread_pred - spread_value) / abs(spread_value) * 100, 1) if spread_value != 0 else 50.0
                    
                    spread_pick = f"{spread_winner} {spread_value:+.1f}"
                else:
                    spread_winner = "Unknown"
                    spread_confidence = 0.0
                    spread_pick = "Unknown"
                
                # Get ML values from the prediction results
                ml_home_value = 100  # Default ML value
                ml_away_value = 100  # Default ML value
                
                if 'ml_predictions' in prediction_results and i < len(prediction_results['ml_predictions']):
                    ml_pred = prediction_results['ml_predictions'][i]
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ml_pred, '__len__') and len(ml_pred) > 0:
                        if ml_pred.ndim == 2:
                            ml_pred = ml_pred[0]  # Flatten 2D array to 1D
                        
                        if len(ml_pred) >= 2:
                            # New model format: [ML_Home, ML_Away, Home-Team-Win]
                            ml_home_value = int(round(ml_pred[0]))
                            ml_away_value = int(round(ml_pred[1]))
                
                # Create row data
                row_data = {
                    'Date': current_date,
                    'Home_Team': home_team,
                    'Away_Team': away_team,
                    'ML_Winner': ml_winner,
                    'ML_Confidence': ml_confidence,
                    'ML_Home_Value': ml_home_value,
                    'ML_Away_Value': ml_away_value,
                    'Home_ML_Odds': home_team_odds[i] if i < len(home_team_odds) else -110,
                    'Away_ML_Odds': away_team_odds[i] if i < len(away_team_odds) else -110,
                    'OU': f"{ou_pick} {ou_value:.2f} ({round(ou_value_confidence, 1)}%)",
                    'Spread_Pick': spread_pick,
                    'Spread_Value': todays_games_spread[i] if i < len(todays_games_spread) else 0,
                    'Spread_Confidence': spread_confidence
                }
                
                csv_data.append(row_data)
        
        # Create DataFrame and save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            print(f"\nPrediction results exported to: {filename}")
            print(f"Exported {len(csv_data)} games with predictions")
            
            # Display summary
            print("\nCSV Export Summary:")
            print("=" * 40)
            for _, row in df.iterrows():
                print(f"{row['Away_Team']} @ {row['Home_Team']}")
                print(f"  ML: {row['ML_Winner']} ({row['ML_Confidence']}%) - Home: {row['ML_Home_Value']}, Away: {row['ML_Away_Value']}")
                print(f"  O/U: {row['OU']}")
                print(f"  Spread: {row['Spread_Pick']} ({row['Spread_Confidence']}%)")
                print()
        else:
            print("No prediction data to export")
            
    except Exception as e:
        print(f"Error exporting predictions to CSV: {e}")

def export_predictions_to_txt(games, prediction_results, todays_games_uo, todays_games_spread, home_team_odds, away_team_odds):
    """Export prediction results to TXT file in the specified format"""
    try:
        # Get current date for filename
        current_date = datetime.now(TZ).strftime("%Y%m%d")
        filename = f"nhl_predictions_{current_date}.txt"
        
        # Create list to store formatted lines
        txt_lines = []
        
        # Process each game
        for i, (home_team, away_team) in enumerate(games):
            if i < len(todays_games_uo):
                # Get OU value and confidence - use predicted value if available, otherwise use default
                ou_value = prediction_results.get('ou_value_predictions', [])[i] if i < len(prediction_results.get('ou_value_predictions', [])) else (todays_games_uo[i] if i < len(todays_games_uo) else 6.0)
                ou_value_confidence = prediction_results.get('ou_confidence', [])[i] if i < len(prediction_results.get('ou_confidence', [])) else 50.0
                
                # Get ML prediction to determine winning team
                ml_winner = home_team  # Default
                ml_confidence = 50.0  # Default
                
                if 'ml_predictions' in prediction_results and i < len(prediction_results['ml_predictions']):
                    ml_pred = prediction_results['ml_predictions'][i]
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ml_pred, '__len__') and len(ml_pred) > 0:
                        if ml_pred.ndim == 2:
                            ml_pred = ml_pred[0]  # Flatten 2D array to 1D
                        
                        if len(ml_pred) == 1:
                            # Old model format: single probability value
                            ml_confidence = round(float(ml_pred[0]), 3)
                            ml_winner = home_team if ml_confidence > 0.5 else away_team
                        else:
                            # New model format: [ML_Home, ML_Away, Home-Team-Win]
                            winner = int(ml_pred[2] > 0.5)  # Home-Team-Win > 0.5 means home team wins
                            ml_winner = home_team if winner == 1 else away_team
                            ml_confidence = round(ml_pred[2] if winner == 1 else (1 - ml_pred[2]), 3)
                
                # Get OU prediction using classification model
                ou_pick = "OVER"  # Default
                ou_confidence = 50.0  # Default
                
                # Try to get OU classification prediction first
                if 'ou_classification_predictions' in prediction_results and i < len(prediction_results['ou_classification_predictions']):
                    ou_class_pred = prediction_results['ou_classification_predictions'][i]
                    ou_class_confidence = prediction_results.get('ou_classification_confidence', [])[i] if i < len(prediction_results.get('ou_classification_confidence', [])) else 50.0
                    
                    # Classification model: 0 = UNDER, 1 = OVER
                    ou_pick = "OVER" if ou_class_pred == 1 else "UNDER"
                    ou_confidence = round(ou_class_confidence, 1)
                
                # Fallback to old regression method if classification not available
                elif 'ou_predictions' in prediction_results and i < len(prediction_results['ou_predictions']):
                    ou_pred = prediction_results['ou_predictions'][i]
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ou_pred, '__len__') and len(ou_pred) > 0:
                        ou_pred = ou_pred[0] if hasattr(ou_pred, '__len__') else float(ou_pred)
                    else:
                        ou_pred = float(ou_pred)
                    
                    # Determine if prediction is over or under
                    if ou_pred > ou_value:
                        ou_pick = "OVER"
                        # Calculate confidence based on how far the prediction is from the line
                        diff = abs(ou_pred - ou_value)
                        # Use the model's accuracy as base confidence (73.8% for the best UO model)
                        # Add confidence based on the difference from the line
                        base_confidence = 73.8  # Model accuracy
                        diff_confidence = min(20.0, diff * 100)  # Max 20% boost from difference
                        ou_confidence = min(95.0, max(50.0, base_confidence + diff_confidence))
                    else:
                        ou_pick = "UNDER"
                        # Calculate confidence based on how far the prediction is from the line
                        diff = abs(ou_pred - ou_value)
                        # Use the model's accuracy as base confidence (73.8% for the best UO model)
                        base_confidence = 73.8  # Model accuracy
                        diff_confidence = min(20.0, diff * 100)  # Max 20% boost from difference
                        ou_confidence = min(95.0, max(50.0, base_confidence + diff_confidence))
                
                
                # Get spread prediction
                spread_winner = away_team  # Default
                spread_value = 1.5  # Default
                spread_confidence = 50.0  # Default
                
                if 'spread_predictions' in prediction_results and i < len(prediction_results['spread_predictions']):
                    spread_pred = prediction_results['spread_predictions'][i]
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(spread_pred, '__len__') and len(spread_pred) > 0:
                        spread_pred = spread_pred[0] if hasattr(spread_pred, '__len__') else float(spread_pred)
                    else:
                        spread_pred = float(spread_pred)
                    
                    # Get spread value
                    spread_value = todays_games_spread[i] if i < len(todays_games_spread) else 1.5
                    
                    # Calculate spread confidence based on how far the prediction is from the line
                    # Use a more reasonable confidence calculation
                    if spread_value != 0:
                        # Calculate confidence as a percentage based on the difference
                        diff = abs(spread_pred - spread_value)
                        # Cap confidence at 95% and use a more reasonable scaling
                        spread_confidence = min(95.0, max(50.0, 50.0 + (diff * 10)))
                    else:
                        spread_confidence = 50.0
                    
                    # Determine spread winner based on prediction vs actual spread
                    if spread_pred > spread_value:
                        spread_winner = home_team
                    else:
                        spread_winner = away_team
                
                # Get ML values from the prediction results
                ml_home_value = 100  # Default ML value
                ml_away_value = 100  # Default ML value
                
                if 'ml_predictions' in prediction_results and i < len(prediction_results['ml_predictions']):
                    ml_pred = prediction_results['ml_predictions'][i]
                    # Handle array conversion - the prediction is a numpy array from XGBoost
                    if hasattr(ml_pred, '__len__') and len(ml_pred) > 0:
                        if ml_pred.ndim == 2:
                            ml_pred = ml_pred[0]  # Flatten 2D array to 1D
                        
                        if len(ml_pred) >= 2:
                            # New model format: [ML_Home, ML_Away, Home-Team-Win]
                            ml_home_value = int(round(ml_pred[0]))
                            ml_away_value = int(round(ml_pred[1]))
                
                # Create the formatted line with team name, OU prediction, and spread value
                # Format: "Team A vs Team B, recommended bet: Team C, Spread:X.X(confidence%), ML:Y(confidence%), OU:Pick Value(confidence%)"
                # Ensure ML confidence is properly capped and converted to percentage
                ml_confidence_pct = min(95.0, max(50.0, round(ml_confidence * 100, 1)))
                spread_confidence_pct = round(spread_confidence, 1)
                ou_confidence_pct = round(ou_confidence, 1)
                ou_value_confidence_pct = round(ou_value_confidence, 1)
                
                # Combine OU pick and value into single field
                ou_combined = f"{ou_pick} {ou_value:.2f}({ou_value_confidence_pct}%)"
                
                line = f"{home_team} vs {away_team}, recommended bet: {ml_winner}, Spread:{spread_value}({spread_confidence_pct}%), ML:{ml_home_value}({ml_confidence_pct}%), OU:{ou_combined}"
                txt_lines.append(line)
        
        # Write to TXT file
        if txt_lines:
            with open(filename, 'w', encoding='utf-8') as f:
                for line in txt_lines:
                    f.write(line + '\n')
            
            print(f"\nPrediction results exported to: {filename}")
            print(f"Exported {len(txt_lines)} games with predictions")
            
            # Display summary
            print("\nTXT Export Summary:")
            print("=" * 40)
            for line in txt_lines:
                print(line)
        else:
            print("No prediction data to export to TXT")
            
    except Exception as e:
        print(f"Error exporting TXT predictions: {e}")

def main():
    print("NHL XGBoost Prediction System")
    print("=" * 50)
    
    # Scrape today's NHL schedule
    schedule_df = scrape_todays_nhl_schedule()
    
    if schedule_df is None or schedule_df.empty:
        print("No NHL games scheduled for today.")
        return
    
    # Use all games for today, regardless of status
    print(f"Processing all {len(schedule_df)} games scheduled for today:")
    for _, game in schedule_df.iterrows():
        print(f"   {game['away']} @ {game['home']} - {game['start_local']} ({game['status']})")
    
    # Convert to games list format
    games = [(row['home'], row['away']) for _, row in schedule_df.iterrows()]
    
    # Load NHL team data
    print("Loading NHL team data...")
    team_df = load_nhl_team_data()
    if team_df is None:
        print("Failed to load team data. Please run Get_Data.py first.")
        return
    
    # Prepare prediction data
    print("Preparing prediction data...")
    result = prepare_nhl_prediction_data(games, team_df, args.ou, args.spread, args.home_odds, args.away_odds)
    if result[0] is None:
        print("Failed to prepare prediction data.")
        return
    
    frame_ml, todays_games_uo, todays_games_spread, _, home_team_odds, away_team_odds = result
    
    print(f"Prepared data for {len(games)} games")
    print(f"   Features: {frame_ml.shape[1]}")
    print()
    
    # Run XGBoost predictions
    if args.xgb or args.A:
        print("XGBoost Model Predictions")
        print("=" * 50)
        try:
            # Get prediction results
            prediction_results = XGBoost_Runner.xgb_runner(
                frame_ml,  # Pass the DataFrame with column names
                todays_games_uo, 
                frame_ml, 
                games, 
                home_team_odds, 
                away_team_odds, 
                args.kc,
                todays_games_spread
            )
            
            # Export results to CSV
            export_predictions_to_csv(games, prediction_results, todays_games_uo, todays_games_spread, home_team_odds, away_team_odds)
            
            # Export results to TXT
            export_predictions_to_txt(games, prediction_results, todays_games_uo, todays_games_spread, home_team_odds, away_team_odds)
            
        except Exception as e:
            print(f"Error running XGBoost predictions: {e}")
            print("\nTo fix this issue:")
            print("   1. First, regenerate the dataset with spread data:")
            print("      python src/Process-Data/Create_Games.py")
            print("   2. Then train the models:")
            print("      python src/Train-Models/XGBoost_Model_ML.py")
            print("      python src/Train-Models/XGBoost_Model_UO.py")
            print("      python src/Train-Models/XGBoost_Model_Spread.py")
        print("=" * 50)
    
    if args.kc:
        print("\nKelly Criterion Analysis")
        print("This shows the recommended bet size based on model edge")
    
    print("\nPrediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NHL XGBoost Prediction System')
    parser.add_argument('-xgb', action='store_true', help='Run XGBoost Model predictions (default)')
    parser.add_argument('-A', action='store_true', help='Run all available models')
    parser.add_argument('-kc', action='store_true', help='Calculate Kelly Criterion bet sizing')
    parser.add_argument('--ou', nargs='+', type=float, help='Specify Over/Under values for each game')
    parser.add_argument('--spread', nargs='+', type=float, help='Specify spread values for each game')
    parser.add_argument('--home-odds', nargs='+', type=int, help='Specify home team moneyline odds')
    parser.add_argument('--away-odds', nargs='+', type=int, help='Specify away team moneyline odds')
    args = parser.parse_args()
    
    # If no arguments provided, default to XGBoost
    if not any([args.xgb, args.A]):
        args.xgb = True
    
    main()
