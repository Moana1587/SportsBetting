import os
import random
import sqlite3
import sys
import time
import re
import json
import asyncio
import aiohttp
import functools
from datetime import datetime, timedelta
import pytz

import pandas as pd
import requests
import toml
from tqdm import tqdm

# TODO: Add tests

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# User agents and language pools for web scraping
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-CA,en;q=0.9"
]

NEXT_DATA_PATTERN = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL)

@functools.lru_cache(maxsize=64)
def normalize_name(name):
    """Cached team name normalization"""
    return (name
            .lower()
            .replace(".", "")
            .replace("'", "")
            .replace("-", " ")
            .replace("&", "and")
            .strip())

def normalize_date_for_matching(date_obj):
    """Normalize a date to ensure consistent matching between APIs"""
    if isinstance(date_obj, str):
        date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()
    return date_obj

async def get_html_async(session, url, semaphore, retries=3, base_delay=2):
    """Async version of get_html with semaphore for rate limiting"""
    for attempt in range(retries):
        async with semaphore:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            }
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        print(f"Request failed with status {resp.status} for {url}")
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        
        if attempt < retries - 1:
            delay = base_delay + random.uniform(0, 2)
            await asyncio.sleep(delay)
    
    return None

def get_mlb_schedule(start_date, end_date):
    """Synchronous MLB schedule fetching"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    schedule_map = {}
    current_start = start
    
    while current_start <= end:
        current_end = min(current_start.replace(year=current_start.year + 1) - timedelta(days=1), end)
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={current_start.strftime('%Y-%m-%d')}&endDate={current_end.strftime('%Y-%m-%d')}"
        resp = requests.get(url)
        data = resp.json()
        
        for date_info in data.get("dates", []):
            date = date_info["date"]
            if date not in schedule_map:
                schedule_map[date] = {}
            for g in date_info.get("games", []):
                away = g["teams"]["away"]["team"]["name"]
                home = g["teams"]["home"]["team"]["name"]
                schedule_map[date][(normalize_name(away), normalize_name(home))] = g["gameType"]

        current_start = current_end + timedelta(days=1)

    return schedule_map

def get_odds_url(date, odds_type):
    """Get the appropriate URL for the given odds type"""
    base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"
    
    if odds_type == "moneyline":
        return f"{base_url}/?date={date}"
    elif odds_type == "pointspread":
        return f"{base_url}/pointspread/full-game/?date={date}"
    elif odds_type == "totals":
        return f"{base_url}/totals/full-game/?date={date}"
    else:
        raise ValueError(f"Unknown odds type: {odds_type}")

def extract_odds_data(odds, odds_type):
    """Extract the appropriate odds data based on the odds type"""
    opening_line = odds.get("openingLine", {})
    current_line = odds.get("currentLine", {})
    
    if odds_type == "moneyline":
        opening_keys = ["homeOdds", "awayOdds"]
        current_keys = ["homeOdds", "awayOdds"]
    elif odds_type == "pointspread":
        opening_keys = ["homeOdds", "awayOdds", "homeSpread", "awaySpread"]
        current_keys = ["homeOdds", "awayOdds", "homeSpread", "awaySpread"]
    else:  # totals
        opening_keys = ["overOdds", "underOdds", "total"]
        current_keys = ["overOdds", "underOdds", "total"]
    
    opening_line_cleaned = {k: opening_line.get(k) for k in opening_keys}
    current_line_cleaned = {k: current_line.get(k) for k in current_keys}
    
    return opening_line_cleaned, current_line_cleaned

async def scrape_mlb_odds_async(session, date, odds_type, game_type_map, semaphore, base_delay=2):
    """Async version of scrape_mlb_odds"""
    
    url = get_odds_url(date, odds_type)
    html = await get_html_async(session, url, semaphore, base_delay=base_delay)
    
    if not html:
        print(f"Failed to fetch {odds_type} odds for {date}")
        return date, odds_type, []

    match = NEXT_DATA_PATTERN.search(html)
    if not match:
        print(f"No __NEXT_DATA__ found for {odds_type} odds on {date}")
        return date, odds_type, []

    try:
        data = json.loads(match.group(1))
        odds_tables = data.get("props", {}).get("pageProps", {}).get("oddsTables", [])
        
        if not odds_tables:
            return date, odds_type, []

        game_rows = odds_tables[0].get("oddsTableModel", {}).get("gameRows", [])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing {odds_type} data for {date}: {e}")
        return date, odds_type, []

    games_for_date = []

    for game in game_rows:
        try:
            game_view = game.get("gameView", {})
            away = normalize_name(game_view.get("awayTeam", {}).get("fullName", "Unknown"))
            home = normalize_name(game_view.get("homeTeam", {}).get("fullName", "Unknown"))

            # Create a game identifier for matching across odds types
            game_key = f"{away}_vs_{home}"
            
            cleaned_game = {
                "gameKey": game_key,
                "gameView": {}
            }
            
            # Copy game view data
            for key in ["startDate", "awayTeam", "awayTeamScore", "homeTeam", "homeTeamScore", "gameStatusText", "venueName"]:
                cleaned_game["gameView"][key] = game_view.get(key)
            
            cleaned_game["gameView"]["gameType"] = game_type_map.get(date, {}).get((away, home), "Unknown")

            cleaned_odds_views = []
            for odds in game.get("oddsViews", []):
                if odds is None:
                    continue
                
                try:
                    sportsbook = odds.get("sportsbook", "Unknown")
                    opening_line_cleaned, current_line_cleaned = extract_odds_data(odds, odds_type)
                    
                    cleaned_book = {
                        "sportsbook": sportsbook, 
                        "openingLine": opening_line_cleaned, 
                        "currentLine": current_line_cleaned
                    }
                    cleaned_odds_views.append(cleaned_book)
                except Exception as e:
                    print(f"Error processing {odds_type} odds for {date}: {e}")
                    continue
            
            cleaned_game["oddsViews"] = cleaned_odds_views
            games_for_date.append(cleaned_game)
            
        except Exception as e:
            print(f"Error processing game for {odds_type} on {date}: {e}")
            continue

    return date, odds_type, games_for_date

def merge_odds_data(all_results, odds_types):
    """Merge odds data from different types into a single structure"""
    merged_data = {}
    
    # Group results by date
    date_results = {}
    for date, odds_type, games in all_results:
        if date not in date_results:
            date_results[date] = {}
        date_results[date][odds_type] = games
    
    # Merge games by game key
    for date, odds_by_type in date_results.items():
        merged_games = {}
        
        # First pass: collect all unique games
        for odds_type, games in odds_by_type.items():
            for game in games:
                game_key = game.get("gameKey")
                if not game_key:
                    continue
                    
                if game_key not in merged_games:
                    # Create a copy of gameView to avoid reference issues
                    game_view_copy = game["gameView"].copy()
                    merged_games[game_key] = {
                        "gameView": game_view_copy,
                        "odds": {}
                    }
                
                # Add odds for this type
                merged_games[game_key]["odds"][odds_type] = game["oddsViews"]
        
        # Convert to list
        merged_data[date] = list(merged_games.values())
    
    return merged_data

async def scrape_range_async(start_date, end_date, fast, max_concurrent, odds_types):
    """Main async scraping function for multiple odds types"""
    print("Fetching MLB schedule...")
    game_type_map = get_mlb_schedule(start_date, end_date)
    
    if not game_type_map:
        print("No games found in date range")
        return {}
    
    dates = sorted(game_type_map.keys())
    print(f"Found {len(dates)} dates to scrape for odds types: {', '.join(odds_types)}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    base_delay = 0.1 if fast else 1.0
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create tasks for each date and odds type combination
        for date in dates:
            for odds_type in odds_types:
                task = scrape_mlb_odds_async(session, date, odds_type, game_type_map, semaphore, base_delay)
                tasks.append(task)
        
        print(f"Scraping {len(tasks)} date/odds-type combinations with {max_concurrent} concurrent requests...")
        
        results = []
        chunk_size = max_concurrent * 2
        
        pbar = None
        try:
            pbar = tqdm(total=len(tasks), desc="Scraping", unit="requests")
            
            for i in range(0, len(tasks), chunk_size):
                chunk_tasks = tasks[i:i + chunk_size]
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                successful_in_chunk = 0
                for result in chunk_results:
                    if isinstance(result, tuple) and len(result) == 3:
                        date, odds_type, games = result
                        results.append((date, odds_type, games))
                        if games:
                            successful_in_chunk += 1
                    else:
                        print(f"\nTask failed: {result}")
                
                pbar.update(len(chunk_tasks))
                total_with_games = sum(1 for _, _, games in results if games)
                pbar.set_postfix({"with_games": total_with_games, "chunk_success": successful_in_chunk})
                
                if not fast and i + chunk_size < len(tasks):
                    delay = base_delay + random.uniform(0, base_delay)
                    await asyncio.sleep(delay)
                    
        finally:
            if pbar:
                pbar.close()
    
    # Merge all odds types into a single data structure
    merged_data = merge_odds_data(results, odds_types)
    
    return merged_data

def process_scraped_data_to_dataframe(scraped_data):
    """Convert scraped data to the expected DataFrame format"""
    df_data = []
    teams_last_played = {}

    for date_str, games in scraped_data.items():
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        for game in games:
            game_view = game.get("gameView", {})
            odds = game.get("odds", {})
            
            # Extract team information
            home_team_data = game_view.get("homeTeam", {})
            away_team_data = game_view.get("awayTeam", {})
            
            home_team = home_team_data.get("fullName", "Unknown")
            away_team = away_team_data.get("fullName", "Unknown")
            
            # Extract scores from gameView
            home_score = game_view.get("homeTeamScore")
            away_score = game_view.get("awayTeamScore")
            
            # Calculate points and win margin
            points = None
            win_margin = None
            if home_score is not None and away_score is not None:
                points = home_score + away_score
                win_margin = home_score - away_score
            
            # Track team rest days
            if home_team not in teams_last_played:
                teams_last_played[home_team] = date_obj
                home_games_rested = timedelta(days=7)
            else:
                home_games_rested = date_obj - teams_last_played[home_team]
                teams_last_played[home_team] = date_obj

            if away_team not in teams_last_played:
                teams_last_played[away_team] = date_obj
                away_games_rested = timedelta(days=7)
            else:
                away_games_rested = date_obj - teams_last_played[away_team]
                teams_last_played[away_team] = date_obj
            
            # Initialize odds data
            odds_data = {
                'Date': date_obj,
                'Home': home_team,
                'Away': away_team,
                'OU': None,
                'Spread': None,
                'ML_Home': None,
                'ML_Away': None,
                'Points': points,
                'Win_Margin': win_margin,
                    'Days_Rest_Home': home_games_rested.days,
                    'Days_Rest_Away': away_games_rested.days
            }
            
            # Process moneyline odds
            if "moneyline" in odds:
                for book in odds["moneyline"]:
                    if book.get("currentLine", {}).get("homeOdds"):
                        odds_data['ML_Home'] = book["currentLine"]["homeOdds"]
                    if book.get("currentLine", {}).get("awayOdds"):
                        odds_data['ML_Away'] = book["currentLine"]["awayOdds"]
            
            # Process spread odds
            if "pointspread" in odds:
                for book in odds["pointspread"]:
                    if book.get("currentLine", {}).get("homeSpread") is not None:
                        odds_data['Spread'] = abs(book["currentLine"]["homeSpread"])
            
            # Process totals odds
            if "totals" in odds:
                for book in odds["totals"]:
                    if book.get("currentLine", {}).get("total") is not None:
                        odds_data['OU'] = book["currentLine"]["total"]
            
            # Calculate missing values if they are null
            if odds_data['OU'] is None:
                if odds_data['Points'] is not None:
                    odds_data['OU'] = odds_data['Points']
                else:
                    odds_data['OU'] = 8.5  # Default MLB over/under
            
            if odds_data['Spread'] is None:
                if odds_data['Win_Margin'] is not None:
                    odds_data['Spread'] = abs(odds_data['Win_Margin'])
                else:
                    odds_data['Spread'] = 1.5  # Default MLB spread
            
            # Only add if we have some odds data
            if any([odds_data['ML_Home'], odds_data['ML_Away'], 
                   odds_data['Spread'], odds_data['OU']]):
                df_data.append(odds_data)
    
    return df_data

def main():
    """Main function to run the scraping process"""
    config = toml.load("../../config.toml")
    con = sqlite3.connect("../../Data/OddsData.sqlite")
    
    for key, value in config['get-odds-data'].items():
        start_date = value['start_date']
        end_date = value['end_date']
        
        print(f"Starting scraping from {start_date} to {end_date}")
        
        # Scrape data using async approach
        odds_types = ["moneyline", "pointspread", "totals"]
        scraped_data = asyncio.run(scrape_range_async(
            start_date, 
            end_date, 
            fast=False,  # Set to True for faster scraping
            max_concurrent=5,  # Adjust based on your needs
            odds_types=odds_types
        ))
        
        # Convert to DataFrame format
        df_data = process_scraped_data_to_dataframe(scraped_data)
        
        # Store in database
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_sql(key, con, if_exists="replace", index=False)
            print(f"Successfully stored {len(df)} odds records for {key}")
        else:
            print(f"No odds data collected for {key}")
    
    con.close()
    print("Odds data collection completed!")

if __name__ == "__main__":
    main()
