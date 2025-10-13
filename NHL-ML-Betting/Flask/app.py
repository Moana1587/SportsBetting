from datetime import date
import json
from flask import Flask, render_template,jsonify
from functools import lru_cache
import subprocess, requests, re, time


@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")

@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")

@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")

@lru_cache()
def fetch_xgboost(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="xgboost")

def fetch_odds_data(sportsbook="fanduel"):
    """Fetch odds data from The Odds API (aggregates multiple sportsbooks)"""
    try:
        # Using The Odds API which aggregates odds from multiple sportsbooks
        api_key = "cb4bb74f84461679822ad58e4fee2d62"  # You'll need to get a free API key from the-odds-api.com
        url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
        
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        # Map sportsbook names to their identifiers in the API
        sportsbook_mapping = {
            'fanduel': 'fanduel',
            'draftkings': 'draftkings', 
            'betmgm': 'betmgm'
        }
        
        if sportsbook not in sportsbook_mapping:
            return {}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        odds_data = {}
        target_book = sportsbook_mapping[sportsbook]
        
        for game in data:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            
            # Find odds from the target sportsbook
            for bookmaker in game.get('bookmakers', []):
                if bookmaker.get('key') == target_book:
                    home_ml = None
                    away_ml = None
                    ou_line = None
                    home_spread = None
                    
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':  # Moneyline
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home_team:
                                    home_ml = outcome.get('price')
                                elif outcome.get('name') == away_team:
                                    away_ml = outcome.get('price')
                        elif market.get('key') == 'totals':  # Over/Under
                            ou_line = market.get('outcomes', [{}])[0].get('point', 5.5)
                        elif market.get('key') == 'spreads':  # Point Spread
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home_team:
                                    home_spread = outcome.get('point', 0)
                    
                    if home_team and away_team:
                        odds_data[f"{away_team}:{home_team}"] = {
                            'home_team_odds': str(home_ml) if home_ml else '-110',
                            'away_team_odds': str(away_ml) if away_ml else '-110',
                            'ou_line': ou_line if ou_line else 5.5,
                            'home_spread': home_spread if home_spread else 0
                        }
                    break
        
        return odds_data
    except Exception as e:
        print(f"Error fetching {sportsbook} odds: {e}")
        # Return mock data for demonstration
        return get_mock_odds_data(sportsbook)

def get_mock_odds_data(sportsbook="fanduel"):
    """Generate mock odds data for demonstration purposes"""
    mock_odds = {
        "fanduel": {
            "Toronto Maple Leafs:Boston Bruins": {
                'home_team_odds': '-120',
                'away_team_odds': '+100',
                'ou_line': 5.5,
                'home_spread': -1.5
            },
            "New York Islanders:New York Rangers": {
                'home_team_odds': '-110',
                'away_team_odds': '-110',
                'ou_line': 6.0,
                'home_spread': 0.0
            },
            "Edmonton Oilers:Calgary Flames": {
                'home_team_odds': '+105',
                'away_team_odds': '-125',
                'ou_line': 6.5,
                'home_spread': 1.5
            }
        },
        "draftkings": {
            "Toronto Maple Leafs:Boston Bruins": {
                'home_team_odds': '-115',
                'away_team_odds': '+105',
                'ou_line': 5.5,
                'home_spread': -1.5
            },
            "New York Islanders:New York Rangers": {
                'home_team_odds': '-105',
                'away_team_odds': '-115',
                'ou_line': 6.0,
                'home_spread': 0.0
            },
            "Edmonton Oilers:Calgary Flames": {
                'home_team_odds': '+110',
                'away_team_odds': '-130',
                'ou_line': 6.5,
                'home_spread': 1.5
            }
        },
        "betmgm": {
            "Toronto Maple Leafs:Boston Bruins": {
                'home_team_odds': '-125',
                'away_team_odds': '+105',
                'ou_line': 5.5,
                'home_spread': -1.5
            },
            "New York Islanders:New York Rangers": {
                'home_team_odds': '-110',
                'away_team_odds': '-110',
                'ou_line': 6.0,
                'home_spread': 0.0
            },
            "Edmonton Oilers:Calgary Flames": {
                'home_team_odds': '+100',
                'away_team_odds': '-120',
                'ou_line': 6.5,
                'home_spread': 1.5
            }
        }
    }
    
    return mock_odds.get(sportsbook, {})

def fetch_game_data(sportsbook="fanduel"):
    """Fetch NHL game predictions and odds data"""
    cmd = ["python", "main.py", "-xgb"]
    stdout = subprocess.check_output(cmd, cwd="../").decode()
    
    # Updated regex patterns for NHL game data format
    # Pattern: "Florida Panthers (78.5%) vs Chicago Blackhawks: OVER 5.5 (92.2%) | Recommended bet: Chicago Blackhawks +1.5 (100.0%)"
    data_re = re.compile(r'(?P<home_team>[\w ]+)(\s+\((?P<home_confidence>[\d+\.]+)%\))?\s+vs\s+(?P<away_team>[\w ]+)(\s+\((?P<away_confidence>[\d+\.]+)%\))?:\s+(?P<ou_pick>OVER|UNDER)\s+(?P<ou_value>[\d+\.]+)\s+\((?P<ou_confidence>[\d+\.]+)%\)', re.MULTILINE)
    ev_re = re.compile(r'(?P<team>[\w ]+)\s+EV:\s+(?P<ev>[-\d+\.]+)', re.MULTILINE)
    spread_re = re.compile(r'\|?\s*Recommended bet:\s+(?P<spread_team>[\w ]+)\s+(?P<spread_value>[+-]?[\d+\.]+)\s+\((?P<spread_confidence>[\d+\.]+)%\)', re.MULTILINE)
    
    # For XGBoost, we don't need odds data, just use the prediction data
    if sportsbook == "xgboost":
        odds_data = {}
    else:
        # Fetch odds data from sportsbook
        odds_data = fetch_odds_data(sportsbook)
    
    games = {}
    for match in data_re.finditer(stdout):
        away_team = match.group('away_team').strip()
        home_team = match.group('home_team').strip()
        game_key = f"{away_team}:{home_team}"
        
        # Get odds for this game
        game_odds = odds_data.get(game_key, {})
        
        if sportsbook == "xgboost":
            # For XGBoost, show prediction confidence as "odds"
            away_conf = round(float(match.group('away_confidence')) if match.group('away_confidence') else 0, 1)
            home_conf = round(float(match.group('home_confidence')) if match.group('home_confidence') else 0, 1)
            
            game_dict = {
                'away_team': away_team,
                'home_team': home_team,
                'away_confidence': away_conf,
                'home_confidence': home_conf,
                'ou_pick': match.group('ou_pick'),
                'ou_value': round(float(match.group('ou_value')), 1),
                'ou_confidence': round(float(match.group('ou_confidence')), 1),
                # For XGBoost, show confidence as "odds" format
                'away_team_odds': f"{away_conf}%",
                'home_team_odds': f"{home_conf}%",
                'ou_line': round(float(match.group('ou_value')), 1),
                'home_spread': 0  # Not used for XGBoost display
            }
        else:
            game_dict = {
                'away_team': away_team,
                'home_team': home_team,
                'away_confidence': round(float(match.group('away_confidence')) if match.group('away_confidence') else 0, 1),
                'home_confidence': round(float(match.group('home_confidence')) if match.group('home_confidence') else 0, 1),
                'ou_pick': match.group('ou_pick'),
                'ou_value': round(float(match.group('ou_value')), 1),
                'ou_confidence': round(float(match.group('ou_confidence')), 1),
                # Use real odds data or defaults
                'away_team_odds': game_odds.get('away_team_odds', '-110'),
                'home_team_odds': game_odds.get('home_team_odds', '-110'),
                'ou_line': round(float(game_odds.get('ou_line', match.group('ou_value'))), 1),
                'home_spread': round(float(game_odds.get('home_spread', 0)), 1)
            }
        
        # Add EV data
        for ev_match in ev_re.finditer(stdout):
            if ev_match.group('team') == game_dict['away_team']:
                game_dict['away_team_ev'] = round(float(ev_match.group('ev')), 2)
            if ev_match.group('team') == game_dict['home_team']:
                game_dict['home_team_ev'] = round(float(ev_match.group('ev')), 2)
        
        # Add spread data
        for spread_match in spread_re.finditer(stdout):
            spread_team = spread_match.group('spread_team').strip()
            spread_value = spread_match.group('spread_value')
            spread_confidence = round(float(spread_match.group('spread_confidence')), 1)
            
            # Check if this spread recommendation is for this game
            if spread_team in [game_dict['home_team'], game_dict['away_team']]:
                game_dict['spread_pick'] = f"{spread_team} {spread_value}"
                game_dict['spread_confidence'] = spread_confidence

        print(json.dumps(game_dict, sort_keys=True, indent=4))
        games[game_key] = game_dict
    return games


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
    draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
    betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())
    xgboost = fetch_xgboost(ttl_hash=get_ttl_hash())

    return render_template('index.html', today=date.today(), data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm, "xgboost": xgboost})




def get_player_data(team_abv):
    """Fetch NHL player data for a given team abbreviation"""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNHLTeamRoster"
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    querystring = {"teamAbv": team_abv}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName'),
                    'shortName': player.get('shortName'),
                    'headshot': player.get('nhlComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'college': player.get('college'),
                    'experience': player.get('exp'),
                    'jerseyNum': player.get('jerseyNum'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay'),
                    'shoots': player.get('shoots', 'N/A'),
                    'nationality': player.get('nationality', 'N/A')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }
        
    except Exception as e:
        print(f"Error in get_player_data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNHLPlayerInfo"
    info_querystring = {"playerID": player_id}
    
    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNHLGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": "2024",
    }
    
    try:
        # Get both responses
        info_response = requests.get(info_url, headers=headers, params=info_querystring)
        games_response = requests.get(games_url, headers=headers, params=games_querystring)
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data['body'].values())
            games.sort(key=lambda x: x['gameID'], reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data['body']
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_info = player_info['injury']
                injury_status = injury_info

            # Combine and return all data
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName'),
                    'position': player_info.get('pos'),
                    'number': player_info.get('jerseyNum'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('team'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('exp'),
                    'headshot': player_info.get('nhlComHeadshot'),
                    'injury': injury_status,
                    'shoots': player_info.get('shoots', 'N/A'),
                    'nationality': player_info.get('nationality', 'N/A')
                }
            })
            
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

        
team_abbreviations = {
    'Anaheim Ducks': 'ANA',
    'Arizona Coyotes': 'ARI',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK',
    'Minnesota Wild': 'MIN',
    'Montreal Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA',
    'St. Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG'
}