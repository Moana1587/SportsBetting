from datetime import date
import json
from flask import Flask, render_template, jsonify
from functools import lru_cache
import subprocess, requests, re, time


@lru_cache()
def fetch_mlb_predictions(ttl_hash=None):
    del ttl_hash
    return fetch_mlb_game_data()

def fetch_mlb_game_data():
    """Fetch MLB game predictions using the updated main.py"""
    cmd = ["python", "main.py", "-xgb"]
    stdout = subprocess.check_output(cmd, cwd="../").decode()
    
    # Updated regex patterns for MLB output format
    data_re = re.compile(r'\n(?P<home_team>[\w ]+)(\((?P<home_confidence>[\d+\.]+)%\))? vs (?P<away_team>[\w ]+)(\((?P<away_confidence>[\d+\.]+)%\))?: (?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d+\.]+) (\((?P<ou_confidence>[\d+\.]+)%\))?', re.MULTILINE)
    ev_re = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d+\.]+)', re.MULTILINE)
    
    games = {}
    for match in data_re.finditer(stdout):
        game_dict = {
            'away_team': match.group('away_team').strip(),
            'home_team': match.group('home_team').strip(),
            'away_confidence': match.group('away_confidence'),
            'home_confidence': match.group('home_confidence'),
            'ou_pick': match.group('ou_pick'),
            'ou_value': match.group('ou_value'),
            'ou_confidence': match.group('ou_confidence'),
            'away_team_ev': None,
            'home_team_ev': None,
            'away_team_odds': '100',  # Default odds
            'home_team_odds': '100'   # Default odds
        }
        
        # Extract EV data
        for ev_match in ev_re.finditer(stdout):
            if ev_match.group('team') == game_dict['away_team']:
                game_dict['away_team_ev'] = ev_match.group('ev')
            if ev_match.group('team') == game_dict['home_team']:
                game_dict['home_team_ev'] = ev_match.group('ev')

        print(json.dumps(game_dict, sort_keys=True, indent=4))
        games[f"{game_dict['away_team']}:{game_dict['home_team']}"] = game_dict
    return games


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    mlb_predictions = fetch_mlb_predictions(ttl_hash=get_ttl_hash())

    return render_template('index.html', today=date.today(), data={"mlb": mlb_predictions})




def get_mlb_team_data(team_name):
    """Fetch MLB team roster data using MLB Stats API"""
    try:
        # Get team ID first
        team_id = mlb_team_ids.get(team_name)
        if not team_id:
            return {
                'success': False,
                'error': f'Team ID not found for {team_name}'
            }
        
        # Fetch roster data
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        response = requests.get(url)
        data = response.json()
        
        if 'roster' in data:
            formatted_players = []
            for player in data['roster']:
                person = player.get('person', {})
                position = player.get('position', {})
                
                formatted_player = {
                    'name': person.get('fullName'),
                    'shortName': person.get('lastName'),
                    'headshot': f"https://img.mlbstatic.com/mlb-photos/image/upload/w_300,q_100/v1/people/{person.get('id')}/headshot/83/current",
                    'injury': 'Healthy',  # MLB API doesn't provide injury info in roster
                    'position': position.get('abbreviation'),
                    'height': person.get('height'),
                    'weight': person.get('weight'),
                    'college': person.get('college'),
                    'experience': person.get('mlbDebutDate'),
                    'jerseyNum': player.get('jerseyNumber'),
                    'playerId': person.get('id'),
                    'birthDate': person.get('birthDate')
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
        print(f"Error in get_mlb_team_data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Fetch and return the MLB team data
    result = get_mlb_team_data(team_name)
    return jsonify(result)


@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    """Fetch MLB player stats using MLB Stats API"""
    try:
        # Get player info
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        response = requests.get(url)
        data = response.json()
        
        if 'people' in data and len(data['people']) > 0:
            player_info = data['people'][0]
            
            # Get recent game stats (last 10 games)
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season=2024"
            stats_response = requests.get(stats_url)
            stats_data = stats_response.json()
            
            recent_games = []
            if 'stats' in stats_data and len(stats_data['stats']) > 0:
                games = stats_data['stats'][0].get('splits', [])
                recent_games = games[:10]  # Last 10 games
            
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('fullName'),
                    'position': player_info.get('primaryPosition', {}).get('abbreviation'),
                    'number': player_info.get('primaryNumber'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('currentTeam', {}).get('name'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('mlbDebutDate'),
                    'headshot': f"https://img.mlbstatic.com/mlb-photos/image/upload/w_300,q_100/v1/people/{player_id}/headshot/83/current",
                    'injury': 'Healthy'  # MLB API doesn't provide injury info easily
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

        
# MLB Team IDs for the MLB Stats API
mlb_team_ids = {
    'Arizona Diamondbacks': 109,
    'Atlanta Braves': 144,
    'Baltimore Orioles': 110,
    'Boston Red Sox': 111,
    'Chicago Cubs': 112,
    'Chicago White Sox': 145,
    'Cincinnati Reds': 113,
    'Cleveland Guardians': 114,
    'Colorado Rockies': 115,
    'Detroit Tigers': 116,
    'Houston Astros': 117,
    'Kansas City Royals': 118,
    'Los Angeles Angels': 108,
    'Los Angeles Dodgers': 119,
    'Miami Marlins': 146,
    'Milwaukee Brewers': 158,
    'Minnesota Twins': 142,
    'New York Mets': 121,
    'New York Yankees': 147,
    'Oakland Athletics': 133,
    'Philadelphia Phillies': 143,
    'Pittsburgh Pirates': 134,
    'San Diego Padres': 135,
    'San Francisco Giants': 137,
    'Seattle Mariners': 136,
    'St. Louis Cardinals': 138,
    'Tampa Bay Rays': 139,
    'Texas Rangers': 140,
    'Toronto Blue Jays': 141,
    'Washington Nationals': 120
}