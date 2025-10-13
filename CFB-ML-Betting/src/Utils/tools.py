import re
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


data_headers = {'Accept': 'application/json, text/plain, */*','Accept-Encoding': 'gzip, deflate, br',
          'Accept-Language': 'en-US,en;q=0.9','Connection': 'keep-alive','Host': 'stats.nba.com',
          'Referer': 'https://stats.nba.com/','Sec-Fetch-Mode': 'cors','Sec-Fetch-Site': 'same-origin',
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
          'x-nba-stats-origin': 'stats','x-nba-stats-token':'true',}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    try:
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('gs').get('g')


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
