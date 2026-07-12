import requests
from datetime import datetime, timedelta
from .db import db
from .config import Config
import time
import math

COMPETITION_TO_SPORT_KEY = {
    'PL': 'soccer_epl',
    'PD': 'soccer_spain_la_liga',
    'BL1': 'soccer_germany_bundesliga',
    'SA': 'soccer_italy_serie_a',
    'FL1': 'soccer_france_ligue_1',
    'DED': 'soccer_netherlands_eredivisie',
    'PPL': 'soccer_portugal_primeira_liga',
    'ELC': 'soccer_england_championship',
    'BSA': 'soccer_brazil_campeonato',
    'CL': 'soccer_uefa_champions_league',
    'EL': 'soccer_uefa_europa_league',
    'WC': 'soccer_fifa_world_cup',
    'EC': 'soccer_uefa_euro',
}

# ------------------------------------------------------------
# ELO UPDATE
# ------------------------------------------------------------
def update_elo(home_team_id, away_team_id, home_goals, away_goals):
    home = db.teams.find_one({'_id': home_team_id})
    away = db.teams.find_one({'_id': away_team_id})
    if not home or not away or home_goals is None or away_goals is None:
        return
    K = 30
    home_elo = home.get('elo_rating', 1500)
    away_elo = away.get('elo_rating', 1500)
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home
    if home_goals > away_goals:
        home_result, away_result = 1, 0
    elif home_goals < away_goals:
        home_result, away_result = 0, 1
    else:
        home_result, away_result = 0.5, 0.5
    new_home_elo = home_elo + K * (home_result - expected_home)
    new_away_elo = away_elo + K * (away_result - expected_away)
    db.teams.update_one({'_id': home_team_id}, {'$set': {'elo_rating': new_home_elo}})
    db.teams.update_one({'_id': away_team_id}, {'$set': {'elo_rating': new_away_elo}})

# ------------------------------------------------------------
# ATTACK / DEFENCE RATINGS
# ------------------------------------------------------------
def update_attack_defence(home_team_id, away_team_id, home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return
    LEAGUE_AVG_HOME = 1.35
    LEAGUE_AVG_AWAY = 1.05
    ALPHA = 0.2
    home = db.teams.find_one({'_id': home_team_id})
    away = db.teams.find_one({'_id': away_team_id})
    if not home or not away:
        return
    home_attack = home.get('attack_rating', 1.0)
    home_defence = home.get('defence_rating', 1.0)
    away_attack = away.get('attack_rating', 1.0)
    away_defence = away.get('defence_rating', 1.0)
    home_observed_attack = home_goals / LEAGUE_AVG_HOME
    away_observed_attack = away_goals / LEAGUE_AVG_AWAY
    home_observed_defence = away_goals / LEAGUE_AVG_HOME
    away_observed_defence = home_goals / LEAGUE_AVG_AWAY
    new_home_attack = home_attack * (1 - ALPHA) + home_observed_attack * ALPHA
    new_home_defence = home_defence * (1 - ALPHA) + home_observed_defence * ALPHA
    new_away_attack = away_attack * (1 - ALPHA) + away_observed_attack * ALPHA
    new_away_defence = away_defence * (1 - ALPHA) + away_observed_defence * ALPHA
    new_home_attack = max(0.3, min(2.5, new_home_attack))
    new_home_defence = max(0.3, min(2.5, new_home_defence))
    new_away_attack = max(0.3, min(2.5, new_away_attack))
    new_away_defence = max(0.3, min(2.5, new_away_defence))
    db.teams.update_one({'_id': home_team_id}, {'$set': {'attack_rating': new_home_attack, 'defence_rating': new_home_defence}})
    db.teams.update_one({'_id': away_team_id}, {'$set': {'attack_rating': new_away_attack, 'defence_rating': new_away_defence}})

# ------------------------------------------------------------
# FETCH AND STORE
# ------------------------------------------------------------
def fetch_matches_in_range(headers, date_from, date_to):
    url = 'https://api.football-data.org/v4/matches'
    params = {'dateFrom': date_from, 'dateTo': date_to}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print("  Rate limit hit, waiting 10 seconds...")
            time.sleep(10)
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  Error {resp.status_code}: {resp.text}")
            return []
        data = resp.json()
        return data.get('matches', [])
    except Exception as e:
        print(f"  Exception: {e}")
        return []

def store_matches(matches, progress_counter):
    stored = 0
    for match in matches:
        home_name = match['homeTeam']['name']
        away_name = match['awayTeam']['name']
        competition = match['competition']['code']
        sport_key = COMPETITION_TO_SPORT_KEY.get(competition, 'soccer')
        event_id = str(match['id'])
        match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
        status = match['status']
        home_goals = match['score']['fullTime']['home'] if status == 'FINISHED' else None
        away_goals = match['score']['fullTime']['away'] if status == 'FINISHED' else None

        if not home_name or not away_name:
            continue

        home = db.teams.find_one({'name': home_name, 'sport': 'football'})
        if not home:
            home_id = db.teams.insert_one({
                'name': home_name, 'sport': 'football', 'strength': 50,
                'elo_rating': 1500, 'home_ppg': 1.5, 'away_ppg': 1.0,
                'coach_win_rate': 0.5, 'matches_coached': 0,
                'latitude': None, 'longitude': None,
                'attack_rating': 1.0, 'defence_rating': 1.0
            }).inserted_id
        else:
            home_id = home['_id']

        away = db.teams.find_one({'name': away_name, 'sport': 'football'})
        if not away:
            away_id = db.teams.insert_one({
                'name': away_name, 'sport': 'football', 'strength': 50,
                'elo_rating': 1500, 'home_ppg': 1.5, 'away_ppg': 1.0,
                'coach_win_rate': 0.5, 'matches_coached': 0,
                'latitude': None, 'longitude': None,
                'attack_rating': 1.0, 'defence_rating': 1.0
            }).inserted_id
        else:
            away_id = away['_id']

        db.matches.update_one(
            {'event_id': event_id},
            {'$set': {
                'home_team_id': home_id,
                'away_team_id': away_id,
                'date': match_date,
                'tournament': match['competition']['name'],
                'stage': match.get('stage', 'group'),
                'leg': 1,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'sport_key': sport_key,
                'event_id': event_id
            }},
            upsert=True
        )

        if home_goals is not None and away_goals is not None:
            update_elo(home_id, away_id, home_goals, away_goals)
            update_attack_defence(home_id, away_id, home_goals, away_goals)
        stored += 1
        progress_counter[0] += 1
        if progress_counter[0] % 50 == 0:
            print(f"    ✅ Total matches stored/updated so far: {progress_counter[0]}")
    return stored

def fetch_football_matches():
    headers = {'X-Auth-Token': Config.FOOTBALL_API_KEY}
    now = datetime.now()
    total_counter = [0]  # mutable counter for progress logging

    past_days = 300
    chunk_size = 10
    print(f"🔄 Fetching past {past_days} days in {chunk_size}-day chunks...")
    for i in range(0, past_days, chunk_size):
        start = now - timedelta(days=past_days - i)
        end = start + timedelta(days=min(chunk_size, past_days - i))
        date_from = start.strftime('%Y-%m-%d')
        date_to = end.strftime('%Y-%m-%d')
        print(f"  Past {date_from} to {date_to}...")
        matches = fetch_matches_in_range(headers, date_from, date_to)
        if matches:
            stored = store_matches(matches, total_counter)
            print(f"    Stored {stored} matches in this chunk.")
        else:
            print("    No matches")
        time.sleep(0.5)

    future_days = 20
    print(f"\n🔄 Fetching future {future_days} days...")
    for i in range(0, future_days, 7):
        start = now + timedelta(days=i)
        end = start + timedelta(days=min(7, future_days - i))
        date_from = start.strftime('%Y-%m-%d')
        date_to = end.strftime('%Y-%m-%d')
        print(f"  Future {date_from} to {date_to}...")
        matches = fetch_matches_in_range(headers, date_from, date_to)
        if matches:
            stored = store_matches(matches, total_counter)
            print(f"    Stored {stored} matches in this chunk.")
        else:
            print("    No matches")
        time.sleep(0.5)

    print(f"\n✅ Football matches ingestion completed. Total matches stored/updated: {total_counter[0]}")

fetch_all_football = fetch_football_matches