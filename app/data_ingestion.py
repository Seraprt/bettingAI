import requests
from datetime import datetime, timedelta
from .db import db
from .config import Config
import time

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

def fetch_matches_in_range(headers, date_from, date_to):
    """Fetch matches for a single date range (max 10 days)."""
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

def store_matches(matches):
    """Store a list of matches in the database."""
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

        # Create/update teams (same as before)
        home = db.teams.find_one({'name': home_name, 'sport': 'football'})
        if not home:
            home_id = db.teams.insert_one({
                'name': home_name, 'sport': 'football', 'strength': 50,
                'elo_rating': 1500, 'home_ppg': 1.5, 'away_ppg': 1.0,
                'coach_win_rate': 0.5, 'matches_coached': 0,
                'latitude': None, 'longitude': None
            }).inserted_id
        else:
            home_id = home['_id']

        away = db.teams.find_one({'name': away_name, 'sport': 'football'})
        if not away:
            away_id = db.teams.insert_one({
                'name': away_name, 'sport': 'football', 'strength': 50,
                'elo_rating': 1500, 'home_ppg': 1.5, 'away_ppg': 1.0,
                'coach_win_rate': 0.5, 'matches_coached': 0,
                'latitude': None, 'longitude': None
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

def fetch_football_matches():
    """Fetch matches from Football-Data.org using multiple 10-day windows."""
    headers = {'X-Auth-Token': Config.FOOTBALL_API_KEY}
    now = datetime.now()

    # 1. Fetch past 60 days in 10-day chunks
    past_days = 60
    chunk_size = 10
    for i in range(0, past_days, chunk_size):
        start = now - timedelta(days=past_days - i)
        end = start + timedelta(days=min(chunk_size, past_days - i))
        date_from = start.strftime('%Y-%m-%d')
        date_to = end.strftime('%Y-%m-%d')
        print(f"Fetching past {date_from} to {date_to}...")
        matches = fetch_matches_in_range(headers, date_from, date_to)
        if matches:
            store_matches(matches)
            print(f"  Stored {len(matches)} matches")
        time.sleep(0.5)  # avoid rate limit

    # 2. Fetch future 14 days (in 7-day chunks to stay safe)
    future_days = 14
    for i in range(0, future_days, 7):
        start = now + timedelta(days=i)
        end = start + timedelta(days=min(7, future_days - i))
        date_from = start.strftime('%Y-%m-%d')
        date_to = end.strftime('%Y-%m-%d')
        print(f"Fetching future {date_from} to {date_to}...")
        matches = fetch_matches_in_range(headers, date_from, date_to)
        if matches:
            store_matches(matches)
            print(f"  Stored {len(matches)} matches")
        time.sleep(0.5)

    print("Football matches ingestion completed.")

# Alias for compatibility
fetch_all_football = fetch_football_matches