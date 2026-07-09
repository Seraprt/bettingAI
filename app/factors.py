from .db import db
from .utils import get_weather, get_news_sentiment, haversine
from datetime import datetime, timedelta
import math
import numpy as np
from bson import ObjectId

def get_team(team_id):
    if isinstance(team_id, str):
        team_id = ObjectId(team_id)
    return db.teams.find_one({'_id': team_id})

def get_form_score(team_id, match_date, num_games=5):
    team_id = ObjectId(team_id) if isinstance(team_id, str) else team_id
    matches = list(db.matches.find({
        '$or': [{'home_team_id': team_id}, {'away_team_id': team_id}],
        'date': {'$lt': match_date},
        'home_goals': {'$ne': None},
        'away_goals': {'$ne': None}
    }).sort('date', -1).limit(num_games))
    if not matches:
        return 0.5
    weights = [5,4,3,2,1][:len(matches)]
    total_points = 0.0
    max_possible = sum(weights) * 3
    for i, m in enumerate(matches):
        if m['home_team_id'] == team_id:
            gf = m['home_goals']
            ga = m['away_goals']
            opp_id = m['away_team_id']
        else:
            gf = m['away_goals']
            ga = m['home_goals']
            opp_id = m['home_team_id']
        if gf > ga:
            pts = 3
        elif gf == ga:
            pts = 1
        else:
            pts = 0
        opp = get_team(opp_id)
        opp_strength = opp['strength'] / 100.0 if opp else 0.5
        adjusted_pts = pts * (0.5 + 0.5 * opp_strength)
        total_points += weights[i] * adjusted_pts
    norm = total_points / max_possible if max_possible > 0 else 0.5
    return min(1.0, norm)

def get_strength_score(team_id):
    """Factor 2: Team strength – now purely based on Elo rating."""
    team = get_team(team_id)
    if not team:
        return 50.0
    elo = team.get('elo_rating', 1500)
    strength = max(0, min(100, (elo - 1000) / 10))
    return strength

def get_availability_score(team_id):
    team_id = ObjectId(team_id) if isinstance(team_id, str) else team_id
    players = list(db.players.find({'team_id': team_id}))
    if not players:
        return 0.5
    total_imp = sum(p.get('importance', 0.5) for p in players)
    available_imp = sum(p.get('importance', 0.5) for p in players if p.get('available', True))
    if total_imp == 0:
        return 0.5
    return available_imp / total_imp

def get_tournament_factor(tournament, stage, leg, aggregate_diff):
    t = tournament.lower() if tournament else ''
    s = stage.lower() if stage else ''
    if 'final' in s:
        return 1.2
    if 'semi' in s:
        return 1.1
    if 'quarter' in s:
        return 1.05
    if 'group' in s:
        return 1.0
    if 'cup' in t and ('round' in s or '1st' in s or '2nd' in s):
        return 0.9
    if leg == 2 and aggregate_diff is not None:
        if abs(aggregate_diff) >= 2:
            return 0.95
    return 1.0

def get_coach_score(team_id):
    team = get_team(team_id)
    if not team or team.get('matches_coached', 0) < 2:
        return 0.5
    return team.get('coach_win_rate', 0.5)

def get_home_away_score(team_id, is_home, opponent_strength):
    team = get_team(team_id)
    if not team:
        return 0.5
    opp_factor = opponent_strength / 100.0 if opponent_strength else 0.5
    if is_home:
        raw = team.get('home_ppg', 1.5) / 3.0
        return max(0, min(1, raw * (1 + 0.2 * (1 - opp_factor))))
    else:
        raw = team.get('away_ppg', 1.0) / 3.0
        return max(0, min(1, raw * (1 - 0.2 * opp_factor)))

def get_h2h_score(home_id, away_id, match_date):
    home_id = ObjectId(home_id) if isinstance(home_id, str) else home_id
    away_id = ObjectId(away_id) if isinstance(away_id, str) else away_id
    matches = list(db.matches.find({
        '$or': [
            {'home_team_id': home_id, 'away_team_id': away_id},
            {'home_team_id': away_id, 'away_team_id': home_id}
        ],
        'date': {'$lt': match_date},
        'home_goals': {'$ne': None},
        'away_goals': {'$ne': None}
    }).sort('date', -1).limit(5))
    if not matches:
        return 0.5
    weights = [5,4,3,2,1][:len(matches)]
    total = 0.0
    for i, m in enumerate(matches):
        if m['home_team_id'] == home_id:
            if m['home_goals'] > m['away_goals']:
                total += weights[i] * 1.0
            elif m['home_goals'] == m['away_goals']:
                total += weights[i] * 0.5
        else:
            if m['away_goals'] > m['home_goals']:
                total += weights[i] * 1.0
            elif m['away_goals'] == m['home_goals']:
                total += weights[i] * 0.5
    return total / sum(weights)

def get_weather_score(lat, lon, match_time):
    weather = get_weather(lat, lon, match_time)
    temp = weather['temp']
    condition = weather['condition'].lower()
    wind = weather['wind_speed']
    multiplier = 1.0
    if temp > 30:
        multiplier -= 0.03
    elif temp < 5:
        multiplier -= 0.03
    if 'rain' in condition or 'snow' in condition:
        multiplier -= 0.05
    if wind > 30:
        multiplier -= 0.02
    if 15 <= temp <= 25 and 'clear' in condition and wind < 15:
        multiplier += 0.02
    return max(0.85, min(1.15, multiplier))

def get_fatigue_score(team_id, match_date, current_venue_lat, current_venue_lon):
    team_id = ObjectId(team_id) if isinstance(team_id, str) else team_id
    last_match = db.matches.find_one({
        '$or': [{'home_team_id': team_id}, {'away_team_id': team_id}],
        'date': {'$lt': match_date}
    }, sort=[('date', -1)])
    if not last_match:
        return 1.0
    rest_days = (match_date - last_match['date']).total_seconds() / 86400.0
    if rest_days >= 4:
        rest_factor = 1.0
    elif rest_days >= 3:
        rest_factor = 0.95
    elif rest_days >= 2:
        rest_factor = 0.85
    else:
        rest_factor = 0.7
    team = get_team(team_id)
    if not team or not team.get('latitude') or not team.get('longitude'):
        travel_penalty = 0.0
    else:
        if last_match['home_team_id'] == team_id:
            last_venue = get_team(last_match['home_team_id'])
        else:
            last_venue = get_team(last_match['away_team_id'])
        if last_venue and last_venue.get('latitude'):
            dist = haversine(last_venue['latitude'], last_venue['longitude'],
                             current_venue_lat, current_venue_lon)
            travel_penalty = min(0.15, (dist / 200) * 0.01)
        else:
            travel_penalty = 0.0
    return max(0.5, min(1.0, rest_factor - travel_penalty))

def get_news_score(team_id):
    team = get_team(team_id)
    if not team:
        return 0.5
    return get_news_sentiment(team['name'])

# ------------------------------------------------------------------
# NEW: Attack and Defence Ratings
# ------------------------------------------------------------------
def get_attack_rating(team_id):
    """Return the team's attacking rating (goals scored relative to average)."""
    team = get_team(team_id)
    if not team:
        return 1.0
    return team.get('attack_rating', 1.0)

def get_defence_rating(team_id):
    """Return the team's defensive rating (goals conceded relative to average)."""
    team = get_team(team_id)
    if not team:
        return 1.0
    return team.get('defence_rating', 1.0)