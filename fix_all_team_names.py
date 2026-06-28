import requests
from app.db import db
from app.config import Config
from bson import ObjectId
import time

HEADERS = {'X-Auth-Token': Config.FOOTBALL_API_KEY}

# Find all teams with empty, null, or "Unknown Team" name
teams = list(db.teams.find({
    '$or': [
        {'name': ''},
        {'name': None},
        {'name': 'Unknown Team'}
    ]
}))
print(f"Found {len(teams)} teams with missing or placeholder names.")

if not teams:
    print("All teams already have valid names.")
    exit()

for team in teams:
    # Find a match that references this team
    match = db.matches.find_one({
        '$or': [
            {'home_team_id': team['_id']},
            {'away_team_id': team['_id']}
        ]
    })
    if not match:
        print(f"No match found for team {team['_id']} (name: {team.get('name')}), skipping.")
        continue

    event_id = match.get('event_id')
    if not event_id:
        print(f"Match {match['_id']} has no event_id, skipping team {team['_id']}.")
        continue

    # Fetch match details from Football-Data.org
    url = f'https://api.football-data.org/v4/matches/{event_id}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"Failed to fetch match {event_id}: {resp.status_code}")
            continue
        data = resp.json()
        home_team_name = data['homeTeam']['name']
        away_team_name = data['awayTeam']['name']
    except Exception as e:
        print(f"Error fetching match {event_id}: {e}")
        continue

    # Determine which team we are updating
    if match.get('home_team_id') == team['_id']:
        new_name = home_team_name
    elif match.get('away_team_id') == team['_id']:
        new_name = away_team_name
    else:
        continue

    # Only update if the name is different and not empty
    if new_name and new_name != team.get('name'):
        db.teams.update_one({'_id': team['_id']}, {'$set': {'name': new_name}})
        print(f"Updated team {team['_id']} from '{team.get('name')}' to '{new_name}'")
    time.sleep(0.1)

print("All team names have been fixed.")