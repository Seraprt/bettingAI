from app.db import db

# Find all teams with empty or None name
empty_teams = list(db.teams.find({'$or': [{'name': ''}, {'name': None}]}))
print(f"Found {len(empty_teams)} teams with empty names.")

for team in empty_teams:
    # Set a placeholder name
    db.teams.update_one({'_id': team['_id']}, {'$set': {'name': 'Unknown Team'}})
    print(f"Updated team {team['_id']}")

# Also update any match that references these teams to show "Unknown" in the API
# (not necessary because the API already uses 'Unknown' if team not found, but we fixed the team)
print("Done.")