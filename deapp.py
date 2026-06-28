from app.db import db
from bson import ObjectId

# Get the first 5 matches where home_team_id or away_team_id doesn't exist
matches = list(db.matches.find().limit(20))
for m in matches:
    home_id = m.get('home_team_id')
    away_id = m.get('away_team_id')
    home = db.teams.find_one({'_id': home_id}) if home_id else None
    away = db.teams.find_one({'_id': away_id}) if away_id else None
    if not home or not away:
        print(f"Match {m['_id']}: {m.get('tournament', '')}")
        print(f"  Home ID: {home_id} -> {'Found' if home else 'MISSING'}")
        print(f"  Away ID: {away_id} -> {'Found' if away else 'MISSING'}")
        print('---')