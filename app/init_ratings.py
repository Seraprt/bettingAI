from app.db import db
from app.data_ingestion import update_attack_defence

def compute_ratings_from_all_matches():
    print("🔄 Recomputing attack/defence ratings from all historical matches...")
    # Get all finished matches, oldest first
    matches = list(db.matches.find({
        'home_goals': {'$ne': None},
        'away_goals': {'$ne': None}
    }).sort('date', 1))

    count = 0
    for m in matches:
        update_attack_defence(
            m['home_team_id'],
            m['away_team_id'],
            m['home_goals'],
            m['away_goals']
        )
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count} matches...")
    print(f"✅ Done. Updated ratings for {count} matches.")

if __name__ == '__main__':
    compute_ratings_from_all_matches()