from bson import ObjectId
from app.db import db
from datetime import datetime

def update_predictions():
    """Check finished matches and update prediction outcomes."""
    predictions = list(db.predictions.find({'actual_outcome': None}))
    for pred in predictions:
        match_id = pred['match_id']
        match = db.matches.find_one({'_id': ObjectId(match_id)})
        if not match:
            continue
        if match.get('home_goals') is not None and match.get('away_goals') is not None:
            # For now, just mark as 'checked' – you can later implement logic per market
            db.predictions.update_one(
                {'_id': pred['_id']},
                {'$set': {'actual_outcome': 'pending_evaluation', 'checked_at': datetime.utcnow()}}
            )
    print("Prediction outcomes updated (marked for evaluation).")