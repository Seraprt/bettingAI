import joblib
import pandas as pd
from .db import db
from .factors import get_form_score, get_strength_score, get_fatigue_score, get_h2h_score

# Load models once
HOME_MODEL = joblib.load('data/xg_home_model.pkl')
AWAY_MODEL = joblib.load('data/xg_away_model.pkl')

def predict_xg_ml(home_team_id, away_team_id, match_date):
    """
    Use trained ML models to predict expected goals.
    """
    # Fetch teams
    home = db.teams.find_one({'_id': home_team_id})
    away = db.teams.find_one({'_id': away_team_id})
    if not home or not away:
        return (1.2, 1.0)

    # Build feature vector (must match training)
    features = {
        'home_strength': home.get('strength', 50),
        'away_strength': away.get('strength', 50),
        'home_form': get_form_score(home_team_id, match_date),
        'away_form': get_form_score(away_team_id, match_date),
        'home_ppg': home.get('home_ppg', 1.5),
        'away_ppg': away.get('away_ppg', 1.0),
        'home_fatigue': get_fatigue_score(home_team_id, match_date, home.get('latitude',0), home.get('longitude',0)),
        'away_fatigue': get_fatigue_score(away_team_id, match_date, away.get('latitude',0), away.get('longitude',0)),
        'h2h_advantage': get_h2h_score(home_team_id, away_team_id, match_date)
    }
    # Convert to DataFrame
    X = pd.DataFrame([features])

    home_xg = HOME_MODEL.predict(X)[0]
    away_xg = AWAY_MODEL.predict(X)[0]

    # Clip to reasonable ranges
    home_xg = max(0.3, min(3.5, home_xg))
    away_xg = max(0.3, min(3.5, away_xg))
    return home_xg, away_xg