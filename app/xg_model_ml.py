import joblib
import os
import numpy as np
from .factors import (
    get_form_score, get_strength_score, get_availability_score,
    get_tournament_factor, get_coach_score, get_home_away_score,
    get_h2h_score, get_fatigue_score, get_news_score
)

MODEL_HOME_PATH = 'app/models/xg_home.pkl'
MODEL_AWAY_PATH = 'app/models/xg_away.pkl'

def _load_models():
    if os.path.exists(MODEL_HOME_PATH) and os.path.exists(MODEL_AWAY_PATH):
        return joblib.load(MODEL_HOME_PATH), joblib.load(MODEL_AWAY_PATH)
    return None, None

def predict_xg_ml(home_id, away_id, match_date):
    """Predict expected goals using the trained XGBoost model."""
    model_home, model_away = _load_models()
    if model_home is None or model_away is None:
        return None, None  # fallback to heuristic

    # Compute features (same as training)
    home_form = get_form_score(home_id, match_date)
    away_form = get_form_score(away_id, match_date)
    home_strength = get_strength_score(home_id)
    away_strength = get_strength_score(away_id)
    home_availability = get_availability_score(home_id)
    away_availability = get_availability_score(away_id)
    tournament_factor = get_tournament_factor(match_date.get('tournament'), match_date.get('stage'), 1, None)
    home_coach = get_coach_score(home_id)
    away_coach = get_coach_score(away_id)
    home_away = get_home_away_score(home_id, True, away_strength)
    away_away = get_home_away_score(away_id, False, home_strength)
    h2h = get_h2h_score(home_id, away_id, match_date)
    home_fatigue = get_fatigue_score(home_id, match_date, 0, 0)
    away_fatigue = get_fatigue_score(away_id, match_date, 0, 0)
    home_news = get_news_score(home_id)
    away_news = get_news_score(away_id)

    X = np.array([[
        home_form, away_form,
        home_strength, away_strength,
        home_availability, away_availability,
        tournament_factor,
        home_coach, away_coach,
        home_away, away_away,
        h2h,
        home_fatigue, away_fatigue,
        home_news, away_news
    ]])

    home_xg = model_home.predict(X)[0]
    away_xg = model_away.predict(X)[0]
    return home_xg, away_xg