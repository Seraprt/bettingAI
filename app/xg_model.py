from .db import db
from .factors import get_form_score

def predict_xg(home_team_id, away_team_id, match_date, use_ml=False):
    """
    Returns (home_xg, away_xg).
    If use_ml=True, it will load a pre-trained model (we'll implement later).
    """
    if use_ml:
        # We'll implement ML version after training
        from .xg_model_ml import predict_xg_ml
        return predict_xg_ml(home_team_id, away_team_id, match_date)
    else:
        # Heuristic version
        home = db.teams.find_one({'_id': home_team_id})
        away = db.teams.find_one({'_id': away_team_id})
        if not home or not away:
            return (1.2, 1.0)

        home_strength = home.get('strength', 50) / 50.0
        away_strength = away.get('strength', 50) / 50.0
        home_adv = 0.3
        home_form = get_form_score(home_team_id, match_date)
        away_form = get_form_score(away_team_id, match_date)

        home_xg = (home_strength * 0.8 + home_form * 0.5 + home_adv) * 0.9
        away_xg = (away_strength * 0.8 + away_form * 0.5) * 0.9

        home_xg = max(0.3, min(3.5, home_xg))
        away_xg = max(0.3, min(3.5, away_xg))
        return home_xg, away_xg