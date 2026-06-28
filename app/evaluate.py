import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.db import db
from app.xg_model_ml import predict_xg_ml
from app.factors import get_form_score, get_strength_score

def evaluate():
    # Get recent matches (last 30 days) that have results
    matches = list(db.matches.find({
        'home_goals': {'$ne': None},
        'away_goals': {'$ne': None}
    }).sort('date', -1).limit(100))

    if len(matches) < 10:
        print("Not enough recent matches to evaluate.")
        return

    home_preds = []
    home_actuals = []
    away_preds = []
    away_actuals = []

    for m in matches:
        h_xg, a_xg = predict_xg_ml(m['home_team_id'], m['away_team_id'], m['date'])
        if h_xg is None:  # fallback to heuristic
            h_xg = m.get('home_xg', 1.2)  # stored earlier
            a_xg = m.get('away_xg', 1.0)
        home_preds.append(h_xg)
        home_actuals.append(m['home_goals'])
        away_preds.append(a_xg)
        away_actuals.append(m['away_goals'])

    home_mae = mean_absolute_error(home_actuals, home_preds)
    home_rmse = mean_squared_error(home_actuals, home_preds, squared=False)
    home_r2 = r2_score(home_actuals, home_preds)

    away_mae = mean_absolute_error(away_actuals, away_preds)
    away_rmse = mean_squared_error(away_actuals, away_preds, squared=False)
    away_r2 = r2_score(away_actuals, away_preds)

    print(f"Home goals: MAE={home_mae:.3f}, RMSE={home_rmse:.3f}, R²={home_r2:.3f}")
    print(f"Away goals: MAE={away_mae:.3f}, RMSE={away_rmse:.3f}, R²={away_r2:.3f}")

if __name__ == '__main__':
    evaluate()