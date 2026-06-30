import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.db import db
from app.factors import (
    get_form_score, get_strength_score, get_availability_score,
    get_tournament_factor, get_coach_score, get_home_away_score,
    get_h2h_score, get_fatigue_score, get_news_score
)
from bson import ObjectId
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_feature_matrix(matches):
    features = []
    targets_home = []
    targets_away = []
    total = len(matches)
    for idx, match in enumerate(matches):
        if idx % 20 == 0:
            logging.info(f"  🔄 Processing match {idx+1}/{total}")
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        match_date = match['date']
        home_goals = match.get('home_goals')
        away_goals = match.get('away_goals')
        if home_goals is None or away_goals is None:
            continue

        home_form = get_form_score(home_id, match_date)
        away_form = get_form_score(away_id, match_date)
        home_strength = get_strength_score(home_id)
        away_strength = get_strength_score(away_id)
        home_availability = get_availability_score(home_id)
        away_availability = get_availability_score(away_id)
        tournament_factor = get_tournament_factor(
            match.get('tournament'), match.get('stage'), match.get('leg', 1), None
        )
        home_coach = get_coach_score(home_id)
        away_coach = get_coach_score(away_id)
        home_away = get_home_away_score(home_id, True, away_strength)
        away_away = get_home_away_score(away_id, False, home_strength)
        h2h = get_h2h_score(home_id, away_id, match_date)
        home_fatigue = get_fatigue_score(home_id, match_date, 0, 0)
        away_fatigue = get_fatigue_score(away_id, match_date, 0, 0)
        home_news = get_news_score(home_id)
        away_news = get_news_score(away_id)

        features.append([
            home_form, away_form,
            home_strength, away_strength,
            home_availability, away_availability,
            tournament_factor,
            home_coach, away_coach,
            home_away, away_away,
            h2h,
            home_fatigue, away_fatigue,
            home_news, away_news
        ])
        targets_home.append(home_goals)
        targets_away.append(away_goals)

    return np.array(features), np.array(targets_home), np.array(targets_away)

def train():
    logging.info("🚀 Starting training on ALL finished matches...")
    logging.info("📊 Loading finished matches (no limit)...")
    
    # Use a cursor with batch_size to avoid timeout issues
    cursor = db.matches.find(
        {'home_goals': {'$ne': None}, 'away_goals': {'$ne': None}}
    ).sort('date', -1)
    matches = list(cursor)
    logging.info(f"✅ Found {len(matches)} finished matches.")

    if len(matches) < 50:
        logging.warning("⚠️ Not enough matches to train. Need at least 50.")
        return

    logging.info("📦 Building feature matrix (this may take a moment)...")
    X, y_home, y_away = build_feature_matrix(matches)
    logging.info(f"✅ Feature matrix shape: {X.shape}")

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    logging.info("🧠 Training home goals model (XGBoost)...")
    model_home = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=1,
        eval_metric='mae'
    )
    model_home.fit(X_train, y_home_train, verbose=True)
    home_pred = model_home.predict(X_test)
    home_mae = mean_absolute_error(y_home_test, home_pred)
    home_rmse = mean_squared_error(y_home_test, home_pred, squared=False)
    home_r2 = r2_score(y_home_test, home_pred)
    logging.info(f"✅ Home goals: MAE={home_mae:.3f}, RMSE={home_rmse:.3f}, R²={home_r2:.3f}")

    logging.info("🧠 Training away goals model (XGBoost)...")
    model_away = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=1,
        eval_metric='mae'
    )
    model_away.fit(X_train, y_away_train, verbose=True)
    away_pred = model_away.predict(X_test)
    away_mae = mean_absolute_error(y_away_test, away_pred)
    away_rmse = mean_squared_error(y_away_test, away_pred, squared=False)
    away_r2 = r2_score(y_away_test, away_pred)
    logging.info(f"✅ Away goals: MAE={away_mae:.3f}, RMSE={away_rmse:.3f}, R²={away_r2:.3f}")

    os.makedirs('app/models', exist_ok=True)
    joblib.dump(model_home, 'app/models/xg_home.pkl')
    joblib.dump(model_away, 'app/models/xg_away.pkl')
    logging.info("💾 Models saved to app/models/")
    logging.info("🎉 Training complete!")
def run_training():
    """Wrapper function to be called from outside (threads, endpoints)."""
    try:
        train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
if __name__ == '__main__':
    train()