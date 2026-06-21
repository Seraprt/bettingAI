import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

def load_historical_football():
    """
    Load historical matches from Football-Data.org or CSV.
    For now, we'll assume you have a CSV with columns:
    home_team, away_team, date, home_goals, away_goals,
    home_strength, away_strength, home_form, away_form, ...
    """
    # Example: load from CSV
    df = pd.read_csv('data/football_historical.csv')
    return df

def load_historical_basketball():
    # similar for basketball
    df = pd.read_csv('data/basketball_historical.csv')
    return df

# ------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------

def prepare_features(df, sport='football'):
    """
    Create feature matrix X and target y.
    For football: predict home_xg and away_xg separately.
    """
    if sport == 'football':
        features = [
            'home_strength', 'away_strength',
            'home_form', 'away_form',
            'home_ppg', 'away_ppg',
            'home_fatigue', 'away_fatigue',
            'h2h_advantage'
        ]
        X = df[features]
        y_home = df['home_goals']
        y_away = df['away_goals']
        return X, y_home, y_away
    else:
        # basketball: features include offensive/defensive ratings, etc.
        pass

# ------------------------------------------------------------
# 3. TRAIN & SAVE MODELS
# ------------------------------------------------------------

def train_football_model():
    df = load_historical_football()
    X, y_home, y_away = prepare_features(df, 'football')

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    # Train home goals model
    model_home = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model_home.fit(X_train, y_home_train)
    home_mae = mean_absolute_error(y_home_test, model_home.predict(X_test))
    print(f"Home goals MAE: {home_mae:.3f}")

    # Train away goals model
    model_away = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model_away.fit(X_train, y_away_train)
    away_mae = mean_absolute_error(y_away_test, model_away.predict(X_test))
    print(f"Away goals MAE: {away_mae:.3f}")

    # Save models
    joblib.dump(model_home, 'data/xg_home_model.pkl')
    joblib.dump(model_away, 'data/xg_away_model.pkl')
    print("Models saved to data/")

def train_basketball_model():
    # Similar using normal regression
    pass

if __name__ == '__main__':
    train_football_model()
    # train_basketball_model()