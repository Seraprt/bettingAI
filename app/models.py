from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Team(db.Model):
    __tablename__ = 'teams'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    sport = db.Column(db.String(30), nullable=False)
    home_ground = db.Column(db.String(100))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    strength = db.Column(db.Float, default=50.0)        # 0-100
    home_ppg = db.Column(db.Float, default=1.5)         # points per game at home
    away_ppg = db.Column(db.Float, default=1.0)
    elo_rating = db.Column(db.Float, default=1500)
    coach_win_rate = db.Column(db.Float, default=0.5)
    matches_coached = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    players = db.relationship('Player', backref='team', lazy=True)

class Player(db.Model):
    __tablename__ = 'players'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    importance = db.Column(db.Float, default=0.5)       # 0-1, based on goals/assists
    available = db.Column(db.Boolean, default=True)

class Match(db.Model):
    __tablename__ = 'matches'
    id = db.Column(db.Integer, primary_key=True)
    home_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    away_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    date = db.Column(db.DateTime, nullable=False)
    tournament = db.Column(db.String(100))
    stage = db.Column(db.String(50))                    # group, knockout, final, etc.
    leg = db.Column(db.Integer, default=1)              # 1 or 2
    aggregate_home = db.Column(db.Integer, nullable=True)
    aggregate_away = db.Column(db.Integer, nullable=True)
    weather_temp = db.Column(db.Float)
    weather_condition = db.Column(db.String(50))
    home_goals = db.Column(db.Integer, nullable=True)
    away_goals = db.Column(db.Integer, nullable=True)
    # Predicted probabilities (stored after computation)
    home_win_prob = db.Column(db.Float)
    draw_prob = db.Column(db.Float)
    away_win_prob = db.Column(db.Float)
    confidence = db.Column(db.Float)
    # Factor scores (cached)
    home_factors = db.Column(db.JSON)
    away_factors = db.Column(db.JSON)

    home_team = db.relationship('Team', foreign_keys=[home_team_id])
    away_team = db.relationship('Team', foreign_keys=[away_team_id])