from apscheduler.schedulers.background import BackgroundScheduler
from .data_ingestion import fetch_football_matches
from .factors import get_strength_score
from .db import db

def update_all_strengths():
    teams = db.teams.find()
    for team in teams:
        new_strength = get_strength_score(team['_id'])
        db.teams.update_one({'_id': team['_id']}, {'$set': {'strength': new_strength}})
    print("Team strengths updated")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_football_matches, 'cron', hour=6, minute=0)
    scheduler.add_job(update_all_strengths, 'cron', hour=7, minute=0)
    scheduler.start()