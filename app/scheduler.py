from apscheduler.schedulers.background import BackgroundScheduler
from .data_ingestion import fetch_all_football
from .prediction_engine import predict
from .db import db
from .train_model import run_training
import logging

def ingest_and_predict():
    """
    Run ingestion (fetch new matches) and then compute predictions for all matches.
    This is called daily by the scheduler.
    """
    logging.info("🔄 Starting scheduled ingestion...")
    try:
        fetch_all_football()
        logging.info("✅ Ingestion complete.")
    except Exception as e:
        logging.error(f"❌ Ingestion failed: {e}")
        return

    logging.info("🔄 Computing predictions for all matches...")
    matches = list(db.matches.find())
    count = 0
    for m in matches:
        if m.get('home_win_prob') is None:
            try:
                predict(m)
                count += 1
            except Exception as e:
                logging.error(f"Failed to predict {m['_id']}: {e}")
    logging.info(f"✅ Predictions computed for {count} matches.")
def run_training_job():
    logging.info("🔄 Running weekly training job...")
    run_training()

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(ingest_and_predict, 'cron', hour=0, minute=0)
    scheduler.add_job(run_training_job, 'cron', day_of_week='sun', hour=3, minute=0)  # weekly
    scheduler.start()
    logging.info("🚀 Scheduler started. Daily ingestion at 00:00 UTC, weekly training on Sunday 03:00 UTC.")
