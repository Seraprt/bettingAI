from apscheduler.schedulers.background import BackgroundScheduler
from .data_ingestion import fetch_all_football
from .prediction_engine import predict
from .db import db
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

def start_scheduler():
    """Start the background scheduler for daily ingestion and prediction."""
    scheduler = BackgroundScheduler()
    # Schedule at midnight UTC every day
    scheduler.add_job(ingest_and_predict, 'cron', hour=0, minute=0)
    scheduler.start()
    logging.info("🚀 Scheduler started. Daily ingestion and prediction at 00:00 UTC.")