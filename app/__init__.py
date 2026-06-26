import os
import threading
import time
import requests
from flask import Flask
from flask_cors import CORS
from .config import Config
from .db import db
from .routes import api
from .scheduler import start_scheduler
from .prediction_engine import predict
import logging

# Keep-alive function (unchanged)
def keep_alive():
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'bettingai-ml4c.onrender.com')
    url = f"https://{host}/api/health"
    while True:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logging.info(f"✅ Keep-alive ping sent to {url}")
            else:
                logging.warning(f"⚠️ Keep-alive ping failed with status {response.status_code}")
        except Exception as e:
            logging.error(f"❌ Keep-alive ping error: {e}")
        time.sleep(4 * 60)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS (adjust origins for production)
    CORS(app, origins=["*"])

    # Register blueprint
    app.register_blueprint(api, url_prefix='/api')

    with app.app_context():
        # Create indexes
        db.teams.create_index('name', unique=True)
        db.matches.create_index([('date', 1)])
        db.matches.create_index([('home_team_id', 1), ('away_team_id', 1)])

        # --- Run prediction on startup (force_predict_all) ---
        logging.info("🚀 Running startup prediction for all matches...")
        matches = list(db.matches.find())
        count = 0
        for m in matches:
            if m.get('home_win_prob') is None:
                try:
                    predict(m)
                    count += 1
                except Exception as e:
                    logging.error(f"Failed to predict {m['_id']}: {e}")
        logging.info(f"✅ Startup prediction: computed for {count} matches.")

        # Start scheduler (only if not debug)
        if not app.debug:
            start_scheduler()
            # Start keep-alive thread
            keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            keep_alive_thread.start()
            logging.info("✅ Keep-alive thread started (pings every 4 minutes)")

    return app