import os
import threading
import time
import requests
from flask import Flask
from flask_cors import CORS
from flask_mail import Mail
from .config import Config
from .db import db
from .routes import api
from .scheduler import start_scheduler
from .train_model import run_training
from .prediction_engine import predict
import logging

# Keep-alive function – pings the backend health endpoint
def keep_alive():
    # Use the current backend hostname (change if needed)
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'bettingai-argh.onrender.com')
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

    # ---- INITIALISE FLASK-MAIL ----
    mail = Mail(app)
    app.extensions['mail'] = mail   # so auth.py can access it

    # ---- CORS – allow your frontend and localhost ----
    CORS(app, origins=[
        "https://xtech-bet-exyq.onrender.com",   # your frontend URL
        "http://localhost:3000",                 # local dev
        "https://bettingai-argh.onrender.com",   # your backend URL (optional)
          # remove this in production if you want to restrict to the above
    ])

    app.register_blueprint(api, url_prefix='/api')

    with app.app_context():
        db.teams.create_index('name', unique=True)
        db.matches.create_index([('date', 1)])
        db.matches.create_index([('home_team_id', 1), ('away_team_id', 1)])

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

        if not app.debug:
            start_scheduler()
            keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            keep_alive_thread.start()
            logging.info("✅ Keep-alive thread started (pings every 4 minutes)")

    return app