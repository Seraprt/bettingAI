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

# Keep-alive function
def keep_alive():
    """Ping the server periodically to prevent Render from sleeping."""
    # Use the external hostname provided by Render, or fallback to a default
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'xlgames.onrender.com')
    url = f"https://{host}/api/health"
    
    while True:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Keep-alive ping sent to {url} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"⚠️ Keep-alive ping failed with status {response.status_code}")
        except Exception as e:
            print(f"❌ Keep-alive ping error: {e}")
        
        # Sleep for 4 minutes (Render sleeps after 15 minutes of inactivity)
        time.sleep(4 * 60)  # 240 seconds

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

        # Start scheduler (only if not debug)
        if not app.debug:
            start_scheduler()

            # Start keep-alive thread (only in production)
            keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            keep_alive_thread.start()
            print("✅ Keep-alive thread started (pings every 4 minutes)")

    return app