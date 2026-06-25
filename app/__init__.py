from flask import Flask
from flask_cors import CORS
from .config import Config
from .db import db
from .routes import api
from .scheduler import start_scheduler

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

  
    CORS(app, origins=[
        "http://localhost:5173",   
        "http://localhost:3000",    
        "https://your-frontend-domain.com",  
        "*"                        
    ])
   

    app.register_blueprint(api, url_prefix='/api')

    with app.app_context():
        
        db.teams.create_index('name', unique=True)
        db.matches.create_index([('date', 1)])
        db.matches.create_index([('home_team_id', 1), ('away_team_id', 1)])

     
        if not app.debug:
            start_scheduler()

    return app