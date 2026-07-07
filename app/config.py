import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MONGO_URI = os.environ.get('DATABASE_URL') + '&connectTimeoutMS=300000&socketTimeoutMS=30000'
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    FOOTBALL_API_KEY = os.environ.get('FOOTBALL_API_KEY')  # keep for fallback
    ODDS_API_KEY = "8645a1f5adca08799ea3b24500f044cc"      # The Odds API key
    RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY')      
    SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
    # Email (Brevo SMTP)
    MAIL_SERVER = 'smtp-relay.brevo.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    FROM_EMAIL = os.environ.get('FROM_EMAIL', 'noreply@yourdomain.com') 
  