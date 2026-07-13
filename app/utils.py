import requests
import math
from datetime import datetime, timedelta
from textblob import TextBlob
from .config import Config
from .db import db
from bson import ObjectId
def get_weather(lat, lon, match_time):
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {
        'lat': lat,
        'lon': lon,
        'appid': Config.WEATHER_API_KEY,
        'units': 'metric'
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        return {
            'temp': data['main']['temp'],
            'condition': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
    except:
        return {'temp': 20, 'condition': 'clear', 'wind_speed': 5}

def get_news_sentiment(team_name, days=3):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': team_name,
        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': Config.NEWS_API_KEY,
        'pageSize': 10
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        articles = resp.json().get('articles', [])
        if not articles:
            return 0.5
        sentiments = []
        for art in articles:
            blob = TextBlob(art['title'] + ' ' + art.get('description', ''))
            sentiments.append(blob.sentiment.polarity)
        avg = sum(sentiments) / len(sentiments)
        return (avg + 1) / 2
    except:
        return 0.5

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def store_prediction(match_id, market, probability, confidence):
    match = db.matches.find_one({'_id': ObjectId(match_id)})
    if not match:
        return
    db.predictions.update_one(
        {'match_id': str(match_id), 'market': market},
        {'$set': {
            'match_id': str(match_id),
            'market': market,
            'probability': probability,
            'confidence': confidence,
            'predicted_at': datetime.utcnow(),
            'match_date': match['date'],
            'actual_outcome': None,
            'home_team': match['home_team_id'],
            'away_team': match['away_team_id'],
            'tournament': match.get('tournament')
        }},
        upsert=True
    )