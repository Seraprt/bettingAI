import requests
from .config import Config
from .market_mapper import map_outcome_to_internal   # <-- changed to relative import

ODDS_API_KEY = Config.ODDS_API_KEY
BASE_URL = "https://api.the-odds-api.com/v4"

def get_sports():
    url = f"{BASE_URL}/sports"
    params = {'apiKey': ODDS_API_KEY}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def get_events_by_sport(sport_key):
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {'apiKey': ODDS_API_KEY}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def get_odds_for_event(sport_key, event_id, markets='h2h,spreads,totals'):
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'eu',
        'markets': markets,
        'oddsFormat': 'decimal'
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def get_odds_best_price(odds_data, internal_market_name, sport='football'):
    if not odds_data.get('bookmakers'):
        return None, None, None
    best_odds = None
    best_bookmaker = None
    for bookmaker in odds_data['bookmakers']:
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            for outcome in market.get('outcomes', []):
                out_name = outcome['name']
                mapped = map_outcome_to_internal(market_key, out_name, sport)
                if mapped == internal_market_name:
                    odds = outcome['price']
                    if best_odds is None or odds > best_odds:
                        best_odds = odds
                        best_bookmaker = bookmaker['title']
    if best_odds:
        return best_bookmaker, best_odds, 1 / best_odds
    return None, None, None