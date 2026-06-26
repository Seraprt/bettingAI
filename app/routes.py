from flask import Blueprint, jsonify, request
from .db import db
from bson import ObjectId
from datetime import datetime, timedelta
import logging
import traceback
from .prediction_engine import (
    predict, get_best_market, get_sure_bets, get_safe_markets, get_time_remaining
)

api = Blueprint('api', __name__)
@api.route('/today_matches', methods=['GET'])
def today_matches():
    now = datetime.utcnow()
    future = now + timedelta(days=14)
    matches = list(db.matches.find({
        'date': {'$gte': now, '$lte': future}
    }).sort('date', 1))
    result = []
    for m in matches:
        home = db.teams.find_one({'_id': m['home_team_id']})
        away = db.teams.find_one({'_id': m['away_team_id']})
        result.append({
            'id': str(m['_id']),
            'home': home['name'] if home else 'Unknown',
            'away': away['name'] if away else 'Unknown',
            'tournament': m.get('tournament'),
            'date': m['date'].isoformat(),
            'time_remaining': get_time_remaining(m['date'])
        })
    return jsonify(result)

@api.route('/predict/<match_id>', methods=['GET'])
def get_prediction(match_id):
    try:
        match = db.matches.find_one({'_id': ObjectId(match_id)})
        if not match:
            return jsonify({'error': 'Match not found'}), 404
        if match.get('home_win_prob') is None:
            predict(match)
            match = db.matches.find_one({'_id': ObjectId(match_id)})
        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})
        safe_markets = get_safe_markets(match)
        return jsonify({
            'match_id': str(match['_id']),
            'home_team': home['name'] if home else 'Unknown',
            'away_team': away['name'] if away else 'Unknown',
            'home_win_prob': round(match.get('home_win_prob', 0), 4),
            'draw_prob': round(match.get('draw_prob', 0), 4),
            'away_win_prob': round(match.get('away_win_prob', 0), 4),
            'confidence': round(match.get('confidence', 0), 4),
            'home_xg': round(match.get('home_xg', 0), 2),
            'away_xg': round(match.get('away_xg', 0), 2),
            'time_remaining': get_time_remaining(match['date']),
            'weather': {
                'temp': match.get('weather_temp'),
                'condition': match.get('weather_condition')
            },
            'factors': {
                'home': match.get('home_factors'),
                'away': match.get('away_factors')
            },
            'safe_markets': safe_markets
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/update_strength/<team_id>', methods=['POST'])
def update_strength(team_id):
    from .factors import get_strength_score
    team = db.teams.find_one({'_id': ObjectId(team_id)})
    if not team:
        return jsonify({'error': 'Team not found'}), 404
    new_strength = get_strength_score(team_id)
    db.teams.update_one({'_id': ObjectId(team_id)}, {'$set': {'strength': new_strength}})
    return jsonify({'status': 'updated', 'strength': new_strength})

@api.route('/predict_all', methods=['POST'])
def predict_all():
    now = datetime.utcnow()
    future = now + timedelta(days=14)
    matches = list(db.matches.find({'date': {'$gte': now, '$lte': future}}))
    count = 0
    for m in matches:
        if m.get('home_win_prob') is None:
            try:
                predict(m)
                count += 1
            except Exception as e:
                logging.error(f"Failed to predict {m['_id']}: {e}")
    return jsonify({'message': f'Predictions computed for {count} matches.'}), 200

@api.route('/best_bets', methods=['GET'])
def best_bets():
    min_confidence = float(request.args.get('min_confidence', 0.2))
    days_ahead = int(request.args.get('days_ahead', 14))
    tournament = request.args.get('tournament')
    min_prob = float(request.args.get('min_prob', 0.2))
    max_prob = float(request.args.get('max_prob', 1))
    limit = int(request.args.get('limit', 50))

    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    query = {'date': {'$gte': now, '$lte': future}}
    if tournament:
        query['tournament'] = tournament

    upcoming = list(db.matches.find(query).sort('date', 1))

    if not upcoming:
        return jsonify({'message': 'No upcoming matches found.'}), 200

    best_bets = []
    all_markets = set()
    for match in upcoming:
        if match.get('home_win_prob') is None:
            try:
                predict(match)
                match = db.matches.find_one({'_id': match['_id']})
            except Exception as e:
                logging.error(f"Prediction failed for {match['_id']}: {e}")
                continue

        best = get_best_market(match)
        if not best:
            continue

        if best['confidence'] < min_confidence:
            continue

        prob = best['probability']
        if prob < min_prob or prob > max_prob:
            continue

        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})

        market = best['market']
        all_markets.add(market)

        best_bets.append({
            'match': f"{home['name'] if home else 'Unknown'} vs {away['name'] if away else 'Unknown'}",
            'tournament': match.get('tournament'),
            'market': market,
            'probability': prob,
            'confidence': best['confidence'],
            'combined_score': best['score'],
            'stake_kelly': best['stake_kelly'],
            'reason': best['reason'],
            'match_id': str(match['_id']),
            'time_remaining': get_time_remaining(match['date'])
        })

    best_bets.sort(key=lambda x: x['combined_score'], reverse=True)
    limited_bets = best_bets[:limit]
    return jsonify({
        'bets': limited_bets,
        'available_markets': sorted(list(all_markets))
    })

@api.route('/sure_bets', methods=['GET'])
def sure_bets():
    min_prob = float(request.args.get('min_prob', 0.6))
    min_confidence = float(request.args.get('min_confidence', 0.5))
    days_ahead = int(request.args.get('days_ahead', 3))

    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    matches = list(db.matches.find({'date': {'$gte': now, '$lte': future}}))

    if not matches:
        return jsonify({'message': 'No upcoming matches found.'}), 200

    sure_list = get_sure_bets(matches, min_prob, min_confidence)
    return jsonify(sure_list)

@api.route('/available_markets', methods=['GET'])
def available_markets():
    markets = [
        'home_win', 'draw', 'away_win',
        '1X', 'X2', '12',
        'over_0.5', 'under_0.5',
        'over_1.5', 'under_1.5',
        'over_2.5', 'under_2.5',
        'over_3.5', 'under_3.5',
        'over_4.5', 'under_4.5',
        'over_5.5', 'under_5.5',
        'over_6.5', 'under_6.5',
        'over_7.5', 'under_7.5',
        'btts_yes', 'btts_no',
        'home_over_0.5', 'home_under_0.5',
        'home_over_1.5', 'home_under_1.5',
        'home_over_2.5', 'home_under_2.5',
        'away_over_0.5', 'away_under_0.5',
        'away_over_1.5', 'away_under_1.5',
        'away_over_2.5', 'away_under_2.5',
        'home_over_2.5_goals', 'away_over_2.5_goals', 'any_team_over_2.5_goals',
        'home_win_to_nil', 'away_win_to_nil',
        'exact_2_goals', 'exact_3_goals', 'exact_4_goals',
        'home_-0.5', 'away_+0.5',
        'home_-1', 'away_+1',
        'home_-1.5', 'away_+1.5',
        'home_-2', 'away_+2',
        'home_-2.5', 'away_+2.5',
        'home_+0.5', 'away_-0.5',
        'odd', 'even',
        'ht_ft_home_home', 'ht_ft_home_draw', 'ht_ft_home_away',
        'ht_ft_draw_home', 'ht_ft_draw_draw', 'ht_ft_draw_away',
        'ht_ft_away_home', 'ht_ft_away_draw', 'ht_ft_away_away'
    ]
    return jsonify(sorted(markets))

@api.route('/debug_matches', methods=['GET'])
def debug_matches():
    matches = list(db.matches.find().sort('date', 1))
    result = []
    for m in matches:
        home = db.teams.find_one({'_id': m['home_team_id']})
        away = db.teams.find_one({'_id': m['away_team_id']})
        result.append({
            'home': home['name'] if home else 'Unknown',
            'away': away['name'] if away else 'Unknown',
            'tournament': m.get('tournament'),
            'date': m['date'].isoformat() if m.get('date') else None,
            'has_prediction': m.get('home_win_prob') is not None,
            'confidence': m.get('confidence'),
            'home_xg': m.get('home_xg'),
            'away_xg': m.get('away_xg')
        })
    return jsonify(result)

@api.route('/test_market/<match_id>', methods=['GET'])
def test_market(match_id):
    match = db.matches.find_one({'_id': ObjectId(match_id)})
    if not match:
        return jsonify({'error': 'Not found'}), 404
    if match.get('home_xg') is None:
        predict(match)
        match = db.matches.find_one({'_id': ObjectId(match_id)})
    from .prediction_engine import compute_all_market_probs
    probs = compute_all_market_probs(match['home_xg'], match['away_xg'])
    best = get_best_market(match)
    return jsonify({
        'xg': {'home': match['home_xg'], 'away': match['away_xg']},
        'all_market_probs': probs,
        'best_market': best
    })

@api.route('/force_predict_all', methods=['POST'])
def force_predict_all():
    matches = list(db.matches.find())
    count = 0
    for m in matches:
        if m.get('home_win_prob') is None:
            try:
                predict(m)
                count += 1
            except Exception as e:
                logging.error(f"Failed to predict {m['_id']}: {e}")
    return jsonify({'message': f'Predictions computed for {count} matches out of {len(matches)} total.'}), 200
@api.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive', 'timestamp': datetime.utcnow().isoformat()}), 200

@api.route('/ingest', methods=['POST'])
def ingest():
    from .data_ingestion import fetch_all_football
    fetch_all_football()
    return jsonify({'message': 'Ingestion triggered.'}), 200