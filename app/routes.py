from flask import Blueprint, jsonify, request, g
from .db import db
import threading
from bson import ObjectId
from datetime import datetime, timedelta
import logging
import traceback
from .init_ratings import compute_ratings_from_all_matches
from functools import wraps
from .prediction_engine import (
    predict, get_best_market, get_sure_bets, get_safe_markets, get_time_remaining,
    compute_all_market_probs, get_market_groups, get_most_likely_score, kelly_fraction
)
from .auth import (
    register_user, login_user, is_premium, is_admin, get_user_by_id,
    create_subscription_request, approve_subscription, decline_subscription,
    revoke_subscription, get_all_subscription_requests, expire_all_expired,
    get_analytics, request_password_reset, reset_password, is_admin_credentials,
    decode_token, generate_token, hash_password, check_password, send_reset_email
)

api = Blueprint('api', __name__)

# ------------------------------------------------------------------
# Global flags for background tasks
# ------------------------------------------------------------------
_training_in_progress = False
_recompute_in_progress = False
_ingestion_in_progress = False
_init_ratings_in_progress = False  # moved here

# ------------------------------------------------------------------
# Authentication helpers (decorators)
# ------------------------------------------------------------------
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid token'}), 401
        token = token.split(' ')[1]
        user_id = decode_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        g.user_id = user_id
        g.user = get_user_by_id(user_id)
        if not g.user:
            return jsonify({'error': 'User not found'}), 401
        return f(*args, **kwargs)
    return decorated

def require_premium(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_premium(g.user_id):
            return jsonify({'error': 'Premium subscription required'}), 403
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_admin(g.user_id):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

# ------------------------------------------------------------------
# Public routes (no auth)
# ------------------------------------------------------------------
@api.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    email_or_phone = data.get('email_or_phone')
    password = data.get('password')
    if not all([username, email_or_phone, password]):
        return jsonify({'error': 'Missing fields'}), 400
    user_id, msg = register_user(username, email_or_phone, password)
    if not user_id:
        return jsonify({'error': msg}), 400
    return jsonify({'message': msg, 'user_id': str(user_id)}), 201

@api.route('/login', methods=['POST'])
def login():
    data = request.json
    username_or_phone = data.get('username_or_phone')
    password = data.get('password')
    if not username_or_phone or not password:
        return jsonify({'error': 'Missing credentials'}), 400
    token, msg = login_user(username_or_phone, password)
    if not token:
        return jsonify({'error': msg}), 401
    user = db.users.find_one({'$or': [{'username': username_or_phone}, {'email_or_phone': username_or_phone}]})
    return jsonify({
        'token': token,
        'user': {
            'id': str(user['_id']),
            'username': user['username'],
            'email_or_phone': user['email_or_phone'],
            'is_premium': user.get('is_premium', False)
        }
    }), 200

@api.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    if '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400
    token, msg = request_password_reset(email)
    if not token:
        return jsonify({'error': msg}), 404
    success = send_reset_email(email, token)
    if not success:
        return jsonify({'error': 'Could not send email'}), 500
    return jsonify({'message': 'Reset link sent to your email'}), 200

@api.route('/reset-password', methods=['POST'])
def reset_password_route():
    data = request.json
    token = data.get('token')
    new_password = data.get('new_password')
    if not token or not new_password:
        return jsonify({'error': 'Token and new password required'}), 400
    msg = reset_password(token, new_password)
    if 'Invalid' in msg:
        return jsonify({'error': msg}), 400
    return jsonify({'message': msg}), 200

@api.route('/admin-login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if is_admin_credentials(username, password):
        admin_user = db.users.find_one({'is_admin': True})
        if not admin_user:
            admin_id = db.users.insert_one({
                'username': 'Obasi excellent',
                'email_or_phone': 'admin@example.com',
                'password': hash_password('Excel1234@$'),
                'is_premium': True,
                'is_admin': True,
                'subscription_expiry': datetime.utcnow() + timedelta(days=365*100),
                'created_at': datetime.utcnow()
            }).inserted_id
            admin_user = db.users.find_one({'_id': admin_id})
        token = generate_token(admin_user['_id'])
        return jsonify({'token': token, 'admin': True}), 200
    return jsonify({'error': 'Invalid admin credentials'}), 401

# ------------------------------------------------------------------
# Protected routes (require auth)
# ------------------------------------------------------------------
@api.route('/profile', methods=['GET'])
@require_auth
def profile():
    user = g.user
    return jsonify({
        'id': str(user['_id']),
        'username': user['username'],
        'email_or_phone': user['email_or_phone'],
        'is_premium': user.get('is_premium', False),
        'is_admin': user.get('is_admin', False),
        'subscription_plan': user.get('subscription_plan'),
        'subscription_expiry': user.get('subscription_expiry').isoformat() if user.get('subscription_expiry') else None
    })

@api.route('/subscribe', methods=['POST'])
@require_auth
def subscribe():
    data = request.json
    plan = data.get('plan')
    if not plan:
        return jsonify({'error': 'Plan required'}), 400
    msg, err = create_subscription_request(g.user_id, plan)
    if err:
        return jsonify({'error': err}), 400
    return jsonify({'message': msg}), 200

@api.route('/check-subscription', methods=['GET'])   # <-- FIXED: added '=' and correct parentheses
@require_auth
def check_subscription():
    premium = is_premium(g.user_id)
    user = g.user
    return jsonify({
        'is_premium': premium,
        'subscription_plan': user.get('subscription_plan'),
        'subscription_expiry': user.get('subscription_expiry').isoformat() if user.get('subscription_expiry') else None
    })

# ------------------------------------------------------------------
# Admin routes
# ------------------------------------------------------------------
@api.route('/admin/requests', methods=['GET'])
@require_auth
@require_admin
def admin_requests():
    requests = get_all_subscription_requests()
    result = []
    for req in requests:
        user = db.users.find_one({'_id': req['user_id']})
        result.append({
            'id': str(req['_id']),
            'user_id': str(req['user_id']),
            'username': user['username'] if user else 'Unknown',
            'plan': req['plan'],
            'amount': req['amount'],
            'status': req['status'],
            'created_at': req['created_at'].isoformat()
        })
    return jsonify(result)

@api.route('/admin/approve/<request_id>', methods=['POST'])
@require_auth
@require_admin
def admin_approve(request_id):
    req = db.subscription_requests.find_one({'_id': ObjectId(request_id)})
    if not req or req['status'] != 'pending':
        return jsonify({'error': 'Request not found or not pending'}), 404
    msg = approve_subscription(req['user_id'])
    return jsonify({'message': msg})

@api.route('/admin/decline/<request_id>', methods=['POST'])
@require_auth
@require_admin
def admin_decline(request_id):
    req = db.subscription_requests.find_one({'_id': ObjectId(request_id)})
    if not req or req['status'] != 'pending':
        return jsonify({'error': 'Request not found or not pending'}), 404
    msg = decline_subscription(req['user_id'])
    return jsonify({'message': msg})

@api.route('/admin/approve-all', methods=['POST'])
@require_auth
@require_admin
def admin_approve_all():
    pending = db.subscription_requests.find({'status': 'pending'})
    count = 0
    for req in pending:
        approve_subscription(req['user_id'])
        count += 1
    return jsonify({'message': f'Approved {count} requests'})

@api.route('/admin/decline-all', methods=['POST'])
@require_auth
@require_admin
def admin_decline_all():
    pending = db.subscription_requests.find({'status': 'pending'})
    count = 0
    for req in pending:
        decline_subscription(req['user_id'])
        count += 1
    return jsonify({'message': f'Declined {count} requests'})

@api.route('/admin/revoke/<user_id>', methods=['POST'])
@require_auth
@require_admin
def admin_revoke(user_id):
    msg = revoke_subscription(user_id)
    return jsonify({'message': msg})

@api.route('/admin/expire-expired', methods=['POST'])
@require_auth
@require_admin
def admin_expire_expired():
    count = expire_all_expired()
    return jsonify({'message': f'Expired {count} users'})

@api.route('/admin/analytics', methods=['GET'])
@require_auth
@require_admin
def admin_analytics():
    analytics = get_analytics()
    return jsonify(analytics)

@api.route('/admin/users', methods=['GET'])
@require_auth
@require_admin
def admin_users():
    users = list(db.users.find().sort('created_at', -1))
    result = []
    for u in users:
        result.append({
            'id': str(u['_id']),
            'username': u['username'],
            'email_or_phone': u['email_or_phone'],
            'is_premium': u.get('is_premium', False),
            'subscription_plan': u.get('subscription_plan'),
            'subscription_expiry': u.get('subscription_expiry').isoformat() if u.get('subscription_expiry') else None,
            'created_at': u['created_at'].isoformat()
        })
    return jsonify(result)

@api.route('/admin/status', methods=['GET'])
@require_auth
@require_admin
def admin_status():
    return jsonify({
        'training': 'running' if _training_in_progress else 'idle',
        'recompute': 'running' if _recompute_in_progress else 'idle',
        'ingestion': 'running' if _ingestion_in_progress else 'idle',
        'init_ratings': 'running' if _init_ratings_in_progress else 'idle'
    }), 200

@api.route('/admin/evaluate_predictions', methods=['POST'])
@require_auth
@require_admin
def evaluate_predictions():
    """
    Evaluate all Sure Bet predictions against actual match results.
    Updates the predictions collection with 'won' or 'lost'.
    """
    from .prediction_engine import evaluate_market

    # Get all predictions that have not been evaluated
    predictions = list(db.predictions.find({'actual_outcome': None}))
    evaluated = 0
    skipped = 0

    for pred in predictions:
        match_id = pred.get('match_id')
        market = pred.get('market')
        if not match_id or not market:
            continue

        match = db.matches.find_one({'_id': ObjectId(match_id)})
        if not match:
            continue

        # Only evaluate if match is finished
        if match.get('home_goals') is None or match.get('away_goals') is None:
            skipped += 1
            continue

        home_goals = match['home_goals']
        away_goals = match['away_goals']

        # Use the helper to determine if the bet won
        won = evaluate_market(market, home_goals, away_goals)

        db.predictions.update_one(
            {'_id': pred['_id']},
            {'$set': {'actual_outcome': 'won' if won else 'lost'}}
        )
        evaluated += 1

    return jsonify({
        'message': f'Evaluated {evaluated} predictions, skipped {skipped} (not finished).'
    }), 200

# ------------------------------------------------------------------
# Prediction endpoints (with access control)
# ------------------------------------------------------------------
@api.route('/today_matches', methods=['GET'])
@require_auth
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
        match_time = m['date'].strftime('%H:%M') if m.get('date') else 'TBD'
        result.append({
            'id': str(m['_id']),
            'home': home['name'] if home else 'Unknown',
            'away': away['name'] if away else 'Unknown',
            'tournament': m.get('tournament'),
            'date': m['date'].isoformat(),
            'time': match_time,
            'time_remaining': get_time_remaining(m['date'])
        })
    return jsonify(result)

@api.route('/predict/<match_id>', methods=['GET'])
@require_auth
@require_premium
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

@api.route('/check-username', methods=['POST'])
def check_username():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'available': False, 'suggestions': []}), 400
    existing = db.users.find_one({'username': username})
    if existing:
        suggestions = []
        base = username
        i = 1
        while len(suggestions) < 5:
            alt = f"{base}{i}"
            if not db.users.find_one({'username': alt}):
                suggestions.append(alt)
            i += 1
            if i > 20:
                break
        return jsonify({'available': False, 'suggestions': suggestions})
    else:
        return jsonify({'available': True, 'suggestions': []})

@api.route('/best_bets', methods=['GET'])
@require_auth
@require_premium
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

# ------------------------------------------------------------------
# Recompute All (background)
# ------------------------------------------------------------------
def recompute_all_task():
    global _recompute_in_progress
    logging.info("🔄 Recompute task started.")
    total = db.matches.count_documents({})
    logging.info(f"📊 Total matches to recompute: {total}")
    count = 0
    cursor = db.matches.find()
    for m in cursor:
        try:
            predict(m)
            count += 1
            if count % 50 == 0:
                logging.info(f"🔄 Recompute progress: {count}/{total} matches processed")
        except Exception as e:
            logging.error(f"❌ Failed to recompute match {m['_id']}: {e}")
    logging.info(f"✅ Recompute completed: {count} matches out of {total}.")
    _recompute_in_progress = False

@api.route('/recompute_all', methods=['POST'])
@require_auth
@require_admin
def recompute_all():
    global _recompute_in_progress
    if _recompute_in_progress:
        logging.warning("⚠️ Recompute already in progress – request rejected.")
        return jsonify({'message': 'Recompute already in progress.'}), 409

    logging.info("🚀 Recompute requested by admin.")
    _recompute_in_progress = True
    thread = threading.Thread(target=recompute_all_task, daemon=True)
    thread.start()
    logging.info("✅ Recompute thread started. Check logs for progress.")
    return jsonify({'message': 'Recompute started in background. Check logs for progress.'}), 202

# ------------------------------------------------------------------
# Sure Bets
# ------------------------------------------------------------------
@api.route('/sure_bets', methods=['GET'])
@require_auth
@require_premium
def sure_bets():
    min_prob = float(request.args.get('min_prob', 0.6))
    min_confidence = float(request.args.get('min_confidence', 0.5))
    days_ahead = int(request.args.get('days_ahead', 6))

    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    matches = list(db.matches.find({'date': {'$gte': now, '$lte': future}}))

    if not matches:
        return jsonify({'message': 'No upcoming matches found.'}), 200

    sure_list = get_sure_bets(matches, min_prob, min_confidence)

    # Store for learning
    for bet in sure_list:
        match = db.matches.find_one({'_id': ObjectId(bet['match_id'])})
        if match:
            db.predictions.update_one(
                {'match_id': bet['match_id'], 'market': bet['market']},
                {'$set': {
                    'match_id': bet['match_id'],
                    'market': bet['market'],
                    'probability': bet['probability'],
                    'confidence': bet['confidence'],
                    'predicted_at': datetime.utcnow(),
                    'match_date': match['date'],
                    'actual_outcome': None,
                    'home_team': match['home_team_id'],
                    'away_team': match['away_team_id'],
                    'tournament': match.get('tournament')
                }},
                upsert=True
            )

    return jsonify(sure_list)

# ------------------------------------------------------------------
# Market Page (groups: GG, 1X2, DC+Under, DC+GG, Draw)
# ------------------------------------------------------------------
@api.route('/market_bets', methods=['GET'])
@require_auth
@require_premium
def market_bets():
    days_ahead = int(request.args.get('days_ahead', 14))
    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    matches = list(db.matches.find({'date': {'$gte': now, '$lte': future}}))

    if not matches:
        return jsonify({'message': 'No upcoming matches found.'}), 200

    # Ensure predictions exist
    for m in matches:
        if m.get('home_win_prob') is None:
            try:
                predict(m)
            except Exception as e:
                logging.error(f"Prediction failed for {m['_id']}: {e}")

    groups = get_market_groups(matches)

    # Add Draw group (using factor-based draw probability)
    draw_group = {'name': 'Draw (Most Likely)', 'bets': []}
    for match in matches:
        if match.get('home_win_prob') is None:
            continue
        draw_prob = match.get('draw_prob', 0)
        confidence = match.get('confidence', 0.5)
        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})
        home_name = home['name'] if home else 'Unknown'
        away_name = away['name'] if away else 'Unknown'
        match_label = f"{home_name} vs {away_name}"
        score = draw_prob * confidence
        draw_group['bets'].append({
            'match': match_label,
            'market': 'Draw',
            'probability': draw_prob,
            'confidence': confidence,
            'score': score,
            'match_id': str(match['_id'])
        })
    draw_group['bets'] = sorted(draw_group['bets'], key=lambda x: x['score'], reverse=True)[:4]
    groups['draw'] = draw_group

    return jsonify(groups)

# ------------------------------------------------------------------
# Prediction Accuracy Details (all predictions, grouped by market type)
# ------------------------------------------------------------------
 # add this at the top if not already imported

@api.route('/prediction_accuracy_details', methods=['GET'])
@require_auth
@require_admin
def prediction_accuracy_details():
    try:
        # Fetch all predictions that have actual_outcome set (won/lost)
        resolved = list(db.predictions.find({'actual_outcome': {'$in': ['won', 'lost']}}))
        pending = db.predictions.count_documents({'actual_outcome': None})
        total_resolved = len(resolved)
        correct = sum(1 for p in resolved if p.get('actual_outcome') == 'won')
        accuracy = round(correct / total_resolved * 100, 2) if total_resolved > 0 else 0

        correct_score_results = []
        other_results = []
        for pred in resolved:
            # Get team names from the stored IDs
            home_team_id = pred.get('home_team')
            away_team_id = pred.get('away_team')
            home_name = 'Unknown'
            away_name = 'Unknown'
            if home_team_id:
                home = db.teams.find_one({'_id': ObjectId(home_team_id)})
                if home:
                    home_name = home.get('name', 'Unknown')
            if away_team_id:
                away = db.teams.find_one({'_id': ObjectId(away_team_id)})
                if away:
                    away_name = away.get('name', 'Unknown')

            is_correct_score = pred.get('market', '').startswith('correct_')
            entry = {
                'match': f"{home_name} vs {away_name}",
                'tournament': pred.get('tournament'),
                'market': pred.get('market'),
                'probability': pred.get('probability'),
                'confidence': pred.get('confidence'),
                'predicted_at': pred.get('predicted_at').isoformat() if pred.get('predicted_at') else None,
                'actual_outcome': pred.get('actual_outcome'),
                'was_correct': pred.get('actual_outcome') == 'won'
            }
            if is_correct_score:
                correct_score_results.append(entry)
            else:
                other_results.append(entry)

        def compute_stats(items):
            total = len(items)
            correct = sum(1 for i in items if i['was_correct'])
            acc = round(correct / total * 100, 2) if total > 0 else 0
            return {'total': total, 'correct': correct, 'accuracy': acc, 'items': items}

        return jsonify({
            'total_resolved': total_resolved,
            'correct': correct,
            'accuracy': accuracy,
            'pending': pending,
            'correct_score': compute_stats(correct_score_results),
            'other_markets': compute_stats(other_results),
            'all': compute_stats(resolved)
        })
    except Exception as e:
        logging.error(f"Error in prediction_accuracy_details: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500# ------------------------------------------------------------------
# Factors Page (admin only) – shows factor-based predictions for all market groups
# ------------------------------------------------------------------
@api.route('/factors_predictions', methods=['GET'])
@require_auth
@require_admin
def factors_predictions():
    from .prediction_engine import (
        compute_all_market_probs, get_most_likely_score, kelly_fraction, poisson
    )
    days_ahead = int(request.args.get('days_ahead', 14))
    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    matches = list(db.matches.find({'date': {'$gte': now, '$lte': future}}))

    if not matches:
        return jsonify({'message': 'No upcoming matches found.'}), 200

    # Ensure predictions exist
    for m in matches:
        if m.get('home_win_prob') is None:
            try:
                predict(m)
            except Exception as e:
                logging.error(f"Prediction failed for {m['_id']}: {e}")

    # Define groups (same as market page + correct_score)
    groups = {
        'gg': {'name': 'Both Teams to Score (GG)', 'bets': []},
        '1x2': {'name': '1X2 (Highest Probability)', 'bets': []},
        'dc_under': {'name': 'Double Chance + Under 3.5', 'bets': []},
        'dc_gg': {'name': 'Double Chance + GG', 'bets': []},
        'correct_score': {'name': 'Correct Score (Most Likely)', 'bets': []}
    }

    for match in matches:
        if match.get('home_win_prob') is None:
            continue
        h_xg = match.get('home_xg', 1.2)
        a_xg = match.get('away_xg', 1.0)
        probs = compute_all_market_probs(h_xg, a_xg)
        confidence = match.get('confidence', 0.5)

        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})
        home_name = home['name'] if home else 'Unknown'
        away_name = away['name'] if away else 'Unknown'
        match_label = f"{home_name} vs {away_name}"
        home_factors = match.get('home_factors', {})
        away_factors = match.get('away_factors', {})

        # 1. GG (Both Teams to Score)
        gg_prob = probs.get('btts_yes', 0)
        groups['gg']['bets'].append({
            'match': match_label,
            'market': 'Both Teams to Score (Yes)',
            'probability': gg_prob,
            'confidence': confidence,
            'score': gg_prob * confidence,
            'match_id': str(match['_id']),
            'home_factors': home_factors,
            'away_factors': away_factors
        })

        # 2. 1X2 – pick highest probability outcome
        home_win = probs.get('home_win', 0)
        draw = probs.get('draw', 0)
        away_win = probs.get('away_win', 0)
        best_market = 'home_win' if home_win >= draw and home_win >= away_win else ('draw' if draw >= away_win else 'away_win')
        best_prob = max(home_win, draw, away_win)
        groups['1x2']['bets'].append({
            'match': match_label,
            'market': best_market.replace('_', ' ').title(),
            'probability': best_prob,
            'confidence': confidence,
            'score': best_prob * confidence,
            'match_id': str(match['_id']),
            'home_factors': home_factors,
            'away_factors': away_factors,
            # Kelly stakes for 1X2 (assuming odds = 2.0)
            'home_stake': kelly_fraction(home_win, 2.0),
            'draw_stake': kelly_fraction(draw, 2.0),
            'away_stake': kelly_fraction(away_win, 2.0)
        })

        # 3. Double Chance + Under 3.5
        dc_1x = probs.get('1X', 0)
        dc_x2 = probs.get('X2', 0)
        best_dc = max(dc_1x, dc_x2)
        best_dc_label = '1X' if dc_1x >= dc_x2 else 'X2'
        under_3_5 = probs.get('under_3.5', 0)
        combined_prob = best_dc * under_3_5
        groups['dc_under']['bets'].append({
            'match': match_label,
            'market': f'{best_dc_label} + Under 3.5',
            'probability': combined_prob,
            'confidence': confidence,
            'score': combined_prob * confidence,
            'match_id': str(match['_id']),
            'home_factors': home_factors,
            'away_factors': away_factors
        })

        # 4. Double Chance + GG
        gg_prob = probs.get('btts_yes', 0)
        combined_gg_prob = best_dc * gg_prob
        groups['dc_gg']['bets'].append({
            'match': match_label,
            'market': f'{best_dc_label} + GG',
            'probability': combined_gg_prob,
            'confidence': confidence,
            'score': combined_gg_prob * confidence,
            'match_id': str(match['_id']),
            'home_factors': home_factors,
            'away_factors': away_factors
        })

        # 5. Correct Score – most likely scoreline and its probability
        correct_score = get_most_likely_score(h_xg, a_xg)
        # Compute probability of that exact score
        h, a = map(int, correct_score.split('-'))
        score_prob = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
        groups['correct_score']['bets'].append({
            'match': match_label,
            'market': f'Correct Score {correct_score}',
            'probability': score_prob,
            'confidence': confidence,
            'score': score_prob * confidence,
            'match_id': str(match['_id']),
            'home_factors': home_factors,
            'away_factors': away_factors
        })

    # Sort and keep top 4 per group
    for key in groups:
        groups[key]['bets'] = sorted(groups[key]['bets'], key=lambda x: x['score'], reverse=True)[:4]

    return jsonify(groups)

# ------------------------------------------------------------------
# Administrative / internal endpoints
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Training (background)
# ------------------------------------------------------------------
@api.route('/train', methods=['POST'])
def trigger_training():
    global _training_in_progress
    if _training_in_progress:
        return jsonify({'message': 'Training already in progress. Please wait.'}), 409

    from .train_model import run_training
    _training_in_progress = True

    def training_task():
        global _training_in_progress
        try:
            run_training()
        except Exception as e:
            logging.error(f"Training failed: {e}")
        finally:
            _training_in_progress = False

    thread = threading.Thread(target=training_task, daemon=True)
    thread.start()
    return jsonify({'message': 'Training started in background. Check logs for progress.'}), 202

# ------------------------------------------------------------------
# Ingestion (background with progress logging)
# ------------------------------------------------------------------
def ingestion_task():
    global _ingestion_in_progress
    from .data_ingestion import fetch_all_football
    logging.info("🚀 Ingestion started in background.")
    try:
        logging.info("📥 Fetching matches from Football-Data.org (past 120 days + future 14 days)...")
        fetch_all_football()
        logging.info("✅ Ingestion completed successfully.")
    except Exception as e:
        logging.error(f"❌ Ingestion failed: {e}")
    finally:
        _ingestion_in_progress = False

@api.route('/admin/ingest', methods=['POST'])
@require_auth
@require_admin
def admin_ingest():
    global _ingestion_in_progress
    if _ingestion_in_progress:
        return jsonify({'message': 'Ingestion already in progress.'}), 409
    _ingestion_in_progress = True
    thread = threading.Thread(target=ingestion_task, daemon=True)
    thread.start()
    return jsonify({'message': 'Ingestion started in background. Check logs for progress.'}), 202

# ------------------------------------------------------------------
# Other endpoints (available_markets, debug, health, etc.)
# ------------------------------------------------------------------
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

@api.route('/reload_models', methods=['POST'])
def reload_models():
    from .prediction_engine import load_ml_models, _models_loaded
    _models_loaded = False
    load_ml_models()
    return jsonify({'message': 'Models reloaded.'}), 200

@api.route('/ingest', methods=['POST'])
def ingest():
    from .data_ingestion import fetch_all_football
    fetch_all_football()
    return jsonify({'message': 'Ingestion triggered.'}), 200

@api.route('/prediction_accuracy', methods=['GET'])
def prediction_accuracy():
    predictions = list(db.predictions.find({'actual_outcome': {'$ne': None}}))
    total = len(predictions)
    if total == 0:
        return jsonify({'message': 'No resolved predictions yet.'}), 200
    correct = sum(1 for p in predictions if p.get('actual_outcome') == 'won')
    lost = sum(1 for p in predictions if p.get('actual_outcome') == 'lost')
    pending = db.predictions.count_documents({'actual_outcome': None})
    return jsonify({
        'total_resolved': total,
        'correct': correct,
        'lost': lost,
        'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
        'pending': pending
    })

@api.route('/admin/learning_insights', methods=['GET'])
@require_auth
@require_admin
def learning_insights():
    """
    Returns statistics about how much data the system has collected per team,
    and an overall learning score (0-1) indicating data quality.
    """
    teams = list(db.teams.find({}))
    team_data = []
    total_matches = 0
    min_matches = float('inf')
    max_matches = 0

    for team in teams:
        team_id = team['_id']
        count = db.matches.count_documents({
            '$or': [
                {'home_team_id': team_id, 'home_goals': {'$ne': None}},
                {'away_team_id': team_id, 'away_goals': {'$ne': None}}
            ]
        })
        total_matches += count
        if count < min_matches:
            min_matches = count
        if count > max_matches:
            max_matches = count
        team_data.append({
            'name': team.get('name', 'Unknown'),
            'matches': count,
            'attack_rating': team.get('attack_rating', 1.0),
            'defence_rating': team.get('defence_rating', 1.0),
            'elo_rating': team.get('elo_rating', 1500)
        })

    team_data_sorted = sorted(team_data, key=lambda x: x['matches'], reverse=True)

    total_teams = len(teams)
    teams_with_data = sum(1 for t in team_data if t['matches'] > 0)
    teams_insufficient = sum(1 for t in team_data if t['matches'] < 5)
    avg_matches = total_matches / total_teams if total_teams > 0 else 0

    pct_teams_with_10 = sum(1 for t in team_data if t['matches'] >= 10) / total_teams if total_teams > 0 else 0
    avg_match_score = min(1.0, avg_matches / 50)
    ratings_diverse = sum(1 for t in team_data if abs(t['attack_rating'] - 1.0) > 0.05 or abs(t['defence_rating'] - 1.0) > 0.05) / total_teams if total_teams > 0 else 0

    learning_score = (pct_teams_with_10 * 0.5) + (avg_match_score * 0.3) + (ratings_diverse * 0.2)
    learning_score = round(min(1.0, learning_score), 3)

    return jsonify({
        'total_teams': total_teams,
        'teams_with_data': teams_with_data,
        'teams_insufficient': teams_insufficient,
        'avg_matches_per_team': round(avg_matches, 2),
        'min_matches': min_matches if min_matches != float('inf') else 0,
        'max_matches': max_matches,
        'learning_score': learning_score,
        'top_teams': team_data_sorted[:20]
    }), 200

@api.route('/admin/init_ratings', methods=['POST'])
@require_auth
@require_admin
def admin_init_ratings():
    global _init_ratings_in_progress
    if _init_ratings_in_progress:
        return jsonify({'message': 'Initialisation already in progress.'}), 409

    from .init_ratings import compute_ratings_from_all_matches

    def init_task():
        global _init_ratings_in_progress
        try:
            compute_ratings_from_all_matches()
        except Exception as e:
            logging.error(f"Init ratings failed: {e}")
        finally:
            _init_ratings_in_progress = False

    _init_ratings_in_progress = True
    thread = threading.Thread(target=init_task, daemon=True)
    thread.start()
    return jsonify({'message': 'Initialisation started in background. Check logs for progress.'}), 202