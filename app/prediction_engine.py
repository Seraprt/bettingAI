import math
import numpy as np
from scipy.stats import poisson, norm
from .db import db
from .factors import (
    get_form_score, get_strength_score, get_availability_score,
    get_tournament_factor, get_coach_score, get_home_away_score,
    get_h2h_score, get_weather_score, get_fatigue_score, get_news_score,
    get_attack_rating, get_defence_rating   # <-- NEW imports
)
from .utils import get_weather
import logging
import os
import joblib
from datetime import datetime
from .cloud_storage import download_file, list_models

WEIGHTS = {
    'form': 0.20,
    'strength': 0.15,
    'availability': 0.15,
    'tournament': 0.05,
    'coach': 0.05,
    'home_away': 0.10,
    'h2h': 0.05,
    'weather': 0.05,
    'fatigue': 0.10,
    'news': 0.10
}

# Global model cache
_model_home = None
_model_away = None
_models_loaded = False

# League average goals (for attack/defence xG)
LEAGUE_AVG_HOME = 1.35
LEAGUE_AVG_AWAY = 1.05


def load_ml_models():
    global _model_home, _model_away, _models_loaded
    if _models_loaded:
        return

    local_home = 'app/models/xg_home.pkl'
    local_away = 'app/models/xg_away.pkl'

    if not os.path.exists(local_home) or not os.path.exists(local_away):
        logging.info("📥 Local models not found, downloading from cloud...")
        if download_file('models/xg_home.pkl', local_home) and download_file('models/xg_away.pkl', local_away):
            logging.info("✅ Models downloaded from cloud.")
        else:
            logging.warning("⚠️ Could not download models from cloud. Using heuristic.")
            _models_loaded = True
            return

    try:
        _model_home = joblib.load(local_home)
        _model_away = joblib.load(local_away)
        _models_loaded = True
        logging.info("✅ ML models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load ML models: {e}")
        _models_loaded = True


# ------------------------------------------------------------------
# 1. Match context analysis
# ------------------------------------------------------------------
def get_match_context(match_doc):
    context = {
        'is_derby': False,
        'is_knockout': False,
        'is_final': False,
        'is_group': False,
        'host_advantage': False,
        'is_world_cup': False,
        'motivation': 1.0
    }
    tournament = match_doc.get('tournament', '').lower()
    stage = match_doc.get('stage', '').lower()

    if 'world cup' in tournament:
        context['is_world_cup'] = True
    if 'final' in stage:
        context['is_final'] = True
        context['motivation'] = 1.2
    elif 'semi' in stage or 'quarter' in stage:
        context['is_knockout'] = True
        context['motivation'] = 1.1
    elif 'group' in stage:
        context['is_group'] = True
        context['motivation'] = 1.0
    else:
        if 'cup' in tournament and 'round' in stage:
            context['motivation'] = 0.9

    h2h = match_doc.get('home_factors', {}).get('h2h', 0.5) if match_doc.get('home_factors') else 0.5
    if 'derby' in tournament or 'clasico' in tournament or 'rival' in tournament:
        context['is_derby'] = True
    elif h2h > 0.8 and 'league' in tournament:
        context['is_derby'] = True

    return context


# ------------------------------------------------------------------
# 2. Compute factors and xG (using attack/defence ratings)
# ------------------------------------------------------------------
def compute_match_factors(match_doc):
    home_id = match_doc['home_team_id']
    away_id = match_doc['away_team_id']
    home = db.teams.find_one({'_id': home_id})
    away = db.teams.find_one({'_id': away_id})
    if not home or not away:
        return None, None, 1.0, None, None

    weather = get_weather(home.get('latitude', 0), home.get('longitude', 0), match_doc['date'])
    weather_mult = get_weather_score(home.get('latitude', 0), home.get('longitude', 0), match_doc['date'])
    match_doc['weather_temp'] = weather['temp']
    match_doc['weather_condition'] = weather['condition']

    context = get_match_context(match_doc)
    agg_diff = None
    if match_doc.get('leg') == 2 and match_doc.get('aggregate_home') is not None:
        agg_diff = match_doc['aggregate_home'] - match_doc['aggregate_away']

    home_factors = {
        'form': get_form_score(home_id, match_doc['date']),
        'strength': get_strength_score(home_id),
        'availability': get_availability_score(home_id),
        'tournament': get_tournament_factor(match_doc.get('tournament'), match_doc.get('stage'),
                                            match_doc.get('leg', 1), agg_diff),
        'coach': get_coach_score(home_id),
        'home_away': get_home_away_score(home_id, True, away.get('strength', 50)),
        'h2h': get_h2h_score(home_id, away_id, match_doc['date']),
        'fatigue': get_fatigue_score(home_id, match_doc['date'],
                                     home.get('latitude', 0), home.get('longitude', 0)),
        'news': get_news_score(home_id)
    }
    away_factors = {
        'form': get_form_score(away_id, match_doc['date']),
        'strength': get_strength_score(away_id),
        'availability': get_availability_score(away_id),
        'tournament': get_tournament_factor(match_doc.get('tournament'), match_doc.get('stage'),
                                            match_doc.get('leg', 1), agg_diff),
        'coach': get_coach_score(away_id),
        'home_away': get_home_away_score(away_id, False, home.get('strength', 50)),
        'h2h': 1 - home_factors['h2h'],
        'fatigue': get_fatigue_score(away_id, match_doc['date'],
                                     away.get('latitude', 0), away.get('longitude', 0)),
        'news': get_news_score(away_id)
    }

    if context['is_world_cup']:
        home_factors['home_away'] = 0.5
        away_factors['home_away'] = 0.5

    # ---- NEW: Compute xG from attack/defence ratings ----
    home_attack = get_attack_rating(home_id)
    home_defence = get_defence_rating(home_id)
    away_attack = get_attack_rating(away_id)
    away_defence = get_defence_rating(away_id)

    # Expected goals using attack * defence * league average
    home_xg = LEAGUE_AVG_HOME * home_attack * away_defence
    away_xg = LEAGUE_AVG_AWAY * away_attack * home_defence

    # Apply weather multiplier
    home_xg *= weather_mult
    away_xg *= weather_mult

    # Clip to reasonable range
    home_xg = max(0.3, min(3.5, home_xg))
    away_xg = max(0.3, min(3.5, away_xg))

    # Convert to Python float for MongoDB
    home_xg = float(home_xg)
    away_xg = float(away_xg)

    return home_factors, away_factors, weather_mult, home_xg, away_xg, context


# ------------------------------------------------------------------
# 3. Predict 1X2 and store
# ------------------------------------------------------------------
def predict(match_doc):
    try:
        home_factors, away_factors, weather_mult, h_xg, a_xg, context = compute_match_factors(match_doc)
        if not home_factors:
            return

        score_diff = 0.0
        for key in WEIGHTS:
            if key == 'weather':
                continue
            diff = home_factors[key] - away_factors[key]
            score_diff += WEIGHTS[key] * diff

        if weather_mult < 1.0:
            score_diff *= (weather_mult - 0.1)
        elif weather_mult > 1.0:
            score_diff *= (weather_mult + 0.1)

        scale = 1.8
        home_win = 1 / (1 + math.exp(-score_diff * scale))
        away_win = 1 / (1 + math.exp(score_diff * scale))
        draw = 1 - home_win - away_win
        if draw < 0:
            draw = 0
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total

        contributions = []
        for key in WEIGHTS:
            if key == 'weather':
                continue
            diff = home_factors[key] - away_factors[key]
            contributions.append(WEIGHTS[key] * diff)
        std_dev = np.std(contributions) if len(contributions) > 1 else 0.5
        agreement = 1 / (1 + std_dev)
        extremity = min(1, abs(score_diff) * 2)
        confidence = 0.5 * agreement + 0.5 * extremity
        confidence = min(1, confidence)

        update_data = {
            'home_win_prob': float(home_win),
            'draw_prob': float(draw),
            'away_win_prob': float(away_win),
            'confidence': float(confidence),
            'home_factors': home_factors,
            'away_factors': away_factors,
            'home_xg': float(h_xg),
            'away_xg': float(a_xg),
            'weather_temp': match_doc.get('weather_temp'),
            'weather_condition': match_doc.get('weather_condition'),
            'context': context
        }

        db.matches.update_one(
            {'_id': match_doc['_id']},
            {'$set': update_data}
        )
    except Exception as e:
        logging.error(f"Error in predict: {e}")
        raise


# ------------------------------------------------------------------
# 4. Market probability helpers (unchanged)
# ------------------------------------------------------------------
def poisson_win_prob(h_xg, a_xg):
    prob = 0.0
    for h in range(0, 11):
        for a in range(0, 11):
            if h > a:
                prob += poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
    return prob

def poisson_draw_prob(h_xg, a_xg):
    prob = 0.0
    for g in range(0, 11):
        prob += poisson.pmf(g, h_xg) * poisson.pmf(g, a_xg)
    return prob

def poisson_cdf_total(threshold, h_xg, a_xg):
    prob = 0.0
    for h in range(0, 11):
        for a in range(0, 11):
            if h + a <= threshold:
                prob += poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
    return prob

def poisson_handicap_prob(h_xg, a_xg, handicap):
    prob = 0.0
    for h in range(0, 11):
        for a in range(0, 11):
            if h + handicap > a:
                prob += poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
    return prob


# ------------------------------------------------------------------
# 5. Compute all market probabilities (unchanged)
# ------------------------------------------------------------------
def compute_all_market_probs(h_xg, a_xg):
    probs = {}

    home_win = poisson_win_prob(h_xg, a_xg)
    draw = poisson_draw_prob(h_xg, a_xg)
    away_win = 1 - home_win - draw
    probs['home_win'] = home_win
    probs['draw'] = draw
    probs['away_win'] = away_win

    probs['1X'] = home_win + draw
    probs['X2'] = draw + away_win
    probs['12'] = home_win + away_win

    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = 1 - poisson_cdf_total(threshold, h_xg, a_xg)
        probs[f'over_{threshold}'] = over
        probs[f'under_{threshold}'] = 1 - over

    p_home_scores = 1 - poisson.pmf(0, h_xg)
    p_away_scores = 1 - poisson.pmf(0, a_xg)
    probs['btts_yes'] = p_home_scores * p_away_scores
    probs['btts_no'] = 1 - probs['btts_yes']

    for thresh in [0.5, 1.5, 2.5]:
        probs[f'home_over_{thresh}'] = 1 - poisson.cdf(thresh, h_xg)
        probs[f'home_under_{thresh}'] = poisson.cdf(thresh, h_xg)
        probs[f'away_over_{thresh}'] = 1 - poisson.cdf(thresh, a_xg)
        probs[f'away_under_{thresh}'] = poisson.cdf(thresh, a_xg)

    probs['home_over_2.5_goals'] = 1 - poisson.cdf(2, h_xg)
    probs['away_over_2.5_goals'] = 1 - poisson.cdf(2, a_xg)
    p_home_less3 = poisson.cdf(2, h_xg)
    p_away_less3 = poisson.cdf(2, a_xg)
    probs['any_team_over_2.5_goals'] = 1 - (p_home_less3 * p_away_less3)

    # Also add "under 2.5" as any_team_under_2.5_goals (for secondary picks)
    probs['any_team_under_2.5_goals'] = p_home_less3 * p_away_less3

    for hcap in [-2, -1.5, -1, 1, 1.5, 2]:
        if hcap < 0:
            probs[f'home_{hcap}'] = poisson_handicap_prob(h_xg, a_xg, hcap)
            probs[f'away_+{abs(hcap)}'] = 1 - probs[f'home_{hcap}']
        else:
            probs[f'away_-{hcap}'] = poisson_handicap_prob(h_xg, a_xg, -hcap)
            probs[f'home_+{hcap}'] = 1 - probs[f'away_-{hcap}']

    np.random.seed(42)
    odd_count = even_count = 0
    for _ in range(10000):
        h = np.random.poisson(h_xg)
        a = np.random.poisson(a_xg)
        if (h + a) % 2 == 1:
            odd_count += 1
        else:
            even_count += 1
    probs['odd'] = odd_count / 10000
    probs['even'] = even_count / 10000

    score_probs = {}
    for h in range(0, 5):
        for a in range(0, 5):
            if h == 0 and a == 0:
                continue
            key = f'{h}-{a}'
            score_probs[key] = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for key, prob in sorted_scores:
        probs[f'correct_{key}'] = prob

    return probs


# ------------------------------------------------------------------
# 6. Helper: most likely correct score
# ------------------------------------------------------------------
def get_most_likely_score(h_xg, a_xg):
    best_score = None
    best_prob = 0
    for h in range(0, 6):
        for a in range(0, 6):
            prob = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
            if prob > best_prob:
                best_prob = prob
                best_score = f"{h}-{a}"
    return best_score


# ------------------------------------------------------------------
# 7. Detailed reason generator (with secondary market)
# ------------------------------------------------------------------
def generate_detailed_reason(match_doc, market, probability, confidence,
                              home_team_name, away_team_name, correct_score,
                              secondary_market=None, secondary_prob=None):
    home_factors = match_doc.get('home_factors', {})
    away_factors = match_doc.get('away_factors', {})
    context = match_doc.get('context', {})
    h_xg = match_doc.get('home_xg', 1.2)
    a_xg = match_doc.get('away_xg', 1.0)

    reason_parts = []

    if context.get('is_final'):
        reason_parts.append("🏆 This is a FINAL – high motivation and intensity expected.")
    elif context.get('is_knockout'):
        reason_parts.append("🔥 Knockout stage – teams will be extra cautious and motivated.")
    elif context.get('is_group'):
        reason_parts.append("📊 Group stage – teams may rotate if already qualified or fight for survival.")
    if context.get('is_derby'):
        reason_parts.append("⚔️ This is a DERBY – form and statistics can be overridden by rivalry and emotion.")

    home_strength = home_factors.get('strength', 50)
    away_strength = away_factors.get('strength', 50)
    home_form = home_factors.get('form', 0.5)
    away_form = away_factors.get('form', 0.5)

    if home_strength > away_strength + 10:
        reason_parts.append(f"💪 {home_team_name} has stronger squad ({home_strength:.0f} vs {away_strength:.0f})")
    elif away_strength > home_strength + 10:
        reason_parts.append(f"💪 {away_team_name} has stronger squad ({away_strength:.0f} vs {home_strength:.0f})")
    else:
        reason_parts.append(f"⚖️ Squad strength is balanced ({home_strength:.0f} vs {away_strength:.0f})")

    if home_form > away_form + 0.2:
        reason_parts.append(f"📈 {home_team_name} is in better form ({home_form:.2f} vs {away_form:.2f})")
    elif away_form > home_form + 0.2:
        reason_parts.append(f"📈 {away_team_name} is in better form ({away_form:.2f} vs {home_form:.2f})")
    else:
        reason_parts.append(f"📊 Form is similar ({home_form:.2f} vs {away_form:.2f})")

    home_adv = home_factors.get('home_away', 0.5)
    away_adv = away_factors.get('home_away', 0.5)
    if home_adv > 0.65:
        reason_parts.append(f"🏠 Strong home advantage for {home_team_name} ({home_adv:.2f})")
    elif away_adv < 0.35:
        reason_parts.append(f"✈️ {away_team_name} struggles away from home ({away_adv:.2f})")

    home_fatigue = home_factors.get('fatigue', 1.0)
    away_fatigue = away_factors.get('fatigue', 1.0)
    if home_fatigue < 0.8:
        reason_parts.append(f"😓 {home_team_name} may be fatigued (recent match/travel)")
    if away_fatigue < 0.8:
        reason_parts.append(f"😓 {away_team_name} may be fatigued (recent match/travel)")

    home_news = home_factors.get('news', 0.5)
    away_news = away_factors.get('news', 0.5)
    if home_news > 0.6:
        reason_parts.append(f"📰 Positive news for {home_team_name}")
    elif home_news < 0.4:
        reason_parts.append(f"📰 Negative news for {home_team_name}")
    if away_news > 0.6:
        reason_parts.append(f"📰 Positive news for {away_team_name}")
    elif away_news < 0.4:
        reason_parts.append(f"📰 Negative news for {away_team_name}")

    h2h = home_factors.get('h2h', 0.5)
    if h2h > 0.7:
        reason_parts.append(f"📊 Historical advantage for {home_team_name} in head-to-head")
    elif h2h < 0.3:
        reason_parts.append(f"📊 Historical advantage for {away_team_name} in head-to-head")

    total_xg = h_xg + a_xg
    if 'over' in market or 'under' in market:
        reason_parts.append(f"⚽ Total expected goals = {total_xg:.2f} (home {h_xg:.2f}, away {a_xg:.2f})")
    elif 'btts_yes' in market:
        reason_parts.append(f"⚽ Both teams have attacking potential (home xG {h_xg:.2f}, away xG {a_xg:.2f})")
    elif 'home_win_to_nil' in market:
        reason_parts.append(f"🧤 {home_team_name} likely to keep a clean sheet (away xG {a_xg:.2f})")
    elif 'away_win_to_nil' in market:
        reason_parts.append(f"🧤 {away_team_name} likely to keep a clean sheet (home xG {h_xg:.2f})")

    if confidence > 0.8:
        reason_parts.append(f"✅ High confidence ({confidence:.0%}) in this selection")
    elif confidence > 0.6:
        reason_parts.append(f"📊 Moderate confidence ({confidence:.0%})")

    if home_fatigue < 0.7 or away_fatigue < 0.7:
        reason_parts.append("⚠️ Fatigue could affect performance")
    if home_news < 0.4 or away_news < 0.4:
        reason_parts.append("⚠️ Negative team news may impact result")

    if 'under' in market:
        reason_parts.append(f"🔒 Under {market.split('_')[1]} goals is supported by low expected total ({total_xg:.2f})")
    elif 'over' in market:
        reason_parts.append(f"⚡ Over {market.split('_')[1]} goals is supported by high expected total ({total_xg:.2f})")
    elif 'correct_' in market:
        score = market.split('_')[1]
        reason_parts.append(f"🎯 Correct score {score} has probability {probability:.1%} based on Poisson distribution")

    reason_parts.append(f"🎯 Predicted correct score: {correct_score}")

    # ---- ADD SECONDARY PICK ----
    if secondary_market and secondary_prob is not None:
        reason_parts.append(f"📌 Secondary pick: {secondary_market} with {(secondary_prob*100):.1f}% probability")

    if not reason_parts:
        reason_parts.append("Factors are balanced, but this market still offers value.")

    return " | ".join(reason_parts)


# ------------------------------------------------------------------
# 8. Get best market (for Best Bets)
# ------------------------------------------------------------------
def get_best_market(match_doc):
    if match_doc.get('home_xg') is None:
        predict(match_doc)
        match_doc = db.matches.find_one({'_id': match_doc['_id']})

    h_xg = match_doc.get('home_xg', 1.2)
    a_xg = match_doc.get('away_xg', 1.0)
    confidence = match_doc.get('confidence', 0.5)
    context = match_doc.get('context', {})

    home = db.teams.find_one({'_id': match_doc['home_team_id']})
    away = db.teams.find_one({'_id': match_doc['away_team_id']})
    home_name = home['name'] if home else 'Unknown'
    away_name = away['name'] if away else 'Unknown'

    probs = compute_all_market_probs(h_xg, a_xg)
    excluded = ['under_0.5', 'over_5.5', 'under_5.5', 'over_6.5', 'under_6.5', 'over_7.5', 'under_7.5']

    best_market = None
    best_score = 0
    best_prob = 0

    for market, prob in probs.items():
        if market.startswith('correct_'):
            continue
        if market in excluded:
            continue
        if context.get('is_derby') and market in ['home_win', 'away_win'] and prob < 0.5:
            continue
        score = prob * confidence
        if score > best_score:
            best_score = score
            best_market = market
            best_prob = prob

    if best_market is None:
        for market, prob in probs.items():
            if market.startswith('correct_'):
                continue
            if market in excluded:
                continue
            if prob > best_prob:
                best_prob = prob
                best_market = market

    if best_market:
        correct_score = get_most_likely_score(h_xg, a_xg)
        reason = generate_detailed_reason(match_doc, best_market, best_prob, confidence,
                                          home_name, away_name, correct_score)
        return {
            'market': best_market,
            'probability': best_prob,
            'confidence': confidence,
            'score': best_score,
            'stake_kelly': kelly_fraction(best_prob, 2.0),
            'reason': reason,
            'correct_score': correct_score
        }
    return None


# ------------------------------------------------------------------
# 9. Kelly staking
# ------------------------------------------------------------------
def kelly_fraction(prob, odds):
    if odds <= 1:
        return 0
    numerator = prob * (odds - 1) - (1 - prob)
    denominator = odds - 1
    if denominator <= 0:
        return 0
    kelly = numerator / denominator
    return max(0, min(1, kelly))


# ------------------------------------------------------------------
# 10. Safe markets
# ------------------------------------------------------------------
def get_safe_markets(match_doc):
    if match_doc.get('home_xg') is None:
        predict(match_doc)
        match_doc = db.matches.find_one({'_id': match_doc['_id']})
    h_xg = match_doc['home_xg']
    a_xg = match_doc['away_xg']
    confidence = match_doc['confidence']
    probs = compute_all_market_probs(h_xg, a_xg)
    safe = []
    for market, prob in probs.items():
        if market.startswith('correct_'):
            continue
        if prob >= 0.7 and confidence >= 0.5:
            safe.append({'market': market, 'probability': round(prob, 4)})
    return safe


# ------------------------------------------------------------------
# 11. Sure bets (ONE per match + secondary pick)
# ------------------------------------------------------------------
def get_sure_bets(matches, min_prob=0.8, min_confidence=0.7, max_matches=20):
    # Define the secondary market list
    secondary_candidates = [
        'home_win', '12', 'away_win', '1X', 'X2',
        'any_team_over_2.5_goals', 'any_team_under_2.5_goals'
    ]

    sure_list = []
    processed = 0
    for match in matches:
        if match.get('home_win_prob') is None:
            continue
        confidence = match.get('confidence', 0)
        if confidence < min_confidence:
            continue

        h_xg = match.get('home_xg', 1.2)
        a_xg = match.get('away_xg', 1.0)
        probs = compute_all_market_probs(h_xg, a_xg)
        context = match.get('context', {})

        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})
        home_name = home['name'] if home else 'Unknown'
        away_name = away['name'] if away else 'Unknown'

        excluded = ['under_0.5', 'over_5.5', 'under_5.5', 'over_6.5', 'under_6.5', 'over_7.5', 'under_7.5']

        best_market = None
        best_score = 0
        best_prob = 0
        for market, prob in probs.items():
            if market.startswith('correct_'):
                continue
            if market in excluded:
                continue
            if prob < min_prob:
                continue
            if context.get('is_derby') and market in ['home_win', 'away_win'] and prob < 0.65:
                continue
            score = prob * confidence
            if score > best_score:
                best_score = score
                best_market = market
                best_prob = prob

        if best_market and best_prob >= min_prob:
            correct_score = get_most_likely_score(h_xg, a_xg)

            # ---- Select secondary market ----
            secondary_market = None
            secondary_prob = None
            # Build a list of candidate scores (market, score) excluding the primary and any that are invalid
            candidates = []
            for mkt in secondary_candidates:
                if mkt == best_market:
                    continue
                prob = probs.get(mkt)
                if prob is None:
                    continue
                # For derby, avoid straight win if prob < 0.5
                if context.get('is_derby') and mkt in ['home_win', 'away_win'] and prob < 0.5:
                    continue
                score = prob * confidence
                candidates.append((score, mkt, prob))
            if candidates:
                # Sort by score descending and pick the best
                candidates.sort(reverse=True, key=lambda x: x[0])
                secondary_market = candidates[0][1]
                secondary_prob = candidates[0][2]

            reason = generate_detailed_reason(
                match, best_market, best_prob, confidence,
                home_name, away_name, correct_score,
                secondary_market, secondary_prob
            )
            sure_list.append({
                'match': f"{home_name} vs {away_name}",
                'tournament': match.get('tournament'),
                'market': best_market,
                'probability': best_prob,
                'confidence': confidence,
                'score': best_score,
                'reason': reason,
                'match_id': str(match['_id']),
                'time_remaining': get_time_remaining(match['date']),
                'correct_score': correct_score
            })
        processed += 1
        if processed >= max_matches:
            break

    sure_list.sort(key=lambda x: x['score'], reverse=True)
    return sure_list[:20]


# ------------------------------------------------------------------
# 12. Time remaining helper
# ------------------------------------------------------------------
def get_time_remaining(match_date):
    from datetime import datetime
    now = datetime.utcnow()
    diff = match_date - now
    if diff.total_seconds() < 0:
        return "🔴 Started"
    elif diff.total_seconds() < 3600:
        mins = int(diff.total_seconds() // 60)
        return f"⏳ {mins}m"
    else:
        hours = int(diff.total_seconds() // 3600)
        mins = int((diff.total_seconds() % 3600) // 60)
        return f"⏳ {hours}h {mins}m"