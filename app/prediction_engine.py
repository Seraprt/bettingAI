import math
import numpy as np
from scipy.stats import poisson, norm
from .db import db
from .factors import (
    get_form_score, get_strength_score, get_availability_score,
    get_tournament_factor, get_coach_score, get_home_away_score,
    get_h2h_score, get_weather_score, get_fatigue_score, get_news_score
)
from .utils import get_weather
import logging

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

# ------------------------------------------------------------------
# 1. Match context analysis
# ------------------------------------------------------------------
def get_match_context(match_doc):
    """Return context flags: derby, knockout, final, group, host_advantage, etc."""
    context = {
        'is_derby': False,
        'is_knockout': False,
        'is_final': False,
        'is_group': False,
        'host_advantage': False,
        'is_world_cup': False,
        'motivation': 1.0  # multiplier
    }
    tournament = match_doc.get('tournament', '').lower()
    stage = match_doc.get('stage', '').lower()

    if 'world cup' in tournament:
        context['is_world_cup'] = True
        # For World Cup, home advantage is only for host (we don't know host, so ignore)
        # But we can set home_away factor to neutral if not host (we'll handle in factors)
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
        # Domestic cup early rounds might have lower motivation
        if 'cup' in tournament and 'round' in stage:
            context['motivation'] = 0.9

    # Derby detection: simple heuristic – if teams are from same city or have high H2H intensity
    # We'll use H2H factor: if h2h > 0.7 and both teams are from same country (approx), consider derby
    # For now, we rely on the user's request: we'll mark as derby if the match is known (we can add a list later)
    # For demonstration, we'll use a placeholder: if tournament is domestic and h2h factor > 0.7
    h2h = match_doc.get('home_factors', {}).get('h2h', 0.5) if match_doc.get('home_factors') else 0.5
    if 'derby' in tournament or 'clasico' in tournament or 'rival' in tournament:
        context['is_derby'] = True
    elif h2h > 0.8 and 'league' in tournament:
        context['is_derby'] = True

    return context

# ------------------------------------------------------------------
# 2. Compute factors and xG
# ------------------------------------------------------------------
def compute_match_factors(match_doc):
    home_id = match_doc['home_team_id']
    away_id = match_doc['away_team_id']
    home = db.teams.find_one({'_id': home_id})
    away = db.teams.find_one({'_id': away_id})
    if not home or not away:
        return None, None, 1.0, None, None

    # Get weather
    weather = get_weather(home.get('latitude', 0), home.get('longitude', 0), match_doc['date'])
    weather_mult = get_weather_score(home.get('latitude', 0), home.get('longitude', 0), match_doc['date'])
    match_doc['weather_temp'] = weather['temp']
    match_doc['weather_condition'] = weather['condition']

    # Tournament context
    context = get_match_context(match_doc)
    # Adjust home/away factor for World Cup (neutral) or finals
    if context['is_world_cup']:
        # For World Cup, home advantage is only for host; we don't have host info, so set to neutral
        # We'll adjust home_away_score later
        pass

    agg_diff = None
    if match_doc.get('leg') == 2 and match_doc.get('aggregate_home') is not None:
        agg_diff = match_doc['aggregate_home'] - match_doc['aggregate_away']

    # Home factors
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

    # For World Cup, reduce home advantage
    if context['is_world_cup']:
        # Make home_away factor neutral (0.5) for both
        home_factors['home_away'] = 0.5
        away_factors['home_away'] = 0.5

    # Convert to xG
    base_home = 1.2
    base_away = 1.0
    home_xg = base_home * (0.5 + 0.5 * home_factors['strength']/100) * (1 + 0.1 * home_factors['form']) + 0.3 * (home_factors['home_away'] - 0.5)
    away_xg = base_away * (0.5 + 0.5 * away_factors['strength']/100) * (1 + 0.1 * away_factors['form'])
    home_xg *= (0.9 + 0.1 * home_factors['fatigue'])
    away_xg *= (0.9 + 0.1 * away_factors['fatigue'])
    home_xg *= weather_mult
    away_xg *= weather_mult
    home_xg = max(0.3, min(3.5, home_xg))
    away_xg = max(0.3, min(3.5, away_xg))

    return home_factors, away_factors, weather_mult, home_xg, away_xg, context

# ------------------------------------------------------------------
# 3. Predict 1X2 and store
# ------------------------------------------------------------------
def predict(match_doc):
    try:
        home_factors, away_factors, weather_mult, h_xg, a_xg, context = compute_match_factors(match_doc)
        if not home_factors:
            return

        # Weighted score difference
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

        # Confidence
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

        # Store
        db.matches.update_one(
            {'_id': match_doc['_id']},
            {'$set': {
                'home_win_prob': home_win,
                'draw_prob': draw,
                'away_win_prob': away_win,
                'confidence': confidence,
                'home_factors': home_factors,
                'away_factors': away_factors,
                'home_xg': h_xg,
                'away_xg': a_xg,
                'weather_temp': match_doc.get('weather_temp'),
                'weather_condition': match_doc.get('weather_condition'),
                'context': context
            }}
        )
    except Exception as e:
        logging.error(f"Error in predict: {e}")
        raise

# ------------------------------------------------------------------
# 4. Market probability helpers
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

def compute_all_market_probs(h_xg, a_xg):
    probs = {}

    # 1X2
    home_win = poisson_win_prob(h_xg, a_xg)
    draw = poisson_draw_prob(h_xg, a_xg)
    away_win = 1 - home_win - draw
    probs['home_win'] = home_win
    probs['draw'] = draw
    probs['away_win'] = away_win

    # Double Chance
    probs['1X'] = home_win + draw
    probs['X2'] = draw + away_win
    probs['12'] = home_win + away_win

    # Over/Under
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
        over = 1 - poisson_cdf_total(threshold, h_xg, a_xg)
        probs[f'over_{threshold}'] = over
        probs[f'under_{threshold}'] = 1 - over

    # BTTS
    p_home_scores = 1 - poisson.pmf(0, h_xg)
    p_away_scores = 1 - poisson.pmf(0, a_xg)
    probs['btts_yes'] = p_home_scores * p_away_scores
    probs['btts_no'] = 1 - probs['btts_yes']

    # Team Totals
    for thresh in [0.5, 1.5, 2.5]:
        probs[f'home_over_{thresh}'] = 1 - poisson.cdf(thresh, h_xg)
        probs[f'home_under_{thresh}'] = poisson.cdf(thresh, h_xg)
        probs[f'away_over_{thresh}'] = 1 - poisson.cdf(thresh, a_xg)
        probs[f'away_under_{thresh}'] = poisson.cdf(thresh, a_xg)

    # Team to score 3+ goals
    probs['home_over_2.5_goals'] = 1 - poisson.cdf(2, h_xg)
    probs['away_over_2.5_goals'] = 1 - poisson.cdf(2, a_xg)
    p_home_less3 = poisson.cdf(2, h_xg)
    p_away_less3 = poisson.cdf(2, a_xg)
    probs['any_team_over_2.5_goals'] = 1 - (p_home_less3 * p_away_less3)

    # Exact total goals
    for exact in [2, 3, 4]:
        prob_exact = 0.0
        for h in range(0, 11):
            a = exact - h
            if 0 <= a <= 10:
                prob_exact += poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
        probs[f'exact_{exact}_goals'] = prob_exact

    # Home win to nil, Away win to nil
    prob_home_win_to_nil = 0.0
    prob_away_win_to_nil = 0.0
    for h in range(1, 11):
        prob_home_win_to_nil += poisson.pmf(h, h_xg) * poisson.pmf(0, a_xg)
    for a in range(1, 11):
        prob_away_win_to_nil += poisson.pmf(0, h_xg) * poisson.pmf(a, a_xg)
    probs['home_win_to_nil'] = prob_home_win_to_nil
    probs['away_win_to_nil'] = prob_away_win_to_nil

    # Handicaps
    handicaps = [-2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5]
    for hcap in handicaps:
        if hcap < 0:
            key = f'home_{hcap}'
            probs[key] = poisson_handicap_prob(h_xg, a_xg, hcap)
            away_key = f'away_+{abs(hcap)}'
            probs[away_key] = 1 - probs[key]
        else:
            # away -hcap is covered by the negative loop
            pass
    # Add home +0.5, away -0.5 for completeness
    probs['home_+0.5'] = 1 - away_win  # home not lose
    probs['away_-0.5'] = away_win

    # Odd/Even (simulation)
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

    # Correct Scores (top 10)
    score_probs = {}
    for h in range(0, 5):
        for a in range(0, 5):
            if h == 0 and a == 0:
                continue
            key = f'{h}-{a}'
            score_probs[key] = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    for key, prob in sorted_scores:
        probs[f'correct_{key}'] = prob

    return probs

# ------------------------------------------------------------------
# 5. Detailed reason generation
# ------------------------------------------------------------------
def generate_detailed_reason(match_doc, market, probability, confidence):
    home_factors = match_doc.get('home_factors', {})
    away_factors = match_doc.get('away_factors', {})
    context = match_doc.get('context', {})
    h_xg = match_doc.get('home_xg', 1.2)
    a_xg = match_doc.get('away_xg', 1.0)
    home = db.teams.find_one({'_id': match_doc['home_team_id']})
    away = db.teams.find_one({'_id': match_doc['away_team_id']})

    reason_parts = []

    # 1. Tournament context
    if context.get('is_final'):
        reason_parts.append("🏆 This is a FINAL – high motivation and intensity expected.")
    elif context.get('is_knockout'):
        reason_parts.append("🔥 Knockout stage – teams will be extra cautious and motivated.")
    elif context.get('is_group'):
        reason_parts.append("📊 Group stage – teams may rotate if already qualified or fight for survival.")
    if context.get('is_derby'):
        reason_parts.append("⚔️ This is a DERBY – form and statistics can be overridden by rivalry and emotion.")

    # 2. Team strength and form
    home_strength = home_factors.get('strength', 50)
    away_strength = away_factors.get('strength', 50)
    home_form = home_factors.get('form', 0.5)
    away_form = away_factors.get('form', 0.5)

    if home_strength > away_strength + 10:
        reason_parts.append(f"💪 {home['name']} has stronger squad ({home_strength:.0f} vs {away_strength:.0f})")
    elif away_strength > home_strength + 10:
        reason_parts.append(f"💪 {away['name']} has stronger squad ({away_strength:.0f} vs {home_strength:.0f})")
    else:
        reason_parts.append(f"⚖️ Squad strength is balanced ({home_strength:.0f} vs {away_strength:.0f})")

    if home_form > away_form + 0.2:
        reason_parts.append(f"📈 {home['name']} is in better form ({home_form:.2f} vs {away_form:.2f})")
    elif away_form > home_form + 0.2:
        reason_parts.append(f"📈 {away['name']} is in better form ({away_form:.2f} vs {home_form:.2f})")
    else:
        reason_parts.append(f"📊 Form is similar ({home_form:.2f} vs {away_form:.2f})")

    # 3. Home/Away advantage
    home_adv = home_factors.get('home_away', 0.5)
    away_adv = away_factors.get('home_away', 0.5)
    if home_adv > 0.65:
        reason_parts.append(f"🏠 Strong home advantage for {home['name']} ({home_adv:.2f})")
    elif away_adv < 0.35:
        reason_parts.append(f"✈️ {away['name']} struggles away from home ({away_adv:.2f})")

    # 4. Fatigue
    home_fatigue = home_factors.get('fatigue', 1.0)
    away_fatigue = away_factors.get('fatigue', 1.0)
    if home_fatigue < 0.8:
        reason_parts.append(f"😓 {home['name']} may be fatigued (recent match/travel)")
    if away_fatigue < 0.8:
        reason_parts.append(f"😓 {away['name']} may be fatigued (recent match/travel)")

    # 5. News
    home_news = home_factors.get('news', 0.5)
    away_news = away_factors.get('news', 0.5)
    if home_news > 0.6:
        reason_parts.append(f"📰 Positive news for {home['name']}")
    elif home_news < 0.4:
        reason_parts.append(f"📰 Negative news for {home['name']}")
    if away_news > 0.6:
        reason_parts.append(f"📰 Positive news for {away['name']}")
    elif away_news < 0.4:
        reason_parts.append(f"📰 Negative news for {away['name']}")

    # 6. Head-to-head
    h2h = home_factors.get('h2h', 0.5)
    if h2h > 0.7:
        reason_parts.append(f"📊 Historical advantage for {home['name']} in head-to-head")
    elif h2h < 0.3:
        reason_parts.append(f"📊 Historical advantage for {away['name']} in head-to-head")

    # 7. Expected goals (xG)
    total_xg = h_xg + a_xg
    if 'over' in market or 'under' in market:
        reason_parts.append(f"⚽ Total expected goals = {total_xg:.2f} (home {h_xg:.2f}, away {a_xg:.2f})")
    elif 'btts_yes' in market:
        reason_parts.append(f"⚽ Both teams have attacking potential (home xG {h_xg:.2f}, away xG {a_xg:.2f})")
    elif 'home_win_to_nil' in market:
        reason_parts.append(f"🧤 {home['name']} likely to keep a clean sheet (away xG {a_xg:.2f})")
    elif 'away_win_to_nil' in market:
        reason_parts.append(f"🧤 {away['name']} likely to keep a clean sheet (home xG {h_xg:.2f})")

    # 8. Confidence and probability
    if confidence > 0.8:
        reason_parts.append(f"✅ High confidence ({confidence:.0%}) in this selection")
    elif confidence > 0.6:
        reason_parts.append(f"📊 Moderate confidence ({confidence:.0%})")

    # 9. Negative factors (if any)
    if home_fatigue < 0.7 or away_fatigue < 0.7:
        reason_parts.append("⚠️ Fatigue could affect performance")
    if home_news < 0.4 or away_news < 0.4:
        reason_parts.append("⚠️ Negative team news may impact result")

    # 10. Market-specific final note
    if 'under' in market:
        reason_parts.append(f"🔒 Under {market.split('_')[1]} goals is supported by low expected total ({total_xg:.2f})")
    elif 'over' in market:
        reason_parts.append(f"⚡ Over {market.split('_')[1]} goals is supported by high expected total ({total_xg:.2f})")
    elif 'correct_' in market:
        score = market.split('_')[1]
        reason_parts.append(f"🎯 Correct score {score} has probability {probability:.1%} based on Poisson distribution")

    if not reason_parts:
        reason_parts.append("Factors are balanced, but this market still offers value.")

    return " | ".join(reason_parts)

# ------------------------------------------------------------------
# 6. Get best market for a match (for Best Bets)
# ------------------------------------------------------------------
def get_best_market(match_doc):
    if match_doc.get('home_xg') is None:
        predict(match_doc)
        match_doc = db.matches.find_one({'_id': match_doc['_id']})

    h_xg = match_doc.get('home_xg', 1.2)
    a_xg = match_doc.get('away_xg', 1.0)
    confidence = match_doc.get('confidence', 0.5)
    context = match_doc.get('context', {})

    probs = compute_all_market_probs(h_xg, a_xg)

    # Exclude trivial markets
    excluded = ['under_6.5', 'under_7.5', 'under_5.5', 'under_4.5', 'over_0.5']
    # For derby or knockout, we might want to avoid certain markets, but we keep all.

    best_market = None
    best_score = 0
    best_prob = 0

    for market, prob in probs.items():
        if market.startswith('correct_'):
            continue
        if market in excluded:
            continue
        # For derby, we might want to avoid straight win markets if probability < 0.5
        if context.get('is_derby') and market in ['home_win', 'away_win'] and prob < 0.5:
            continue
        # For finals, increase weight on 1X2 and BTTS
        score = prob * confidence
        if score > best_score:
            best_score = score
            best_market = market
            best_prob = prob

    # Fallback: if none, pick highest probability
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
        reason = generate_detailed_reason(match_doc, best_market, best_prob, confidence)
        return {
            'market': best_market,
            'probability': best_prob,
            'confidence': confidence,
            'score': best_score,
            'stake_kelly': kelly_fraction(best_prob, 2.0),
            'reason': reason
        }
    return None

# ------------------------------------------------------------------
# 7. Kelly staking
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
# 8. Get safe markets (for Prediction Detail)
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
# 9. Sure bets: analyse all matches and return the most predictable
# ------------------------------------------------------------------
def get_sure_bets(matches, min_prob=0.8, min_confidence=0.7):
    """
    For a list of matches, find markets with probability >= min_prob and confidence >= min_confidence.
    Return list with detailed reasons.
    """
    sure_list = []
    for match in matches:
        if match.get('home_win_prob') is None:
            predict(match)
            match = db.matches.find_one({'_id': match['_id']})
        confidence = match.get('confidence', 0)
        if confidence < min_confidence:
            continue

        h_xg = match.get('home_xg', 1.2)
        a_xg = match.get('away_xg', 1.0)
        probs = compute_all_market_probs(h_xg, a_xg)
        context = match.get('context', {})

        home = db.teams.find_one({'_id': match['home_team_id']})
        away = db.teams.find_one({'_id': match['away_team_id']})

        # For each market, check if probability >= min_prob and we are not in derby with straight win
        for market, prob in probs.items():
            if market.startswith('correct_'):
                continue
            if prob < min_prob:
                continue
            # Derby: avoid straight win if not strong evidence
            if context.get('is_derby') and market in ['home_win', 'away_win'] and prob < 0.65:
                continue
            # Finals: give more weight to BTTS or 1X2
            # Already handled by probability threshold

            reason = generate_detailed_reason(match, market, prob, confidence)
            sure_list.append({
                'match': f"{home['name'] if home else 'Unknown'} vs {away['name'] if away else 'Unknown'}",
                'tournament': match.get('tournament'),
                'market': market,
                'probability': prob,
                'confidence': confidence,
                'score': prob * confidence,
                'reason': reason,
                'match_id': str(match['_id']),
                'time_remaining': get_time_remaining(match['date'])  # we'll need to import this
            })

    # Sort by score descending
    sure_list.sort(key=lambda x: x['score'], reverse=True)
    return sure_list[:20]  # return top 20

# We need to import get_time_remaining from routes or utils, but we'll define it here for completeness.
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