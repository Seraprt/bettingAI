import re

def map_outcome_to_internal(market_key, outcome_name, sport='football'):
    """
    Convert The Odds API outcome to our internal market name.
    sport: 'football' or 'basketball'
    """
    # Normalise
    name = outcome_name.strip()
    if sport == 'football':
        return _map_football(market_key, name)
    elif sport == 'basketball':
        return _map_basketball(market_key, name)
    return None

def _map_football(market_key, name):
    # h2h
    if market_key == 'h2h':
        if name.lower() == 'home':
            return 'home_win'
        elif name.lower() == 'away':
            return 'away_win'
        elif name.lower() == 'draw':
            return 'draw'
    # totals
    elif market_key == 'totals':
        # e.g., "Over 2.5", "Under 3.5"
        parts = name.split(' ')
        if len(parts) == 2:
            typ, val = parts[0], float(parts[1])
            if typ.lower() == 'over':
                return f'over_{val}'
            elif typ.lower() == 'under':
                return f'under_{val}'
    # spreads (handicaps)
    elif market_key == 'spreads':
        # e.g., "Home -1", "Away +1"
        # We'll parse simple patterns
        if 'home' in name.lower() or 'Home' in name:
            # extract the number
            nums = re.findall(r'[-+]?\d+\.?\d*', name)
            if nums:
                spread = float(nums[0])
                return f'home_{spread}'
        elif 'away' in name.lower() or 'Away' in name:
            nums = re.findall(r'[-+]?\d+\.?\d*', name)
            if nums:
                spread = float(nums[0])
                return f'away_+{spread}' if spread > 0 else f'away_{spread}'
    # alternative: some APIs return "Over 2.5" in h2h? No, but we handle.
    return None

def _map_basketball(market_key, name):
    # Similar but basketball point totals
    if market_key == 'h2h':
        if name.lower() == 'home':
            return 'home_win'
        elif name.lower() == 'away':
            return 'away_win'
    elif market_key == 'totals':
        parts = name.split(' ')
        if len(parts) == 2:
            typ, val = parts[0], float(parts[1])
            if typ.lower() == 'over':
                return f'over_{val}'
            elif typ.lower() == 'under':
                return f'under_{val}'
    elif market_key == 'spreads':
        if 'home' in name.lower() or 'Home' in name:
            nums = re.findall(r'[-+]?\d+\.?\d*', name)
            if nums:
                spread = float(nums[0])
                return f'home_{spread}'
        elif 'away' in name.lower() or 'Away' in name:
            nums = re.findall(r'[-+]?\d+\.?\d*', name)
            if nums:
                spread = float(nums[0])
                return f'away_+{spread}' if spread > 0 else f'away_{spread}'
    return None

def kelly_fraction(prob, odds):
    if odds <= 1:
        return 0
    numerator = prob * (odds - 1) - (1 - prob)
    denominator = odds - 1
    if denominator <= 0:
        return 0
    kelly = numerator / denominator
    return max(0, min(1, kelly))