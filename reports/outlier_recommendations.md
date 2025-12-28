# Outlier Recommendations

## Core Profile
- date_filter: last_90_days
- leagues_filter: primary leagues only
- bet_types: ['Gamelines', 'Player Props']
- devig_books: ['DraftKings', 'FanDuel', 'BetMGM', 'PointsBet', 'Caesars']
- devig_method: power
- ev_min_pct: 2.0
- kelly_min_pct: 1.0
- vig_max_pct: 4.0
- market_width_max_pct: 4.0
- fair_value_odds_min: 1.4
- fair_value_odds_max: 5.0
- market_limits_min: 5
- market_limits_max: 500
- variation_max: 3.0
- bet_size_strategy: {'type': 'fractional_kelly', 'kelly_multiplier': 0.25, 'max_pct_bankroll': 0.02}

## Expansion Profile
- date_filter: last_180_days
- leagues_filter: expanded
- bet_types: ['Gamelines', 'Player Props', 'Team Props', 'Game Props']
- devig_books: ['DraftKings', 'FanDuel', 'BetMGM', 'PointsBet', 'Caesars']
- devig_method: power
- ev_min_pct: 1.0
- kelly_min_pct: 0.5
- vig_max_pct: 6.0
- market_width_max_pct: 6.0
- fair_value_odds_min: 1.3
- fair_value_odds_max: 8.0
- market_limits_min: 5
- market_limits_max: 250
- variation_max: 4.0
- bet_size_strategy: {'type': 'fractional_kelly', 'kelly_multiplier': 0.2, 'max_pct_bankroll': 0.025}