# Outlier Recommendations

## Core Profile
- date_filter: During the week
- leagues: ['NBA', 'NHL', 'NFL', 'NCAAM', 'NCAAFB', 'other']
- bet_types: ['Player Props']
- devig_required_books: ['FanDuel']
- devig_optional_books: ['Pinnacle', 'BookMaker', 'DraftKings', 'Caesars']
- devig_min_books_required: 1
- devig_method: Multiplicative
- devig_weights: {'FanDuel': 0.5, 'Pinnacle': 0.125, 'BookMaker': 0.125, 'DraftKings': 0.125, 'Caesars': 0.125}
- kelly_multiplier: 1/4
- ev_min_pct: 9.09
- kelly_min_pct: 0.0
- vig_max_pct: 8.0
- market_width_max: 40.0
- fair_value_min_american: -118.0
- fair_value_max_american: 120.0
- market_limits: Not supported
- variation_max_pct: 3.0
- stake_cap_pct_bankroll: 0.02
- stake_kelly_fraction: 0.25

## Expansion Profile
- date_filter: This month
- leagues: ['NBA', 'NHL', 'NFL', 'NCAAM', 'NCAAFB', 'other']
- bet_types: ['Gamelines']
- devig_required_books: ['Pinnacle', 'Circa', 'BookMaker']
- devig_optional_books: []
- devig_min_books_required: 2
- devig_method: Multiplicative
- devig_weights: {'Pinnacle': 0.3333333333333333, 'Circa': 0.3333333333333333, 'BookMaker': 0.3333333333333333}
- kelly_multiplier: Full
- ev_min_pct: 7.06
- kelly_min_pct: 0.0
- vig_max_pct: 8.0
- market_width_max: 20.0
- fair_value_min_american: -200.0
- fair_value_max_american: 134.0
- market_limits: Not supported
- variation_max_pct: 3.0
- stake_cap_pct_bankroll: 0.01
- stake_kelly_fraction: 1.0

## Devig Weights
- FanDuel: 0.5000 (research-derived)
- Pinnacle: 0.1250 (research-derived)
- BookMaker: 0.1250 (research-derived)
- DraftKings: 0.1250 (research-derived)
- Caesars: 0.1250 (research-derived)