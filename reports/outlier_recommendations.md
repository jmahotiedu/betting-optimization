# Outlier Recommendations

## Core Profile
- date_filter (research-derived): During the week
- leagues (data-derived): ['NBA', 'NHL', 'NFL', 'NCAAM', 'NCAAFB', 'other']
- bet_types (data-derived): ['Player Props']
- devig_required_books (research-derived): ['FanDuel']
- devig_optional_books (research-derived): ['Pinnacle', 'BookMaker', 'DraftKings', 'Caesars']
- devig_min_books_required (research-derived): 1
- devig_method (data-derived): Multiplicative
- devig_weights (research-derived): {'FanDuel': 0.5, 'Pinnacle': 0.125, 'BookMaker': 0.125, 'DraftKings': 0.125, 'Caesars': 0.125}
- kelly_multiplier (data-derived): 1/4
- ev_min_pct (data-derived): 9.09
- kelly_min_pct (data-derived): 0.0
- vig_max_pct (research-derived): 8.0
- market_width_max (research-derived): 40.0
- fair_value_min_american (data-derived): -118.0
- fair_value_max_american (data-derived): 120.0
- market_limits (research-derived): Not supported
- variation_max_pct (research-derived): 3.0
- stake_cap_pct_bankroll (data-derived): 0.02
- stake_kelly_fraction (data-derived): 0.25

## Expansion Profile
- date_filter (research-derived): This month
- leagues (data-derived): ['NBA', 'NHL', 'NFL', 'NCAAM', 'NCAAFB', 'other']
- bet_types (data-derived): ['Gamelines']
- devig_required_books (research-derived): ['Pinnacle', 'Circa', 'BookMaker']
- devig_optional_books (research-derived): []
- devig_min_books_required (research-derived): 2
- devig_method (data-derived): Multiplicative
- devig_weights (research-derived): {'Pinnacle': 0.3333333333333333, 'Circa': 0.3333333333333333, 'BookMaker': 0.3333333333333333}
- kelly_multiplier (data-derived): Full
- ev_min_pct (data-derived): 7.06
- kelly_min_pct (data-derived): 0.0
- vig_max_pct (research-derived): 8.0
- market_width_max (research-derived): 20.0
- fair_value_min_american (data-derived): -200.0
- fair_value_max_american (data-derived): 134.0
- market_limits (research-derived): Not supported
- variation_max_pct (research-derived): 3.0
- stake_cap_pct_bankroll (data-derived): 0.01
- stake_kelly_fraction (data-derived): 1.0

## Devig Weights
- FanDuel: 0.5000 (research-derived)
- Pinnacle: 0.1250 (research-derived)
- BookMaker: 0.1250 (research-derived)
- DraftKings: 0.1250 (research-derived)
- Caesars: 0.1250 (research-derived)