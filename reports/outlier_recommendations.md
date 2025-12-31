# Outlier Recommendations

## Data Sources
- total_transactions: total_transactions.csv
- today_transactions: transactions(1).csv
- latest_date: 2025-12-30
- ev_unit: decimal

## Preset Core Profile
## Core Settings
- profile: Props Core
- date_filter: Any time
- bet_types: ['Player Props']
- required_books: []
- optional_books: ['DraftKings', 'BookMaker.eu', 'FanDuel', 'Caesars', 'BetOnline.ag', 'Pinnacle', 'BetMGM']
- min_books: 1
- weights: {'DraftKings': 71.21, 'BookMaker.eu': 60.0, 'FanDuel': 67.47, 'Caesars': 57.05, 'BetOnline.ag': 50.0, 'Pinnacle': 100.0, 'BetMGM': 65.69}
- devig_method: Power
- variation_max_pct: 3.0
- vig_max_pct: None
- fair_value_max_american: None
- ev_min_pct: 5.76
- kelly_multiplier: 1/4
- weights_source: priors_blended_with_props_ml
- weights_generated_at: 2025-12-31T11:07:11.244104Z
- ev_min_pct_source: total_transactions_ev_p60_clv_pos_floor_1pct

## Preset Expansion Profile
## Expansion Settings
- profile: Gamelines Expansion
- date_filter: In the next 24 hours
- bet_types: ['Gamelines']
- required_books: []
- optional_books: ['BetOnline.ag', 'BookMaker.eu', 'Circa', 'Pinnacle']
- min_books: 2
- weights: {'BetOnline.ag': 41.99, 'BookMaker.eu': 44.33, 'Circa': 76.5, 'Pinnacle': 100.0}
- devig_method: Probit
- variation_max_pct: 3.0
- vig_max_pct: 8.0
- fair_value_max_american: 200.0
- ev_min_pct: 1.0
- kelly_multiplier: 1/2
- weights_source: odds_api_only
- weights_generated_at: 2025-12-31T11:07:11.244104Z
- ev_min_pct_source: total_transactions_ev_p50_floor_1pct

## Overlay Core Profile (Execution Discipline Only)
## Core Overlay
- profile: core
- leagues: ['Bundesliga', 'NBA', 'NCAAFB', 'NCAAM', 'NFL', 'NHL', 'TENNIS', 'UEFA Champions League', 'UFC', 'other']
- markets: ['3-Pointers', 'Assists', 'Blocks', 'Moneyline', 'Other', 'Point Spread', 'Points', 'Points + Assists', 'Points + Rebounds', 'Points + Rebounds + Assists', 'Rebounds', 'Rebounds + Assists', 'Steals', 'Steals + Blocks', 'Team Total', 'Total', 'Total Assists', 'Total Points', 'Total Receiving Yards', 'Total Receptions', 'Total Rushing Yards', 'Total Saves', 'Total Shots on Goal']
- bet_types: ['Gamelines', 'Player Props']
- sportsbooks: ['Bally Bet', 'Bet365', 'BetMGM', 'BetRivers - Indiana', 'Caesars', 'DraftKings', 'FanDuel', 'Fanatics', 'Hard Rock', 'Kalshi', 'Novig', 'Onyx', 'ProphetX', 'theScore Bet']
- ev_min_pct: 0.0
- min_bets_per_combo: 0
- min_roi_per_combo: 0.0
- min_worst_decile_clv_per_combo: 0.0
- odds_min_decimal: 1.01
- odds_max_decimal: 20.0
- odds_min_american: -10000.0
- odds_max_american: 1900.0
- time_window_days: None
- stake_cap_pct_bankroll: 0.02
- max_drawdown: 0.25
- drawdown_constraint_met: False
- overlay_role: execution_discipline_only

## Overlay Expansion Profile (Execution Discipline Only)
## Expansion Overlay
- profile: expansion
- leagues: ['NBA', 'NHL']
- markets: ['Total Assists', 'Total Saves']
- bet_types: ['Player Props']
- sportsbooks: ['BetMGM', 'ProphetX']
- ev_min_pct: 0.0
- min_bets_per_combo: 5
- min_roi_per_combo: 0.0
- min_worst_decile_clv_per_combo: 0.0
- odds_min_decimal: 1.77
- odds_max_decimal: 2.3
- odds_min_american: -130.0
- odds_max_american: 130.0
- time_window_days: 90
- stake_cap_pct_bankroll: 0.02
- max_drawdown: 0.25
- drawdown_constraint_met: True
- overlay_role: execution_discipline_only

## Notes
- Weights derived from settled bets; props blend sharp priors with props ML.
- Gamelines weights come from outcome-based Odds-API.io snapshots.
- EV thresholds are based on total_transactions percentiles and floored at 1%.