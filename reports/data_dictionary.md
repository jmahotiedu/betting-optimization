# Data Dictionary

## transactions.csv
Rows: 1,858
Columns: 17

| column | dtype | missing_pct |
| --- | --- | --- |
| bet_id | object | 0.00% |
| sportsbook | object | 0.00% |
| type | object | 0.00% |
| status | object | 0.00% |
| odds | float64 | 0.11% |
| closing_line | float64 | 25.83% |
| ev | float64 | 30.84% |
| amount | float64 | 0.00% |
| profit | float64 | 0.00% |
| time_placed | object | 0.00% |
| time_settled | object | 1.18% |
| time_placed_iso | object | 0.00% |
| time_settled_iso | object | 1.18% |
| bet_info | object | 0.00% |
| tags | float64 | 100.00% |
| sports | object | 0.00% |
| leagues | object | 0.00% |

## os_markets_clean.csv
Rows: 1,549
Columns: 8

| column | dtype | missing_pct |
| --- | --- | --- |
| market | object | 0.00% |
| sportsbook | object | 0.00% |
| league | object | 0.00% |
| wins | float64 | 0.00% |
| losses | float64 | 0.00% |
| profit | float64 | 0.00% |
| roi_pct | float64 | 0.00% |
| est_user_bpd | float64 | 0.00% |

## os_settings.txt
Contains allowable ranges for OddsShopper settings: OS Rating, EV, Odds Range, EV Age, Time to Event Start, and size strategy choices (Flat vs OS Kelly).

## Assumptions
- `odds` and `closing_line` are decimal odds unless detected as American (absolute value >= 100 and not in decimal range).
- `profit` appears net profit (stake excluded).
- `amount` is stake size.