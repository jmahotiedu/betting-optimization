# EV Unit Disambiguation

## Schema summary

transactions.csv rows: 1,858
columns: bet_id, sportsbook, type, status, odds, closing_line, ev, amount, profit, time_placed, time_settled, time_placed_iso, time_settled_iso, bet_info, tags, sports, leagues

## Canonical functions used

- `reports/src/odds_utils.py`: `to_decimal`, `american_to_decimal`, `decimal_to_american`, `detect_odds_type`

## Decision framework

See `reports/ev_unit_gate.md` for deterministic gate thresholds and pass/fail outcomes.

## Gate summary

| candidate | M1 pass_count | M2 pass_count | M3 slope | passes_all |
| --- | --- | --- | --- | --- |
| A (Decimal EV) | 3 | 3 | 0.986437 | True |
| B (Percent EV) | 0 | 1 | 98.643702 | False |

## Decision

EV units determined: A (Decimal EV)
