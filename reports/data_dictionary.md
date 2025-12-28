# Data Dictionary

## transactions.csv

- **bet_id**: missing 0.0%
- **sportsbook**: missing 0.0%
- **type**: missing 0.0%
- **status**: missing 0.0%
- **odds**: missing 0.1%
- **closing_line**: missing 25.8%
- **ev**: missing 30.8%
- **amount**: missing 0.0%
- **profit**: missing 0.0%
- **time_placed**: missing 0.0%
- **time_settled**: missing 1.2%
- **time_placed_iso**: missing 0.0%
- **time_settled_iso**: missing 1.2%
- **bet_info**: missing 0.0%
- **tags**: missing 100.0%
- **sports**: missing 0.0%
- **leagues**: missing 0.0%

## os_markets_clean.csv

- **market**: missing 0.0%
- **sportsbook**: missing 0.0%
- **league**: missing 0.0%
- **wins**: missing 0.0%
- **losses**: missing 0.0%
- **profit**: missing 0.0%
- **roi_pct**: missing 0.0%
- **est_user_bpd**: missing 0.0%

## Odds Format Notes
- Odds appear as decimal (e.g., 2.27) with occasional American-style values possible.
- The parser treats values between 0 and 1 as decimal-minus-one and adds 1.