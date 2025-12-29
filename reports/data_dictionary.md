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

## os_samples

- **Event Name**: missing 0.0%
- **Participant Name**: missing 0.0%
- **Offer Name**: missing 0.0%
- **Event Date**: missing 0.0%
- **Sportsbook**: missing 0.0%
- **League**: missing 0.0%
- **Line**: missing 0.0%
- **Side**: missing 0.0%
- **Odds at Placement**: missing 0.0%
- **Current Odds**: missing 0.0%
- **CLV**: missing 0.0%
- **OS Rating**: missing 0.0%

## Odds Format Notes
- Odds appear as decimal in transactions, with possible American values in samples.
- The parser treats values between 0 and 1 as decimal-minus-one and adds 1.

## OS Samples
- OS sample rows are loaded from an uploaded file in / or /workspace/betting-optimization.

## Odds Format Distribution
- odds types: {'decimal': 1828, 'unknown': 20, 'american': 7, 'decimal_minus_one': 1}
- closing_line types: {'decimal': 1365, 'unknown': 12, 'american': 1}