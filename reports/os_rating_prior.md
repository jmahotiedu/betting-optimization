# OS Rating Prior

## Sample Type
- Detected source: row-level offers
- Sample rows: 20

## Join Logic
- Matched samples to transactions by league + market + sportsbook.
- Fallback to league + market average, then overall average.

## Coverage
- Transactions with direct combo match: 483
- Transactions with market-level match: 20
- Sample combo coverage: 53.8%

## OS Rating Coverage (non-null os_rating_pred)
- Overall: 100.0%
- By league (lowest 10):
  - Bundesliga: 100.0%
  - NBA: 100.0%
  - NCAAFB: 100.0%
  - NCAAM: 100.0%
  - NFL: 100.0%
  - NHL: 100.0%
  - TENNIS: 100.0%
  - UEFA Champions League: 100.0%
  - UFC: 100.0%
  - other: 100.0%
- By market (lowest 10):
  - 3-Pointers: 100.0%
  - Assists: 100.0%
  - Moneyline: 100.0%
  - Other: 100.0%
  - Point Spread: 100.0%
  - Points: 100.0%
  - Points + Assists: 100.0%
  - Points + Rebounds: 100.0%
  - Points + Rebounds + Assists: 100.0%
  - Rebounds: 100.0%
- By sportsbook (lowest 10):
  - Bally Bet: 100.0%
  - Bet365: 100.0%
  - BetMGM: 100.0%
  - BetRivers - Indiana: 100.0%
  - Caesars Sportsbook: 100.0%
  - Draftkings Sportsbook: 100.0%
  - Fanatics: 100.0%
  - Fanduel Sportsbook: 100.0%
  - Hard Rock Sportsbook: 100.0%
  - Kalshi: 100.0%

## Unmatched Sample Combos (first 10)
- NBA | 3-Pointers | BetRivers
- NBA | Rebounds + Assists | BetMGM
- NBA | Assists | Circa
- NBA | 3-Pointers | BetMGM
- NBA | Points + Rebounds + Assists | Hard Rock Sportsbook
- NBA | Rebounds + Assists | Hard Rock Sportsbook

## Notes
- Sample file has limited rows; model uses hierarchical averages rather than a high-variance regression.