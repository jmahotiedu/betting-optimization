# OS Rating Model

## Join Logic
- Matched samples to transactions by league + market + sportsbook.
- Fallback to league + market average, then overall average.

## Coverage
- Transactions with direct combo match: 0
- Transactions with market-level match: 10

## Notes
- Sample file has limited rows; model uses hierarchical averages rather than a high-variance regression.