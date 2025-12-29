# Data Quality

## Anomalies

- Missing odds detected; rows excluded from calculations requiring odds.
- Missing closing_line detected; CLV computed only where available.

## Handling
- Odds parsed to decimal using type detection; unknown odds treated as decimal.
- CLV computed when closing_line is present.
- Stakes coerced to positive for ROI summaries.