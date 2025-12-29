# EV Audit

## Column Provenance
- EV column used: `ev`
- EV-like columns detected: ev

## Detected EV Formats (counts)
- Missing: 573
- String percent: 0
- Zero: 18
- Decimal fraction (0 < ev <= 0.2): 851
- Percent-like numeric (0.2 < ev <= 100): 106
- Out-of-range numeric (ev < 0 or ev > 100): 310

## Summary Statistics by Format
### String percent
- No rows
### Zero
- min: -0.0
- median: 0.0
- max: -0.0
- example rows:
  - row 103: ev=-0.0000
  - row 108: ev=-0.0000
  - row 138: ev=-0.0000
  - row 195: ev=-0.0000
  - row 370: ev=-0.0000

### Decimal fraction (0 < ev <= 0.2)
- min: 0.0002
- median: 0.0649
- max: 0.2
- example rows:
  - row 0: ev=0.0305
  - row 1: ev=0.1111
  - row 2: ev=0.0742
  - row 3: ev=0.1098
  - row 4: ev=0.0535

### Percent-like numeric (0.2 < ev <= 100)
- min: 0.2057
- median: 0.35975
- max: 1.0823
- example rows:
  - row 140: ev=0.2350
  - row 221: ev=0.2100
  - row 222: ev=0.2100
  - row 251: ev=0.3125
  - row 252: ev=0.2676

### Out-of-range numeric (ev < 0 or ev > 100)
- min: -0.6146
- median: -0.03985
- max: -0.0001
- example rows:
  - row 8: ev=-0.1247
  - row 12: ev=-0.1530
  - row 17: ev=-0.1241
  - row 21: ev=-0.0584
  - row 22: ev=-0.0780

## Notes
- Decimal vs percent-like splits above are heuristic based on magnitude and are for audit only.