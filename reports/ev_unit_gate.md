# EV Unit Gate

## Candidate definitions

- Candidate A (decimal EV): `ev_decimal = ev`
- Candidate B (percent EV): `ev_decimal = ev / 100.0`

## Derived quantities

- expected_roi = ev_decimal
- expected_profit = amount * expected_roi

## Decision thresholds (declared before computing)

- M1 plausibility bounds: fraction in [0.001, 0.99] for at least 2 of 3 cutoffs
  - Cutoffs derived from percent thresholds {1.45, 1.05, 2.0} interpreted as 1.45%, 1.05%, 2.0%
- M2 sign test: median realized ROI for EV>=cutoff must be >= median ROI for EV<cutoff for at least 2 of 3 cutoffs
- M3 CLV regression slope must fall in [0.1, 10.0]
  - Justification: EV and CLV are both ROI-like decimals; a slope within an order of magnitude of 1 indicates unit coherence.

## Results

rows_total: 1858

### A (Decimal EV)

#### M1 — EV threshold consistency

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573

| threshold_percent | cutoff | fraction_ev_ge_cutoff | passes |
| --- | --- | --- | --- |
| 1.45 | 0.014500 | 0.688716 | True |
| 1.05 | 0.010500 | 0.698833 | True |
| 2.0 | 0.020000 | 0.663035 | True |

M1 pass_count: 3 -> passes: True

#### M2 — Realized ROI sign test

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573
- missing amount: 0
- missing profit: 0
- amount == 0: 0

| threshold_percent | cutoff | n_ev_ge | n_ev_lt | median_roi_ev_ge | median_roi_ev_lt | note | passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.45 | 0.014500 | 885 | 400 | -1.000000 | -1.000000 | medians_equal | True |
| 1.05 | 0.010500 | 898 | 387 | -1.000000 | -1.000000 | medians_equal | True |
| 2.0 | 0.020000 | 852 | 433 | -0.999924 | -1.000000 |  | True |

M2 pass_count: 3 -> passes: True

#### M3 — CLV regression sanity

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573
- missing odds: 2
- missing closing_line: 480
- invalid odds decimal: 2
- invalid close decimal: 480

regression: clv ~ expected_roi (with intercept), slope=0.986437, intercept=0.000918, R^2=0.990072, n=1285
M3 passes: True

Candidate passes all metrics: True

### B (Percent EV)

#### M1 — EV threshold consistency

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573

| threshold_percent | cutoff | fraction_ev_ge_cutoff | passes |
| --- | --- | --- | --- |
| 1.45 | 1.450000 | 0.000000 | False |
| 1.05 | 1.050000 | 0.000778 | False |
| 2.0 | 2.000000 | 0.000000 | False |

M1 pass_count: 0 -> passes: False

#### M2 — Realized ROI sign test

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573
- missing amount: 0
- missing profit: 0
- amount == 0: 0

| threshold_percent | cutoff | n_ev_ge | n_ev_lt | median_roi_ev_ge | median_roi_ev_lt | note | passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.45 | 1.450000 | 0 | 1285 | nan | -1.000000 |  | False |
| 1.05 | 1.050000 | 1 | 1284 | -1.000000 | -1.000000 | medians_equal | True |
| 2.0 | 2.000000 | 0 | 1285 | nan | -1.000000 |  | False |

M2 pass_count: 1 -> passes: False

#### M3 — CLV regression sanity

rows_total: 1858
rows_used: 1285
rows_excluded: 573
- missing ev: 573
- missing odds: 2
- missing closing_line: 480
- invalid odds decimal: 2
- invalid close decimal: 480

regression: clv ~ expected_roi (with intercept), slope=98.643702, intercept=0.000918, R^2=0.990072, n=1285
M3 passes: False

Candidate passes all metrics: False

## Decision

EV units determined: A (Decimal EV)
