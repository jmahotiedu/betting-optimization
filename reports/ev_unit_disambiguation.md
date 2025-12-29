# EV Unit Disambiguation

## Candidate Interpretations
- A) `ev` is decimal fraction (e.g., 0.0145 = 1.45%)
- B) `ev` is percent (e.g., 1.45 = 1.45%)
- C) `ev` is expected ROI decimal (same numeric scale as A, may include negatives)
- D) Other (no direct evidence in transactions.csv)

## Deterministic Consistency Tests
- Test 1: EV vs CLV Spearman correlation (requires odds + closing_line)
- Test 1b: Median CLV for EV>0 vs EV<=0 (sign consistency)
- Test 2: EV quintile vs realized ROI monotonicity (Spearman on bucket means)
- Test 3: Fair-odds recomputation (not applicable; no fair-odds columns detected)

## Results Table

| interpretation                                    |   clv_spearman | clv_spearman_pass   | clv_median_pass   |   roi_bucket_spearman | roi_bucket_pass   |
|:--------------------------------------------------|---------------:|:--------------------|:------------------|----------------------:|:------------------|
| A) ev as decimal fraction                         |       0.998768 | True                | True              |                     1 | True              |
| B) ev as percent (divide by 100)                  |       0.998768 | True                | True              |                     1 | True              |
| C) ev as expected ROI decimal (same numeric as A) |       0.998768 | True                | True              |                     1 | True              |

## Evidence Snippets
### A) ev as decimal fraction
- EV vs CLV Spearman: 0.9987676044652878
- EV>0 vs EV<=0 median CLV pass: True
- EV bucket vs ROI Spearman: 0.9999999999999999

### B) ev as percent (divide by 100)
- EV vs CLV Spearman: 0.9987676044652878
- EV>0 vs EV<=0 median CLV pass: True
- EV bucket vs ROI Spearman: 0.9999999999999999

### C) ev as expected ROI decimal (same numeric as A)
- EV vs CLV Spearman: 0.9987676044652878
- EV>0 vs EV<=0 median CLV pass: True
- EV bucket vs ROI Spearman: 0.9999999999999999

## Conclusion
- Pass counts (passes/available tests):
  - A) ev as decimal fraction: 3/3
  - B) ev as percent (divide by 100): 3/3
  - C) ev as expected ROI decimal (same numeric as A): 3/3
- Selected interpretation: **Undetermined**
- Determination: Multiple interpretations have equivalent cross-field consistency; EV units remain ambiguous.
- Required data to resolve: explicit EV unit definition from data source, or a column providing fair-odds/model probability to recompute EV.

## Required Final Checks
- Are EV units now unambiguous? No
- Was any EV value guessed or inferred without evidence? No
- Were any defaults silently applied? No
- Would a third party reproduce the same EV interpretation? No