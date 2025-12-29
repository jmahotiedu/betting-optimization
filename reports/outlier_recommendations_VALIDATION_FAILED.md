# VALIDATION_FAILED

## Integrity Status
- FAILED

## Critical Warnings
- EV unit ambiguity detected. EV values must be explicitly labeled as percent or decimal. Execution halted.
- DO NOT USE: Recommendations are blocked until integrity checks pass.

## OS Sample Summary
- 52 rows loaded; 6 markets; 9 sportsbooks
- Core minimum samples required: 30
- Expansion minimum samples required: 30
- Threshold rationale: minimum OS sample count required to avoid sparse or misleading priors.

## Integrity Checklist

- OS Rating samples availability: **PASS**
  - 52 rows
- OS sample count thresholds: **PASS**
  - 52 rows >= Core 30 and Expansion 30
- EV unit consistency: **FAIL**
  - EV unit ambiguity detected. EV values must be explicitly labeled as percent or decimal. Execution halted.
- OS Rating coverage majority: **FAIL**
  - Not evaluated (blocked by prior failure)
- Sample size sufficiency: **FAIL**
  - Not evaluated (blocked by prior failure)
- Odds bounds binding (>=10% tail mass): **FAIL**
  - Not evaluated (blocked by prior failure)
- Odds conversion consistency: **FAIL**
  - Not evaluated (blocked by prior failure)
- Devig feasibility (two-way markets): **FAIL**
  - Not evaluated (blocked by prior failure)

## Outcome
- Recommendations were not generated due to failed integrity checks.