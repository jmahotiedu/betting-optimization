# EV Normalization Proposal

## Proposed Rule (Pending Approval)
- **Rule:** Treat `ev` as a decimal fraction **only when 0.0 ≤ ev ≤ 1.0**. Any numeric `ev` outside that range is **rejected as ambiguous** and would be excluded from downstream calculations. Missing `ev` stays missing.

## Exact Transformation Logic
1. Parse `ev` as numeric where possible.
2. Keep values where `0.0 ≤ ev ≤ 1.0` (decimal fraction interpretation).
3. Mark all numeric values where `ev < 0.0` or `ev > 1.0` as **invalid** and remove those rows from EV-based filtering.
4. Leave missing values as missing.

## Rows Affected (from audit)
- **Total rows:** 1,858
- **Rows kept (0.0 ≤ ev ≤ 1.0):** 869 (851 decimal-fraction + 18 zeros)
- **Rows rejected as ambiguous:** 416 (106 percent-like numeric + 310 out-of-range numeric)
- **Rows missing EV:** 573

## Why This Rule Is Defensible
- No string percent values exist to disambiguate units.
- Treating values outside `[0,1]` as invalid avoids **guessing** whether they represent percent or another scale.
- This rule is strict, auditable, and avoids silent unit inference.

## Required Confirmation
- Explicit confirmation that EV is stored as decimal fractions in `transactions.csv`, or approval to discard all non-[0,1] values as ambiguous.
- Confirmation of whether zero EV values should be retained or excluded.

**Note:** This proposal is not applied until explicit approval is given.
