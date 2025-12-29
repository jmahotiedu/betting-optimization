import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_DIR))

from reports.src.odds_utils import to_decimal  # noqa: E402
REPORTS_DIR = REPO_DIR / "reports"
TRANSACTIONS_PATH = REPO_DIR / "transactions.csv"

EV_RELATED_COLS = [
    "ev",
    "odds",
    "closing_line",
    "closing_odds",
    "amount",
    "profit",
    "clv",
    "bet_info",
    "sportsbook",
    "time_placed_iso",
    "time_placed",
    "time_settled",
    "type",
    "status",
    "sports",
    "leagues",
]

THRESHOLDS_PERCENT = [1.45, 1.05, 2.0]
M1_PLAUSIBLE_MIN = 0.001
M1_PLAUSIBLE_MAX = 0.99
M1_REQUIRED_PASS_COUNT = 2

M2_REQUIRED_PASS_COUNT = 2

M3_SLOPE_MIN = 0.1
M3_SLOPE_MAX = 10.0


def _clean_value(value: object, limit: int = 80) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _write_schema_probe(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    sample_cols = [c for c in EV_RELATED_COLS if c in df.columns]
    sample = df[sample_cols].head(25).copy()
    sample = sample.apply(lambda col: col.map(_clean_value))

    lines: List[str] = []
    lines.append("# EV Schema Probe")
    lines.append("")
    lines.append("## Columns in transactions.csv")
    lines.append("")
    lines.append(", ".join(columns))
    lines.append("")
    lines.append("## 25-row sample of EV-related fields")
    lines.append("")

    if sample_cols:
        header = "| " + " | ".join(sample_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(sample_cols)) + " |"
        lines.append(header)
        lines.append(separator)
        for _, row in sample.iterrows():
            line = "| " + " | ".join(row[c] for c in sample_cols) + " |"
            lines.append(line)
    else:
        lines.append("_No EV-related fields detected in transactions.csv._")

    (REPORTS_DIR / "ev_schema_probe.md").write_text("\n".join(lines))


def _regression_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan, math.nan, math.nan, len(x)
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else math.nan
    return slope, intercept, r2, len(x)


def _rows_exclusion_summary(
    total: int, used_mask: pd.Series, reasons: Dict[str, pd.Series]
) -> Tuple[int, int, List[str]]:
    used = int(used_mask.sum())
    excluded = total - used
    excluded_mask = ~used_mask
    reason_lines = []
    reason_union = pd.Series(False, index=used_mask.index)
    for label, mask in reasons.items():
        overlap = mask & excluded_mask
        count = int(overlap.sum())
        reason_union = reason_union | overlap
        reason_lines.append(f"- {label}: {count}")
    unexplained = int((excluded_mask & ~reason_union).sum())
    if unexplained:
        reason_lines.append(f"- unexplained_excluded: {unexplained}")
    return used, excluded, reason_lines


def _compute_gate(df: pd.DataFrame) -> Tuple[str, Dict[str, object]]:
    rows_total = len(df)
    working = df.copy()
    working["ev"] = pd.to_numeric(working.get("ev"), errors="coerce")
    working["amount"] = pd.to_numeric(working.get("amount"), errors="coerce")
    working["profit"] = pd.to_numeric(working.get("profit"), errors="coerce")

    if "odds" in working.columns:
        working["odds_decimal"] = working["odds"].apply(to_decimal)
    close_col = "closing_line" if "closing_line" in working.columns else (
        "closing_odds" if "closing_odds" in working.columns else None
    )
    if close_col:
        working["close_decimal"] = working[close_col].apply(to_decimal)

    candidates = {
        "A (Decimal EV)": working["ev"],
        "B (Percent EV)": working["ev"] / 100.0,
    }

    gate_results: Dict[str, object] = {
        "rows_total": rows_total,
        "thresholds_percent": THRESHOLDS_PERCENT,
        "m1_bounds": (M1_PLAUSIBLE_MIN, M1_PLAUSIBLE_MAX, M1_REQUIRED_PASS_COUNT),
        "m2_required": M2_REQUIRED_PASS_COUNT,
        "m3_bounds": (M3_SLOPE_MIN, M3_SLOPE_MAX),
        "candidates": {},
        "close_col": close_col,
    }

    for name, ev_series in candidates.items():
        candidate = {}

        # Metric M1
        m1_rows_used_mask = working["ev"].notna()
        m1_reasons = {"missing ev": working["ev"].isna()}
        m1_used, m1_excluded, m1_reason_lines = _rows_exclusion_summary(
            rows_total, m1_rows_used_mask, m1_reasons
        )

        m1_fractions = {}
        m1_pass_flags = {}
        pass_count = 0
        for threshold in THRESHOLDS_PERCENT:
            cutoff = threshold / 100.0 if name.startswith("A") else threshold
            values = working.loc[m1_rows_used_mask, "ev"]
            fraction = float((values >= cutoff).mean()) if len(values) else math.nan
            m1_fractions[threshold] = {"cutoff": cutoff, "fraction": fraction}
            plausible = M1_PLAUSIBLE_MIN <= fraction <= M1_PLAUSIBLE_MAX
            m1_pass_flags[threshold] = plausible
            if plausible:
                pass_count += 1
        m1_pass = pass_count >= M1_REQUIRED_PASS_COUNT

        candidate["m1"] = {
            "rows_used": m1_used,
            "rows_excluded": m1_excluded,
            "exclusion_reasons": m1_reason_lines,
            "fractions": m1_fractions,
            "passes": m1_pass_flags,
            "pass_count": pass_count,
            "passes_metric": m1_pass,
        }

        # Metric M2
        m2_mask = (
            working["ev"].notna()
            & working["amount"].notna()
            & working["profit"].notna()
            & (working["amount"] != 0)
        )
        m2_reasons = {
            "missing ev": working["ev"].isna(),
            "missing amount": working["amount"].isna(),
            "missing profit": working["profit"].isna(),
            "amount == 0": working["amount"].fillna(0) == 0,
        }
        m2_used, m2_excluded, m2_reason_lines = _rows_exclusion_summary(
            rows_total, m2_mask, m2_reasons
        )
        m2_results = {}
        m2_pass_count = 0
        for threshold in THRESHOLDS_PERCENT:
            cutoff = threshold / 100.0 if name.startswith("A") else threshold
            subset = working.loc[m2_mask, ["ev", "amount", "profit"]].copy()
            subset["roi"] = subset["profit"] / subset["amount"]
            high = subset[subset["ev"] >= cutoff]["roi"]
            low = subset[subset["ev"] < cutoff]["roi"]
            median_high = float(high.median()) if len(high) else math.nan
            median_low = float(low.median()) if len(low) else math.nan
            passes = bool(median_high >= median_low) if len(high) and len(low) else False
            medians_equal = bool(median_high == median_low) if len(high) and len(low) else False
            note = "medians_equal" if medians_equal else ""
            if passes:
                m2_pass_count += 1
            m2_results[threshold] = {
                "cutoff": cutoff,
                "median_roi_high": median_high,
                "median_roi_low": median_low,
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "note": note,
                "passes": passes,
            }
        m2_pass = m2_pass_count >= M2_REQUIRED_PASS_COUNT

        candidate["m2"] = {
            "rows_used": m2_used,
            "rows_excluded": m2_excluded,
            "exclusion_reasons": m2_reason_lines,
            "results": m2_results,
            "pass_count": m2_pass_count,
            "passes_metric": m2_pass,
        }

        # Metric M3
        if close_col and "odds" in working.columns:
            ev_valid = working["ev"].notna()
            odds_raw = working["odds"]
            close_raw = working[close_col]
            odds_decimal = working["odds_decimal"]
            close_decimal = working["close_decimal"]
            m3_mask = (
                ev_valid
                & odds_raw.notna()
                & close_raw.notna()
                & odds_decimal.notna()
                & close_decimal.notna()
                & (odds_decimal > 1)
                & (close_decimal > 1)
            )
            m3_reasons = {
                "missing ev": working["ev"].isna(),
                "missing odds": odds_raw.isna(),
                f"missing {close_col}": close_raw.isna(),
                "invalid odds decimal": odds_decimal.isna() | (odds_decimal <= 1),
                "invalid close decimal": close_decimal.isna() | (close_decimal <= 1),
            }
            m3_used, m3_excluded, m3_reason_lines = _rows_exclusion_summary(
                rows_total, m3_mask, m3_reasons
            )

            subset = working.loc[m3_mask, ["odds_decimal", "close_decimal"]].copy()
            subset["clv"] = (subset["odds_decimal"] / subset["close_decimal"]) - 1
            expected_roi = ev_series.loc[m3_mask]
            slope, intercept, r2, n = _regression_with_intercept(
                expected_roi.to_numpy(), subset["clv"].to_numpy()
            )
            m3_pass = bool(M3_SLOPE_MIN <= slope <= M3_SLOPE_MAX) if n >= 2 else False
            candidate["m3"] = {
                "rows_used": m3_used,
                "rows_excluded": m3_excluded,
                "exclusion_reasons": m3_reason_lines,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "n": n,
                "passes_metric": m3_pass,
            }
        else:
            candidate["m3"] = {
                "rows_used": 0,
                "rows_excluded": rows_total,
                "exclusion_reasons": [
                    "missing odds or closing_line/closing_odds columns",
                ],
                "slope": math.nan,
                "intercept": math.nan,
                "r2": math.nan,
                "n": 0,
                "passes_metric": False,
            }

        candidate["passes_all"] = (
            candidate["m1"]["passes_metric"]
            and candidate["m2"]["passes_metric"]
            and candidate["m3"]["passes_metric"]
        )

        gate_results["candidates"][name] = candidate

    valid_candidates = [name for name, data in gate_results["candidates"].items() if data["passes_all"]]
    if len(valid_candidates) == 1:
        decision = f"EV units determined: {valid_candidates[0]}"
        decision_key = "determined"
    else:
        decision = "EV units undetermined"
        decision_key = "undetermined"

    gate_results["decision"] = decision
    gate_results["decision_key"] = decision_key
    gate_results["valid_candidates"] = valid_candidates
    return decision_key, gate_results


def _write_gate_report(gate: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# EV Unit Gate")
    lines.append("")
    lines.append("## Candidate definitions")
    lines.append("")
    lines.append("- Candidate A (decimal EV): `ev_decimal = ev`")
    lines.append("- Candidate B (percent EV): `ev_decimal = ev / 100.0`")
    lines.append("")
    lines.append("## Derived quantities")
    lines.append("")
    lines.append("- expected_roi = ev_decimal")
    lines.append("- expected_profit = amount * expected_roi")
    lines.append("")
    lines.append("## Decision thresholds (declared before computing)")
    lines.append("")
    lines.append(
        f"- M1 plausibility bounds: fraction in [{M1_PLAUSIBLE_MIN}, {M1_PLAUSIBLE_MAX}] for at least {M1_REQUIRED_PASS_COUNT} of 3 cutoffs"
    )
    lines.append(
        "  - Cutoffs derived from percent thresholds {1.45, 1.05, 2.0} interpreted as 1.45%, 1.05%, 2.0%"
    )
    lines.append(
        f"- M2 sign test: median realized ROI for EV>=cutoff must be >= median ROI for EV<cutoff for at least {M2_REQUIRED_PASS_COUNT} of 3 cutoffs"
    )
    lines.append(
        f"- M3 CLV regression slope must fall in [{M3_SLOPE_MIN}, {M3_SLOPE_MAX}]"
    )
    lines.append(
        "  - Justification: EV and CLV are both ROI-like decimals; a slope within an order of magnitude of 1 indicates unit coherence."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(f"rows_total: {gate['rows_total']}")
    lines.append("")

    thresholds = gate["thresholds_percent"]

    for name, candidate in gate["candidates"].items():
        lines.append(f"### {name}")
        lines.append("")

        # M1
        m1 = candidate["m1"]
        lines.append("#### M1 — EV threshold consistency")
        lines.append("")
        lines.append(f"rows_total: {gate['rows_total']}")
        lines.append(f"rows_used: {m1['rows_used']}")
        lines.append(f"rows_excluded: {m1['rows_excluded']}")
        lines.extend(m1["exclusion_reasons"])
        lines.append("")
        lines.append("| threshold_percent | cutoff | fraction_ev_ge_cutoff | passes |")
        lines.append("| --- | --- | --- | --- |")
        for threshold in thresholds:
            entry = m1["fractions"][threshold]
            fraction = entry["fraction"]
            lines.append(
                f"| {threshold} | {entry['cutoff']:.6f} | {fraction:.6f} | {m1['passes'][threshold]} |"
            )
        lines.append("")
        lines.append(f"M1 pass_count: {m1['pass_count']} -> passes: {m1['passes_metric']}")
        lines.append("")

        # M2
        m2 = candidate["m2"]
        lines.append("#### M2 — Realized ROI sign test")
        lines.append("")
        lines.append(f"rows_total: {gate['rows_total']}")
        lines.append(f"rows_used: {m2['rows_used']}")
        lines.append(f"rows_excluded: {m2['rows_excluded']}")
        lines.extend(m2["exclusion_reasons"])
        lines.append("")
        lines.append("| threshold_percent | cutoff | n_ev_ge | n_ev_lt | median_roi_ev_ge | median_roi_ev_lt | note | passes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for threshold in thresholds:
            entry = m2["results"][threshold]
            lines.append(
                f"| {threshold} | {entry['cutoff']:.6f} | {entry['n_high']} | {entry['n_low']} | {entry['median_roi_high']:.6f} | {entry['median_roi_low']:.6f} | {entry['note']} | {entry['passes']} |"
            )
        lines.append("")
        lines.append(f"M2 pass_count: {m2['pass_count']} -> passes: {m2['passes_metric']}")
        lines.append("")

        # M3
        m3 = candidate["m3"]
        lines.append("#### M3 — CLV regression sanity")
        lines.append("")
        lines.append(f"rows_total: {gate['rows_total']}")
        lines.append(f"rows_used: {m3['rows_used']}")
        lines.append(f"rows_excluded: {m3['rows_excluded']}")
        lines.extend(m3["exclusion_reasons"])
        lines.append("")
        lines.append(
            f"regression: clv ~ expected_roi (with intercept), slope={m3['slope']:.6f}, intercept={m3['intercept']:.6f}, R^2={m3['r2']:.6f}, n={m3['n']}"
        )
        lines.append(f"M3 passes: {m3['passes_metric']}")
        lines.append("")

        lines.append(f"Candidate passes all metrics: {candidate['passes_all']}")
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(gate["decision"])
    lines.append("")

    (REPORTS_DIR / "ev_unit_gate.md").write_text("\n".join(lines))


def _write_disambiguation_report(df: pd.DataFrame, gate: Dict[str, object]) -> None:
    columns = list(df.columns)
    lines: List[str] = []
    lines.append("# EV Unit Disambiguation")
    lines.append("")
    lines.append("## Schema summary")
    lines.append("")
    lines.append(f"transactions.csv rows: {len(df):,}")
    lines.append(f"columns: {', '.join(columns)}")
    lines.append("")
    lines.append("## Canonical functions used")
    lines.append("")
    lines.append("- `reports/src/odds_utils.py`: `to_decimal`, `american_to_decimal`, `decimal_to_american`, `detect_odds_type`")
    lines.append("")
    lines.append("## Decision framework")
    lines.append("")
    lines.append("See `reports/ev_unit_gate.md` for deterministic gate thresholds and pass/fail outcomes.")
    lines.append("")
    lines.append("## Gate summary")
    lines.append("")
    lines.append("| candidate | M1 pass_count | M2 pass_count | M3 slope | passes_all |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, candidate in gate["candidates"].items():
        m1_pass_count = candidate["m1"]["pass_count"]
        m2_pass_count = candidate["m2"]["pass_count"]
        m3_slope = candidate["m3"]["slope"]
        lines.append(
            f"| {name} | {m1_pass_count} | {m2_pass_count} | {m3_slope:.6f} | {candidate['passes_all']} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(gate["decision"])
    lines.append("")

    (REPORTS_DIR / "ev_unit_disambiguation.md").write_text("\n".join(lines))


def _write_validation_failed(gate: Dict[str, object]) -> None:
    path = REPORTS_DIR / "VALIDATION_FAILED.md"
    if gate["decision_key"] == "determined":
        if path.exists():
            path.unlink()
        return

    lines = [
        "# VALIDATION_FAILED",
        "",
        "EV units could not be uniquely determined under the strict EV unit gate.",
        "",
        "---",
        "User confirmation (paste one):",
        "",
        "- \"EV is decimal\": `ev = 0.0145 means 1.45%`",
        "or",
        "- \"EV is percent\": `ev = 1.45 means 1.45%`",
        "",
        "Additional notes / link to exporter documentation:",
        "[PASTE HERE]",
        "---",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    df = pd.read_csv(TRANSACTIONS_PATH)
    _write_schema_probe(df)
    decision_key, gate = _compute_gate(df)
    _write_gate_report(gate)
    _write_disambiguation_report(df, gate)
    _write_validation_failed(gate)


if __name__ == "__main__":
    main()
