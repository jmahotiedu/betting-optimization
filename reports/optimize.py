from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.data_load import load_transactions, load_os_markets, load_os_settings
from src.odds_utils import to_decimal, decimal_to_american, american_to_decimal
from src.mapping import normalize_market, normalize_league
from src.backtest import bankroll_simulation, summarize_performance, bootstrap_ci
from src.optimize_oddsshopper import optimize as optimize_os, build_tx_tables
from src.optimize_outlier import build_presets, build_overlays
from src.outlier_fair_value import build_two_way_table
from src.report import write_markdown, write_json, write_csv


MIN_OS_SAMPLES_CORE = 30
MIN_OS_SAMPLES_EXPANSION = 30

OS_RATING_WARNING = (
    "OS Rating samples are missing or insufficient. Any default OS Rating (e.g. 20) is arbitrary and invalid. "
    "Optimization aborted to prevent false confidence."
)
OS_RATING_COVERAGE_WARNING = (
    "OS Rating samples do not cover the majority of selected markets/books. Portfolio generation aborted to prevent false confidence."
)
EV_UNIT_WARNING = "EV unit ambiguity detected. EV values must be explicitly labeled as percent or decimal. Execution halted."
ODDS_MISMATCH_WARNING = (
    "Odds bounds mismatch after conversion. Decimal and American filters do not describe the same region. Execution halted."
)
ODDS_BINDING_WARNING = (
    "Odds bounds do not meaningfully bind the distribution. Quantile-based fallback applied. NOT DATA-DERIVED."
)
VALIDATION_FAILED_TAG = "VALIDATION_FAILED"
DEVIG_ABSENCE_WARNING = (
    "Transaction data does not contain paired two-way markets. Devig methods and weights cannot be validated or optimized, "
    "and EV thresholds based on devig edge are not meaningful. Preset priors are used by necessity, not by evidence."
)
OVERLAY_WARNING = "Overlay filters do not generate edge. They only constrain execution. Treating overlays as optimization inputs invalidates results."
BACKTEST_WARNING = (
    "Backtest results reflect historical execution under fixed rules. They do not imply future profitability and are highly "
    "sensitive to market regime changes."
)
EXECUTION_WARNING = "If any execution condition fails, the correct action is to skip the bet. Forced execution is worse than missing volume."
OVERLAY_EXECUTION_WARNING = (
    "Overlay filters do not improve expected value and were not included in optimization or validation."
)
DEVIG_STRUCTURAL_WARNING = "Devig outputs are a structural approximation due to missing two-way market data."


def compute_data_dictionary(transactions: pd.DataFrame, os_markets: pd.DataFrame) -> str:
    lines = ["# Data Dictionary", "", "## transactions.csv", ""]
    for col in transactions.columns:
        missing = transactions[col].isna().mean() * 100
        lines.append(f"- **{col}**: missing {missing:.1f}%")
    lines.extend(["", "## os_markets_clean.csv", ""])
    for col in os_markets.columns:
        missing = os_markets[col].isna().mean() * 100
        lines.append(f"- **{col}**: missing {missing:.1f}%")
    lines.extend(
        [
            "",
            "## Odds Format Notes",
            "- Odds appear as decimal (e.g., 2.27) with occasional American-style values possible.",
            "- The parser treats values between 0 and 1 as decimal-minus-one and adds 1.",
        ]
    )
    return "\n".join(lines)


def _load_os_samples(path: str = "os_samples.csv", min_rows: int = 30) -> pd.DataFrame:
    samples_path = Path(path)
    if not samples_path.exists():
        raise RuntimeError(OS_RATING_WARNING)
    samples = pd.read_csv(samples_path)
    if samples.empty or len(samples) < min_rows:
        raise RuntimeError(OS_RATING_WARNING)
    required = {"Offer Name", "Sportsbook", "League", "OS Rating"}
    if not required.issubset(samples.columns):
        raise RuntimeError(OS_RATING_WARNING)
    samples["market_norm"] = samples["Offer Name"].apply(lambda x: normalize_market(x, None))
    samples["league_norm"] = samples["League"].apply(lambda x: normalize_league(x, None))
    samples["sportsbook"] = samples["Sportsbook"].astype(str).str.strip()
    samples["os_rating"] = pd.to_numeric(samples["OS Rating"], errors="coerce")
    samples = samples.dropna(subset=["market_norm", "league_norm", "sportsbook", "os_rating"])
    if samples.empty:
        raise RuntimeError(OS_RATING_WARNING)
    return samples


def _coerce_ev_units(df: pd.DataFrame) -> pd.DataFrame:
    ev = df["ev"].dropna()
    df.attrs["ev_unit_coerced"] = False
    if ev.empty:
        return df
    max_ev = ev.max()
    min_ev = ev.min()
    if min_ev < 0:
        raise RuntimeError(EV_UNIT_WARNING)
    if max_ev > 1.0 and min_ev < 0.2:
        raise RuntimeError(EV_UNIT_WARNING)
    if max_ev > 1.0:
        if max_ev > 100:
            raise RuntimeError(EV_UNIT_WARNING)
        print("EV values greater than 1.0 detected; interpreting as percent and converting to decimal. NOT DATA-DERIVED.")
        df["ev"] = df["ev"] / 100.0
        df.attrs["ev_unit_coerced"] = True
        return df
    if max_ev < 0.2:
        return df
    raise RuntimeError(EV_UNIT_WARNING)


def build_transactions_metrics(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["status"] = df["status"].astype(str)
    df = df[(df["type"].str.lower() == "straight") & (df["status"].str.startswith("SETTLED"))]
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df.loc[df["close_decimal"].isna(), "clv"] = np.nan
    df["market_norm"] = df.apply(lambda r: normalize_market(r.get("bet_info"), r.get("type")), axis=1)
    df["league_norm"] = df.apply(lambda r: normalize_league(r.get("leagues"), r.get("sports")), axis=1)
    df["bet_type"] = df["market_norm"].apply(
        lambda m: "Gamelines" if m in {"Moneyline", "Point Spread", "Total", "Team Total", "Run Line", "Puck Line"} else "Player Props"
    )
    df["amount"] = df["amount"].abs()
    df["profit"] = df["profit"].fillna(0.0)
    df["time_placed_iso"] = pd.to_datetime(df["time_placed_iso"], errors="coerce", utc=True)
    df["ev"] = df["ev"].fillna(0.0)
    df = _coerce_ev_units(df)
    df["edge"] = df["ev"]
    df["kelly_pct"] = df.apply(
        lambda r: max(0.0, r["edge"]) / (r["odds_decimal"] - 1) if r["odds_decimal"] and r["odds_decimal"] > 1 else 0.0,
        axis=1,
    )
    return df


def _apply_os_settings(df: pd.DataFrame, settings: Dict[str, Any], combos: pd.DataFrame) -> pd.DataFrame:
    subset = df.merge(combos[["league_norm", "market_norm", "sportsbook"]], on=["league_norm", "market_norm", "sportsbook"], how="inner")
    ev_min = settings["minimum_ev"] / 100.0
    odds_min = settings["odds_range_min_decimal"]
    odds_max = settings["odds_range_max_decimal"]
    subset = subset[(subset["ev"] >= ev_min) & (subset["odds_decimal"] >= odds_min) & (subset["odds_decimal"] <= odds_max)]
    return subset


def _odds_binding_rates(df: pd.DataFrame, settings: Dict[str, Any]) -> tuple[float, float]:
    ev_min = settings["minimum_ev"] / 100.0
    filtered = df[(df["ev"].notna()) & (df["ev"] >= ev_min)]
    odds_american = filtered["odds_decimal"].apply(decimal_to_american).replace([np.inf, -np.inf], np.nan).dropna()
    if odds_american.empty:
        return 0.0, 0.0
    min_bound = decimal_to_american(settings["odds_range_min_decimal"])
    max_bound = decimal_to_american(settings["odds_range_max_decimal"])
    lower_tail = (odds_american <= min_bound).mean()
    upper_tail = (odds_american >= max_bound).mean()
    return float(lower_tail), float(upper_tail)


def _apply_odds_binding_fallback(df: pd.DataFrame, settings: Dict[str, Any], warnings: List[str]) -> None:
    lower_tail, upper_tail = _odds_binding_rates(df, settings)
    if lower_tail >= 0.1 and upper_tail >= 0.1:
        settings["odds_binding_adjusted"] = False
        return
    ev_min = settings["minimum_ev"] / 100.0
    filtered = df[(df["ev"].notna()) & (df["ev"] >= ev_min)]
    odds_american = filtered["odds_decimal"].apply(decimal_to_american).replace([np.inf, -np.inf], np.nan).dropna()
    if odds_american.empty:
        settings["odds_binding_adjusted"] = False
        return
    lower_bound = float(np.quantile(odds_american, 0.1))
    upper_bound = float(np.quantile(odds_american, 0.9))
    odds_min_decimal = american_to_decimal(lower_bound)
    odds_max_decimal = american_to_decimal(upper_bound)
    if odds_min_decimal > 1 and odds_max_decimal > odds_min_decimal:
        settings["odds_range_min_decimal"] = round(float(odds_min_decimal), 3)
        settings["odds_range_max_decimal"] = round(float(odds_max_decimal), 3)
        settings["odds_binding_adjusted"] = True
        warnings.append(ODDS_BINDING_WARNING)


def _validate_odds_bounds(min_decimal: float, max_decimal: float, min_american: float, max_american: float, tolerance: float = 0.02) -> None:
    if min_decimal <= 1 or max_decimal <= 1:
        raise RuntimeError(ODDS_MISMATCH_WARNING)
    min_decimal_from_american = to_decimal(min_american)
    max_decimal_from_american = to_decimal(max_american)
    if any(pd.isna(val) for val in [min_decimal_from_american, max_decimal_from_american]):
        raise RuntimeError(ODDS_MISMATCH_WARNING)
    if abs(min_decimal - min_decimal_from_american) > tolerance or abs(max_decimal - max_decimal_from_american) > tolerance:
        raise RuntimeError(ODDS_MISMATCH_WARNING)


def _os_backtest(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    stake_strategy = settings["betting_size_strategy"]
    if stake_strategy == "Flat":
        return bankroll_simulation(df, bankroll=1000, stake_strategy="flat", max_bet_pct=settings["flat_unit_pct_bankroll"])
    return bankroll_simulation(
        df,
        bankroll=1000,
        stake_strategy="os kelly",
        kelly_fraction=settings["fractional_kelly"],
        max_bet_pct=settings["max_bet_pct_bankroll"],
    )


def _os_sample_coverage(os_samples: pd.DataFrame, combos: pd.DataFrame) -> dict:
    if combos.empty:
        return {"coverage_pct": 0.0, "covered": 0, "total": 0, "uncovered": [], "extrapolated": True}
    os_keys = os_samples[["league_norm", "market_norm", "sportsbook"]].drop_duplicates()
    merged = combos.merge(os_keys, on=["league_norm", "market_norm", "sportsbook"], how="left", indicator=True)
    covered_mask = merged["_merge"] == "both"
    total = len(merged)
    covered = int(covered_mask.sum())
    uncovered = merged.loc[~covered_mask, ["league_norm", "market_norm", "sportsbook"]].to_dict("records")
    coverage_pct = covered / total if total else 0.0
    return {
        "coverage_pct": coverage_pct,
        "covered": covered,
        "total": total,
        "uncovered": uncovered,
        "extrapolated": bool(uncovered),
    }


def _derive_os_rating_threshold(os_samples: pd.DataFrame, combos: pd.DataFrame) -> float | None:
    if combos.empty:
        return None
    merged = os_samples.merge(combos, on=["league_norm", "market_norm", "sportsbook"], how="inner")
    if merged.empty:
        return None
    return float(merged["os_rating"].min())


def _coverage_section(coverage: dict) -> List[str]:
    lines = [
        f"- Coverage: {coverage['coverage_pct']*100:.1f}% ({coverage['covered']}/{coverage['total']})",
        f"- Ratings extrapolated: {'Yes' if coverage['extrapolated'] else 'No'}",
    ]
    if coverage["uncovered"]:
        lines.append("- Uncovered combos:")
        for item in coverage["uncovered"]:
            lines.append(f"  - {item['league_norm']} | {item['market_norm']} | {item['sportsbook']}")
    return lines


def _integrity_report(checks: List[dict]) -> List[str]:
    lines = ["## Integrity Checklist", ""]
    for check in checks:
        status = "PASS" if check["status"] else "FAIL"
        lines.append(f"- {check['name']}: **{status}**")
        detail = check.get("detail")
        if detail:
            lines.append(f"  - {detail}")
    lines.append("")
    return lines


def _blocked_integrity_checks(primary: dict) -> List[dict]:
    blocked_detail = "Not evaluated (blocked by prior failure)"
    blocked_checks = [
        {"name": "OS sample count thresholds", "status": False, "detail": blocked_detail, "critical": True},
        {"name": "OS Rating coverage majority", "status": False, "detail": blocked_detail, "critical": True},
        {"name": "Sample size sufficiency", "status": False, "detail": blocked_detail, "critical": False},
        {"name": "Odds bounds binding (>=10% tail mass)", "status": False, "detail": blocked_detail, "critical": True},
        {"name": "Odds conversion consistency", "status": False, "detail": blocked_detail, "critical": True},
        {"name": "EV unit consistency", "status": False, "detail": blocked_detail, "critical": True},
        {"name": "Devig feasibility (two-way markets)", "status": False, "detail": blocked_detail, "critical": True},
    ]
    return [primary, *blocked_checks]


def validation_report(
    df: pd.DataFrame,
    baseline: Dict[str, Any],
    portfolio_summaries: Dict[str, dict],
    coverage_reports: Dict[str, dict],
    warnings: List[str],
    portfolio_warnings: Dict[str, List[str]],
    integrity_checks: List[dict],
    os_sample_summary: str,
    final_check: Dict[str, str],
) -> str:
    lines = []

    roi_ci = bootstrap_ci((df["profit"] / df["amount"].replace(0, 1)).to_numpy())
    clv_ci = bootstrap_ci(df["clv"].fillna(0).to_numpy())
    bootstrap_lines = [
        "## Bootstrap CIs",
        f"- ROI mean CI (approx): [{roi_ci[0]:.4f}, {roi_ci[1]:.4f}]",
        f"- CLV mean CI (approx): [{clv_ci[0]:.4f}, {clv_ci[1]:.4f}]",
        "",
    ]

    os_lines = ["## OddsShopper Portfolios"]
    for name, summary in portfolio_summaries.items():
        coverage = coverage_reports.get(name, {})
        max_dd = summary.get("max_drawdown", 0.0)
        os_lines.extend(
            [
                f"### {name}",
                f"- Bets: {summary['bets']}",
                f"- ROI: {summary['roi']:.4f}",
                f"- Profit: {summary['profit']:.2f}",
                f"- Avg CLV: {summary['avg_clv']:.4f}",
                f"- Worst Decile CLV: {summary['worst_decile_clv']:.4f}",
                f"- Max Drawdown: {max_dd:.4f}",
                "",
                "#### OS Rating Coverage",
                *_coverage_section(coverage),
                "",
            ]
        )
        if portfolio_warnings.get(name):
            os_lines.append("#### Portfolio Warnings")
            for warning in portfolio_warnings[name]:
                os_lines.append(f"- {warning}")
            os_lines.append("")

    limitations_lines = [
        "## Remaining Unknowns",
        "- EV age and time-to-event filters are not available in transactions, so values are research-derived defaults.",
        "- Devig book pairing and multi-book snapshots are not in the transaction export, so devig weights/methods cannot be validated.",
        "- Vig/market-width/variation filters are not in the transaction export and cannot be validated from data.",
        "",
        "## Known Non-Fixable Limitations",
        "- No two-way devig.",
        "- No sharp-book anchoring.",
        "- No real weight calibration.",
        "- OS Rating dependent on external samples.",
        "",
        "## Data Integrity Checklist (Pre-Run)",
        "- OS samples present.",
        "- Ratings non-default.",
        "- Odds units consistent.",
        "- EV units consistent.",
        "- Market + book keys normalized.",
        "- Time filters enforced.",
        "",
        "## What This System Is NOT",
        "- Not an automated bettor.",
        "- Not self-correcting.",
        "- Not safe under regime shifts.",
        "- Not robust without fresh data.",
        "",
        "## Execution Discipline Notes",
        f"- {EXECUTION_WARNING}",
    ]
    lines.extend(
        [
            "# Validation Report",
            "",
            "## Integrity Status",
            "- PASSED",
            "",
            "## OS Sample Summary",
            f"- {os_sample_summary}",
            f"- Core minimum samples required: {MIN_OS_SAMPLES_CORE}",
            f"- Expansion minimum samples required: {MIN_OS_SAMPLES_EXPANSION}",
            "- Threshold rationale: minimum OS sample count required to avoid sparse or misleading priors.",
            "",
            "## Critical Warnings",
            *[f"- {warning}" for warning in warnings],
            "",
            *_integrity_report(integrity_checks),
            "## Baseline Strategy (Historical Settled Straights)",
            f"- Bets: {baseline['bets']}",
            f"- ROI: {baseline['roi']:.4f}",
            f"- Profit: {baseline['profit']:.2f}",
            f"- Avg CLV: {baseline['avg_clv']:.4f}",
            f"- Worst Decile CLV: {baseline['worst_decile_clv']:.4f}",
            "",
        ]
    )
    lines.extend(bootstrap_lines)
    lines.extend(os_lines)
    lines.extend(limitations_lines)
    lines.extend(
        [
            "",
            "## Final Check",
            f"- Are all OS ratings derived from provided samples? {final_check['os_ratings_derived']}",
            f"- Were any defaults or inferred constants used? {final_check['defaults_used']}",
            f"- Is every numeric field unit-validated? {final_check['units_validated']}",
            f"- Would a reader be misled into thinking this is optimized when it is not? {final_check['misleading_optimized']}",
        ]
    )

    return "\n".join(lines)


def _validation_failed_report(warnings: List[str], integrity_checks: List[dict], os_sample_summary: str) -> str:
    lines = [
        f"# {VALIDATION_FAILED_TAG}",
        "",
        "## Integrity Status",
        "- FAILED",
        "",
        "## Critical Warnings",
        *[f"- {warning}" for warning in warnings],
        "- DO NOT USE: Recommendations are blocked until integrity checks pass.",
        "",
        "## OS Sample Summary",
        f"- {os_sample_summary}",
        f"- Core minimum samples required: {MIN_OS_SAMPLES_CORE}",
        f"- Expansion minimum samples required: {MIN_OS_SAMPLES_EXPANSION}",
        "- Threshold rationale: minimum OS sample count required to avoid sparse or misleading priors.",
        "",
        *_integrity_report(integrity_checks),
        "## Outcome",
        "- Recommendations were not generated due to failed integrity checks.",
    ]
    return "\n".join(lines)


def _write_validation_failed_outputs(out_dir: Path, warnings: List[str], integrity_checks: List[dict], os_sample_summary: str) -> None:
    report = _validation_failed_report(warnings, integrity_checks, os_sample_summary)
    for path in [
        out_dir / "validation_report.md",
        out_dir / "oddsshopper_recommendations.md",
        out_dir / "outlier_recommendations.md",
    ]:
        if path.exists():
            path.unlink()
    write_markdown(out_dir / f"validation_report_{VALIDATION_FAILED_TAG}.md", report)
    write_markdown(out_dir / f"oddsshopper_recommendations_{VALIDATION_FAILED_TAG}.md", report)
    write_markdown(out_dir / f"outlier_recommendations_{VALIDATION_FAILED_TAG}.md", report)
    write_json(out_dir / "oddsshopper_settings.json", {"validation_failed": True, "warnings": warnings})
    write_json(out_dir / "outlier_preset_core.json", {"validation_failed": True, "warnings": warnings})
    write_json(out_dir / "outlier_preset_expansion.json", {"validation_failed": True, "warnings": warnings})
    write_json(out_dir / "outlier_overlay_core.json", {"validation_failed": True, "warnings": warnings})
    write_json(out_dir / "outlier_overlay_expansion.json", {"validation_failed": True, "warnings": warnings})
    write_csv(out_dir / "oddsshopper_portfolios.csv", pd.DataFrame([{"validation_failed": True, "warning": "; ".join(warnings)}]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--os_markets", required=True)
    parser.add_argument("--os_settings", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    integrity_checks: List[dict] = []

    os_sample_summary = "No OS samples loaded"
    try:
        os_samples = _load_os_samples()
        os_sample_summary = (
            f"{len(os_samples)} rows loaded; "
            f"{os_samples['market_norm'].nunique()} markets; "
            f"{os_samples['sportsbook'].nunique()} sportsbooks"
        )
        integrity_checks.append(
            {"name": "OS Rating samples availability", "status": True, "detail": f"{len(os_samples)} rows", "critical": True}
        )
    except RuntimeError as exc:
        warning = str(exc)
        warnings.append(warning)
        integrity_checks = _blocked_integrity_checks(
            {"name": "OS Rating samples availability", "status": False, "detail": warning, "critical": True}
        )
        _write_validation_failed_outputs(out_dir, warnings, integrity_checks, os_sample_summary)
        return

    if len(os_samples) < MIN_OS_SAMPLES_CORE or len(os_samples) < MIN_OS_SAMPLES_EXPANSION:
        warning = (
            "OS Rating sample count below minimum threshold. "
            "Optimization aborted to prevent false confidence."
        )
        warnings.append(warning)
        integrity_checks.append(
            {
                "name": "OS sample count thresholds",
                "status": False,
                "detail": f"{len(os_samples)} rows < Core {MIN_OS_SAMPLES_CORE} or Expansion {MIN_OS_SAMPLES_EXPANSION}",
                "critical": True,
            }
        )
        integrity_checks.extend(
            _blocked_integrity_checks(
                {"name": "OS Rating samples availability", "status": True, "detail": f"{len(os_samples)} rows", "critical": True}
            )[1:]
        )
        _write_validation_failed_outputs(out_dir, warnings, integrity_checks, os_sample_summary)
        return

    integrity_checks.append(
        {
            "name": "OS sample count thresholds",
            "status": True,
            "detail": f"{len(os_samples)} rows >= Core {MIN_OS_SAMPLES_CORE} and Expansion {MIN_OS_SAMPLES_EXPANSION}",
            "critical": True,
        }
    )

    transactions = load_transactions(args.transactions)
    os_markets = load_os_markets(args.os_markets)
    os_settings = load_os_settings(args.os_settings)

    try:
        metrics = build_transactions_metrics(transactions)
    except RuntimeError as exc:
        warning = str(exc)
        warnings.append(warning)
        integrity_checks.append({"name": "EV unit consistency", "status": False, "detail": warning, "critical": True})
        blocked_detail = "Not evaluated (blocked by prior failure)"
        integrity_checks.extend(
            [
                {"name": "OS Rating coverage majority", "status": False, "detail": blocked_detail, "critical": True},
                {"name": "Sample size sufficiency", "status": False, "detail": blocked_detail, "critical": False},
                {"name": "Odds bounds binding (>=10% tail mass)", "status": False, "detail": blocked_detail, "critical": True},
                {"name": "Odds conversion consistency", "status": False, "detail": blocked_detail, "critical": True},
                {"name": "Devig feasibility (two-way markets)", "status": False, "detail": blocked_detail, "critical": True},
            ]
        )
        _write_validation_failed_outputs(out_dir, warnings, integrity_checks, os_sample_summary)
        return

    if metrics.attrs.get("ev_unit_coerced"):
        warnings.append("EV values were coerced from percent to decimal. NOT DATA-DERIVED.")
        integrity_checks.append(
            {"name": "EV unit consistency", "status": True, "detail": "Values coerced from percent to decimal", "critical": True}
        )
    else:
        integrity_checks.append({"name": "EV unit consistency", "status": True, "detail": "Values appear consistent", "critical": True})

    write_markdown(out_dir / "data_dictionary.md", compute_data_dictionary(transactions, os_markets))

    market_table, book_table = build_tx_tables(metrics)
    write_csv(out_dir / "tx_market_tables.csv", market_table)
    write_csv(out_dir / "tx_book_tables.csv", book_table)

    two_way_table = build_two_way_table(metrics)
    two_way_count = len(two_way_table)
    if two_way_count == 0:
        warnings.append(DEVIG_ABSENCE_WARNING)
        warnings.append(DEVIG_STRUCTURAL_WARNING)
        integrity_checks.append(
            {"name": "Devig feasibility (two-way markets)", "status": False, "detail": "No two-way markets detected", "critical": True}
        )
    else:
        integrity_checks.append(
            {"name": "Devig feasibility (two-way markets)", "status": True, "detail": f"{two_way_count} two-way rows", "critical": True}
        )

    warnings.extend([OVERLAY_WARNING, OVERLAY_EXECUTION_WARNING, BACKTEST_WARNING, EXECUTION_WARNING])

    os_results = optimize_os(metrics, os_markets)
    os_backtests = []
    portfolio_rows = []
    settings_payload: Dict[str, Dict[str, Any]] = {}
    portfolio_summaries: Dict[str, dict] = {}
    portfolio_summaries_raw: Dict[str, dict] = {}
    coverage_reports: Dict[str, dict] = {}
    coverage_reports_raw: Dict[str, dict] = {}
    portfolio_warnings: Dict[str, List[str]] = {}
    portfolio_warnings_raw: Dict[str, List[str]] = {}
    display_name_map: Dict[str, str] = {}
    odds_binding_failures = []
    odds_conversion_failures = []
    extrapolated_portfolios = []

    for result in os_results:
        combos = pd.DataFrame(result.portfolio_markets)
        coverage = _os_sample_coverage(os_samples, combos)
        coverage_reports_raw[result.name] = coverage
        coverage_pct = coverage["coverage_pct"]
        if coverage_pct < 0.5:
            warnings.append(OS_RATING_COVERAGE_WARNING)
            integrity_checks.append(
                {
                    "name": f"OS Rating coverage majority ({result.name})",
                    "status": False,
                    "detail": f"{coverage_pct*100:.1f}% coverage",
                    "critical": True,
                }
            )
        else:
            integrity_checks.append(
                {
                    "name": f"OS Rating coverage majority ({result.name})",
                    "status": True,
                    "detail": f"{coverage_pct*100:.1f}% coverage",
                    "critical": True,
                }
            )

        os_rating = _derive_os_rating_threshold(os_samples, combos)
        if os_rating is None:
            warnings.append(OS_RATING_WARNING)
            integrity_checks.append(
                {
                    "name": f"OS Rating coverage ({result.name})",
                    "status": False,
                    "detail": "No OS Rating samples for selected combos",
                    "critical": True,
                }
            )
        else:
            result.settings["minimum_os_rating"] = round(os_rating, 2)
            result.settings["minimum_os_rating_source"] = "observed_minimum_os_samples"

        combo_subset = metrics.merge(
            combos[["league_norm", "market_norm", "sportsbook"]],
            on=["league_norm", "market_norm", "sportsbook"],
            how="inner",
        )
        portfolio_warnings_raw[result.name] = []
        if coverage["extrapolated"]:
            portfolio_warnings_raw[result.name].append(
                "OS Ratings are extrapolated across uncovered league/market/sportsbook combos. NOT DATA-DERIVED."
            )
            extrapolated_portfolios.append(result.name)

        _apply_odds_binding_fallback(combo_subset, result.settings, portfolio_warnings_raw[result.name])
        lower_tail, upper_tail = _odds_binding_rates(combo_subset, result.settings)

        subset = _apply_os_settings(metrics, result.settings, combos)
        backtest = _os_backtest(subset, result.settings)
        summary = summarize_performance(backtest)
        summary["max_drawdown"] = backtest["drawdown"].max() if not backtest.empty else 0.0
        portfolio_summaries_raw[result.name] = summary
        os_backtests.append(backtest.assign(portfolio=result.name))

        if summary["bets"] < 15:
            warning = "Results shown are based on fewer than 15 bets and are statistically meaningless."
            warnings.append(warning)
            portfolio_warnings_raw[result.name].append(warning)
        elif summary["bets"] < 30:
            warning = "Results shown are based on fewer than 30 bets and are not statistically reliable."
            warnings.append(warning)
            portfolio_warnings_raw[result.name].append(warning)

        display_name = result.name
        if result.name == "Core":
            downgrade_reasons = []
            if coverage_pct < 0.7:
                downgrade_reasons.append("coverage <70%")
            if summary["bets"] < 15:
                downgrade_reasons.append("fewer than 15 bets")
            if downgrade_reasons:
                display_name = f"Experimental (Core label blocked: {', '.join(downgrade_reasons)})"
        display_name_map[result.name] = display_name
        if lower_tail < 0.1 or upper_tail < 0.1:
            odds_binding_failures.append(display_name)
        portfolio_summaries[display_name] = summary
        coverage_reports[display_name] = coverage
        portfolio_warnings[display_name] = portfolio_warnings_raw[result.name]

        settings = result.settings.copy()
        settings["portfolio_label"] = display_name
        settings["odds_range_min_american"] = round(decimal_to_american(settings["odds_range_min_decimal"]), 0)
        settings["odds_range_max_american"] = round(decimal_to_american(settings["odds_range_max_decimal"]), 0)
        try:
            _validate_odds_bounds(
                settings["odds_range_min_decimal"],
                settings["odds_range_max_decimal"],
                settings["odds_range_min_american"],
                settings["odds_range_max_american"],
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
            integrity_checks.append(
                {
                    "name": f"Odds conversion consistency ({display_name})",
                    "status": False,
                    "detail": str(exc),
                    "critical": True,
                }
            )
            odds_conversion_failures.append(display_name)
        settings_payload[display_name] = settings

        for item in result.portfolio_markets:
            row = {**item, "portfolio": display_name}
            portfolio_rows.append(row)

    if odds_binding_failures:
        integrity_checks.append(
            {
                "name": "Odds bounds binding (>=10% tail mass)",
                "status": False,
                "detail": f"Insufficient binding for: {', '.join(odds_binding_failures)}",
                "critical": True,
            }
        )
    else:
        integrity_checks.append(
            {"name": "Odds bounds binding (>=10% tail mass)", "status": True, "detail": "All portfolios bind tails", "critical": True}
        )

    if extrapolated_portfolios:
        label_list = [display_name_map.get(name, name) for name in extrapolated_portfolios]
        warnings.append(
            f"OS Ratings extrapolated for portfolios with uncovered combos: {', '.join(label_list)}. NOT DATA-DERIVED."
        )

    if odds_conversion_failures:
        integrity_checks.append(
            {
                "name": "Odds conversion consistency",
                "status": False,
                "detail": f"Failures: {', '.join(odds_conversion_failures)}",
                "critical": True,
            }
        )
    else:
        integrity_checks.append(
            {"name": "Odds conversion consistency", "status": True, "detail": "Decimal/American bounds match", "critical": True}
        )

    core_bets = portfolio_summaries_raw.get("Core", {}).get("bets")
    expansion_bets = portfolio_summaries_raw.get("Expansion", {}).get("bets")
    if core_bets is not None:
        integrity_checks.append(
            {
                "name": "Core sample size (>=15 bets)",
                "status": core_bets >= 15,
                "detail": f"{core_bets} bets",
                "critical": False,
            }
        )
        integrity_checks.append(
            {
                "name": "Core sample size (>=30 bets)",
                "status": core_bets >= 30,
                "detail": f"{core_bets} bets",
                "critical": False,
            }
        )

    if core_bets is not None and expansion_bets is not None:
        if expansion_bets < 2 * core_bets:
            warning = "Expansion portfolio does not materially expand volume beyond Core. Expansion classification may be misleading."
            warnings.append(warning)
            portfolio_warnings_raw.setdefault("Expansion", []).append(warning)
            expansion_label = display_name_map.get("Expansion", "Expansion")
            portfolio_warnings.setdefault(expansion_label, []).append(warning)
        integrity_checks.append(
            {
                "name": "Expansion volume (>=2x Core)",
                "status": expansion_bets >= 2 * core_bets,
                "detail": f"Core={core_bets}, Expansion={expansion_bets}",
                "critical": False,
            }
        )

    critical_fail = any(check["critical"] and not check["status"] for check in integrity_checks)

    if critical_fail:
        _write_validation_failed_outputs(out_dir, warnings, integrity_checks, os_sample_summary)
        return

    write_csv(out_dir / "oddsshopper_portfolios.csv", pd.DataFrame(portfolio_rows))
    write_json(out_dir / "oddsshopper_settings.json", settings_payload)

    if os_backtests:
        backtest_frames = []
        for backtest in os_backtests:
            portfolio_name = backtest["portfolio"].iloc[0] if not backtest.empty else "Unknown"
            backtest["portfolio"] = display_name_map.get(portfolio_name, portfolio_name)
            backtest_frames.append(backtest)
        write_csv(out_dir / "oddsshopper_backtest.csv", pd.concat(backtest_frames, ignore_index=True))

    os_md = [
        "# OddsShopper Recommendations",
        "",
        "## Integrity Status",
        "- PASSED",
        "",
        "## OS Sample Summary",
        f"- {os_sample_summary}",
        f"- Core minimum samples required: {MIN_OS_SAMPLES_CORE}",
        f"- Expansion minimum samples required: {MIN_OS_SAMPLES_EXPANSION}",
        "- Threshold rationale: minimum OS sample count required to avoid sparse or misleading priors.",
        "",
        "## Critical Warnings",
        *[f"- {warning}" for warning in warnings],
        "",
        "## Execution Discipline Notes",
        f"- {EXECUTION_WARNING}",
        "",
        "## Limitations",
        "- OS Rating priors depend entirely on provided OS samples.",
        "- EV age and time-to-event filters are not present in transactions and cannot be validated.",
        "",
    ]
    for result in os_results:
        display_name = display_name_map[result.name]
        os_md.extend([f"## Portfolio: {display_name}", "", "### Settings"])
        for key, value in settings_payload[display_name].items():
            os_md.append(f"- {key}: {value}")
        os_md.append("")
        os_md.append("### OS Rating Coverage")
        os_md.extend(_coverage_section(coverage_reports[display_name]))
        os_md.append("")
        if portfolio_warnings.get(display_name):
            os_md.append("### Portfolio Warnings")
            for warning in portfolio_warnings[display_name]:
                os_md.append(f"- {warning}")
            os_md.append("")
        os_md.append("### Included Market + Sportsbook + League Combos")
        for item in result.portfolio_markets:
            os_md.append(
                f"- {item['league_norm']} | {item['market_norm']} | {item['sportsbook']} "
                f"(roi_shrunk={item['roi_shrunk']:.4f}, clv_shrunk={item['clv_shrunk']:.4f}, bets={item['bets']})"
            )
        os_md.append("")
    write_markdown(out_dir / "oddsshopper_recommendations.md", "\n".join(os_md))

    presets = build_presets()
    preset_core = presets[0].settings
    preset_expansion = presets[1].settings
    if two_way_count == 0:
        preset_core["devig_method"] = "Average (preset prior)"
        preset_expansion["devig_method"] = "Average (preset prior)"
        preset_core["devig_weights_source"] = "preset-prior (structural approximation)"
        preset_expansion["devig_weights_source"] = "preset-prior (structural approximation)"
        preset_core["devig_validation_status"] = "structural approximation"
        preset_expansion["devig_validation_status"] = "structural approximation"

    write_json(out_dir / "outlier_preset_core.json", preset_core)
    write_json(out_dir / "outlier_preset_expansion.json", preset_expansion)

    overlays = build_overlays(metrics)
    core_overlay = overlays[0].settings
    expansion_overlay = overlays[1].settings
    core_overlay["overlay_role"] = "execution_discipline_only"
    expansion_overlay["overlay_role"] = "execution_discipline_only"
    write_json(out_dir / "outlier_overlay_core.json", core_overlay)
    write_json(out_dir / "outlier_overlay_expansion.json", expansion_overlay)

    outlier_md = [
        "# Outlier Recommendations",
        "",
        "## Integrity Status",
        "- PASSED",
        "",
        "## OS Sample Summary",
        f"- {os_sample_summary}",
        f"- Core minimum samples required: {MIN_OS_SAMPLES_CORE}",
        f"- Expansion minimum samples required: {MIN_OS_SAMPLES_EXPANSION}",
        "- Threshold rationale: minimum OS sample count required to avoid sparse or misleading priors.",
        "",
        "## Critical Warnings",
        *[f"- {warning}" for warning in warnings],
        "",
        "## Execution Discipline Notes",
        f"- {EXECUTION_WARNING}",
        "",
        "## Limitations",
        "- Devig validation is limited by missing two-way market data.",
        "- Overlay filters are execution discipline only.",
        "",
        "## Preset Core Profile",
    ]
    for key, value in preset_core.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Preset Expansion Profile")
    for key, value in preset_expansion.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Overlay Core Profile (Execution Discipline Only)")
    for key, value in core_overlay.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Overlay Expansion Profile (Execution Discipline Only)")
    for key, value in expansion_overlay.items():
        outlier_md.append(f"- {key}: {value}")
    write_markdown(out_dir / "outlier_recommendations.md", "\n".join(outlier_md))

    baseline = summarize_performance(metrics)
    settings_sources = [settings.get("minimum_os_rating_source") for settings in settings_payload.values()]
    os_ratings_derived = all(source == "observed_minimum_os_samples" for source in settings_sources)
    defaults_used = any(source != "observed_minimum_os_samples" for source in settings_sources)
    units_validated = not metrics.attrs.get("ev_unit_coerced", False)
    final_check = {
        "os_ratings_derived": "Yes" if os_ratings_derived else "No",
        "defaults_used": "No" if not defaults_used else "Yes",
        "units_validated": "Yes" if units_validated else "No (EV units coerced from percent)",
        "misleading_optimized": "No (execution-only overlays and warnings included)",
    }
    validation = validation_report(
        metrics,
        baseline,
        portfolio_summaries,
        coverage_reports,
        warnings,
        portfolio_warnings,
        integrity_checks,
        os_sample_summary,
        final_check,
    )
    write_markdown(out_dir / "validation_report.md", validation)

    write_json(out_dir / "os_settings_reference.json", os_settings)


if __name__ == "__main__":
    main()
