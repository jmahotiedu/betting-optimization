from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_load import load_transactions, load_os_markets, load_os_settings
from src.odds_utils import to_decimal
from src.mapping import normalize_market, normalize_league
from src.backtest import bankroll_simulation, summarize_performance, bootstrap_ci
from src.optimize_oddsshopper import optimize as optimize_os
from src.optimize_outlier import optimize as optimize_outlier
from src.report import write_markdown, write_json, write_csv


def compute_quality(transactions: pd.DataFrame) -> str:
    anomalies = []
    if transactions["odds"].isna().any():
        anomalies.append("Missing odds detected; rows excluded from calculations requiring odds.")
    if transactions["closing_line"].isna().any():
        anomalies.append("Missing closing_line detected; CLV computed only where available.")
    if (transactions["amount"] <= 0).any():
        anomalies.append("Non-positive stakes detected; used absolute value for ROI calc.")
    if transactions["profit"].isna().any():
        anomalies.append("Missing profit values; treated as 0 in ROI calculations.")

    lines = ["# Data Quality", "", "## Anomalies", ""]
    if anomalies:
        lines.extend([f"- {item}" for item in anomalies])
    else:
        lines.append("- No critical anomalies detected.")

    lines.extend(
        [
            "",
            "## Handling",
            "- Odds parsed to decimal using type detection; unknown odds treated as decimal.",
            "- CLV computed when closing_line is present.",
            "- Stakes coerced to positive for ROI summaries.",
        ]
    )
    return "\n".join(lines)


def build_transactions_metrics(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df["market_norm"] = df.apply(lambda r: normalize_market(r.get("bet_info"), r.get("type")), axis=1)
    df["league_norm"] = df.apply(lambda r: normalize_league(r.get("leagues"), r.get("sports")), axis=1)
    df["amount"] = df["amount"].abs()
    df["profit"] = df["profit"].fillna(0.0)
    return df


def validation_report(df: pd.DataFrame, out_dir: Path) -> str:
    baseline = summarize_performance(df)
    roi_ci = bootstrap_ci(df["profit"] / df["amount"].replace(0, 1))
    clv_ci = bootstrap_ci(df["clv"].fillna(0).to_numpy())

    lines = [
        "# Validation Report",
        "",
        "## Baseline Strategy (Historical)",
        f"- Bets: {baseline['bets']}",
        f"- ROI: {baseline['roi']:.4f}",
        f"- Profit: {baseline['profit']:.2f}",
        f"- Avg CLV: {baseline['avg_clv']:.4f}",
        f"- Worst Decile CLV: {baseline['worst_decile_clv']:.4f}",
        "",
        "## Bootstrap CIs",
        f"- ROI mean CI (approx): [{roi_ci[0]:.4f}, {roi_ci[1]:.4f}]",
        f"- CLV mean CI (approx): [{clv_ci[0]:.4f}, {clv_ci[1]:.4f}]",
        "",
        "## Remaining Unknowns",
        "- Limited visibility into OS Rating, EV age, and time-to-event filters in the transaction export.",
        "- Outlier book weighting relies on proxy consensus (closing_line).",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--os_markets", required=True)
    parser.add_argument("--os_settings", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(args.transactions)
    os_markets = load_os_markets(args.os_markets)
    os_settings = load_os_settings(args.os_settings)

    metrics = build_transactions_metrics(transactions)
    write_markdown(out_dir / "data_quality.md", compute_quality(metrics))

    os_results = optimize_os(metrics, os_markets)
    os_result = os_results[0]
    write_csv(out_dir / "oddsshopper_backtest.csv", os_result.backtest)
    write_csv(out_dir / "oddsshopper_portfolios.csv", pd.DataFrame(os_result.portfolio_markets))
    write_json(out_dir / "oddsshopper_settings.json", os_result.settings)

    os_md = [
        "# OddsShopper Recommendations",
        "",
        f"## Portfolio: {os_result.name}",
        "", 
        "### Settings",
    ]
    for key, value in os_result.settings.items():
        os_md.append(f"- {key}: {value}")
    os_md.append("")
    os_md.append("### Included Market + Sportsbook + League Combos")
    for item in os_result.portfolio_markets:
        os_md.append(
            f"- {item['market_norm']} | {item['league_norm']} | {item['sportsbook']} "
            f"(roi_shrunk={item['roi_shrunk']:.4f}, clv_shrunk={item['clv_shrunk']:.4f})"
        )
    write_markdown(out_dir / "oddsshopper_recommendations.md", "\n".join(os_md))

    outlier_profiles = optimize_outlier(metrics)
    outlier_backtest = pd.concat([p.backtest.assign(profile=p.name) for p in outlier_profiles], ignore_index=True)
    write_csv(out_dir / "outlier_backtest.csv", outlier_backtest)
    write_json(out_dir / "outlier_devig_weights.json", outlier_profiles[0].devig_weights)

    core_settings = outlier_profiles[0].settings
    expansion_settings = outlier_profiles[1].settings
    write_json(out_dir / "outlier_settings_core.json", core_settings)
    write_json(out_dir / "outlier_settings_expansion.json", expansion_settings)

    outlier_md = [
        "# Outlier Recommendations",
        "",
        "## Core Profile",
    ]
    for key, value in core_settings.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Expansion Profile")
    for key, value in expansion_settings.items():
        outlier_md.append(f"- {key}: {value}")
    write_markdown(out_dir / "outlier_recommendations.md", "\n".join(outlier_md))

    sensitivity_lines = [
        "# Outlier Sensitivity Notes",
        "",
        "- Higher EV and Kelly thresholds improve CLV but reduce volume; monitor sample size.",
        "- Increasing market width or vig thresholds admits more bets but increases drawdown volatility.",
        "- Devig method sensitivity: shin/power methods tend to be more conservative on heavy-favorite markets.",
    ]
    write_markdown(out_dir / "outlier_sensitivity.md", "\n".join(sensitivity_lines))

    validation = validation_report(metrics, out_dir)
    write_markdown(out_dir / "validation_report.md", validation)

    write_json(out_dir / "os_settings_reference.json", os_settings)


if __name__ == "__main__":
    main()
