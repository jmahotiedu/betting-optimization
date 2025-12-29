from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List
import glob

import numpy as np
import pandas as pd

from src.data_load import load_transactions, load_os_markets, load_os_settings
from src.odds_utils import to_decimal, decimal_to_american
from src.mapping import normalize_market, normalize_league
from src.backtest import bankroll_simulation, summarize_performance, bootstrap_ci
from src.optimize_oddsshopper import optimize as optimize_os, build_tx_tables
from src.optimize_outlier import optimize as optimize_outlier
from src.report import write_markdown, write_json, write_csv


def compute_data_dictionary(transactions: pd.DataFrame, os_markets: pd.DataFrame, os_samples: pd.DataFrame | None) -> str:
    lines = ["# Data Dictionary", "", "## transactions.csv", ""]
    for col in transactions.columns:
        missing = transactions[col].isna().mean() * 100
        lines.append(f"- **{col}**: missing {missing:.1f}%")
    lines.extend(["", "## os_markets_clean.csv", ""])
    for col in os_markets.columns:
        missing = os_markets[col].isna().mean() * 100
        lines.append(f"- **{col}**: missing {missing:.1f}%")
    if os_samples is not None:
        lines.extend(["", "## os_samples", ""])
        for col in os_samples.columns:
            missing = os_samples[col].isna().mean() * 100
            lines.append(f"- **{col}**: missing {missing:.1f}%")
    lines.extend(
        [
            "",
            "## Odds Format Notes",
            "- Odds appear as decimal in transactions, with possible American values in samples.",
            "- The parser treats values between 0 and 1 as decimal-minus-one and adds 1.",
        ]
    )
    return "\n".join(lines)


def _find_os_samples() -> Path | None:
    candidates = [
        Path(p)
        for p in glob.glob("/*")
        if ("sample" in p.lower() or "os_" in p.lower())
        and p.lower().endswith((".csv", ".xlsx", ".parquet"))
        and not p.lower().endswith("os_markets_clean.csv")
    ]
    return candidates[0] if candidates else None


def load_os_samples() -> pd.DataFrame | None:
    path = _find_os_samples()
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return None


def build_transactions_metrics(transactions: pd.DataFrame, os_samples: pd.DataFrame | None) -> pd.DataFrame:
    df = transactions.copy()
    df["status"] = df["status"].astype(str)
    df = df[(df["type"].str.lower() == "straight") & (df["status"].str.startswith("SETTLED"))]
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df.loc[df["close_decimal"].isna(), "clv"] = np.nan
    df["market_norm"] = df.apply(lambda r: normalize_market(r.get("bet_info"), r.get("type")), axis=1)
    df["league_norm"] = df.apply(lambda r: normalize_league(r.get("leagues"), r.get("sports")), axis=1)
    df["amount"] = df["amount"].abs()
    df["profit"] = df["profit"].fillna(0.0)
    df["time_placed_iso"] = pd.to_datetime(df["time_placed_iso"], errors="coerce", utc=True)
    df["ev"] = df["ev"].fillna(0.0)
    df["edge"] = df["ev"]
    df["kelly_pct"] = df.apply(
        lambda r: max(0.0, r["edge"]) / (r["odds_decimal"] - 1) if r["odds_decimal"] and r["odds_decimal"] > 1 else 0.0,
        axis=1,
    )

    df["os_rating_pred"] = _derive_os_rating(df, os_samples)
    return df


def _derive_os_rating(transactions: pd.DataFrame, os_samples: pd.DataFrame | None) -> pd.Series:
    if os_samples is None or os_samples.empty:
        return pd.Series([20.0] * len(transactions), index=transactions.index)

    samples = os_samples.copy()
    samples.columns = [c.strip().lower().replace(" ", "_") for c in samples.columns]
    samples["league_norm"] = samples.get("league", "").astype(str).str.strip()
    samples["sportsbook"] = samples.get("sportsbook", "").astype(str).str.strip()
    samples["market_norm"] = samples.get("offer_name", "").apply(lambda x: normalize_market(x, None))
    samples["os_rating"] = pd.to_numeric(samples.get("os_rating"), errors="coerce")

    combo_avg = (
        samples.dropna(subset=["os_rating"])
        .groupby(["league_norm", "market_norm", "sportsbook"])["os_rating"]
        .mean()
        .reset_index()
    )
    market_avg = (
        samples.dropna(subset=["os_rating"])
        .groupby(["league_norm", "market_norm"])["os_rating"]
        .mean()
        .reset_index()
    )
    overall_mean = float(samples["os_rating"].dropna().mean()) if samples["os_rating"].notna().any() else 20.0

    merged = transactions.merge(combo_avg, on=["league_norm", "market_norm", "sportsbook"], how="left")
    merged = merged.merge(market_avg, on=["league_norm", "market_norm"], how="left", suffixes=("", "_market"))
    os_rating = merged["os_rating"].fillna(merged["os_rating_market"]).fillna(overall_mean)
    return os_rating


def os_rating_report(transactions: pd.DataFrame, os_samples: pd.DataFrame | None) -> str:
    if os_samples is None or os_samples.empty:
        return "# OS Rating Model\n\nNo OS sample file detected; using fallback rating of 20.\n"

    samples = os_samples.copy()
    samples.columns = [c.strip().lower().replace(" ", "_") for c in samples.columns]
    samples["league_norm"] = samples.get("league", "").astype(str).str.strip()
    samples["sportsbook"] = samples.get("sportsbook", "").astype(str).str.strip()
    samples["market_norm"] = samples.get("offer_name", "").apply(lambda x: normalize_market(x, None))
    combo_keys = samples.dropna(subset=["os_rating"]).merge(
        transactions[["league_norm", "market_norm", "sportsbook"]],
        on=["league_norm", "market_norm", "sportsbook"],
        how="inner",
    )
    market_keys = samples.dropna(subset=["os_rating"]).merge(
        transactions[["league_norm", "market_norm"]].drop_duplicates(),
        on=["league_norm", "market_norm"],
        how="inner",
    )

    lines = [
        "# OS Rating Model",
        "",
        "## Join Logic",
        "- Matched samples to transactions by league + market + sportsbook.",
        "- Fallback to league + market average, then overall average.",
        "",
        "## Coverage",
        f"- Transactions with direct combo match: {len(combo_keys):,}",
        f"- Transactions with market-level match: {len(market_keys):,}",
        "",
    ]
    lines.append("## Notes")
    lines.append("- Sample file has limited rows; model uses hierarchical averages rather than a high-variance regression.")
    return "\n".join(lines)


def _apply_os_settings(df: pd.DataFrame, settings: Dict[str, Any], combos: pd.DataFrame) -> pd.DataFrame:
    if combos.empty:
        return df.iloc[0:0].copy()
    subset = df.merge(combos[["league_norm", "market_norm", "sportsbook"]], on=["league_norm", "market_norm", "sportsbook"], how="inner")
    ev_min = settings["minimum_ev"] / 100.0
    odds_min = settings["odds_range_min_decimal"]
    odds_max = settings["odds_range_max_decimal"]
    subset = subset[(subset["ev"] >= ev_min) & (subset["odds_decimal"] >= odds_min) & (subset["odds_decimal"] <= odds_max)]
    subset = subset[subset["os_rating_pred"] >= settings["minimum_os_rating"]]
    return subset


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


def _outlier_backtest(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    ev_min = settings["ev_min_pct"] / 100.0
    kelly_min = settings["kelly_min_pct"] / 100.0
    filtered = df[(df["ev"] >= ev_min) & (df["kelly_pct"] >= kelly_min)]
    return bankroll_simulation(
        filtered,
        bankroll=1000,
        stake_strategy="os kelly",
        kelly_fraction=settings["stake_kelly_fraction"],
        max_bet_pct=settings["stake_cap_pct_bankroll"],
    )


def validation_report(df: pd.DataFrame, os_results: List, outlier_profiles: List) -> str:
    baseline = summarize_performance(df)
    lines = [
        "# Validation Report",
        "",
        "## Baseline Strategy (Historical Settled Straights)",
        f"- Bets: {baseline['bets']}",
        f"- ROI: {baseline['roi']:.4f}",
        f"- Profit: {baseline['profit']:.2f}",
        f"- Avg CLV: {baseline['avg_clv']:.4f}",
        f"- Worst Decile CLV: {baseline['worst_decile_clv']:.4f}",
        "",
    ]

    roi_ci = bootstrap_ci((df["profit"] / df["amount"].replace(0, 1)).to_numpy())
    clv_ci = bootstrap_ci(df["clv"].fillna(0).to_numpy())
    lines.extend(
        [
            "## Bootstrap CIs",
            f"- ROI mean CI (approx): [{roi_ci[0]:.4f}, {roi_ci[1]:.4f}]",
            f"- CLV mean CI (approx): [{clv_ci[0]:.4f}, {clv_ci[1]:.4f}]",
            "",
        ]
    )

    lines.append("## OddsShopper Portfolios")
    for result in os_results:
        combos = pd.DataFrame(result.portfolio_markets)
        subset = _apply_os_settings(df, result.settings, combos)
        backtest = _os_backtest(subset, result.settings)
        summary = summarize_performance(backtest)
        max_dd = backtest["drawdown"].max() if not backtest.empty else 0.0
        lines.extend(
            [
                f"### {result.name}",
                f"- Bets: {summary['bets']}",
                f"- ROI: {summary['roi']:.4f}",
                f"- Profit: {summary['profit']:.2f}",
                f"- Avg CLV: {summary['avg_clv']:.4f}",
                f"- Worst Decile CLV: {summary['worst_decile_clv']:.4f}",
                f"- Max Drawdown: {max_dd:.4f}",
                "",
            ]
        )

    lines.append("## Outlier Profiles")
    for profile in outlier_profiles:
        backtest = _outlier_backtest(df, profile.settings)
        summary = summarize_performance(backtest)
        max_dd = backtest["drawdown"].max() if not backtest.empty else 0.0
        lines.extend(
            [
                f"### {profile.name}",
                f"- Bets: {summary['bets']}",
                f"- ROI: {summary['roi']:.4f}",
                f"- Profit: {summary['profit']:.2f}",
                f"- Avg CLV: {summary['avg_clv']:.4f}",
                f"- Worst Decile CLV: {summary['worst_decile_clv']:.4f}",
                f"- Max Drawdown: {max_dd:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Remaining Unknowns",
            "- EV age and time-to-event filters are not available in transactions; values come from OS settings bounds.",
            "- Vig/market-width/variation filters are not in the transaction export and rely on Outlier documentation.",
        ]
    )

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
    os_samples = load_os_samples()

    metrics = build_transactions_metrics(transactions, os_samples)
    write_markdown(out_dir / "data_dictionary.md", compute_data_dictionary(transactions, os_markets, os_samples))
    write_markdown(out_dir / "os_rating_model.md", os_rating_report(metrics, os_samples))

    combo_table, market_table, book_table = build_tx_tables(metrics)
    write_csv(out_dir / "tx_combo_table.csv", combo_table)
    write_csv(out_dir / "tx_market_table.csv", market_table)
    write_csv(out_dir / "tx_book_table.csv", book_table)

    os_results = optimize_os(metrics, os_markets, min_bets=50)
    os_backtests = []
    for result in os_results:
        combos = pd.DataFrame(result.portfolio_markets)
        subset = _apply_os_settings(metrics, result.settings, combos)
        backtest = _os_backtest(subset, result.settings)
        os_backtests.append(backtest.assign(portfolio=result.name))

    portfolio_rows = []
    for result in os_results:
        for item in result.portfolio_markets:
            row = {**item, "portfolio": result.name}
            portfolio_rows.append(row)
    write_csv(out_dir / "oddsshopper_portfolios.csv", pd.DataFrame(portfolio_rows))

    settings_payload = {}
    for result in os_results:
        settings = result.settings.copy()
        settings["odds_range_min_american"] = round(decimal_to_american(settings["odds_range_min_decimal"]), 0)
        settings["odds_range_max_american"] = round(decimal_to_american(settings["odds_range_max_decimal"]), 0)
        settings_payload[result.name] = settings
    write_json(out_dir / "oddsshopper_settings.json", settings_payload)

    os_md = ["# OddsShopper Recommendations", ""]
    for result in os_results:
        os_md.extend([f"## Portfolio: {result.name}", "", "### Settings"])
        for key, value in settings_payload[result.name].items():
            os_md.append(f"- {key}: {value}")
        os_md.append("")
        os_md.append("### Included Market + Sportsbook + League Combos")
        for item in result.portfolio_markets:
            os_md.append(
                f"- {item['league_norm']} | {item['market_norm']} | {item['sportsbook']} "
                f"(roi_shrunk={item['roi_shrunk']:.4f}, clv_shrunk={item['clv_shrunk']:.4f}, bets={item['bets']}, avg_os_rating={item['avg_os_rating']:.2f})"
            )
        os_md.append("")
    write_markdown(out_dir / "oddsshopper_recommendations.md", "\n".join(os_md))

    if os_backtests:
        write_csv(out_dir / "oddsshopper_backtest.csv", pd.concat(os_backtests, ignore_index=True))

    outlier_profiles = optimize_outlier(metrics)
    outlier_backtest = pd.concat([p.backtest.assign(profile=p.name) for p in outlier_profiles], ignore_index=True)
    write_csv(out_dir / "outlier_backtest.csv", outlier_backtest)

    weights_payload = {
        "weights": outlier_profiles[0].devig_weights,
        "sources": outlier_profiles[0].devig_weight_sources,
    }
    write_json(out_dir / "outlier_devig_weights.json", weights_payload)

    core_settings = outlier_profiles[0].settings
    expansion_settings = outlier_profiles[1].settings
    write_json(out_dir / "outlier_settings_core.json", core_settings)
    write_json(out_dir / "outlier_settings_expansion.json", expansion_settings)

    outlier_md = ["# Outlier Recommendations", "", "## Core Profile"]
    for key, value in core_settings.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Expansion Profile")
    for key, value in expansion_settings.items():
        outlier_md.append(f"- {key}: {value}")
    outlier_md.append("")
    outlier_md.append("## Devig Weights")
    for book, weight in outlier_profiles[0].devig_weights.items():
        source = outlier_profiles[0].devig_weight_sources.get(book, "unknown")
        outlier_md.append(f"- {book}: {weight:.4f} ({source})")
    write_markdown(out_dir / "outlier_recommendations.md", "\n".join(outlier_md))

    validation = validation_report(metrics, os_results, outlier_profiles)
    write_markdown(out_dir / "validation_report.md", validation)

    write_json(out_dir / "os_settings_reference.json", os_settings)


if __name__ == "__main__":
    main()
