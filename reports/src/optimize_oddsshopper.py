from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna

from .backtest import bankroll_simulation, summarize_performance


@dataclass
class PortfolioResult:
    name: str
    settings: Dict[str, Any]
    portfolio_markets: List[Dict[str, Any]]
    backtest: pd.DataFrame


def build_tx_tables(transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_combo = transactions.groupby(["league_norm", "market_norm", "sportsbook"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "stake": g["amount"].sum(),
                "profit": g["profit"].sum(),
                "roi": g["profit"].sum() / g["amount"].sum() if g["amount"].sum() else 0.0,
                "avg_clv": g["clv"].mean(),
                "median_clv": g["clv"].median(),
                "worst_decile_clv": g["clv"].quantile(0.1),
                "avg_os_rating": g["os_rating_pred"].mean(),
            }
        )
    )
    combo_table = grouped_combo.reset_index()

    grouped_market = transactions.groupby(["league_norm", "market_norm"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "stake": g["amount"].sum(),
                "profit": g["profit"].sum(),
                "roi": g["profit"].sum() / g["amount"].sum() if g["amount"].sum() else 0.0,
                "avg_clv": g["clv"].mean(),
                "median_clv": g["clv"].median(),
                "worst_decile_clv": g["clv"].quantile(0.1),
                "avg_os_rating": g["os_rating_pred"].mean(),
            }
        )
    )
    market_table = grouped_market.reset_index()

    grouped_book = transactions.groupby(["sportsbook"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "stake": g["amount"].sum(),
                "profit": g["profit"].sum(),
                "roi": g["profit"].sum() / g["amount"].sum() if g["amount"].sum() else 0.0,
                "avg_clv": g["clv"].mean(),
                "median_clv": g["clv"].median(),
                "worst_decile_clv": g["clv"].quantile(0.1),
            }
        )
    )
    book_table = grouped_book.reset_index()
    return combo_table, market_table, book_table


def shrink_combo_performance(combo_table: pd.DataFrame, os_markets: pd.DataFrame, k: float = 25.0) -> pd.DataFrame:
    os_markets = os_markets.rename(columns={"market": "market_norm", "league": "league_norm"})
    merged = combo_table.merge(os_markets, on=["market_norm", "league_norm", "sportsbook"], how="left")
    merged["roi_pct"] = merged["roi_pct"].fillna(0.0)
    merged["roi_prior"] = merged["roi_pct"] / 100.0
    merged["roi_shrunk"] = (merged["roi"] * merged["bets"] + merged["roi_prior"] * k) / (merged["bets"] + k)
    merged["clv_shrunk"] = (merged["avg_clv"] * merged["bets"] + 0.0 * k) / (merged["bets"] + k)
    return merged


def _apply_settings(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    subset = df[(df["ev"] >= settings["minimum_ev"] / 100.0)]
    subset = subset[(subset["odds_decimal"] >= settings["odds_range_min_decimal"]) & (subset["odds_decimal"] <= settings["odds_range_max_decimal"])]
    subset = subset[subset["os_rating_pred"] >= settings["minimum_os_rating"]]
    return subset


def _evaluate_settings(df: pd.DataFrame, settings: Dict[str, Any], min_sample_bets: int) -> Tuple[float, pd.DataFrame]:
    subset = _apply_settings(df, settings)
    if subset.empty:
        return -np.inf, pd.DataFrame()
    if len(subset) < min_sample_bets:
        return -np.inf, pd.DataFrame()
    if settings["betting_size_strategy"] == "Flat":
        backtest = bankroll_simulation(subset, bankroll=1000, stake_strategy="flat", max_bet_pct=settings["flat_unit_pct_bankroll"])
    else:
        backtest = bankroll_simulation(
            subset,
            bankroll=1000,
            stake_strategy="os kelly",
            kelly_fraction=settings["fractional_kelly"],
            max_bet_pct=settings["max_bet_pct_bankroll"],
        )
    summary = summarize_performance(backtest)
    max_dd = backtest["drawdown"].max() if not backtest.empty else 0.0
    if summary["avg_clv"] < 0 or summary["worst_decile_clv"] < -0.05 or max_dd > 0.25:
        return -np.inf, backtest
    score = summary["roi"] + 0.0005 * len(subset)
    return score, backtest


def _optimize_settings(df: pd.DataFrame, base_min_rating: float, min_sample_bets: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = df.dropna(subset=["time_placed_iso"]).sort_values("time_placed_iso")
    cutoff = int(len(df) * 0.7)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:] if cutoff > 0 else df

    rating_quantiles = np.atleast_1d(np.nanquantile(train["os_rating_pred"], [0.2, 0.4, 0.6, 0.8])).tolist()
    rating_candidates = sorted(set(rating_quantiles + [base_min_rating]))
    ev_quantiles = np.atleast_1d(np.nanquantile(train["ev"], [0.2, 0.4, 0.6, 0.8])).tolist()
    ev_candidates = sorted(set(ev_quantiles + [0.01, 0.02, 0.05]))
    odds_min_candidates = [1.4, 1.6, 1.8, 2.0]
    odds_max_candidates = [3.0, 4.0, 5.0, 6.0]

    def make_settings(min_os: float, ev_min: float, odds_min: float, odds_max: float, strategy: str, kelly_frac: float, cap: float) -> Dict[str, Any]:
        return {
            "minimum_os_rating": float(min_os),
            "minimum_ev": round(ev_min * 100, 2),
            "odds_range_min_decimal": round(odds_min, 3),
            "odds_range_max_decimal": round(odds_max, 3),
            "ev_age_max_minutes": 120,
            "time_to_event_start_max_hours": 48,
            "betting_size_strategy": strategy,
            "flat_unit_pct_bankroll": cap if strategy == "Flat" else None,
            "fractional_kelly": kelly_frac if strategy == "OS Kelly" else None,
            "max_bet_pct_bankroll": cap if strategy == "OS Kelly" else None,
        }

    best = (-np.inf, None)
    for min_os in rating_candidates:
        for ev_min in ev_candidates:
            for odds_min in odds_min_candidates:
                for odds_max in odds_max_candidates:
                    if odds_min >= odds_max:
                        continue
                    settings = make_settings(min_os, ev_min, odds_min, odds_max, "Flat", 0.25, 0.01)
                    score, _ = _evaluate_settings(test, settings, min_sample_bets)
                    if score > best[0]:
                        best = (score, (min_os, ev_min, odds_min, odds_max))

    if best[1] is None:
        min_os, ev_min, odds_min, odds_max = base_min_rating, 0.01, 1.4, 5.0
    else:
        min_os, ev_min, odds_min, odds_max = best[1]

    def objective(trial: optuna.Trial) -> float:
        min_os_trial = trial.suggest_float("min_os", max(2.0, min_os * 0.8), min(100.0, min_os * 1.2))
        ev_min_trial = trial.suggest_float("ev_min", max(0.0, ev_min * 0.5), ev_min * 1.5)
        odds_min_trial = trial.suggest_float("odds_min", 1.3, 2.5)
        odds_max_trial = trial.suggest_float("odds_max", 3.0, 8.0)
        if odds_min_trial >= odds_max_trial:
            return -np.inf
        settings = make_settings(min_os_trial, ev_min_trial, odds_min_trial, odds_max_trial, "Flat", 0.25, 0.01)
        score, _ = _evaluate_settings(test, settings, min_sample_bets)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    params = study.best_params if study.best_params else {"min_os": min_os, "ev_min": ev_min, "odds_min": odds_min, "odds_max": odds_max}

    stake_options = []
    for cap in [0.005, 0.01, 0.015, 0.02]:
        settings = make_settings(params["min_os"], params["ev_min"], params["odds_min"], params["odds_max"], "Flat", 0.25, cap)
        score, backtest = _evaluate_settings(test, settings, min_sample_bets)
        if score > -np.inf:
            stake_options.append((score, settings, backtest))
    for kelly_frac in [0.125, 0.25, 0.5]:
        for cap in [0.01, 0.02, 0.03]:
            settings = make_settings(params["min_os"], params["ev_min"], params["odds_min"], params["odds_max"], "OS Kelly", kelly_frac, cap)
            score, backtest = _evaluate_settings(test, settings, min_sample_bets)
            if score > -np.inf:
                stake_options.append((score, settings, backtest))

    if not stake_options:
        settings = make_settings(params["min_os"], params["ev_min"], params["odds_min"], params["odds_max"], "Flat", 0.25, 0.01)
        _, backtest = _evaluate_settings(test, settings, min_sample_bets)
        return settings, backtest

    _, settings, backtest = max(stake_options, key=lambda x: x[0])
    return settings, backtest


def optimize(transactions: pd.DataFrame, os_markets: pd.DataFrame, min_bets: int = 50) -> List[PortfolioResult]:
    combo_table, _, _ = build_tx_tables(transactions)
    shrunk = shrink_combo_performance(combo_table, os_markets)

    core = shrunk[(shrunk["bets"] >= min_bets) & (shrunk["roi_shrunk"] >= 0.03) & (shrunk["avg_clv"] >= 0.02) & (shrunk["worst_decile_clv"] >= -0.05)]
    expansion = shrunk[(shrunk["bets"] >= min_bets) & (shrunk["roi_shrunk"] >= -0.01) & (shrunk["avg_clv"] >= 0.01) & (shrunk["worst_decile_clv"] >= -0.07)]
    experimental = shrunk[(shrunk["bets"] < min_bets) & (shrunk["roi_shrunk"] >= 0.0) & (shrunk["avg_clv"] >= 0.02) & (shrunk["worst_decile_clv"] >= -0.05)]

    portfolios = []
    for name, selection, min_rating, min_sample in [
        ("Core", core, transactions["os_rating_pred"].quantile(0.4), 50),
        ("Expansion", expansion, transactions["os_rating_pred"].quantile(0.25), 80),
    ]:
        combos = selection[["league_norm", "market_norm", "sportsbook", "roi_shrunk", "clv_shrunk", "bets", "avg_os_rating"]]
        combos = combos.sort_values(["roi_shrunk", "clv_shrunk"], ascending=False)
        combo_records = combos.to_dict("records")
        if combos.empty:
            settings, backtest = _optimize_settings(transactions, float(min_rating), min_sample)
        else:
            filtered = transactions.merge(
                combos[["league_norm", "market_norm", "sportsbook"]],
                on=["league_norm", "market_norm", "sportsbook"],
                how="inner",
            )
            settings, backtest = _optimize_settings(filtered, float(min_rating), min_sample)
        portfolios.append(PortfolioResult(name=name, settings=settings, portfolio_markets=combo_records, backtest=backtest))

    if not experimental.empty:
        combos = experimental[["league_norm", "market_norm", "sportsbook", "roi_shrunk", "clv_shrunk", "bets", "avg_os_rating"]]
        combos = combos.sort_values(["roi_shrunk", "clv_shrunk"], ascending=False)
        combo_records = combos.to_dict("records")
        filtered = transactions.merge(
            combos[["league_norm", "market_norm", "sportsbook"]],
            on=["league_norm", "market_norm", "sportsbook"],
            how="inner",
        )
        settings, backtest = _optimize_settings(filtered, float(transactions["os_rating_pred"].quantile(0.3)), 10)
        portfolios.append(PortfolioResult(name="Experimental", settings=settings, portfolio_markets=combo_records, backtest=backtest))

    return portfolios
