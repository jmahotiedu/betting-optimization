from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .odds_utils import to_decimal, implied_prob_from_decimal, decimal_to_american
from .devig import devig
from .backtest import bankroll_simulation, summarize_performance


@dataclass
class OutlierProfile:
    name: str
    settings: Dict[str, Any]
    devig_weights: Dict[str, float]
    devig_weight_sources: Dict[str, str]
    backtest: pd.DataFrame


def _edge_from_method(df: pd.DataFrame, method: str) -> pd.Series:
    odds_decimal = df["odds"].apply(to_decimal)
    close_decimal = df["closing_line"].apply(to_decimal)
    implied = odds_decimal.apply(implied_prob_from_decimal)
    close_prob = close_decimal.apply(implied_prob_from_decimal)
    vig_free = implied.apply(lambda p: devig(method, [p, 1 - p])[0] if p > 0 else np.nan)
    return vig_free - close_prob


def _calibrate_weights(df: pd.DataFrame, devig_books: List[str], method: str, min_bets: int = 25) -> Tuple[Dict[str, float], Dict[str, str]]:
    weights = {}
    sources = {}
    mse_by_book = {}
    for book in devig_books:
        subset = df[df["sportsbook"] == book]
        subset = subset.dropna(subset=["closing_line", "odds"]).copy()
        if len(subset) < min_bets:
            continue
        implied = subset["odds"].apply(to_decimal).apply(implied_prob_from_decimal)
        close_prob = subset["closing_line"].apply(to_decimal).apply(implied_prob_from_decimal)
        devig_probs = implied.apply(lambda p: devig(method, [p, 1 - p])[0] if p > 0 else np.nan)
        mse = float(((devig_probs - close_prob) ** 2).mean())
        if mse == 0 or np.isnan(mse):
            continue
        mse_by_book[book] = mse

    if len(mse_by_book) < 2:
        equal_weight = 1.0 / len(devig_books) if devig_books else 0.0
        for book in devig_books:
            weights[book] = equal_weight
            sources[book] = "research-derived"
        return weights, sources

    inv = {book: 1 / mse for book, mse in mse_by_book.items()}
    total = sum(inv.values())
    for book in devig_books:
        if book in inv:
            weights[book] = inv[book] / total
            sources[book] = "data-derived"
        else:
            weights[book] = 0.0
            sources[book] = "insufficient-data"
    return weights, sources


def _stake_settings(df: pd.DataFrame) -> Tuple[str, float, float]:
    options = []
    for label, kelly_fraction in [("Full", 1.0), ("1/2", 0.5), ("1/4", 0.25), ("1/8", 0.125)]:
        for cap in [0.01, 0.02, 0.03]:
            backtest = bankroll_simulation(df, bankroll=1000, stake_strategy="os kelly", kelly_fraction=kelly_fraction, max_bet_pct=cap)
            summary = summarize_performance(backtest)
            max_drawdown = backtest["drawdown"].max() if not backtest.empty else 0.0
            if max_drawdown <= 0.25:
                options.append((summary["roi"], label, kelly_fraction, cap))
    if not options:
        return "1/4", 0.25, 0.02
    best = max(options, key=lambda x: x[0])
    return best[1], best[2], best[3]


def _thresholds(df: pd.DataFrame) -> Tuple[float, float]:
    ev_candidates = sorted(set(np.nanquantile(df["ev"], [0.2, 0.4, 0.6, 0.8]).tolist() + [0.01, 0.02, 0.05]))
    kelly_candidates = [0.0, 0.01, 0.02, 0.05]

    best = (-np.inf, 0.01, 0.0)
    for ev_min in ev_candidates:
        subset = df[df["ev"] >= ev_min]
        if subset.empty:
            continue
        kelly_pct = subset["kelly_pct"].fillna(0.0)
        for k_min in kelly_candidates:
            filtered = subset[kelly_pct >= k_min]
            if filtered.empty:
                continue
            backtest = bankroll_simulation(filtered, bankroll=1000, stake_strategy="flat", max_bet_pct=0.02)
            summary = summarize_performance(backtest)
            max_drawdown = backtest["drawdown"].max() if not backtest.empty else 0.0
            if max_drawdown > 0.25:
                continue
            if summary["roi"] > best[0]:
                best = (summary["roi"], ev_min, k_min)

    return float(best[1]), float(best[2])


def _odds_bounds(df: pd.DataFrame, lower_q: float, upper_q: float) -> Tuple[float, float]:
    odds = df["odds_decimal"].dropna()
    if odds.empty:
        return -200.0, 200.0
    lower = np.quantile(odds, lower_q)
    upper = np.quantile(odds, upper_q)
    return float(decimal_to_american(lower)), float(decimal_to_american(upper))


def optimize(transactions: pd.DataFrame) -> List[OutlierProfile]:
    df = transactions.copy()
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df["edge"] = df["ev"].fillna(0.0)
    df["kelly_pct"] = df.apply(
        lambda r: max(0.0, r["edge"]) / (r["odds_decimal"] - 1) if r["odds_decimal"] and r["odds_decimal"] > 1 else 0.0,
        axis=1,
    )

    methods = ["multiplicative", "additive", "power", "shin", "probit", "average", "worst case"]
    correlations = {}
    for method in methods:
        edge = _edge_from_method(df, method)
        corr = edge.corr(df["clv"])
        correlations[method] = float(corr) if corr == corr else 0.0
    best_method = max(correlations.items(), key=lambda x: x[1])[0]

    book_counts = df["sportsbook"].value_counts().head(6)
    devig_books = book_counts.index.tolist()

    weights, weight_sources = _calibrate_weights(df, devig_books, best_method)

    ev_min, kelly_min = _thresholds(df)
    kelly_label, kelly_fraction, cap = _stake_settings(df)

    core_odds_min, core_odds_max = _odds_bounds(df, 0.1, 0.9)
    exp_odds_min, exp_odds_max = _odds_bounds(df, 0.05, 0.95)

    leagues_by_volume = df["league_norm"].value_counts().index.tolist()
    core_leagues = leagues_by_volume[:3]
    expansion_leagues = leagues_by_volume[:6]

    bet_types = sorted({
        "Gamelines" if m in {"Moneyline", "Point Spread", "Total", "Team Total", "Run Line", "Puck Line"} else "Player Props"
        for m in df["market_norm"].dropna().unique()
    })

    core_settings = {
        "date_filter": "Any time",
        "leagues": core_leagues,
        "bet_types": bet_types,
        "devig_books": devig_books,
        "devig_method": best_method.title() if best_method != "worst case" else "Worst Case",
        "kelly_multiplier": kelly_label,
        "ev_min_pct": round(ev_min * 100, 2),
        "kelly_min_pct": round(kelly_min * 100, 2),
        "vig_max_pct": 4.0,
        "market_width_max": 40.0,
        "fair_value_min_american": round(core_odds_min, 0),
        "fair_value_max_american": round(core_odds_max, 0),
        "market_limits": "Not supported",
        "variation_max_pct": 3.0,
        "stake_cap_pct_bankroll": cap,
        "stake_kelly_fraction": kelly_fraction,
    }

    expansion_settings = {
        "date_filter": "This month",
        "leagues": expansion_leagues,
        "bet_types": bet_types,
        "devig_books": devig_books,
        "devig_method": best_method.title() if best_method != "worst case" else "Worst Case",
        "kelly_multiplier": kelly_label,
        "ev_min_pct": round(max(0.0, ev_min * 0.8) * 100, 2),
        "kelly_min_pct": round(max(0.0, kelly_min * 0.8) * 100, 2),
        "vig_max_pct": 6.0,
        "market_width_max": 50.0,
        "fair_value_min_american": round(exp_odds_min, 0),
        "fair_value_max_american": round(exp_odds_max, 0),
        "market_limits": "Not supported",
        "variation_max_pct": 4.0,
        "stake_cap_pct_bankroll": cap,
        "stake_kelly_fraction": kelly_fraction,
    }

    return [
        OutlierProfile("Core", core_settings, weights, weight_sources, df.copy()),
        OutlierProfile("Expansion", expansion_settings, weights, weight_sources, df.copy()),
    ]
