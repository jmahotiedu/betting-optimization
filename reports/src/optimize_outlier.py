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


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw.values())
    if total == 0:
        return {k: 0.0 for k in raw}
    return {k: v / total for k, v in raw.items()}


def _calibrate_weights(df: pd.DataFrame, books: List[str], method: str, min_bets: int = 50) -> Tuple[Dict[str, float], Dict[str, str]]:
    mse_by_book = {}
    sources = {}
    for book in books:
        subset = df[df["sportsbook"] == book].dropna(subset=["closing_line", "odds"]).copy()
        if len(subset) < min_bets:
            sources[book] = "insufficient-data"
            continue
        implied = subset["odds"].apply(to_decimal).apply(implied_prob_from_decimal)
        close_prob = subset["closing_line"].apply(to_decimal).apply(implied_prob_from_decimal)
        devig_probs = implied.apply(lambda p: devig(method, [p, 1 - p])[0] if p > 0 else np.nan)
        mse = float(((devig_probs - close_prob) ** 2).mean())
        if mse and not np.isnan(mse):
            mse_by_book[book] = mse
            sources[book] = "data-derived"

    if len(mse_by_book) < 2:
        for book in books:
            sources.setdefault(book, "research-derived")
        return {}, sources

    inv = {book: 1 / mse for book, mse in mse_by_book.items()}
    return _normalize_weights(inv), sources


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
        for k_min in kelly_candidates:
            filtered = subset[subset["kelly_pct"] >= k_min]
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


def _odds_bounds(df: pd.DataFrame, lower_q: float, upper_q: float, min_cap: float, max_cap: float) -> Tuple[float, float]:
    odds = df["odds_decimal"].dropna()
    if odds.empty:
        return min_cap, max_cap
    lower = decimal_to_american(float(np.quantile(odds, lower_q)))
    upper = decimal_to_american(float(np.quantile(odds, upper_q)))
    return max(min_cap, lower), min(max_cap, upper)


def _preset_priors() -> Dict[str, Dict[str, Any]]:
    gamelines = {
        "name": "Gamelines",
        "required_books": ["Pinnacle", "Circa", "BookMaker"],
        "optional_books": [],
        "weights": _normalize_weights({"Pinnacle": 1.0, "Circa": 1.0, "BookMaker": 1.0}),
        "devig_method": "Average",
        "variation_max_pct": 3.0,
        "market_width_max": 20.0,
        "vig_max_pct": 8.0,
        "fair_value_min": -200.0,
        "fair_value_max": 200.0,
        "kelly_multiplier": "1/2",
        "min_books_required": 2,
    }
    nba_props = {
        "name": "NBA Props",
        "required_books": ["FanDuel"],
        "optional_books": ["Pinnacle", "BookMaker", "DraftKings", "Caesars"],
        "weights": _normalize_weights({"FanDuel": 100.0, "Pinnacle": 25.0, "BookMaker": 25.0, "DraftKings": 25.0, "Caesars": 25.0}),
        "devig_method": "Average",
        "variation_max_pct": 3.0,
        "market_width_max": 40.0,
        "vig_max_pct": 8.0,
        "fair_value_min": -200.0,
        "fair_value_max": 200.0,
        "kelly_multiplier": "1/4",
        "min_books_required": 1,
    }
    return {"gamelines": gamelines, "nba_props": nba_props}


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

    gameline_markets = {"Moneyline", "Point Spread", "Total", "Team Total", "Run Line", "Puck Line"}
    df["bet_type"] = df["market_norm"].apply(lambda m: "Gamelines" if m in gameline_markets else "Player Props")

    priors = _preset_priors()
    gamelines_df = df[df["bet_type"] == "Gamelines"]
    props_df = df[df["bet_type"] == "Player Props"]

    core_prior = priors["gamelines"] if gamelines_df["clv"].mean() >= props_df["clv"].mean() else priors["nba_props"]
    expansion_prior = priors["nba_props"] if core_prior["name"] == "Gamelines" else priors["gamelines"]

    methods = ["multiplicative", "additive", "power", "shin", "probit", "average", "worst case"]
    correlations = {}
    for method in methods:
        edge = _edge_from_method(df, method)
        corr = edge.corr(df["clv"])
        correlations[method] = float(corr) if corr == corr else 0.0
    best_method = max(correlations.items(), key=lambda x: x[1])[0]

    profiles = []
    core_type = "Gamelines" if core_prior["name"] == "Gamelines" else "Player Props"
    expansion_type = "Gamelines" if expansion_prior["name"] == "Gamelines" else "Player Props"
    for name, prior, subset in [
        ("Core", core_prior, df[df["bet_type"] == core_type]),
        ("Expansion", expansion_prior, df[df["bet_type"] == expansion_type]),
    ]:
        subset = subset if not subset.empty else df
        ev_min, kelly_min = _thresholds(subset)
        kelly_label, kelly_fraction, cap = _stake_settings(subset)
        fair_min, fair_max = _odds_bounds(subset, 0.2 if name == "Core" else 0.1, 0.8 if name == "Core" else 0.9, prior["fair_value_min"], prior["fair_value_max"])

        books = prior["required_books"] + prior["optional_books"]
        weights, sources = _calibrate_weights(df, books, best_method)
        if not weights:
            weights = prior["weights"]
            sources = {book: "research-derived" for book in weights}

        settings = {
            "date_filter": "During the week" if name == "Core" else "This month",
            "leagues": df["league_norm"].value_counts().index.tolist()[:6],
            "bet_types": ["Gamelines"] if prior["name"] == "Gamelines" else ["Player Props"],
            "devig_required_books": prior["required_books"],
            "devig_optional_books": prior["optional_books"],
            "devig_min_books_required": prior["min_books_required"],
            "devig_method": best_method.title() if best_method != "worst case" else "Worst Case",
            "devig_weights": weights,
            "kelly_multiplier": kelly_label,
            "ev_min_pct": round(ev_min * 100, 2),
            "kelly_min_pct": round(kelly_min * 100, 2),
            "vig_max_pct": prior["vig_max_pct"],
            "market_width_max": prior["market_width_max"],
            "fair_value_min_american": round(fair_min, 0),
            "fair_value_max_american": round(fair_max, 0),
            "market_limits": "Not supported",
            "variation_max_pct": prior["variation_max_pct"],
            "stake_cap_pct_bankroll": cap,
            "stake_kelly_fraction": kelly_fraction,
        }

        filtered = subset[(subset["ev"] >= ev_min) & (subset["kelly_pct"] >= kelly_min)]
        backtest = bankroll_simulation(filtered, bankroll=1000, stake_strategy="os kelly", kelly_fraction=kelly_fraction, max_bet_pct=cap)
        profiles.append(OutlierProfile(name, settings, weights, sources, backtest))

    return profiles
