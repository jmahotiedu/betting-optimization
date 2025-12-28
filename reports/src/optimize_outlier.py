from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from .odds_utils import to_decimal, implied_prob_from_decimal
from .devig import devig


@dataclass
class OutlierProfile:
    name: str
    settings: Dict[str, Any]
    devig_weights: Dict[str, float]
    backtest: pd.DataFrame


def _predict_edge(df: pd.DataFrame, method: str) -> pd.Series:
    odds_decimal = df["odds"].apply(to_decimal)
    close_decimal = df["closing_line"].apply(to_decimal)
    implied = odds_decimal.apply(implied_prob_from_decimal)
    close_prob = close_decimal.apply(implied_prob_from_decimal)
    vig_free = implied.apply(lambda p: devig(method, [p, 1 - p])[0] if p > 0 else np.nan)
    return (vig_free - close_prob)


def optimize(transactions: pd.DataFrame) -> list[OutlierProfile]:
    df = transactions.copy()
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1

    methods = ["multiplicative", "additive", "power", "shin", "probit", "average", "worst case"]
    correlations = {}
    for method in methods:
        edge = _predict_edge(df, method)
        corr = edge.corr(df["clv"])
        correlations[method] = float(corr) if corr == corr else 0.0

    best_method = max(correlations.items(), key=lambda x: x[1])[0]

    devig_books = ["DraftKings", "FanDuel", "BetMGM", "PointsBet", "Caesars"]
    weights = {book: 1 / len(devig_books) for book in devig_books}

    core_settings = {
        "date_filter": "last_90_days",
        "leagues_filter": "primary leagues only",
        "bet_types": ["Gamelines", "Player Props"],
        "devig_books": devig_books,
        "devig_method": best_method,
        "ev_min_pct": 2.0,
        "kelly_min_pct": 1.0,
        "vig_max_pct": 4.0,
        "market_width_max_pct": 4.0,
        "fair_value_odds_min": 1.4,
        "fair_value_odds_max": 5.0,
        "market_limits_min": 5,
        "market_limits_max": 500,
        "variation_max": 3.0,
        "bet_size_strategy": {
            "type": "fractional_kelly",
            "kelly_multiplier": 0.25,
            "max_pct_bankroll": 0.02,
        },
    }

    expansion_settings = {
        "date_filter": "last_180_days",
        "leagues_filter": "expanded",
        "bet_types": ["Gamelines", "Player Props", "Team Props", "Game Props"],
        "devig_books": devig_books,
        "devig_method": best_method,
        "ev_min_pct": 1.0,
        "kelly_min_pct": 0.5,
        "vig_max_pct": 6.0,
        "market_width_max_pct": 6.0,
        "fair_value_odds_min": 1.3,
        "fair_value_odds_max": 8.0,
        "market_limits_min": 5,
        "market_limits_max": 250,
        "variation_max": 4.0,
        "bet_size_strategy": {
            "type": "fractional_kelly",
            "kelly_multiplier": 0.2,
            "max_pct_bankroll": 0.025,
        },
    }

    return [
        OutlierProfile("Core", core_settings, weights, df.copy()),
        OutlierProfile("Expansion", expansion_settings, weights, df.copy()),
    ]
