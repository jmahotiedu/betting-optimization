from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .odds_utils import to_decimal
from .mapping import normalize_market, normalize_league
from .backtest import bankroll_simulation, summarize_performance


@dataclass
class PortfolioResult:
    name: str
    settings: Dict[str, Any]
    portfolio_markets: List[Dict[str, Any]]
    backtest: pd.DataFrame


def build_market_performance(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["market_norm"] = df.apply(lambda r: normalize_market(r.get("bet_info"), r.get("type")), axis=1)
    df["league_norm"] = df.apply(lambda r: normalize_league(r.get("leagues"), r.get("sports")), axis=1)
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df["time_placed_iso"] = pd.to_datetime(df["time_placed_iso"], errors="coerce", utc=True)
    max_time = df["time_placed_iso"].max()
    df["recency_days"] = (max_time - df["time_placed_iso"]).dt.days.fillna(0)
    df["recency_weight"] = np.exp(-df["recency_days"] / 30)

    grouped = df.groupby(["market_norm", "league_norm"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "profit": (g["profit"] * g["recency_weight"]).sum(),
                "stake": (g["amount"] * g["recency_weight"]).sum(),
                "roi": (g["profit"] * g["recency_weight"]).sum() / (g["amount"] * g["recency_weight"]).sum()
                if (g["amount"] * g["recency_weight"]).sum()
                else 0.0,
                "avg_clv": (g["clv"] * g["recency_weight"]).sum() / g["recency_weight"].sum(),
            }
        )
    )
    return grouped.reset_index()


def shrink_market_performance(market_perf: pd.DataFrame, os_markets: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    os_markets = os_markets.rename(columns={"market": "market_norm", "league": "league_norm"})
    merged = os_markets.merge(market_perf, on=["market_norm", "league_norm"], how="left")
    merged["bets"] = merged["bets"].fillna(0.0)
    merged["roi"] = merged["roi"].fillna(0.0)
    merged["avg_clv"] = merged["avg_clv"].fillna(0.0)
    merged["roi_pct"] = merged["roi_pct"].fillna(0.0)
    merged["roi_prior"] = merged["roi_pct"] / 100.0
    merged["roi_shrunk"] = (
        merged["roi"] * merged["bets"] + merged["roi_prior"] * k
    ) / (merged["bets"] + k)
    merged["clv_shrunk"] = (
        merged["avg_clv"] * merged["bets"] + 0.01 * k
    ) / (merged["bets"] + k)
    return merged


def _apply_filters(df: pd.DataFrame, ev_min: float, odds_min: float, odds_max: float) -> pd.DataFrame:
    return df[(df["ev"] >= ev_min) & (df["odds_decimal"] >= odds_min) & (df["odds_decimal"] <= odds_max)]


def optimize(transactions: pd.DataFrame, os_markets: pd.DataFrame) -> List[PortfolioResult]:
    df = transactions.copy()
    df["odds_decimal"] = df["odds"].apply(to_decimal)
    df["close_decimal"] = df["closing_line"].apply(to_decimal)
    df["clv"] = (df["odds_decimal"] / df["close_decimal"]) - 1
    df["time_placed_iso"] = pd.to_datetime(df["time_placed_iso"], errors="coerce", utc=True)
    df = df.dropna(subset=["time_placed_iso"]).sort_values("time_placed_iso")

    market_perf = build_market_performance(df)
    market_perf = shrink_market_performance(market_perf, os_markets)

    good_markets = market_perf[market_perf["roi_shrunk"] > 0].copy()
    portfolio_markets = (
        good_markets[["market_norm", "league_norm", "sportsbook", "roi_shrunk", "clv_shrunk"]]
        .dropna(subset=["sportsbook"])
        .sort_values(["roi_shrunk", "clv_shrunk"], ascending=False)
        .to_dict("records")
    )

    ev_grid = [0.01, 0.02, 0.05]
    odds_min_grid = [1.4, 1.7]
    odds_max_grid = [3.5, 6.0]

    best = None
    for ev_min in ev_grid:
        for odds_min in odds_min_grid:
            for odds_max in odds_max_grid:
                subset = _apply_filters(df, ev_min, odds_min, odds_max)
                if subset.empty:
                    continue
                backtest = bankroll_simulation(subset, bankroll=1000, stake_strategy="flat", max_bet_pct=0.02)
                summary = summarize_performance(backtest)
                max_drawdown = backtest["drawdown"].max() if not backtest.empty else 0.0
                if summary["avg_clv"] < 0.01 or max_drawdown > 0.35:
                    continue
                score = summary["roi"]
                if best is None or score > best[0]:
                    best = (score, ev_min, odds_min, odds_max, backtest)

    if best is None:
        ev_min, odds_min, odds_max = 0.01, 1.5, 5.0
        backtest = bankroll_simulation(df, bankroll=1000, stake_strategy="flat", max_bet_pct=0.02)
    else:
        _, ev_min, odds_min, odds_max, backtest = best

    settings = {
        "minimum_os_rating": 20,
        "minimum_ev": round(ev_min * 100, 2),
        "odds_range_min": round(odds_min, 2),
        "odds_range_max": round(odds_max, 2),
        "ev_age_max_minutes": 120,
        "time_to_event_start_max_hours": 48,
        "betting_size_strategy": "Flat",
        "flat_unit_pct_bankroll": 0.02,
        "fractional_kelly": None,
    }

    return [
        PortfolioResult(
            name="Primary",
            settings=settings,
            portfolio_markets=portfolio_markets,
            backtest=backtest,
        )
    ]
