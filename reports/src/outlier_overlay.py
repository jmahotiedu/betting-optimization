from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .backtest import bankroll_simulation, summarize_performance
from .odds_utils import decimal_to_american

GAMELINE_MARKETS = {"Moneyline", "Point Spread", "Total", "Team Total", "Run Line", "Puck Line"}


@dataclass
class OverlayCandidate:
    min_bets: int
    min_roi: float
    min_worst_decile_clv: float
    ev_min: float
    odds_quantiles: Tuple[float, float]
    time_window_days: int | None


def _bet_type(market: str) -> str:
    return "Gamelines" if market in GAMELINE_MARKETS else "Player Props"


def _apply_overlay_filters(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    filtered = df.copy()
    if settings.get("leagues"):
        filtered = filtered[filtered["league_norm"].isin(settings["leagues"])]
    if settings.get("markets"):
        filtered = filtered[filtered["market_norm"].isin(settings["markets"])]
    if settings.get("sportsbooks"):
        filtered = filtered[filtered["sportsbook"].isin(settings["sportsbooks"])]
    if settings.get("bet_types"):
        filtered = filtered[filtered["bet_type"].isin(settings["bet_types"])]

    ev_min = settings.get("ev_min_pct", 0.0) / 100.0
    filtered = filtered[filtered["ev"] >= ev_min]

    odds_min = settings.get("odds_min_decimal")
    odds_max = settings.get("odds_max_decimal")
    if odds_min is not None and odds_max is not None:
        filtered = filtered[(filtered["odds_decimal"] >= odds_min) & (filtered["odds_decimal"] <= odds_max)]

    time_window_days = settings.get("time_window_days")
    if time_window_days:
        latest = filtered["time_placed_iso"].max()
        if pd.notna(latest):
            cutoff = latest - pd.Timedelta(days=time_window_days)
            filtered = filtered[filtered["time_placed_iso"] >= cutoff]
    return filtered


def _evaluate_overlay(df: pd.DataFrame, settings: Dict[str, Any], enforce_drawdown: bool = True) -> Tuple[float | Tuple[float, float], dict, pd.DataFrame]:
    filtered = _apply_overlay_filters(df, settings)
    if filtered.empty:
        return -np.inf, {}, filtered
    backtest = bankroll_simulation(filtered, bankroll=1000, stake_strategy="flat", max_bet_pct=settings["stake_cap_pct_bankroll"])
    summary = summarize_performance(backtest)
    max_dd = backtest["drawdown"].max() if not backtest.empty else 0.0
    if enforce_drawdown and max_dd > settings["max_drawdown"]:
        return -np.inf, summary, backtest
    score = (summary["worst_decile_clv"], summary["roi"])
    return score, summary, backtest


def _build_combo_table(train: pd.DataFrame) -> pd.DataFrame:
    grouped = train.groupby(["league_norm", "market_norm", "sportsbook"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "roi": g["profit"].sum() / g["amount"].sum() if g["amount"].sum() else 0.0,
                "worst_decile_clv": g["clv"].quantile(0.1),
            }
        )
    )
    return grouped.reset_index()


def _odds_bounds(train: pd.DataFrame, lower_q: float, upper_q: float) -> Tuple[float, float]:
    odds = train["odds_decimal"].dropna()
    if odds.empty:
        return 1.01, 20.0
    lower = float(np.quantile(odds, lower_q))
    upper = float(np.quantile(odds, upper_q))
    if lower >= upper:
        lower = float(np.quantile(odds, 0.1))
        upper = float(np.quantile(odds, 0.9))
    return lower, upper


def _candidate_grid(profile: str) -> List[OverlayCandidate]:
    if profile == "core":
        return [
            OverlayCandidate(20, 0.02, 0.02, ev_min, odds_q, time_window)
            for ev_min in [0.01, 0.02, 0.05]
            for odds_q in [(0.2, 0.8), (0.25, 0.75)]
            for time_window in [30, 60, 90]
        ]
    return [
        OverlayCandidate(min_bets, min_roi, min_clv, ev_min, odds_q, time_window)
        for min_bets in [5, 10]
        for min_roi in [0.0, 0.01]
        for min_clv in [0.0, 0.01]
        for ev_min in [0.0, 0.01, 0.02]
        for odds_q in [(0.1, 0.9), (0.05, 0.95)]
        for time_window in [90, 180, 365, None]
    ]


def optimize_outlier_overlay(df: pd.DataFrame, profile: str = "core") -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = df.copy().dropna(subset=["time_placed_iso"])
    df = df.sort_values("time_placed_iso")
    df["bet_type"] = df["market_norm"].apply(_bet_type)
    cutoff = int(len(df) * 0.7)
    train = df.iloc[:cutoff]
    holdout = df.iloc[cutoff:] if cutoff > 0 else df

    combo_table = _build_combo_table(train)
    best_score = (-np.inf, -np.inf)
    best_settings = None
    best_backtest = holdout.copy()

    for candidate in _candidate_grid(profile):
        eligible = combo_table[
            (combo_table["bets"] >= candidate.min_bets)
            & (combo_table["roi"] >= candidate.min_roi)
            & (combo_table["worst_decile_clv"] >= candidate.min_worst_decile_clv)
        ]
        if eligible.empty:
            continue
        leagues = sorted(eligible["league_norm"].unique().tolist())
        markets = sorted(eligible["market_norm"].unique().tolist())
        sportsbooks = sorted(eligible["sportsbook"].unique().tolist())
        bet_types = sorted({_bet_type(m) for m in markets})
        odds_min, odds_max = _odds_bounds(train, *candidate.odds_quantiles)

        settings = {
            "profile": profile,
            "leagues": leagues,
            "markets": markets,
            "bet_types": bet_types,
            "sportsbooks": sportsbooks,
            "ev_min_pct": round(candidate.ev_min * 100, 2),
            "min_bets_per_combo": candidate.min_bets,
            "min_roi_per_combo": round(candidate.min_roi, 4),
            "min_worst_decile_clv_per_combo": round(candidate.min_worst_decile_clv, 4),
            "odds_min_decimal": round(odds_min, 3),
            "odds_max_decimal": round(odds_max, 3),
            "odds_min_american": round(decimal_to_american(odds_min), 0),
            "odds_max_american": round(decimal_to_american(odds_max), 0),
            "time_window_days": candidate.time_window_days,
            "stake_cap_pct_bankroll": 0.02,
            "max_drawdown": 0.25,
        }

        score, summary, backtest = _evaluate_overlay(holdout, settings)
        if score == -np.inf:
            continue
        if score > best_score:
            best_score = score
            best_settings = settings
            best_settings["drawdown_constraint_met"] = True
            best_backtest = backtest

    if best_settings is None:
        best_score = (-np.inf, -np.inf)
        for candidate in _candidate_grid(profile):
            eligible = combo_table[
                (combo_table["bets"] >= candidate.min_bets)
                & (combo_table["roi"] >= candidate.min_roi)
                & (combo_table["worst_decile_clv"] >= candidate.min_worst_decile_clv)
            ]
            if eligible.empty:
                continue
            leagues = sorted(eligible["league_norm"].unique().tolist())
            markets = sorted(eligible["market_norm"].unique().tolist())
            sportsbooks = sorted(eligible["sportsbook"].unique().tolist())
            bet_types = sorted({_bet_type(m) for m in markets})
            odds_min, odds_max = _odds_bounds(train, *candidate.odds_quantiles)

            settings = {
                "profile": profile,
                "leagues": leagues,
                "markets": markets,
                "bet_types": bet_types,
                "sportsbooks": sportsbooks,
                "ev_min_pct": round(candidate.ev_min * 100, 2),
                "min_bets_per_combo": candidate.min_bets,
                "min_roi_per_combo": round(candidate.min_roi, 4),
                "min_worst_decile_clv_per_combo": round(candidate.min_worst_decile_clv, 4),
                "odds_min_decimal": round(odds_min, 3),
                "odds_max_decimal": round(odds_max, 3),
                "odds_min_american": round(decimal_to_american(odds_min), 0),
                "odds_max_american": round(decimal_to_american(odds_max), 0),
                "time_window_days": candidate.time_window_days,
                "stake_cap_pct_bankroll": 0.02,
                "max_drawdown": 0.25,
            }

            score, summary, backtest = _evaluate_overlay(holdout, settings, enforce_drawdown=False)
            if score == -np.inf:
                continue
            if score > best_score:
                best_score = score
                best_settings = settings
                best_backtest = backtest

        if best_settings is not None:
            best_settings["drawdown_constraint_met"] = False
        else:
            best_settings = {
                "profile": profile,
                "leagues": sorted(df["league_norm"].unique().tolist()),
                "markets": sorted(df["market_norm"].unique().tolist()),
                "bet_types": sorted(df["bet_type"].unique().tolist()),
                "sportsbooks": sorted(df["sportsbook"].unique().tolist()),
                "ev_min_pct": 0.0,
                "min_bets_per_combo": 0,
                "min_roi_per_combo": 0.0,
                "min_worst_decile_clv_per_combo": 0.0,
                "odds_min_decimal": 1.01,
                "odds_max_decimal": 20.0,
                "odds_min_american": round(decimal_to_american(1.01), 0),
                "odds_max_american": round(decimal_to_american(20.0), 0),
                "time_window_days": None,
                "stake_cap_pct_bankroll": 0.02,
                "max_drawdown": 0.25,
                "drawdown_constraint_met": False,
            }
            best_backtest = holdout.copy()

    return best_settings, best_backtest
