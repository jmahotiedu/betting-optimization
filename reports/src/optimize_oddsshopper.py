from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna

from .odds_utils import to_decimal
from .backtest import bankroll_simulation, summarize_performance


@dataclass
class PortfolioResult:
    name: str
    settings: Dict[str, Any]
    portfolio_markets: List[Dict[str, Any]]
    backtest: pd.DataFrame


def build_tx_tables(transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            }
        )
    )
    market_table = grouped_market.reset_index()

    grouped_book = transactions.groupby(["league_norm", "market_norm", "sportsbook"]).apply(
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
    return market_table, book_table


def shrink_book_performance(book_table: pd.DataFrame, os_markets: pd.DataFrame, k: float = 10.0) -> pd.DataFrame:
    os_markets = os_markets.rename(columns={"market": "market_norm", "league": "league_norm"})
    merged = os_markets.merge(book_table, on=["market_norm", "league_norm", "sportsbook"], how="left")
    merged["bets"] = merged["bets"].fillna(0.0)
    merged["roi"] = merged["roi"].fillna(0.0)
    merged["avg_clv"] = merged["avg_clv"].fillna(0.0)
    merged["roi_pct"] = merged["roi_pct"].fillna(0.0)
    merged["roi_prior"] = merged["roi_pct"] / 100.0
    merged["roi_shrunk"] = (merged["roi"] * merged["bets"] + merged["roi_prior"] * k) / (merged["bets"] + k)
    merged["clv_shrunk"] = (merged["avg_clv"] * merged["bets"] + 0.0 * k) / (merged["bets"] + k)
    return merged


def _evaluate_settings(df: pd.DataFrame, ev_min: float, odds_min: float, odds_max: float, stake_strategy: str, kelly_fraction: float, max_bet_pct: float) -> Tuple[float, pd.DataFrame]:
    subset = df[(df["ev"].notna()) & (df["ev"] >= ev_min)]
    subset = subset[(subset["odds_decimal"] >= odds_min) & (subset["odds_decimal"] <= odds_max)]
    if subset.empty:
        return -np.inf, pd.DataFrame()
    backtest = bankroll_simulation(
        subset,
        bankroll=1000,
        stake_strategy=stake_strategy,
        kelly_fraction=kelly_fraction,
        max_bet_pct=max_bet_pct,
    )
    summary = summarize_performance(backtest)
    max_drawdown = backtest["drawdown"].max() if not backtest.empty else 0.0
    if summary["avg_clv"] < 0 or summary["worst_decile_clv"] < -0.05 or max_drawdown > 0.25:
        return -np.inf, backtest
    score = summary["roi"]
    return score, backtest


def _optimize_settings(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = df.copy().dropna(subset=["time_placed_iso"]).sort_values("time_placed_iso")
    cutoff = int(len(df) * 0.7)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:] if cutoff > 0 else df

    ev_candidates = sorted(set(np.nanquantile(train["ev"], [0.2, 0.4, 0.6, 0.8]).tolist() + [0.01, 0.02, 0.05]))
    odds_min_candidates = [1.4, 1.6, 1.8, 2.0]
    odds_max_candidates = [3.0, 4.0, 5.0, 6.0]

    best = (-np.inf, None)
    for ev_min in ev_candidates:
        for odds_min in odds_min_candidates:
            for odds_max in odds_max_candidates:
                if odds_min >= odds_max:
                    continue
                score, _ = _evaluate_settings(test, ev_min, odds_min, odds_max, "flat", 0.25, 0.02)
                if score > best[0]:
                    best = (score, (ev_min, odds_min, odds_max))

    if best[1] is None:
        ev_min, odds_min, odds_max = 0.01, 1.4, 5.0
    else:
        ev_min, odds_min, odds_max = best[1]

    def objective(trial: optuna.Trial) -> float:
        ev_min_trial = trial.suggest_float("ev_min", max(0.0, ev_min * 0.5), ev_min * 1.5)
        odds_min_trial = trial.suggest_float("odds_min", 1.3, 2.2)
        odds_max_trial = trial.suggest_float("odds_max", 3.0, 8.0)
        if odds_min_trial >= odds_max_trial:
            return -np.inf
        score, _ = _evaluate_settings(test, ev_min_trial, odds_min_trial, odds_max_trial, "flat", 0.25, 0.02)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params if study.best_params else {"ev_min": ev_min, "odds_min": odds_min, "odds_max": odds_max}

    ev_min = float(best_params["ev_min"])
    odds_min = float(best_params["odds_min"])
    odds_max = float(best_params["odds_max"])

    stake_options = []
    for max_bet_pct in [0.005, 0.01, 0.015, 0.02, 0.025]:
        score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, "flat", 0.25, max_bet_pct)
        if score > -np.inf:
            stake_options.append((score, "Flat", 0.25, max_bet_pct, backtest))
    for kelly_fraction in [0.125, 0.25, 0.5]:
        for max_bet_pct in [0.01, 0.02, 0.03]:
            score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, "os kelly", kelly_fraction, max_bet_pct)
            if score > -np.inf:
                stake_options.append((score, "OS Kelly", kelly_fraction, max_bet_pct, backtest))

    if not stake_options:
        stake_strategy, kelly_fraction, max_bet_pct = "Flat", 0.25, 0.02
    else:
        _, stake_strategy, kelly_fraction, max_bet_pct, _ = max(stake_options, key=lambda x: x[0])

    score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, stake_strategy, kelly_fraction, max_bet_pct)
    if backtest.empty:
        backtest = test.copy()

    settings = {
        "minimum_os_rating": 20,
        "minimum_ev": round(ev_min * 100, 2),
        "odds_range_min_decimal": round(odds_min, 3),
        "odds_range_max_decimal": round(odds_max, 3),
        "ev_age_max_minutes": 120,
        "time_to_event_start_max_hours": 48,
        "betting_size_strategy": stake_strategy,
        "flat_unit_pct_bankroll": max_bet_pct if stake_strategy == "Flat" else None,
        "fractional_kelly": kelly_fraction if stake_strategy == "OS Kelly" else None,
        "max_bet_pct_bankroll": max_bet_pct if stake_strategy == "OS Kelly" else None,
    }
    return settings, backtest


def optimize(transactions: pd.DataFrame, os_markets: pd.DataFrame) -> List[PortfolioResult]:
    market_table, book_table = build_tx_tables(transactions)

    min_bets = max(5, int(np.quantile(book_table["bets"], 0.2))) if not book_table.empty else 5
    candidate = book_table[(book_table["bets"] >= min_bets)]
    candidate = candidate[
        (candidate["roi"] > 0)
        | ((candidate["avg_clv"] > 0) & (candidate["worst_decile_clv"] > -0.05))
    ]

    shrunk = shrink_book_performance(candidate, os_markets)
    shrunk = shrunk.sort_values(["roi_shrunk", "clv_shrunk"], ascending=False)

    core = shrunk[(shrunk["roi_shrunk"] >= 0) & (shrunk["avg_clv"] >= 0) & (shrunk["worst_decile_clv"] >= -0.02)]
    expansion = shrunk[(shrunk["roi_shrunk"] >= -0.01) & (shrunk["avg_clv"] >= -0.01) & (shrunk["worst_decile_clv"] >= -0.05)]

    portfolios = []
    for name, selection in [("Core", core), ("Expansion", expansion)]:
        combos = selection[["league_norm", "market_norm", "sportsbook", "roi_shrunk", "clv_shrunk", "bets"]]
        if combos.empty:
            combos = shrunk[["league_norm", "market_norm", "sportsbook", "roi_shrunk", "clv_shrunk", "bets"]]
        combo_records = combos.sort_values(["roi_shrunk", "clv_shrunk"], ascending=False).to_dict("records")

        filtered = transactions.merge(
            combos[["league_norm", "market_norm", "sportsbook"]],
            on=["league_norm", "market_norm", "sportsbook"],
            how="inner",
        )
        settings, backtest = _optimize_settings(filtered)

        portfolios.append(
            PortfolioResult(
                name=name,
                settings=settings,
                portfolio_markets=combo_records,
                backtest=backtest,
            )
        )

    return portfolios
