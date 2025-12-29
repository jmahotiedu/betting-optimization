from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna

from .odds_utils import to_decimal, decimal_to_american, american_to_decimal
from .backtest import bankroll_simulation, summarize_performance


@dataclass
class PortfolioResult:
    name: str
    settings: Dict[str, Any]
    portfolio_markets: List[Dict[str, Any]]
    backtest: pd.DataFrame


EV_FLOOR = 0.01
TAIL_BINDING_STRICT = 0.05
TAIL_BINDING_RELAXED = 0.03


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


def _american_odds_series(df: pd.DataFrame) -> pd.Series:
    odds_american = df["odds_decimal"].apply(decimal_to_american).replace([np.inf, -np.inf], np.nan)
    return odds_american.dropna()


def _bound_limits(train: pd.DataFrame) -> Tuple[float, float, float, float] | None:
    odds_american = _american_odds_series(train)
    if odds_american.empty:
        return None
    q01, q25, q75, q99 = np.quantile(odds_american, [0.01, 0.25, 0.75, 0.99])
    return float(q01), float(q25), float(q75), float(q99)


def _binding_rates(df: pd.DataFrame, odds_min: float, odds_max: float, ev_min: float | None = None) -> Tuple[float, float]:
    if ev_min is not None:
        df = df[(df["ev"].notna()) & (df["ev"] >= ev_min)]
    odds_american = _american_odds_series(df)
    if odds_american.empty:
        return 0.0, 0.0
    min_bound = decimal_to_american(odds_min)
    max_bound = decimal_to_american(odds_max)
    lower_tail = (odds_american <= min_bound).mean()
    upper_tail = (odds_american >= max_bound).mean()
    return float(lower_tail), float(upper_tail)


def _evaluate_settings(
    df: pd.DataFrame,
    ev_min: float,
    odds_min: float,
    odds_max: float,
    stake_strategy: str,
    kelly_fraction: float,
    max_bet_pct: float,
    bound_limits: Tuple[float, float, float, float] | None,
) -> Tuple[float, pd.DataFrame]:
    if ev_min < EV_FLOOR:
        return -np.inf, pd.DataFrame()
    min_bound = decimal_to_american(odds_min)
    max_bound = decimal_to_american(odds_max)
    if bound_limits is not None:
        q01, q25, q75, q99 = bound_limits
        if min_bound < q01 or min_bound > q25:
            return -np.inf, pd.DataFrame()
        if max_bound < q75 or max_bound > q99:
            return -np.inf, pd.DataFrame()
    subset = df[(df["ev"].notna()) & (df["ev"] >= ev_min)]
    if subset.empty:
        return -np.inf, pd.DataFrame()
    lower_tail, upper_tail = _binding_rates(subset, odds_min, odds_max, ev_min=None)
    if not (
        (lower_tail >= TAIL_BINDING_STRICT and upper_tail >= TAIL_BINDING_STRICT)
        or (lower_tail >= TAIL_BINDING_RELAXED and upper_tail >= TAIL_BINDING_RELAXED)
    ):
        return -np.inf, pd.DataFrame()
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
    bound_limits = _bound_limits(train)

    ev_candidates = sorted(set(np.nanquantile(train["ev"], [0.2, 0.4, 0.6, 0.8]).tolist() + [0.01, 0.02, 0.05]))
    odds_min_candidates = [1.4, 1.6, 1.8, 2.0]
    odds_max_candidates = [3.0, 4.0, 5.0, 6.0]

    best = (-np.inf, None)
    for ev_min in ev_candidates:
        for odds_min in odds_min_candidates:
            for odds_max in odds_max_candidates:
                if odds_min >= odds_max:
                    continue
                score, _ = _evaluate_settings(test, ev_min, odds_min, odds_max, "flat", 0.25, 0.02, bound_limits)
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
        score, _ = _evaluate_settings(test, ev_min_trial, odds_min_trial, odds_max_trial, "flat", 0.25, 0.02, bound_limits)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params if study.best_params else {"ev_min": ev_min, "odds_min": odds_min, "odds_max": odds_max}

    ev_min = float(best_params["ev_min"])
    odds_min = float(best_params["odds_min"])
    odds_max = float(best_params["odds_max"])

    score_check, _ = _evaluate_settings(test, ev_min, odds_min, odds_max, "flat", 0.25, 0.02, bound_limits)
    if score_check == -np.inf and bound_limits is not None:
        q01, q25, q75, q99 = bound_limits
        odds_min_candidate = american_to_decimal(q25)
        odds_max_candidate = american_to_decimal(q75)
        if odds_min_candidate > 1 and odds_max_candidate > odds_min_candidate:
            odds_min = float(odds_min_candidate)
            odds_max = float(odds_max_candidate)
        ev_min = max(EV_FLOOR, ev_min)

    stake_options = []
    for max_bet_pct in [0.005, 0.01, 0.015, 0.02, 0.025]:
        score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, "flat", 0.25, max_bet_pct, bound_limits)
        if score > -np.inf:
            stake_options.append((score, "Flat", 0.25, max_bet_pct, backtest))
    for kelly_fraction in [0.125, 0.25, 0.5]:
        for max_bet_pct in [0.01, 0.02, 0.03]:
            score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, "os kelly", kelly_fraction, max_bet_pct, bound_limits)
            if score > -np.inf:
                stake_options.append((score, "OS Kelly", kelly_fraction, max_bet_pct, backtest))

    if not stake_options:
        stake_strategy, kelly_fraction, max_bet_pct = "Flat", 0.25, 0.02
    else:
        _, stake_strategy, kelly_fraction, max_bet_pct, _ = max(stake_options, key=lambda x: x[0])

    score, backtest = _evaluate_settings(test, ev_min, odds_min, odds_max, stake_strategy, kelly_fraction, max_bet_pct, bound_limits)
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


def _apply_os_settings(df: pd.DataFrame, settings: Dict[str, Any], combos: pd.DataFrame) -> pd.DataFrame:
    subset = df.merge(combos[["league_norm", "market_norm", "sportsbook"]], on=["league_norm", "market_norm", "sportsbook"], how="inner")
    ev_min = settings["minimum_ev"] / 100.0
    odds_min = settings["odds_range_min_decimal"]
    odds_max = settings["odds_range_max_decimal"]
    subset = subset[(subset["ev"] >= ev_min) & (subset["odds_decimal"] >= odds_min) & (subset["odds_decimal"] <= odds_max)]
    return subset


def _ensure_expansion_separation(
    core_result: PortfolioResult,
    expansion_result: PortfolioResult,
    transactions: pd.DataFrame,
) -> PortfolioResult:
    df = transactions.copy().dropna(subset=["time_placed_iso"]).sort_values("time_placed_iso")
    cutoff = int(len(df) * 0.7)
    holdout = df.iloc[cutoff:] if cutoff > 0 else df

    core_combos = pd.DataFrame(core_result.portfolio_markets)
    expansion_combos = pd.DataFrame(expansion_result.portfolio_markets)
    core_holdout = _apply_os_settings(holdout, core_result.settings, core_combos)
    expansion_holdout = _apply_os_settings(holdout, expansion_result.settings, expansion_combos)

    core_bets = len(core_holdout)
    expansion_bets = len(expansion_holdout)

    core_min = core_result.settings["odds_range_min_decimal"]
    core_max = core_result.settings["odds_range_max_decimal"]
    exp_min = expansion_result.settings["odds_range_min_decimal"]
    exp_max = expansion_result.settings["odds_range_max_decimal"]
    core_ev = core_result.settings["minimum_ev"]
    exp_ev = expansion_result.settings["minimum_ev"]

    wider_bounds = exp_min <= core_min * 0.95 and exp_max >= core_max * 1.05
    lower_ev = exp_ev <= core_ev * 0.85

    if core_bets > 0 and expansion_bets >= 2 * core_bets:
        return expansion_result
    if wider_bounds and lower_ev:
        return expansion_result

    filtered = transactions.merge(
        expansion_combos[["league_norm", "market_norm", "sportsbook"]],
        on=["league_norm", "market_norm", "sportsbook"],
        how="inner",
    )
    filtered = filtered.dropna(subset=["time_placed_iso"]).sort_values("time_placed_iso")
    train_cutoff = int(len(filtered) * 0.7)
    train = filtered.iloc[:train_cutoff] if train_cutoff > 0 else filtered
    limits = _bound_limits(train)
    if limits is not None:
        q01, q25, q75, q99 = limits
        exp_min_bound = q01
        exp_max_bound = q99
        exp_min_decimal = american_to_decimal(exp_min_bound)
        exp_max_decimal = american_to_decimal(exp_max_bound)
        if exp_min_decimal > 1 and exp_max_decimal > exp_min_decimal:
            expansion_result.settings["odds_range_min_decimal"] = round(exp_min_decimal, 3)
            expansion_result.settings["odds_range_max_decimal"] = round(exp_max_decimal, 3)

    lowered_ev = max(EV_FLOOR * 100, min(exp_ev, core_ev * 0.8))
    expansion_result.settings["minimum_ev"] = round(lowered_ev, 2)
    return expansion_result


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

    if len(portfolios) == 2:
        core_result = portfolios[0]
        expansion_result = portfolios[1]
        portfolios[1] = _ensure_expansion_separation(core_result, expansion_result, transactions)

    return portfolios
