from __future__ import annotations

import numpy as np
import pandas as pd


def bankroll_simulation(df: pd.DataFrame, bankroll: float, stake_strategy: str, kelly_fraction: float = 0.25, max_bet_pct: float = 0.02) -> pd.DataFrame:
    df = df.sort_values("time_placed_iso")
    bankrolls = []
    drawdowns = []
    peak = bankroll
    for _, row in df.iterrows():
        odds = row["odds_decimal"]
        edge = row.get("edge", 0.0)
        if stake_strategy.lower() == "flat":
            stake = min(row.get("amount", bankroll * max_bet_pct), bankroll * max_bet_pct)
        else:
            kelly = max(0.0, edge) / (odds - 1) if odds > 1 else 0.0
            stake = bankroll * min(kelly * kelly_fraction, max_bet_pct)
        amount = row.get("amount", 0.0)
        profit = row["profit"]
        scaled_profit = profit * (stake / amount) if amount else 0.0
        bankroll += scaled_profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
        bankrolls.append(bankroll)
        drawdowns.append(drawdown)
    out = df.copy()
    out["bankroll"] = bankrolls
    out["drawdown"] = drawdowns
    return out


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(sample.mean())
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def summarize_performance(df: pd.DataFrame) -> dict:
    roi = df["profit"].sum() / df["amount"].sum() if df["amount"].sum() else 0.0
    avg_clv = df["clv"].mean()
    worst_decile = df["clv"].quantile(0.1)
    return {
        "roi": roi,
        "profit": df["profit"].sum(),
        "bets": len(df),
        "avg_clv": avg_clv,
        "worst_decile_clv": worst_decile,
    }
