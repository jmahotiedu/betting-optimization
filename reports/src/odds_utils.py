from __future__ import annotations

import math
from typing import Iterable


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        return float("nan")
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return float("nan")
    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100
    return -100 / (decimal_odds - 1)


def detect_odds_type(value: float) -> str:
    if value is None or math.isnan(value):
        return "unknown"
    if value >= 1.01 and value <= 20:
        return "decimal"
    if abs(value) >= 100:
        return "american"
    return "unknown"


def to_decimal(value: float) -> float:
    if value is None or math.isnan(value):
        return float("nan")
    kind = detect_odds_type(value)
    if kind == "decimal":
        return float(value)
    if kind == "american":
        return american_to_decimal(float(value))
    return float(value)


def implied_prob_from_decimal(decimal_odds: float) -> float:
    if decimal_odds <= 1 or math.isnan(decimal_odds):
        return float("nan")
    return 1.0 / decimal_odds


def normalize_probabilities(probs: Iterable[float]) -> list[float]:
    probs = [max(0.0, float(p)) for p in probs]
    total = sum(probs)
    if total == 0:
        return [float("nan") for _ in probs]
    return [p / total for p in probs]
