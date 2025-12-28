from __future__ import annotations

import math
from typing import Iterable, Optional, Union


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
    if 0 < value < 1.01:
        return "decimal_minus_one"
    if value >= 1.01 and value <= 20:
        return "decimal"
    if abs(value) >= 100:
        return "american"
    return "unknown"


def _coerce_float(value: Optional[Union[str, float, int]]) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return float("nan")
        if text.startswith("+"):
            text = text[1:]
        try:
            return float(text)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_decimal(value: Optional[Union[str, float, int]]) -> float:
    value = _coerce_float(value)
    if math.isnan(value):
        return float("nan")
    kind = detect_odds_type(value)
    if kind == "decimal":
        return float(value)
    if kind == "decimal_minus_one":
        return float(value) + 1.0
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
