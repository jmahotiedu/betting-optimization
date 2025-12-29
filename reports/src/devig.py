from __future__ import annotations

import math
from typing import Iterable, List

from scipy.stats import norm

from .odds_utils import normalize_probabilities


def _ensure_probs(probs: Iterable[float]) -> List[float]:
    probs = [float(p) for p in probs]
    return [p if p > 0 else 0.0 for p in probs]


def multiplicative(probs: Iterable[float]) -> List[float]:
    return normalize_probabilities(_ensure_probs(probs))


def additive(probs: Iterable[float]) -> List[float]:
    probs = _ensure_probs(probs)
    overround = sum(probs) - 1.0
    n = len(probs)
    adjusted = [max(0.0, p - overround / n) for p in probs]
    return normalize_probabilities(adjusted)


def power(probs: Iterable[float]) -> List[float]:
    probs = _ensure_probs(probs)
    if any(p <= 0 for p in probs):
        return multiplicative(probs)

    def f(k: float) -> float:
        return sum(p**k for p in probs) - 1.0

    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    adjusted = [p**k for p in probs]
    return normalize_probabilities(adjusted)


def shin(probs: Iterable[float]) -> List[float]:
    probs = _ensure_probs(probs)
    if any(p <= 0 for p in probs):
        return multiplicative(probs)

    def implied(z: float) -> List[float]:
        adjusted = []
        for p in probs:
            term = math.sqrt(z**2 + 4 * (1 - z) * p**2)
            q = (term - z) / (2 * (1 - z))
            adjusted.append(q)
        return adjusted

    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(implied(mid)) > 1:
            lo = mid
        else:
            hi = mid
    adjusted = implied((lo + hi) / 2)
    return normalize_probabilities(adjusted)


def probit(probs: Iterable[float]) -> List[float]:
    probs = _ensure_probs(probs)
    if any(p <= 0 or p >= 1 for p in probs):
        return multiplicative(probs)
    z = [norm.ppf(p) for p in probs]

    def f(shift: float) -> float:
        return sum(norm.cdf(z_i + shift) for z_i in z) - 1.0

    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2
    adjusted = [norm.cdf(z_i + shift) for z_i in z]
    return normalize_probabilities(adjusted)


def average(probs: Iterable[float]) -> List[float]:
    p1 = multiplicative(probs)
    p2 = additive(probs)
    return normalize_probabilities([(a + b) / 2 for a, b in zip(p1, p2)])


def worst_case(probs: Iterable[float]) -> List[float]:
    p1 = multiplicative(probs)
    p2 = additive(probs)
    adjusted = [min(a, b) for a, b in zip(p1, p2)]
    return normalize_probabilities(adjusted)


def devig(method: str, probs: Iterable[float]) -> List[float]:
    method = method.lower().strip()
    mapping = {
        "multiplicative": multiplicative,
        "additive": additive,
        "power": power,
        "shin": shin,
        "probit": probit,
        "average": average,
        "worst case": worst_case,
        "worst_case": worst_case,
    }
    if method not in mapping:
        raise ValueError(f"Unknown devig method: {method}")
    return mapping[method](probs)
