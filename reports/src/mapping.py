from __future__ import annotations

import re
from typing import Optional

MARKET_PATTERNS = [
    (re.compile(r"\bspread\b", re.I), "Spread"),
    (re.compile(r"\btotal\b", re.I), "Total"),
    (re.compile(r"\bover\b|\bunder\b", re.I), "Total"),
    (re.compile(r"\bmoneyline\b|\bml\b", re.I), "Moneyline"),
    (re.compile(r"\b1h\b|\bfirst half\b", re.I), "First Half"),
    (re.compile(r"\bplayer\b|\bpoints\b|\brebounds\b|\bassists\b", re.I), "Player Props"),
    (re.compile(r"\bteam\b", re.I), "Team Props"),
]


def normalize_market(bet_info: Optional[str], bet_type: Optional[str]) -> str:
    text = " ".join([str(bet_info or ""), str(bet_type or "")]).strip()
    if not text:
        return "Unknown"
    for pattern, name in MARKET_PATTERNS:
        if pattern.search(text):
            return name
    return "Other"


def normalize_league(league: Optional[str], sport: Optional[str]) -> str:
    if league:
        return str(league).strip()
    if sport:
        return str(sport).strip()
    return "Unknown"
