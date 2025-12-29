from __future__ import annotations

import re
from typing import Optional

MARKET_PATTERNS = [
    (re.compile(r"\bpoints\s*\+\s*rebounds\s*\+\s*assists\b|\bPRA\b", re.I), "Points + Rebounds + Assists"),
    (re.compile(r"\bpoints\s*\+\s*rebounds\b", re.I), "Points + Rebounds"),
    (re.compile(r"\bpoints\s*\+\s*assists\b", re.I), "Points + Assists"),
    (re.compile(r"\brebounds\s*\+\s*assists\b", re.I), "Rebounds + Assists"),
    (re.compile(r"\bsteals\s*\+\s*blocks\b", re.I), "Steals + Blocks"),
    (re.compile(r"\btotal\s+receiving\s+yards\b|\breceiving\s+yards\b", re.I), "Total Receiving Yards"),
    (re.compile(r"\btotal\s+rushing\s+yards\b|\brushing\s+yards\b", re.I), "Total Rushing Yards"),
    (re.compile(r"\btotal\s+receptions\b|\breceptions\b", re.I), "Total Receptions"),
    (re.compile(r"\btotal\s+shots\s+on\s+goal\b|\bshots\s+on\s+goal\b", re.I), "Total Shots on Goal"),
    (re.compile(r"\btotal\s+saves\b|\bsaves\b", re.I), "Total Saves"),
    (re.compile(r"\btotal\s+player\s+goals\b|\bplayer\s+goals\b", re.I), "Total Player Goals"),
    (re.compile(r"\bteam\s+total\s+goals\b", re.I), "Team Total Goals"),
    (re.compile(r"\btotal\s+strikeouts\b|\bstrikeouts\b", re.I), "Total Strikeouts"),
    (re.compile(r"\btotal\s+bases\b|\btotal\s+base\b|\bbases\b", re.I), "Total Bases"),
    (re.compile(r"\btotal\s+hits\b|\bhits\b", re.I), "Total Hits"),
    (re.compile(r"\btotal\s+runs\b|\bruns\b", re.I), "Total Runs"),
    (re.compile(r"\bteam\s+total\b", re.I), "Team Total"),
    (re.compile(r"\btotal\s+points\b", re.I), "Total Points"),
    (re.compile(r"\btotal\s+assists\b", re.I), "Total Assists"),
    (re.compile(r"\bpoints\b", re.I), "Points"),
    (re.compile(r"\brebounds\b", re.I), "Rebounds"),
    (re.compile(r"\bassists\b", re.I), "Assists"),
    (re.compile(r"\bblocks\b", re.I), "Blocks"),
    (re.compile(r"\bsteals\b", re.I), "Steals"),
    (re.compile(r"\bturnovers\b", re.I), "Turnovers"),
    (re.compile(r"\b3[-\s]?pointers\b|\b3pt\b|\b3ptm\b", re.I), "3-Pointers"),
    (re.compile(r"\brun\s+line\b", re.I), "Run Line"),
    (re.compile(r"\bpuck\s+line\b", re.I), "Puck Line"),
    (re.compile(r"\bpoint\s+spread\b|\bspread\b", re.I), "Point Spread"),
    (re.compile(r"\bmoneyline\b|\bml\b", re.I), "Moneyline"),
    (re.compile(r"\btotal\b|\bover\b|\bunder\b", re.I), "Total"),
]


def normalize_market(bet_info: Optional[str], bet_type: Optional[str] = None) -> str:
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
