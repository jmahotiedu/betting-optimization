from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .devig import devig
from .odds_utils import to_decimal, implied_prob_from_decimal, decimal_to_american


@dataclass(frozen=True)
class OutlierPresetProfile:
    name: str
    required_books: Tuple[str, ...]
    optional_books: Tuple[str, ...]
    min_books: int
    weights: Dict[str, float]
    devig_method: str
    variation_max_pct: Optional[float]
    vig_max_pct: Optional[float]
    fair_value_max_american: Optional[float]
    ev_min_pct: float
    kelly_multiplier: str
    date_filter: str
    bet_types: Tuple[str, ...]


_SIDE_REGEX = re.compile(r"\b(over|under)\s+([0-9]+(?:\.[0-9]+)?)\s+(.*)", re.I)
_EVENT_SPLIT = re.compile(r"\s+(?:@|at|vs|v)\s+", re.I)
_BOOK_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _parse_two_way_info(bet_info: Optional[str]) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    if not bet_info:
        return None, None, None, None
    text = str(bet_info).strip()
    match = _SIDE_REGEX.search(text)
    if not match:
        return None, None, None, None
    side = match.group(1).lower()
    try:
        line = float(match.group(2))
    except ValueError:
        line = None
    remainder = match.group(3).strip()
    event = None
    selection = remainder
    parts = _EVENT_SPLIT.split(remainder, maxsplit=1)
    if len(parts) == 2:
        selection = parts[0].strip()
        event = parts[1].strip()
    return side, line, selection or None, event or None


def _book_key(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = _BOOK_NORMALIZE.sub("", str(name).lower())
    cleaned = cleaned.replace("sportsbook", "")
    return cleaned


def _market_key(
    league: Optional[str],
    market: Optional[str],
    line: Optional[float],
    selection: Optional[str],
    event: Optional[str],
) -> Optional[str]:
    if line is None or not selection:
        return None
    bits = [
        str(league or "").strip().lower(),
        str(market or "").strip().lower(),
        f"{line:.4f}",
        str(selection).strip().lower(),
        str(event or "").strip().lower(),
    ]
    return "|".join(bits)


def build_two_way_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        side, line, selection, event = _parse_two_way_info(row.get("bet_info"))
        key = _market_key(row.get("league_norm"), row.get("market_norm"), line, selection, event)
        if key is None or side is None:
            continue
        odds_decimal = to_decimal(row.get("odds"))
        implied = implied_prob_from_decimal(odds_decimal)
        if math.isnan(implied):
            continue
        rows.append(
            {
                "row_id": idx,
                "market_key": key,
                "sportsbook": row.get("sportsbook"),
                "sportsbook_key": _book_key(row.get("sportsbook")),
                "side": side,
                "implied_prob": implied,
                "odds_decimal": odds_decimal,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["row_id", "market_key", "sportsbook", "side", "implied_prob", "odds_decimal"])
    return pd.DataFrame(rows)


def compute_devig_probs(two_way: pd.DataFrame, method: str) -> pd.DataFrame:
    if two_way.empty:
        return pd.DataFrame(columns=["market_key", "sportsbook", "p_over", "p_under", "p_over_devig", "p_under_devig", "vig"])
    grouped = two_way.pivot_table(
        index=["market_key", "sportsbook", "sportsbook_key"],
        columns="side",
        values="implied_prob",
        aggfunc="mean",
    )
    if "over" not in grouped.columns or "under" not in grouped.columns:
        return pd.DataFrame(columns=["market_key", "sportsbook", "p_over", "p_under", "p_over_devig", "p_under_devig", "vig"])
    grouped = grouped.reset_index()
    grouped = grouped.dropna(subset=["over", "under"])
    rows = []
    for _, row in grouped.iterrows():
        p_over = float(row["over"])
        p_under = float(row["under"])
        devigged = devig(method, [p_over, p_under])
        p_over_devig, p_under_devig = devigged[0], devigged[1]
        vig = (p_over + p_under) - 1
        rows.append(
            {
                "market_key": row["market_key"],
                "sportsbook": row["sportsbook"],
                "sportsbook_key": row["sportsbook_key"],
                "p_over": p_over,
                "p_under": p_under,
                "p_over_devig": p_over_devig,
                "p_under_devig": p_under_devig,
                "vig": vig,
            }
        )
    return pd.DataFrame(rows)


def _normalize_weights(weights: Dict[str, float], books: Iterable[str]) -> Dict[str, float]:
    selected = {book: weights.get(book, 0.0) for book in books}
    total = sum(selected.values())
    if total <= 0:
        return {}
    return {book: weight / total for book, weight in selected.items()}


def apply_profile_fair_value(df: pd.DataFrame, profile: OutlierPresetProfile) -> pd.DataFrame:
    two_way = build_two_way_table(df)
    devig_table = compute_devig_probs(two_way, profile.devig_method)
    available_books = set(profile.required_books) | set(profile.optional_books)
    available_keys = {_book_key(book) for book in available_books}
    weight_keys = { _book_key(book): weight for book, weight in profile.weights.items() }
    required_keys = {_book_key(book) for book in profile.required_books}

    devig_by_key = {}
    if not devig_table.empty:
        devig_table = devig_table[devig_table["sportsbook_key"].isin(available_keys)]
        for _, row in devig_table.iterrows():
            key = row["market_key"]
            devig_by_key.setdefault(key, {})[row["sportsbook_key"]] = row

    out = df.copy()
    p_fair_list = []
    fair_decimal_list = []
    fair_american_list = []
    edge_list = []

    for idx, row in out.iterrows():
        side, line, selection, event = _parse_two_way_info(row.get("bet_info"))
        key = _market_key(row.get("league_norm"), row.get("market_norm"), line, selection, event)
        odds_decimal = to_decimal(row.get("odds"))
        sportsbook_key = _book_key(row.get("sportsbook"))
        p_fair = np.nan
        fair_decimal = np.nan
        fair_american = np.nan
        edge = np.nan

        if key and side in {"over", "under"}:
            books_for_key = devig_by_key.get(key, {})
            if books_for_key:
                if any(book not in books_for_key for book in required_keys):
                    books_for_key = {}
                if len(books_for_key) >= max(1, profile.min_books):
                    weights = _normalize_weights(weight_keys, books_for_key.keys())
                    if weights:
                        devig_probs = []
                        for book, values in books_for_key.items():
                            prob = values["p_over_devig"] if side == "over" else values["p_under_devig"]
                            devig_probs.append(prob)
                        devig_min = min(devig_probs)
                        devig_max = max(devig_probs)
                        variation = devig_max - devig_min
                        if profile.variation_max_pct is None or variation <= (profile.variation_max_pct / 100.0):
                            vig = np.mean([values["vig"] for values in books_for_key.values()])
                            if profile.vig_max_pct is None or vig <= (profile.vig_max_pct / 100.0):
                                p_fair = 0.0
                                for book, weight in weights.items():
                                    values = books_for_key[book]
                                    prob = values["p_over_devig"] if side == "over" else values["p_under_devig"]
                                    p_fair += weight * prob
            else:
                if sportsbook_key in available_keys and (not required_keys or sportsbook_key in required_keys):
                    implied_prob = implied_prob_from_decimal(odds_decimal)
                    if not math.isnan(implied_prob):
                        p_fair = implied_prob

        if p_fair and p_fair > 0:
            fair_decimal = 1.0 / p_fair
            fair_american = decimal_to_american(fair_decimal)
            if profile.fair_value_max_american is None or (
                not math.isnan(fair_american) and fair_american <= profile.fair_value_max_american
            ):
                if fair_decimal > 0 and not math.isnan(odds_decimal):
                    edge = (odds_decimal / fair_decimal) - 1
            else:
                p_fair = np.nan
                fair_decimal = np.nan
                fair_american = np.nan
                edge = np.nan

        p_fair_list.append(p_fair)
        fair_decimal_list.append(fair_decimal)
        fair_american_list.append(fair_american)
        edge_list.append(edge)

    out["p_fair"] = p_fair_list
    out["fair_decimal"] = fair_decimal_list
    out["fair_american"] = fair_american_list
    out["devig_edge"] = edge_list
    return out


def preset_props_core() -> OutlierPresetProfile:
    weights = {
        "FanDuel": 100.0,
        "Pinnacle": 25.0,
        "BookMaker": 25.0,
        "DraftKings": 25.0,
        "Caesars": 25.0,
    }
    return OutlierPresetProfile(
        name="Props Core",
        required_books=("FanDuel",),
        optional_books=("Pinnacle", "BookMaker", "DraftKings", "Caesars"),
        min_books=1,
        weights=weights,
        devig_method="average",
        variation_max_pct=3.0,
        vig_max_pct=None,
        fair_value_max_american=None,
        ev_min_pct=1.0,
        kelly_multiplier="1/4",
        date_filter="Any time",
        bet_types=("Player Props",),
    )


def preset_gamelines_expansion() -> OutlierPresetProfile:
    weights = {
        "Pinnacle": 1.0,
        "Circa": 1.0,
        "BookMaker": 1.0,
    }
    return OutlierPresetProfile(
        name="Gamelines Expansion",
        required_books=tuple(),
        optional_books=("Pinnacle", "Circa", "BookMaker"),
        min_books=1,
        weights=weights,
        devig_method="average",
        variation_max_pct=3.0,
        vig_max_pct=8.0,
        fair_value_max_american=200.0,
        ev_min_pct=1.0,
        kelly_multiplier="1/2",
        date_filter="In the next 24 hours",
        bet_types=("Gamelines",),
    )
