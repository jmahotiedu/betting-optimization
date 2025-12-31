from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.odds_utils import to_decimal


import re

_BOOK_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _book_key(name: str) -> str:
    cleaned = _BOOK_NORMALIZE.sub("", str(name).lower())
    cleaned = cleaned.replace("sportsbook", "")
    return cleaned


_BOOK_CANONICAL = {
    "Fanduel Sportsbook": "FanDuel",
    "Draftkings Sportsbook": "DraftKings",
    "Hard Rock Sportsbook": "Hard Rock",
    "Caesars Sportsbook": "Caesars",
}


def _canonical_book_name(name: str) -> str:
    text = str(name).strip()
    candidate = _BOOK_CANONICAL.get(text, text)
    return candidate if _book_key(candidate) == _book_key(text) else text


def _normalize_probs(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        return [math.nan for _ in probs]
    return [p / total for p in probs]


def _point_abs(outcome: dict) -> float | None:
    point = outcome.get("point")
    if point is None:
        return None
    try:
        return abs(float(point))
    except (TypeError, ValueError):
        return None


def _extract_records(payload: dict, markets: List[str]) -> List[dict]:
    records: List[dict] = []
    data = payload.get("data", payload)
    if not isinstance(data, list):
        return records
    for event in data:
        event_id = event.get("id")
        for book in event.get("bookmakers", []):
            book_name = _canonical_book_name(book.get("title") or book.get("key") or "")
            for market in book.get("markets", []):
                market_key = market.get("key")
                if market_key not in markets:
                    continue
                groups: Dict[float | None, List[dict]] = {}
                for outcome in market.get("outcomes", []):
                    groups.setdefault(_point_abs(outcome), []).append(outcome)
                for point_abs, outcomes in groups.items():
                    if len(outcomes) != 2:
                        continue
                    probs = []
                    for outcome in outcomes:
                        price = outcome.get("price")
                        decimal_price = to_decimal(price)
                        if math.isnan(decimal_price) or decimal_price <= 1:
                            probs = []
                            break
                        probs.append(1.0 / decimal_price)
                    if len(probs) != 2:
                        continue
                    devig = _normalize_probs(probs)
                    for outcome, prob in zip(outcomes, devig):
                        name = outcome.get("name")
                        records.append(
                            {
                                "event_id": event_id,
                                "market_key": market_key,
                                "point_abs": point_abs,
                                "outcome_name": name,
                                "book": book_name,
                                "devig_prob": prob,
                            }
                        )
    return records


def _load_snapshots(snapshot_dirs: List[Path], markets: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for snapshot_dir in snapshot_dirs:
        if not snapshot_dir.exists():
            continue
        for path in snapshot_dir.rglob("*.json"):
            content = json.loads(path.read_text())
            payload = content.get("response", content)
            rows.extend(_extract_records(payload, markets))
    if not rows:
        return pd.DataFrame(columns=["event_id", "market_key", "point_abs", "outcome_name", "book", "devig_prob"])
    return pd.DataFrame(rows)


def _compute_weights(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, dict]]:
    if df.empty:
        return {}, {}
    consensus = (
        df.groupby(["event_id", "market_key", "point_abs", "outcome_name"])["devig_prob"].mean().rename("consensus_prob")
    )
    merged = df.join(consensus, on=["event_id", "market_key", "point_abs", "outcome_name"])
    merged["abs_diff"] = (merged["devig_prob"] - merged["consensus_prob"]).abs()

    stats = merged.groupby("book").agg(observations=("abs_diff", "size"), avg_abs_diff=("abs_diff", "mean")).reset_index()
    stats["score"] = stats.apply(
        lambda r: (1.0 / (r["avg_abs_diff"] + 1e-6)) * math.sqrt(r["observations"]) if r["observations"] else 0.0,
        axis=1,
    )
    total_score = stats["score"].sum()
    if total_score <= 0:
        return {}, {}
    stats["weight"] = stats["score"] / total_score
    weights = {row["book"]: round(row["weight"], 6) for _, row in stats.iterrows()}
    details = {
        row["book"]: {
            "observations": int(row["observations"]),
            "avg_abs_diff": float(row["avg_abs_diff"]),
            "score": float(row["score"]),
        }
        for _, row in stats.iterrows()
    }
    return weights, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive bookmaker weights from Odds API snapshots.")
    parser.add_argument(
        "--snapshots-dir",
        default="reports/odds_api_snapshots",
        help="Comma-separated snapshot directories with JSON files.",
    )
    parser.add_argument("--markets", default="h2h,spreads,totals", help="Comma-separated markets to include.")
    parser.add_argument("--out", default="reports/odds_api_book_weights.json", help="Output JSON path.")
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    snapshot_dirs = [Path(p.strip()) for p in args.snapshots_dir.split(",") if p.strip()]
    df = _load_snapshots(snapshot_dirs, markets)
    weights, details = _compute_weights(df)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "snapshots_dir": [str(p) for p in snapshot_dirs],
        "markets": markets,
        "weights": weights,
        "book_stats": details,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
