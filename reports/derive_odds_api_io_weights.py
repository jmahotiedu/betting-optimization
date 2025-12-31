from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _normalize_probs(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        return [math.nan for _ in probs]
    return [p / total for p in probs]


def _iter_events(payload) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "response" in payload:
            return _iter_events(payload["response"])
        if "bookmakers" in payload:
            return [payload]
    return []


def _extract_records(events: List[dict]) -> List[dict]:
    records: List[dict] = []
    for event in events:
        event_id = event.get("id")
        bookmakers = event.get("bookmakers", {}) or {}
        for book, markets in bookmakers.items():
            for market in markets or []:
                name = str(market.get("name") or "").strip()
                odds_list = market.get("odds") or []
                if name == "ML":
                    for line in odds_list:
                        home = _to_float(line.get("home"))
                        away = _to_float(line.get("away"))
                        draw = _to_float(line.get("draw")) if "draw" in line else math.nan
                        if math.isnan(home) or math.isnan(away) or home <= 1 or away <= 1 or not math.isnan(draw):
                            continue
                        records.extend(
                            [
                                {
                                    "event_id": event_id,
                                    "market": "ML",
                                    "line": None,
                                    "book": book,
                                    "outcome": "home",
                                    "decimal": home,
                                },
                                {
                                    "event_id": event_id,
                                    "market": "ML",
                                    "line": None,
                                    "book": book,
                                    "outcome": "away",
                                    "decimal": away,
                                },
                            ]
                        )
                elif name == "Spread":
                    for line in odds_list:
                        hdp = _to_float(line.get("hdp"))
                        home = _to_float(line.get("home"))
                        away = _to_float(line.get("away"))
                        if math.isnan(hdp) or math.isnan(home) or math.isnan(away) or home <= 1 or away <= 1:
                            continue
                        records.extend(
                            [
                                {
                                    "event_id": event_id,
                                    "market": "Spread",
                                    "line": abs(hdp),
                                    "book": book,
                                    "outcome": "home",
                                    "decimal": home,
                                },
                                {
                                    "event_id": event_id,
                                    "market": "Spread",
                                    "line": abs(hdp),
                                    "book": book,
                                    "outcome": "away",
                                    "decimal": away,
                                },
                            ]
                        )
                elif name == "Totals":
                    for line in odds_list:
                        hdp = _to_float(line.get("hdp"))
                        over = _to_float(line.get("over"))
                        under = _to_float(line.get("under"))
                        if math.isnan(hdp) or math.isnan(over) or math.isnan(under) or over <= 1 or under <= 1:
                            continue
                        records.extend(
                            [
                                {
                                    "event_id": event_id,
                                    "market": "Totals",
                                    "line": abs(hdp),
                                    "book": book,
                                    "outcome": "over",
                                    "decimal": over,
                                },
                                {
                                    "event_id": event_id,
                                    "market": "Totals",
                                    "line": abs(hdp),
                                    "book": book,
                                    "outcome": "under",
                                    "decimal": under,
                                },
                            ]
                        )
    return records


def _build_devig(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["event_id", "market", "line", "book"])
    rows: List[dict] = []
    for (event_id, market, line, book), group in grouped:
        grouped_outcomes = group.groupby("outcome")["decimal"].mean()
        if len(grouped_outcomes) != 2:
            continue
        outcomes = grouped_outcomes.index.tolist()
        decimals = grouped_outcomes.values.tolist()
        implied = [1.0 / d for d in decimals]
        devig = _normalize_probs(implied)
        overround = sum(implied) - 1.0
        for outcome, prob in zip(outcomes, devig):
            rows.append(
                {
                    "event_id": event_id,
                    "market": market,
                    "line": line,
                    "book": book,
                    "outcome": outcome,
                    "devig_prob": prob,
                    "overround": overround,
                }
            )
    return pd.DataFrame(rows)


def _compute_weights(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, dict]]:
    if df.empty:
        return {}, {}
    consensus = df.groupby(["event_id", "market", "line", "outcome"])["devig_prob"].mean().rename("consensus_prob")
    merged = df.join(consensus, on=["event_id", "market", "line", "outcome"])
    merged["abs_diff"] = (merged["devig_prob"] - merged["consensus_prob"]).abs()

    stats = (
        merged.groupby("book")
        .agg(observations=("abs_diff", "size"), avg_abs_diff=("abs_diff", "mean"), avg_overround=("overround", "mean"))
        .reset_index()
    )
    stats["score"] = stats.apply(
        lambda r: (1.0 / (r["avg_abs_diff"] + 1e-6))
        * (1.0 / (r["avg_overround"] + 1e-6))
        * math.sqrt(r["observations"])
        if r["observations"]
        else 0.0,
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
            "avg_overround": float(row["avg_overround"]),
            "score": float(row["score"]),
        }
        for _, row in stats.iterrows()
    }
    return weights, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive bookmaker weights from Odds-API.io snapshots.")
    parser.add_argument(
        "--snapshots-dir",
        default="reports/odds_api_io_snapshots",
        help="Comma-separated snapshot directories with JSON files.",
    )
    parser.add_argument("--out", default="reports/odds_api_io_book_weights.json", help="Output JSON path.")
    args = parser.parse_args()

    snapshot_dirs = [Path(p.strip()) for p in args.snapshots_dir.split(",") if p.strip()]
    rows: List[dict] = []
    for snapshot_dir in snapshot_dirs:
        if not snapshot_dir.exists():
            continue
        for path in snapshot_dir.rglob("*.json"):
            payload = json.loads(path.read_text())
            events = _iter_events(payload)
            rows.extend(_extract_records(events))

    df = pd.DataFrame(rows)
    devig = _build_devig(df)
    weights, details = _compute_weights(devig)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "snapshots_dir": [str(p) for p in snapshot_dirs],
        "markets": ["ML", "Spread", "Totals"],
        "weights": weights,
        "book_stats": details,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
