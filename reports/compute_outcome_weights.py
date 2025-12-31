from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SNAPSHOT_DIRS = [
    "reports/odds_api_io_snapshots",
    "reports/odds_api_io_snapshots_pinnacle",
    "reports/odds_api_io_snapshots_pinny_betonline",
    "reports/odds_api_io_snapshots_pinny_bookmaker",
    "reports/odds_api_io_snapshots_pinny_circa2",
]

RESULTS_PATH = Path("reports/event_results_aligned.csv")
OUTPUT_PATH = Path("reports/odds_api_io_book_weights.json")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_id"] = df["event_id"].astype(str)
    df["winner_home"] = np.where(df["home_score"] > df["away_score"], 1, 0)
    df = df.dropna(subset=["winner_home"])
    return df


def iter_snapshots() -> List[Path]:
    files: List[Path] = []
    for root in SNAPSHOT_DIRS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        files.extend(sorted(root_path.glob("*.json")))
    return files


def parse_snapshot(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    events = payload.get("response", [])
    rows: List[dict] = []
    for event in events:
        event_id = str(event.get("id"))
        sport = event.get("sport") or event.get("sport_key")
        bookmakers: Dict[str, list] = event.get("bookmakers", {}) or {}
        for book, markets in bookmakers.items():
            for market in markets or []:
                if market.get("name") != "ML":
                    continue
                odds_list = market.get("odds") or []
                for line in odds_list:
                    try:
                        home = float(line.get("home"))
                        away = float(line.get("away"))
                    except (TypeError, ValueError):
                        continue
                    if home <= 1 or away <= 1:
                        continue
                    implied_home = 1.0 / home
                    implied_away = 1.0 / away
                    total = implied_home + implied_away
                    if total <= 0:
                        continue
                    prob_home = implied_home / total
                    rows.append(
                        {
                            "event_id": event_id,
                            "sport_key": sport,
                            "book": book,
                            "prob_home": prob_home,
                        }
                    )
    return rows


def compute_weights(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, dict]]:
    if df.empty:
        return {}, {}
    # metrics
    eps = 1e-9
    metrics = (
        df.groupby("book")
        .apply(
            lambda g: pd.Series(
                {
                    "observations": len(g),
                    "brier": float(((g["prob_home"] - g["winner_home"]) ** 2).mean()),
                    "logloss": float(
                        -np.mean(
                            g["winner_home"] * np.log(g["prob_home"] + eps)
                            + (1 - g["winner_home"]) * np.log(1 - g["prob_home"] + eps)
                        )
                    ),
                }
            )
        )
        .reset_index()
    )
    metrics["score"] = metrics.apply(
        lambda r: (1.0 / (r["logloss"] + eps)) * math.sqrt(r["observations"]) if r["observations"] else 0.0, axis=1
    )
    total_score = metrics["score"].sum()
    if total_score <= 0:
        return {}, {}
    metrics["weight"] = metrics["score"] / total_score
    weights = {row["book"]: round(row["weight"], 6) for _, row in metrics.iterrows()}
    details = {
        row["book"]: {
            "observations": int(row["observations"]),
            "brier": float(row["brier"]),
            "logloss": float(row["logloss"]),
            "score": float(row["score"]),
        }
        for _, row in metrics.iterrows()
    }
    return weights, details


def main() -> None:
    results = load_results(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    if results.empty:
        print("No event results found; skipping weight computation.")
        return

    records: List[dict] = []
    for path in iter_snapshots():
        records.extend(parse_snapshot(path))

    snap_df = pd.DataFrame(records)
    if snap_df.empty:
        print("No snapshot records parsed.")
        return

    merged = snap_df.merge(results[["event_id", "winner_home"]], on="event_id", how="inner")
    weights, details = compute_weights(merged)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "odds_api_io_snapshots + event_results_aligned.csv",
        "weights": weights,
        "details": details,
        "coverage": {
            "events_scored": merged["event_id"].nunique(),
            "observations": len(merged),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote weights for {len(weights)} books to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
