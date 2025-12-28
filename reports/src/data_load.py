from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["time_placed", "time_settled", "time_placed_iso", "time_settled_iso"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_os_markets(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_os_settings(path: str) -> Dict[str, Any]:
    text = Path(path).read_text()
    settings = {
        "raw": text,
        "os_rating_range": (2, 100),
        "ev_range": (0, 100),
        "odds_range": (-1500, 1500),
        "ev_age_minutes": (0, 720),
        "time_to_event_hours": (0, 504),
        "size_strategies": ["Flat", "OS Kelly"],
    }
    return settings


def load_optional_json(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())
