from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

from .outlier_overlay import optimize_outlier_overlay


@dataclass
class OutlierPreset:
    name: str
    settings: Dict[str, Any]


@dataclass
class OutlierOverlay:
    name: str
    settings: Dict[str, Any]
    backtest: pd.DataFrame


def build_presets(weight_settings: Dict[str, Any]) -> List[OutlierPreset]:
    profiles = weight_settings.get("profiles", {})
    props = profiles.get("props_core", {})
    games = profiles.get("gamelines_expansion", {})

    core_settings = {
        "profile": "Props Core",
        "date_filter": "Any time",
        "bet_types": ["Player Props"],
        "required_books": props.get("required_books", []),
        "optional_books": props.get("optional_books", []),
        "min_books": props.get("min_books", 1),
        "weights": props.get("weights", {}),
        "devig_method": str(props.get("devig_method", "power")).title(),
        "variation_max_pct": 3.0,
        "vig_max_pct": None,
        "fair_value_max_american": None,
        "ev_min_pct": props.get("ev_min_pct", 1.0),
        "kelly_multiplier": "1/4",
        "weights_source": props.get("weights_source"),
        "weights_generated_at": weight_settings.get("generated_at"),
        "ev_min_pct_source": props.get("ev_min_pct_source"),
    }
    expansion_settings = {
        "profile": "Gamelines Expansion",
        "date_filter": "In the next 24 hours",
        "bet_types": ["Gamelines"],
        "required_books": games.get("required_books", []),
        "optional_books": games.get("optional_books", []),
        "min_books": games.get("min_books", 1),
        "weights": games.get("weights", {}),
        "devig_method": str(games.get("devig_method", "probit")).title(),
        "variation_max_pct": 3.0,
        "vig_max_pct": games.get("vig_max_pct", 8.0),
        "fair_value_max_american": games.get("fair_value_max_american", 200.0),
        "ev_min_pct": games.get("ev_min_pct", 1.0),
        "kelly_multiplier": "1/2",
        "weights_source": games.get("weights_source"),
        "weights_generated_at": weight_settings.get("generated_at"),
        "ev_min_pct_source": games.get("ev_min_pct_source"),
    }
    return [OutlierPreset("Core", core_settings), OutlierPreset("Expansion", expansion_settings)]


def build_overlays(transactions: pd.DataFrame) -> List[OutlierOverlay]:
    overlays = []
    for name in ["Core", "Expansion"]:
        settings, backtest = optimize_outlier_overlay(transactions, profile=name.lower())
        overlays.append(OutlierOverlay(name, settings, backtest))
    return overlays
