from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

from .outlier_fair_value import preset_props_core, preset_gamelines_expansion
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


def build_presets() -> List[OutlierPreset]:
    core = preset_props_core()
    expansion = preset_gamelines_expansion()
    core_settings = {
        "profile": core.name,
        "date_filter": core.date_filter,
        "bet_types": list(core.bet_types),
        "required_books": list(core.required_books),
        "optional_books": list(core.optional_books),
        "min_books": core.min_books,
        "weights": core.weights,
        "devig_method": core.devig_method.title(),
        "variation_max_pct": core.variation_max_pct,
        "vig_max_pct": core.vig_max_pct,
        "fair_value_max_american": core.fair_value_max_american,
        "ev_min_pct": core.ev_min_pct,
        "kelly_multiplier": core.kelly_multiplier,
    }
    expansion_settings = {
        "profile": expansion.name,
        "date_filter": expansion.date_filter,
        "bet_types": list(expansion.bet_types),
        "required_books": list(expansion.required_books),
        "optional_books": list(expansion.optional_books),
        "min_books": expansion.min_books,
        "weights": expansion.weights,
        "devig_method": expansion.devig_method.title(),
        "variation_max_pct": expansion.variation_max_pct,
        "vig_max_pct": expansion.vig_max_pct,
        "fair_value_max_american": expansion.fair_value_max_american,
        "ev_min_pct": expansion.ev_min_pct,
        "kelly_multiplier": expansion.kelly_multiplier,
    }
    return [OutlierPreset("Core", core_settings), OutlierPreset("Expansion", expansion_settings)]


def build_overlays(transactions: pd.DataFrame) -> List[OutlierOverlay]:
    overlays = []
    for name in ["Core", "Expansion"]:
        settings, backtest = optimize_outlier_overlay(transactions, profile=name.lower())
        overlays.append(OutlierOverlay(name, settings, backtest))
    return overlays
