from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .mapping import normalize_market, normalize_league
from .odds_utils import to_decimal
from .outlier_overlay import GAMELINE_MARKETS


import re

_BOOK_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _book_key(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = _BOOK_NORMALIZE.sub("", str(name).lower())
    cleaned = cleaned.replace("sportsbook", "")
    return cleaned


_BOOK_CANONICAL = {
    "Fanduel Sportsbook": "FanDuel",
    "Draftkings Sportsbook": "DraftKings",
    "Hard Rock Sportsbook": "Hard Rock",
    "Caesars Sportsbook": "Caesars",
}

# Books that should not be used for devig/fair-value weighting.
DISALLOWED_PROPS = {"Novig", "ProphetX", "Bally Bet"}
DISALLOWED_GAMELINES = {"Novig", "ProphetX", "Bally Bet"}

# Prior weights for props to include sharp books when transaction data is not used.
PROP_PRIORS = {
    "Pinnacle": 1.0,
    "BookMaker.eu": 0.6,
    "BetOnline.ag": 0.5,
    "FanDuel": 0.4,
    "DraftKings": 0.3,
    "Caesars": 0.25,
    "BetMGM": 0.25,
}
PROP_PRIOR_ALPHA = 1.0  # 1.0 => use priors only for props weighting
PROP_ML_ALPHA = 0.3  # blend factor for props ML-derived weights (if available)
PROP_ML_MIN_SAMPLES = 40
PROP_ML_RIDGE_LAMBDA = 1.0


def _canonical_book_name(name: Optional[str]) -> str:
    if not name:
        return ""
    text = str(name).strip()
    candidate = _BOOK_CANONICAL.get(text, text)
    return candidate if _book_key(candidate) == _book_key(text) else text


def ev_unit_from_gate(report_path: Optional[Path] = None) -> Optional[str]:
    path = report_path if report_path else Path(__file__).resolve().parents[1] / "ev_unit_gate.md"
    if not path.exists():
        return None
    text = path.read_text().lower()
    if "ev units determined: a (decimal ev)" in text:
        return "decimal"
    if "ev units determined: b (percent ev)" in text:
        return "percent"
    return None


def _coerce_ev_units(df: pd.DataFrame, ev_unit: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df.attrs["ev_unit_coerced"] = False
    df.attrs["ev_unit"] = ev_unit or "unknown"

    ev = pd.to_numeric(df.get("ev"), errors="coerce")
    if ev_unit == "decimal":
        df["ev"] = ev
        return df
    if ev_unit == "percent":
        df["ev"] = ev / 100.0
        df.attrs["ev_unit_coerced"] = True
        return df

    max_ev = ev.max()
    min_ev = ev.min()
    if pd.isna(max_ev):
        df["ev"] = ev
        return df
    if max_ev > 1.0 and max_ev <= 100 and min_ev >= 0:
        df["ev"] = ev / 100.0
        df.attrs["ev_unit_coerced"] = True
        return df
    df["ev"] = ev
    return df


def prepare_transactions(df: pd.DataFrame, ev_unit: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "type" in out.columns:
        out = out[out["type"].astype(str).str.lower() == "straight"]
    out["time_placed_iso"] = pd.to_datetime(out.get("time_placed_iso"), errors="coerce", utc=True)
    out["market_norm"] = out.apply(lambda r: normalize_market(r.get("bet_info"), r.get("type")), axis=1)
    out["league_norm"] = out.apply(lambda r: normalize_league(r.get("leagues"), r.get("sports")), axis=1)
    out["bet_type"] = out["market_norm"].apply(lambda m: "Gamelines" if m in GAMELINE_MARKETS else "Player Props")
    out["odds_decimal"] = out["odds"].apply(to_decimal)
    out["close_decimal"] = out["closing_line"].apply(to_decimal)
    out["clv"] = (out["odds_decimal"] / out["close_decimal"]) - 1
    out.loc[out["close_decimal"].isna(), "clv"] = np.nan
    out["amount"] = out["amount"].abs()
    out["profit"] = out["profit"].fillna(0.0)
    out = _coerce_ev_units(out, ev_unit)
    out["sportsbook"] = out["sportsbook"].apply(_canonical_book_name)
    return out


def _book_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["bet_type", "sportsbook"]).apply(
        lambda g: pd.Series(
            {
                "bets": len(g),
                "stake": g["amount"].sum(),
                "profit": g["profit"].sum(),
                "roi": g["profit"].sum() / g["amount"].sum() if g["amount"].sum() else 0.0,
                "avg_clv": g["clv"].mean(),
                "median_clv": g["clv"].median(),
            }
        )
    )
    return grouped.reset_index()


def _weights_from_stats(
    stats: pd.DataFrame,
    sample_floor: int,
    shrink: int,
    roi_weight: float,
) -> pd.DataFrame:
    stats = stats.copy()
    stats["bets"] = stats["bets"].fillna(0)
    stats["avg_clv"] = stats["avg_clv"].fillna(0.0)
    stats["roi"] = stats["roi"].fillna(0.0)
    stats["sample_factor"] = stats["bets"] / (stats["bets"] + shrink)
    stats["signal"] = stats["avg_clv"].clip(lower=0.0) + roi_weight * stats["roi"].clip(lower=0.0)
    stats["score"] = stats["signal"] * np.sqrt(stats["bets"].clip(lower=0)) * stats["sample_factor"]
    stats.loc[stats["bets"] < sample_floor, "score"] = 0.0
    return stats


def _combine_weights(total_stats: pd.DataFrame, today_stats: pd.DataFrame, alpha: float) -> pd.DataFrame:
    merged = total_stats.merge(
        today_stats[["bet_type", "sportsbook", "score"]].rename(columns={"score": "score_today"}),
        on=["bet_type", "sportsbook"],
        how="left",
    )
    merged["score_today"] = merged["score_today"].fillna(0.0)
    merged["score_combined"] = merged["score"] + alpha * merged["score_today"]
    return merged


def _normalize_weights(group: pd.DataFrame, min_weight: float) -> Dict[str, float]:
    total = group["score_combined"].sum()
    if total <= 0:
        return {}
    weights = (group.set_index("sportsbook")["score_combined"] / total).to_dict()
    weights = {k: v for k, v in weights.items() if v >= min_weight}
    if not weights:
        return {}
    norm = sum(weights.values())
    return {book: weight / norm for book, weight in weights.items()}


def _canonicalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    merged: Dict[str, float] = {}
    for book, weight in weights.items():
        canonical = _canonical_book_name(book)
        key = _book_key(canonical)
        if not key:
            continue
        merged.setdefault(canonical, 0.0)
        merged[canonical] += float(weight)
    total = sum(merged.values())
    if total <= 0:
        return {}
    return {book: weight / total for book, weight in merged.items()}


def _blend_weight_maps(
    base_weights: Dict[str, float],
    external_weights: Dict[str, float],
    alpha: float,
    min_weight: float,
    allowed_books: set[str],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for book in allowed_books:
        base = base_weights.get(book, 0.0)
        ext = external_weights.get(book, 0.0)
        combined[book] = (1 - alpha) * base + alpha * ext
    combined = {book: weight for book, weight in combined.items() if weight > 0}
    total = sum(combined.values())
    if total <= 0:
        return {}
    normalized = {book: weight / total for book, weight in combined.items()}
    filtered = {book: weight for book, weight in normalized.items() if weight >= min_weight}
    if not filtered:
        return {}
    norm = sum(filtered.values())
    return {book: weight / norm for book, weight in filtered.items()}


def _load_external_weights(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload.get("weights", {}) if isinstance(payload, dict) else {}


def _latest_date(df: pd.DataFrame) -> Optional[str]:
    if df.empty or "time_placed_iso" not in df.columns:
        return None
    latest = df["time_placed_iso"].dropna().max()
    if pd.isna(latest):
        return None
    return str(latest.date())


def build_outlier_weight_settings(
    total_metrics: pd.DataFrame,
    today_metrics: Optional[pd.DataFrame],
    total_path: str,
    today_path: Optional[str],
    ev_unit: Optional[str],
    odds_api_weights_path: Optional[Path] = None,
    odds_api_weight_alpha: float = 0.4,
    min_weight: float = 0.01,
    sample_floor_total: int = 20,
    sample_floor_today: int = 3,
    shrink_total: int = 200,
    shrink_today: int = 10,
    roi_weight: float = 0.5,
    today_blend_weight: float = 0.3,
) -> Dict[str, Any]:
    total_settled = total_metrics[total_metrics["status"].astype(str).str.startswith("SETTLED")]
    today_settled = None
    latest_date = None
    if today_metrics is not None and not today_metrics.empty:
        latest_date = _latest_date(today_metrics)
        if latest_date:
            today_slice = today_metrics[today_metrics["time_placed_iso"].dt.date.astype(str) == latest_date]
        else:
            today_slice = today_metrics
        today_settled = today_slice[today_slice["status"].astype(str).str.startswith("SETTLED")]

    stats_total = _book_stats(total_settled)
    stats_today = _book_stats(today_settled) if today_settled is not None else pd.DataFrame(columns=stats_total.columns)

    external_weights = _canonicalize_weight_map(_load_external_weights(odds_api_weights_path))

    stats_total = _weights_from_stats(stats_total, sample_floor_total, shrink_total, roi_weight)
    stats_today = _weights_from_stats(stats_today, sample_floor_today, shrink_today, roi_weight)

    combined = _combine_weights(stats_total, stats_today, today_blend_weight)
    weights_by_type = {}
    for bet_type, group in combined.groupby("bet_type"):
        weights_by_type[bet_type] = _normalize_weights(group, min_weight)

    # Props: ignore transaction weights and use priors only.
    priors_clean = {book: weight for book, weight in PROP_PRIORS.items() if book not in DISALLOWED_PROPS}
    total_prior = sum(priors_clean.values())
    props_prior_norm = {book: weight / total_prior for book, weight in priors_clean.items()} if total_prior > 0 else {}

    def _props_ml_weights(df: pd.DataFrame, allowed_books: set[str]) -> Dict[str, float]:
        # Simple ridge regression on win indicator by sportsbook (non-negative, normalized), only if enough samples.
        props = df[df["bet_type"] == "Player Props"]
        if props.empty or len(props) < PROP_ML_MIN_SAMPLES:
            return {}
        # label: 1 if profit > 0 else 0
        y = (props["profit"] > 0).astype(float).values
        books = props["sportsbook"].tolist()
        unique_books = sorted({b for b in books if b not in DISALLOWED_PROPS and b in allowed_books})
        if not unique_books:
            return {}
        book_to_idx = {b: i for i, b in enumerate(unique_books)}
        X = np.zeros((len(books), len(unique_books)))
        for row_idx, book in enumerate(books):
            if book in book_to_idx:
                X[row_idx, book_to_idx[book]] = 1.0
        XtX = X.T @ X
        ridge = XtX + PROP_ML_RIDGE_LAMBDA * np.eye(len(unique_books))
        try:
            coef = np.linalg.solve(ridge, X.T @ y)
        except np.linalg.LinAlgError:
            return {}
        coef = np.clip(coef, 0, None)
        total = coef.sum()
        if total <= 0:
            return {}
        return {book: float(weight / total) for book, weight in zip(unique_books, coef)}

    allowed_props_books = set(priors_clean.keys())
    ml_weights = _props_ml_weights(total_settled, allowed_props_books)
    if ml_weights:
        # Blend priors with ML to allow data-driven nudge while keeping sharp priors dominant.
        blended = {}
        all_books = set(props_prior_norm) | set(ml_weights)
        for book in all_books:
            prior = props_prior_norm.get(book, 0.0)
            learned = ml_weights.get(book, 0.0)
            blended[book] = (1 - PROP_ML_ALPHA) * prior + PROP_ML_ALPHA * learned
        norm = sum(blended.values())
        weights_by_type["Player Props"] = {b: w / norm for b, w in blended.items() if w > 0}
        props_weights_source = "priors_blended_with_props_ml"
    else:
        weights_by_type["Player Props"] = props_prior_norm
        props_weights_source = "priors_only"

    # Gamelines: if external weights exist, use them exclusively; otherwise keep transaction-derived but drop disallowed.
    if external_weights:
        weights_by_type["Gamelines"] = {book: weight for book, weight in external_weights.items() if book not in DISALLOWED_GAMELINES}
    elif "Gamelines" in weights_by_type:
        weights_by_type["Gamelines"] = {
            book: w for book, w in weights_by_type["Gamelines"].items() if book not in DISALLOWED_GAMELINES
        }

    ev_stats = total_metrics[pd.to_numeric(total_metrics.get("ev"), errors="coerce").notna()].copy()
    ev_by_type = ev_stats.groupby("bet_type")["ev"]
    props_ev = ev_by_type.get_group("Player Props").quantile(0.6) if "Player Props" in ev_by_type.groups else 0.01
    gamelines_ev = ev_by_type.get_group("Gamelines").quantile(0.5) if "Gamelines" in ev_by_type.groups else 0.01

    props_ev_source = "total_transactions_ev_p60_floor_1pct"
    if "Player Props" in ev_stats["bet_type"].unique():
        props_mask = (ev_stats["bet_type"] == "Player Props") & (ev_stats["clv"] > 0)
        props_clv_pos = ev_stats[props_mask]
        if len(props_clv_pos) >= max(sample_floor_total, 20):
            props_ev = props_clv_pos["ev"].quantile(0.6)
            props_ev_source = "total_transactions_ev_p60_clv_pos_floor_1pct"
    props_ev_floor = max(float(props_ev), 0.01)
    gamelines_ev_floor = max(float(gamelines_ev), 0.01)

    def scale_weights(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}
        max_weight = max(weights.values())
        if max_weight <= 0:
            return {}
        scale = 100.0 / max_weight
        return {book: round(weight * scale, 2) for book, weight in weights.items()}

    props_weights = scale_weights(weights_by_type.get("Player Props", {}))
    gamelines_weights = scale_weights(weights_by_type.get("Gamelines", {}))
    props_weights_norm = {book: round(weight, 6) for book, weight in weights_by_type.get("Player Props", {}).items()}
    gamelines_weights_norm = {book: round(weight, 6) for book, weight in weights_by_type.get("Gamelines", {}).items()}
    # Prefer dual-sharp devig for gamelines when coverage supports it.
    gamelines_min_books = 2 if len(gamelines_weights_norm) >= 2 else 1

    def select_required_books(
        bet_type: str,
        weights_norm: Dict[str, float],
        min_bets: int,
        min_weight_share: float,
        top_n: int = 1,
    ) -> list[str]:
        if not weights_norm:
            return []
        bet_counts = (
            stats_total[stats_total["bet_type"] == bet_type][["sportsbook", "bets"]]
            .set_index("sportsbook")["bets"]
            .to_dict()
        )
        ordered = sorted(weights_norm.items(), key=lambda x: x[1], reverse=True)
        required = []
        for book, weight in ordered:
            if bet_counts.get(book, 0) >= min_bets and weight >= min_weight_share:
                required.append(book)
            if len(required) >= top_n:
                break
        return required

    props_required = []

    gamelines_weights_source = "odds_api_only" if external_weights else "transactions_only"

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "latest_date": latest_date,
        "sources": {
            "total": total_path,
            "today": today_path,
            "odds_api_weights": str(odds_api_weights_path) if external_weights else None,
        },
        "ev_unit": ev_unit or "unknown",
        "method": {
            "signal": "max(0, avg_clv) + roi_weight * max(0, roi)",
            "roi_weight": roi_weight,
            "sample_floor_total": sample_floor_total,
            "sample_floor_today": sample_floor_today,
            "shrink_total": shrink_total,
            "shrink_today": shrink_today,
            "today_blend_weight": today_blend_weight,
            "odds_api_weight_alpha": odds_api_weight_alpha if external_weights else None,
            "min_weight": min_weight,
            "weights_scale": "max=100",
            "prop_prior_alpha": PROP_PRIOR_ALPHA if "Player Props" in weights_by_type else None,
        },
        "profiles": {
            "props_core": {
                "bet_type": "Player Props",
                "weights": props_weights,
                "weights_normalized": props_weights_norm,
                "required_books": props_required,
                "optional_books": list(props_weights.keys()),
                "min_books": max(1, len(props_required)),
                "devig_method": "power",
                "ev_min_pct": round(props_ev_floor * 100, 2),
                "ev_min_pct_source": props_ev_source,
                "weights_source": props_weights_source,
            },
            "gamelines_expansion": {
                "bet_type": "Gamelines",
            "weights": gamelines_weights,
            "weights_normalized": gamelines_weights_norm,
            "required_books": [],
            "optional_books": list(gamelines_weights.keys()),
            "min_books": gamelines_min_books,
            "devig_method": "probit",
            "ev_min_pct": round(gamelines_ev_floor * 100, 2),
            "ev_min_pct_source": "total_transactions_ev_p50_floor_1pct",
            "weights_source": gamelines_weights_source,
            },
        },
    }
