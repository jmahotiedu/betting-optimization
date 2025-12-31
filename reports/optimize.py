from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.data_load import load_transactions
from src.optimize_outlier import build_presets, build_overlays
from src.outlier_weights import build_outlier_weight_settings, ev_unit_from_gate, prepare_transactions
from src.report import write_markdown, write_json


def _format_section(title: str, settings: Dict[str, Any]) -> List[str]:
    lines = [f"## {title}"]
    for key, value in settings.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Outlier-only presets and overlays.")
    parser.add_argument("--transactions-total", default="total_transactions.csv", help="Path to total transactions CSV.")
    parser.add_argument("--transactions-today", default="transactions(1).csv", help="Path to today's transactions CSV.")
    parser.add_argument("--out-dir", default="reports", help="Output directory.")
    parser.add_argument("--ev-unit-gate", default=None, help="Optional path to ev_unit_gate.md.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_raw = load_transactions(args.transactions_total)
    today_path = Path(args.transactions_today)
    today_raw = load_transactions(str(today_path)) if today_path.exists() else None

    ev_unit = ev_unit_from_gate(Path(args.ev_unit_gate) if args.ev_unit_gate else None)

    total_metrics = prepare_transactions(total_raw, ev_unit=ev_unit)
    today_metrics = prepare_transactions(today_raw, ev_unit=ev_unit) if today_raw is not None else None

    odds_api_io_weights_path = out_dir / "odds_api_io_book_weights.json"
    weight_settings = build_outlier_weight_settings(
        total_metrics,
        today_metrics,
        total_path=args.transactions_total,
        today_path=str(today_path) if today_raw is not None else None,
        ev_unit=ev_unit,
        odds_api_weights_path=odds_api_io_weights_path if odds_api_io_weights_path.exists() else None,
    )
    write_json(out_dir / "outlier_weights.json", weight_settings)

    presets = build_presets(weight_settings)
    preset_core = presets[0].settings
    preset_expansion = presets[1].settings
    write_json(out_dir / "outlier_preset_core.json", preset_core)
    write_json(out_dir / "outlier_preset_expansion.json", preset_expansion)

    overlays = build_overlays(total_metrics)
    core_overlay = overlays[0].settings
    expansion_overlay = overlays[1].settings
    core_overlay["overlay_role"] = "execution_discipline_only"
    expansion_overlay["overlay_role"] = "execution_discipline_only"
    write_json(out_dir / "outlier_overlay_core.json", core_overlay)
    write_json(out_dir / "outlier_overlay_expansion.json", expansion_overlay)

    notes = [
        "Weights derived from settled bets; props blend sharp priors with props ML.",
        "Gamelines weights come from outcome-based Odds-API.io snapshots.",
        "EV thresholds are based on total_transactions percentiles and floored at 1%.",
    ]

    outlier_md = [
        "# Outlier Recommendations",
        "",
        "## Data Sources",
        f"- total_transactions: {args.transactions_total}",
        f"- today_transactions: {args.transactions_today if today_raw is not None else 'NOT FOUND'}",
        f"- latest_date: {weight_settings.get('latest_date')}",
        f"- ev_unit: {weight_settings.get('ev_unit')}",
        "",
        "## Preset Core Profile",
        *_format_section("Core Settings", preset_core),
        "## Preset Expansion Profile",
        *_format_section("Expansion Settings", preset_expansion),
        "## Overlay Core Profile (Execution Discipline Only)",
        *_format_section("Core Overlay", core_overlay),
        "## Overlay Expansion Profile (Execution Discipline Only)",
        *_format_section("Expansion Overlay", expansion_overlay),
        "## Notes",
        *[f"- {note}" for note in notes],
    ]
    write_markdown(out_dir / "outlier_recommendations.md", "\n".join(outlier_md))


if __name__ == "__main__":
    main()
