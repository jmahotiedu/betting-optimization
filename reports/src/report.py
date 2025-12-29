from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def write_markdown(path: str, content: str) -> None:
    Path(path).write_text(content)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def format_recommendations(title: str, bullets: List[str]) -> str:
    lines = [f"# {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    return "\n".join(lines)
