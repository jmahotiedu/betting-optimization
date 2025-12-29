from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "reports" / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
