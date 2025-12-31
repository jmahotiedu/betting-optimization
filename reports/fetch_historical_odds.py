from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from src.odds_api import get_historical_odds


def _parse_iso(dt: str) -> datetime:
    text = dt.replace("Z", "+00:00")
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _format_stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_times(start: datetime, end: datetime, step_minutes: int) -> List[datetime]:
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(minutes=step_minutes)
    return times


def _estimate_cost(markets: List[str], regions: List[str], requests: int) -> int:
    cost_per_request = 10 * len(markets) * len(regions)
    return cost_per_request * requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical odds snapshots from The Odds API.")
    parser.add_argument("--sports", required=True, help="Comma-separated sport keys, e.g. basketball_nba,icehockey_nhl")
    parser.add_argument("--regions", default="us", help="Comma-separated regions (default: us)")
    parser.add_argument("--markets", default="h2h,spreads,totals", help="Comma-separated markets (default: h2h,spreads,totals)")
    parser.add_argument("--start", required=True, help="Start timestamp ISO8601, e.g. 2025-12-30T00:00:00Z")
    parser.add_argument("--end", required=True, help="End timestamp ISO8601, e.g. 2025-12-31T00:00:00Z")
    parser.add_argument("--step-minutes", type=int, default=60, help="Snapshot interval in minutes (default: 60)")
    parser.add_argument("--out-dir", default="reports/odds_api_snapshots", help="Output directory")
    parser.add_argument("--max-requests", type=int, default=0, help="Hard cap on requests (0 = no cap)")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between requests to avoid rate limits")
    parser.add_argument("--odds-format", default="decimal", help="Odds format for API response (default: decimal)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests/cost only")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY environment variable.")

    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    start = _parse_iso(args.start)
    end = _parse_iso(args.end)
    times = _build_times(start, end, max(1, args.step_minutes))

    planned_requests = len(sports) * len(times)
    if args.max_requests and planned_requests > args.max_requests:
        times = times[: max(1, args.max_requests // max(1, len(sports)))]
        planned_requests = len(sports) * len(times)

    estimated_cost = _estimate_cost(markets, regions, planned_requests)
    if args.dry_run:
        print(f"planned_requests={planned_requests}")
        print(f"estimated_cost={estimated_cost}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dt in times:
        stamp = _format_stamp(dt)
        for sport in sports:
            out_path = out_dir / f"{sport}__{stamp}.json"
            if out_path.exists():
                continue
            payload, headers = get_historical_odds(
                api_key=api_key,
                sport_key=sport,
                regions=",".join(regions),
                markets=",".join(markets),
                date_iso=dt.isoformat().replace("+00:00", "Z"),
                odds_format=args.odds_format,
                date_format="iso",
            )
            out_path.write_text(
                json.dumps(
                    {
                        "requested": {
                            "sport": sport,
                            "regions": regions,
                            "markets": markets,
                            "date": dt.isoformat().replace("+00:00", "Z"),
                            "odds_format": args.odds_format,
                        },
                        "headers": headers,
                        "response": payload,
                    },
                    indent=2,
                )
            )
            time.sleep(max(0.0, args.sleep_seconds))


if __name__ == "__main__":
    main()
