from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from src.odds_api_io import get_events, get_odds_multi


def _parse_targets(raw: str) -> List[Tuple[str, str]]:
    targets = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid target '{item}', expected sport:league.")
        sport, league = item.split(":", 1)
        targets.append((sport.strip(), league.strip()))
    return targets


def _default_window(days_forward: int) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end = (now + timedelta(days=days_forward)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return start, end


def _chunk(items: List[int], size: int) -> List[List[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch odds snapshots from Odds-API.io.")
    parser.add_argument(
        "--targets",
        default="basketball:usa-nba,ice-hockey:usa-nhl,american-football:usa-nfl",
        help="Comma-separated sport:league pairs.",
    )
    parser.add_argument("--status", default="pending", help="Event status filter (default: pending).")
    parser.add_argument("--from-ts", default=None, help="RFC3339 start timestamp (default: now).")
    parser.add_argument("--to-ts", default=None, help="RFC3339 end timestamp (default: now+2d).")
    parser.add_argument("--days-forward", type=int, default=2, help="Days forward if no to-ts provided.")
    parser.add_argument("--max-events", type=int, default=10, help="Max events per target (default: 10).")
    parser.add_argument("--bookmakers", default="Pinnacle,Circa", help="Comma-separated bookmaker names.")
    parser.add_argument("--bookmaker-filter", default=None, help="Optional bookmaker filter for events (e.g. Pinnacle).")
    parser.add_argument("--out-dir", default="reports/odds_api_io_snapshots", help="Output directory.")
    parser.add_argument("--chunk-size", type=int, default=10, help="Events per /odds/multi request.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Pause between requests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests only.")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_IO_KEY")
    if not api_key:
        raise SystemExit("Missing ODDS_API_IO_KEY environment variable.")

    targets = _parse_targets(args.targets)
    from_ts = args.from_ts
    to_ts = args.to_ts
    if not from_ts or not to_ts:
        from_ts, to_ts = _default_window(args.days_forward)

    total_events = 0
    total_requests = 0
    target_events: Dict[str, List[int]] = {}

    for sport, league in targets:
        events, _headers = get_events(
            api_key=api_key,
            sport=sport,
            league=league,
            status=args.status,
            from_ts=from_ts,
            to_ts=to_ts,
            bookmaker=args.bookmaker_filter,
        )
        events = sorted(events, key=lambda e: e.get("date") or "")
        event_ids = [int(e["id"]) for e in events[: args.max_events] if e.get("id") is not None]
        target_key = f"{sport}:{league}"
        target_events[target_key] = event_ids
        total_events += len(event_ids)
        total_requests += max(1, (len(event_ids) + args.chunk_size - 1) // args.chunk_size) if event_ids else 0

    if args.dry_run:
        print(f"targets={len(targets)} events={total_events} requests={total_requests}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for target_key, event_ids in target_events.items():
        if not event_ids:
            continue
        chunks = _chunk(event_ids, args.chunk_size)
        for idx, chunk in enumerate(chunks, start=1):
            payload, headers = get_odds_multi(api_key=api_key, event_ids=chunk, bookmakers=args.bookmakers)
            out_path = out_dir / f"{target_key.replace(':','_')}__{stamp}__chunk{idx}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "requested": {
                            "target": target_key,
                            "event_ids": chunk,
                            "bookmakers": args.bookmakers,
                            "from": from_ts,
                            "to": to_ts,
                            "status": args.status,
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
