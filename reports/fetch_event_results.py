"""Fetch and align event results for odds snapshot events."""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import urllib.request


ESPN_SCOREBOARD_URLS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}

SPORT_ALIASES = {
    "basketball_usa-nba": "basketball_nba",
    "american-football_usa-nfl": "americanfootball_nfl",
    "ice-hockey_usa-nhl": "icehockey_nhl",
}


TEAM_NAME_OVERRIDES = {
    "losangelesclippers": "laclippers",
    "laclippers": "laclippers",
}


def normalize_team(name: str) -> str:
    normalized = "".join(ch.lower() for ch in name if ch.isalnum())
    return TEAM_NAME_OVERRIDES.get(normalized, normalized)


def parse_iso_datetime(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_odds_api_snapshot_events(snapshot_paths: list[pathlib.Path]) -> dict[str, dict[str, str]]:
    events: dict[str, dict[str, str]] = {}
    for path in snapshot_paths:
        data = json.loads(path.read_text())
        response = data.get("response")
        if not response:
            continue
        if isinstance(response, dict):
            items = response.get("data", [])
        else:
            items = response
        for item in items:
            if "home_team" in item:
                sport_key = item["sport_key"]
                events[item["id"]] = {
                    "event_id": item["id"],
                    "sport_key": sport_key,
                    "commence_time": item["commence_time"],
                    "home_team": item["home_team"],
                    "away_team": item["away_team"],
                }
            else:
                sport_slug = item["sport"]["slug"]
                league_slug = item["league"]["slug"]
                sport_key = f"{sport_slug}_{league_slug}"
                events[item["id"]] = {
                    "event_id": str(item["id"]),
                    "sport_key": sport_key,
                    "commence_time": item["date"],
                    "home_team": item["home"],
                    "away_team": item["away"],
                }
    return events


def fetch_scoreboard(sport_key: str, date: dt.date) -> dict:
    url = ESPN_SCOREBOARD_URLS[sport_key]
    query = f"{url}?dates={date.strftime('%Y%m%d')}"
    with urllib.request.urlopen(query) as response:
        return json.load(response)


def build_espn_events(scoreboard: dict) -> list[dict[str, str | int | bool | dt.date]]:
    events: list[dict[str, str | int | bool | dt.date]] = []
    for event in scoreboard.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        event_date = parse_iso_datetime(event["date"]).date()
        status = competition.get("status", {}).get("type", {})
        completed = bool(status.get("completed")) or status.get("state") == "post"
        events.append(
            {
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "home_score": int(home.get("score", 0)) if completed else None,
                "away_score": int(away.get("score", 0)) if completed else None,
                "completed": completed,
                "event_date": event_date,
                "source_url": (event.get("links") or [{}])[0].get("href"),
            }
        )
    return events


def match_event(
    event: dict[str, str],
    espn_events: list[dict[str, str | int | bool | dt.date]],
) -> dict[str, str | int | None]:
    target_date = parse_iso_datetime(event["commence_time"]).date()
    home_norm = normalize_team(event["home_team"])
    away_norm = normalize_team(event["away_team"])

    best_match = None
    best_delta = None
    for candidate in espn_events:
        if normalize_team(candidate["home_team"]) != home_norm:
            continue
        if normalize_team(candidate["away_team"]) != away_norm:
            continue
        delta = abs((candidate["event_date"] - target_date).days)
        if delta > 1:
            continue
        if best_delta is None or delta < best_delta:
            best_match = candidate
            best_delta = delta

    if not best_match:
        return {
            "home_score": None,
            "away_score": None,
            "source_url": None,
        }

    return {
        "home_score": best_match["home_score"],
        "away_score": best_match["away_score"],
        "source_url": best_match["source_url"],
    }


def main() -> None:
    report_dir = pathlib.Path(__file__).resolve().parent
    snapshot_paths = sorted(report_dir.glob("odds_api_snapshots/**/*.json"))
    snapshot_paths += sorted(report_dir.glob("odds_api_io_snapshots*/**/*.json"))

    events = load_odds_api_snapshot_events(snapshot_paths)

    dates_by_sport: dict[str, set[dt.date]] = {}
    for event in events.values():
        sport_key = event["sport_key"]
        canonical = SPORT_ALIASES.get(sport_key, sport_key)
        if canonical not in ESPN_SCOREBOARD_URLS:
            continue
        dates_by_sport.setdefault(canonical, set()).add(
            parse_iso_datetime(event["commence_time"]).date()
        )

    espn_cache: dict[str, list[dict[str, str | int | bool | dt.date]]] = {}
    for sport_key, dates in dates_by_sport.items():
        sport_events: list[dict[str, str | int | bool | dt.date]] = []
        for event_date in sorted(dates):
            scoreboard = fetch_scoreboard(sport_key, event_date)
            sport_events.extend(build_espn_events(scoreboard))
        espn_cache[sport_key] = sport_events

    output_path = report_dir / "event_results_aligned.csv"
    fieldnames = [
        "event_id",
        "sport_key",
        "commence_time",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "source_url",
    ]
    rows = []
    for event in events.values():
        sport_key = event["sport_key"]
        canonical = SPORT_ALIASES.get(sport_key, sport_key)
        match = match_event(event, espn_cache.get(canonical, []))
        rows.append(
            {
                **event,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "source_url": match["source_url"],
            }
        )

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    notes_path = report_dir / "event_results_notes.md"
    total = len(rows)
    with_scores = sum(1 for row in rows if row["home_score"] is not None)
    by_sport = {}
    for row in rows:
        by_sport.setdefault(row["sport_key"], {"total": 0, "with_scores": 0})
        by_sport[row["sport_key"]]["total"] += 1
        if row["home_score"] is not None:
            by_sport[row["sport_key"]]["with_scores"] += 1

    lines = [
        "# Event Results Alignment Notes",
        "",
        "Sources:",
        "- ESPN public scoreboard APIs (site.api.espn.com)",
        "",
        f"Total events: {total}",
        f"Events with scores: {with_scores} ({with_scores / total:.1%})",
        "",
        "## Coverage by sport_key",
    ]
    for sport_key in sorted(by_sport):
        total_sport = by_sport[sport_key]["total"]
        with_scores_sport = by_sport[sport_key]["with_scores"]
        lines.append(
            f"- {sport_key}: {with_scores_sport}/{total_sport} "
            f"({with_scores_sport / total_sport:.1%})"
        )
    lines.append("")
    lines.append(
        "Notes: Events without scores are typically scheduled for later today or "
        "missing from ESPN's completed scoreboard on the queried date range."
    )
    notes_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
