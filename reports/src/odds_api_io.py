from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


HOST = "https://api2.odds-api.io/v3"


class OddsApiIoError(RuntimeError):
    pass


def _request_json(path: str, params: Dict[str, Any], method: str = "GET") -> Tuple[Any, Dict[str, str]]:
    query = urllib.parse.urlencode(params)
    url = f"{HOST}{path}?{query}"
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "betting-optimization"})
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body) if body else {}
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return data, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise OddsApiIoError(f"Odds-API.io request failed: {exc.code} {exc.reason} {body}".strip()) from exc
    except urllib.error.URLError as exc:
        raise OddsApiIoError(f"Odds-API.io request failed: {exc.reason}") from exc


def get_sports(api_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    return _request_json("/sports", {"apiKey": api_key})


def get_bookmakers(api_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    return _request_json("/bookmakers", {"apiKey": api_key})


def get_leagues(api_key: str, sport: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    return _request_json("/leagues", {"apiKey": api_key, "sport": sport})


def get_events(
    api_key: str,
    sport: str,
    league: Optional[str] = None,
    status: Optional[str] = None,
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    bookmaker: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    params: Dict[str, Any] = {"apiKey": api_key, "sport": sport}
    if league:
        params["league"] = league
    if status:
        params["status"] = status
    if from_ts:
        params["from"] = from_ts
    if to_ts:
        params["to"] = to_ts
    if bookmaker:
        params["bookmaker"] = bookmaker
    return _request_json("/events", params)


def get_odds(api_key: str, event_id: int, bookmakers: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    params = {"apiKey": api_key, "eventId": event_id, "bookmakers": bookmakers}
    return _request_json("/odds", params)


def get_odds_multi(api_key: str, event_ids: List[int], bookmakers: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    params = {"apiKey": api_key, "eventIds": ",".join(str(eid) for eid in event_ids), "bookmakers": bookmakers}
    return _request_json("/odds/multi", params)
