from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple


HOST = "https://api.the-odds-api.com"


class OddsApiError(RuntimeError):
    pass


def _request_json(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    query = urllib.parse.urlencode(params)
    url = f"{HOST}{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "betting-optimization"})
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body) if body else {}
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return data, headers
    except urllib.error.HTTPError as exc:
        raise OddsApiError(f"Odds API request failed: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise OddsApiError(f"Odds API request failed: {exc.reason}") from exc


def get_sports(api_key: str, all_sports: bool = False) -> Tuple[Dict[str, Any], Dict[str, str]]:
    params = {"apiKey": api_key}
    if all_sports:
        params["all"] = "true"
    return _request_json("/v4/sports", params)


def get_historical_odds(
    api_key: str,
    sport_key: str,
    regions: str,
    markets: str,
    date_iso: str,
    odds_format: str = "decimal",
    date_format: str = "iso",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "date": date_iso,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    path = f"/v4/historical/sports/{sport_key}/odds"
    return _request_json(path, params)
