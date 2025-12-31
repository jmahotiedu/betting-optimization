Action Log (no chain-of-thought)
================================

2025-12-31
- Initialized action log per user request; will append commands, file changes, data sources, and outputs (no internal reasoning).
- Read `reports/outlier_weights.json` and `reports/odds_api_book_weights.json` to confirm current weights and sources.
- Parsed `total_transactions.csv` and `transactions(1).csv` to list sportsbook coverage and bet types.
- Checked latest snapshot headers in `reports/odds_api_snapshots`; remaining credits reported as 4010.
- Updated `reports/src/outlier_weights.py` to allow external-only books in gamelines blending and to scale weights so max=100 (weights no longer forced to sum to 100).
- Updated `reports/derive_odds_api_weights.py` to accept multiple snapshot directories for combined weighting.
- Attempted historical Odds API fetch for EU region (7 days, 4h cadence, markets h2h/spreads/totals) into `reports/odds_api_snapshots_eu`; dry-run estimated 129 requests / 3870 credits.
- Actual fetch failed with 401 Unauthorized; tested all provided API keys against historical endpoint and all returned 401.
- Ran `python reports/optimize.py` to regenerate outlier outputs with updated weight scaling (warnings from pandas/numpy about future behavior and empty slice).
- Added Odds-API.io integration: `reports/src/odds_api_io.py`, `reports/fetch_odds_api_io.py`, `reports/derive_odds_api_io_weights.py`.
- Updated `reports/optimize.py` to prefer `reports/odds_api_io_book_weights.json` when present.
- Fetched Odds-API.io snapshots for NBA/NHL/NFL pending events (Pinnacle/Circa only) into `reports/odds_api_io_snapshots`.
- Derived weights to `reports/odds_api_io_book_weights.json` and re-ran `python reports/optimize.py`.
i - User supplied a new Odds-API.io key (free plan, 2-bookmaker limit); awaiting bookmaker selection before making requests.
- Detected new PDFs in repo root: `Basketball.pdf`, `American Football.html.pdf`; extracted summary lines (total games + bookmaker list).
- New Odds-API.io key used (redacted). Verified Pinnacle/Circa odds call, fetched fresh snapshots (NBA/NHL/NFL pending) into `reports/odds_api_io_snapshots`.
- Re-derived Odds-API.io weights and re-ran `python reports/optimize.py` (with warnings about pandas/numpy behavior).
- Added bookmaker filter support in `reports/src/odds_api_io.py` and `reports/fetch_odds_api_io.py`.
- Re-fetched snapshots with `--bookmaker-filter Pinnacle` into `reports/odds_api_io_snapshots_pinnacle`.
- Re-derived `reports/odds_api_io_book_weights.json` and regenerated Outlier outputs via `python reports/optimize.py`.
- Updated `reports/src/outlier_weights.py` to tighten props EV threshold using CLV-positive samples and set required props book based on weight/share.
- Updated `reports/derive_odds_api_io_weights.py` to support multiple snapshot directories and collapse duplicate outcomes.
- Fetched additional Odds-API.io snapshots for pairs Pinnacle+BookMaker.eu, Pinnacle+BetOnline.ag, Pinnacle+Circa into dedicated dirs.
- Rebuilt combined Odds-API.io weights and regenerated Outlier outputs via `python reports/optimize.py`.
