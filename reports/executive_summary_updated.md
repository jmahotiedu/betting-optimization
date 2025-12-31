# Executive Summary (Updated with Current Outlier-Only Pipeline)

- **Sharp vs Soft Books:** Pinnacle, Circa, BookMaker.eu, and BetOnline.ag are treated as sharp price-setters; retail books (DraftKings, FanDuel, BetMGM, Caesars) are used mainly for props and as targets for execution. Devig and fair-value construction anchor on sharps first.
- **Outcome-Calibrated Weights:** Gamelines weights come from outcome-based scoring of Odds-API.io snapshots aligned to ESPN results (events_scored=32, observations=284). Current normalized weights: Pinnacle 0.3805, Circa 0.2911, BookMaker.eu 0.1687, BetOnline.ag 0.1598.
- **Props Weights:** Sharp priors blended with a simple props ML ridge (non-negative) on your settled tickets. Current scaled weights: Pinnacle 100, DraftKings 71.21, FanDuel 67.47, BetMGM 65.69, Caesars 57.05, BookMaker.eu 60, BetOnline.ag 50.
- **EV Units:** Decimal ROI confirmed (EV gate). EV floors: props 5.76% (p60 CLV>0, floored 1%), gamelines 1.00% (p50, floored 1%).
- **Devig Methods:** Gamelines use Probit; Props use Power. These remain bias-aware without overfitting to thin samples.
- **Required/Optional:** Gamelines now require at least 2 sharp books (min_books=2) based on Odds-API.io snapshot coverage (31/32 events with >=2 of Pinnacle/Circa/BookMaker.eu/BetOnline.ag). Props remain min_books=1 to avoid coverage drop on retail-only edges.
- **Overlays:** Execution-discipline overlays preserved; "weights_source" and "ev_min_pct_source" carried through to presets.

## Current Presets (Paths)
- `reports/outlier_preset_core.json` (Props Core)  
  - Optional books: Pinnacle, DraftKings, FanDuel, BetMGM, Caesars, BookMaker.eu, BetOnline.ag  
  - Devig: Power; EV min: 5.76%; Kelly: 1/4; min_books: 1  
  - Weights source: priors_blended_with_props_ml; generated_at: 2025-12-31T11:07:11Z

- `reports/outlier_preset_expansion.json` (Gamelines Expansion)  
  - Optional books: Pinnacle, Circa, BookMaker.eu, BetOnline.ag  
  - Devig: Probit; EV min: 1.00%; Kelly: 1/2; min_books: 2  
  - Weights source: odds_api_only (outcome-calibrated); generated_at: 2025-12-31T11:07:11Z

## Data Sources
- Transactions: `total_transactions.csv`, `transactions(1).csv`
- Outcome alignment: `reports/event_results_aligned.csv` (scores from ESPN) with coverage per `reports/event_results_notes.md`
- Snapshot weights: `reports/odds_api_io_book_weights.json` (built via `reports/compute_outcome_weights.py`)

## How to Reproduce
1) Update outcome weights (optional): `python reports/compute_outcome_weights.py`
2) Regenerate presets/overlays: `python reports/optimize.py`

## Next Tunings (if desired)
- Raise `min_books` to 2 for gamelines once dual-sharp coverage is consistent.
- Expand outcome alignment beyond current 32 events to further stabilize weights.
- Add sport-specific weight sets if additional seasons/sports are ingested. 
