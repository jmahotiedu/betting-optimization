# Outlier Research Notes (Sources + Priors)

## Devig methods
- Outlier documents seven devig methods: Multiplicative, Probit, Additive, Shin, Power, Worst Case, and Average, used to remove vig when estimating fair probabilities. Source: https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods

## Multi-book devig and custom weighting
- Outlier Pro supports selecting multiple devig (sharp) books and assigning custom weights to each book to influence fair value calculations. Source: https://help.outlier.bet/en/articles/10714084-multi-book-devig-custom-weighting-now-available-to-pro-users

## Filter presets and UI settings
- Outlier’s Positive EV filter presets list key settings including Devig Books, Min Books Required, Custom Weight, Devig Method, Variation, Kelly Multiplier, EV%, Kelly%, Vig%, Fair Value, Market Width, Bet Types, Date, and Leagues. Source: https://help.outlier.bet/en/articles/11908672-how-to-read-use-positive-ev-filter-presets-in-outlier

### Date filter values
- Any time
- In the next 24 hours
- In the next 3 days
- During the week
- In the next 2 weeks
- This month

### Variation and Market Width meaning
- Variation is presented as a % threshold to limit discrepancies across devig books in the EV filter presets (smaller = more stable markets).
- Market Width is a numeric maximum used to filter markets by dispersion/uncertainty; Outlier recommends tighter values for gamelines and higher values for player props.

## Preset priors used in the optimizer
These are treated as priors and then tuned with transaction CLV/ROI where possible:

### Gamelines preset (prior)
- Devig Books: Pinnacle, Circa, BookMaker
- Devig Method: Average
- Variation: ~3%
- Kelly Multiplier: 1/2
- Fair Value: -200 to +200
- Vig: ~8%

### NBA Props preset (prior)
- Devig Required: FanDuel
- Devig Optional: Pinnacle, BookMaker, DraftKings, Caesars
- Custom Weighting example: FanDuel 100, others 25
- Devig Method: Average
- Variation: ~3%
- Kelly Multiplier: 1/4
- Fair Value: -200 to +200
- Market Width max: ~40

Source for both preset summaries: https://help.outlier.bet/en/articles/11908672-how-to-read-use-positive-ev-filter-presets-in-outlier
