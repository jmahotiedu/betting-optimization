# Outlier Research Notes (Sources)

## Devig methods
- Outlier documents seven devig methods: Multiplicative, Probit, Additive, Shin, Power, Worst Case, and Average, and explains they remove vig to estimate fair probabilities. Source: https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods

## Multi-book devig and custom weighting
- Outlier Pro supports selecting multiple devig (sharp) books and assigning custom weights to each book to influence the fair value calculation. Source: https://help.outlier.bet/en/articles/10714084-multi-book-devig-custom-weighting-now-available-to-pro-users

## Filter definitions (Variation, Market Width, Date, Bet Types)
- The filter preset guide explains Variation and Market Width settings and shows they are used to control uncertainty; it also lists Date and Bet Types filters as part of the EV+ settings UI. Source: https://help.outlier.bet/en/articles/11908672-how-to-read-use-positive-ev-filter-presets-in-outlier

### Date filter values
The required UI values in the prompt match the filter options described in the Outlier preset guide:
- Any time
- In the next 24 hours
- In the next 3 days
- During the week
- In the next 2 weeks
- This month

### Variation and Market Width meaning
- Variation is presented as a % threshold to limit discrepancies across devig books in the EV filter presets (smaller = more stable).
- Market Width is shown as a numeric maximum; Outlier recommends tighter values for gamelines and higher values for player props due to variance.

(These definitions are summarized from the filter preset guide: https://help.outlier.bet/en/articles/11908672-how-to-read-use-positive-ev-filter-presets-in-outlier)
