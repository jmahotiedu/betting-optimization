# Outlier Odds Research Notes

## Sources
- Outlier Help Center — **How to Devig Odds - Comparing the Methods** (lists multiplicative, probit, additive, shin, power, worst case, average; explains devigging and method choice). https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods
- Outlier Help Center — **Multi-Book Devig & Custom Weighting Now Available to Pro Users!** (explains selecting multiple devig books and custom weighting). https://help.outlier.bet/en/articles/10714084-multi-book-devig-custom-weighting-now-available-to-pro-users
- Outlier Help Center — **How to Read & Use Positive EV Filter Presets in Outlier** (lists filter settings including devig books, min books, custom weights, devig method, variation, Kelly multiplier, EV%, Kelly%, vig%, fair value, market width, bet types, date, leagues). https://help.outlier.bet/en/articles/11908672-how-to-read-use-positive-ev-filter-presets-in-outlier

## Key Definitions (from sources)
- **Devig methods**: Outlier enumerates multiplicative, probit, additive, shin, power, worst case, and average devig approaches and notes method choice affects probability estimates and EV evaluation. (Devig Methods article)
- **Multi-book devig + weighting**: Outlier allows selecting multiple devig books and assigning custom weights per book to reflect which books are sharper for a market; weights can be reduced to de-emphasize less trusted books. (Multi-book devig article)
- **Filter settings listed in Outlier presets**:
  - Devig Books
  - Minimum number of books required
  - Custom weight
  - Devig method
  - Variation
  - Kelly multiplier
  - Expected Value percentage (EV%)
  - Kelly percentage
  - Vig percentage
  - Fair value bounds
  - Market width
  - Bet types
  - Date
  - Leagues (Positive EV filter presets article)

## How these map to this project
- Implemented devig methods (multiplicative, probit, additive, shin, power, worst case, average) in `reports/src/devig.py`.
- Built settings outputs (`outlier_settings_core.json`, `outlier_settings_expansion.json`, `outlier_devig_weights.json`) to align with the documented filters and multi-book weighting.
