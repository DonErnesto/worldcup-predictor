# Prediction Approach

Initial model scope:

- Predict 90-minute match outcome probabilities: team 1 win, draw, team 2 win.
- Predict 90-minute scoreline as goals for team 1 and goals for team 2.
- Use historical World Cup results as labels.
- Use FIFA ranking features from the latest available ranking before each tournament when available.
- Add tournament-stage and host indicators once the base parser is stable.
- Keep extra-time and penalty shootout outcomes separate from 90-minute labels.

Baseline model candidates:

- Logistic regression for interpretable first pass.
- Gradient boosted trees once feature tables are reliable.
- Poisson goal model later if we want scoreline probabilities.

Out of scope for the first pass:

- Friendly matches; the current raw snapshot does not include them.
- A final/advancement outcome label for knockout matches.
- Complete qualification-match modeling; only scattered qualification reference files are present.
