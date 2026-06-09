# Prediction Approach

Initial model scope:

- Predict match outcome probabilities: team 1 win, draw, team 2 win.
- Use historical World Cup results as labels.
- Use FIFA ranking features for current team strength.
- Add tournament-stage and host indicators once the base parser is stable.

Baseline model candidates:

- Logistic regression for interpretable first pass.
- Gradient boosted trees once feature tables are reliable.
- Poisson goal model later if we want scoreline probabilities.
