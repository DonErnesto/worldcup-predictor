# World Championship Football Predictor

Small machine-learning app scaffold for predicting upcoming FIFA World Cup matches.

## Repository Layout

- `data/raw/` - downloaded source snapshots, kept unchanged.
- `data/processed/` - cleaned features and model-ready tables.
- `models/` - trained model artifacts.
- `notebooks/` - exploratory analysis.
- `src/` - application and training code.
- `tests/` - automated tests.
- `test_results/` - local test outputs and reports.
- `docs/` - project notes and design docs.
- `scripts/` - repeatable data-processing and training scripts.

## Raw Data Captured

- OpenFootball World Cup archive: `data/raw/openfootball-worldcup/`
- FIFA men's ranking JSON: `data/raw/fifa-rankings/men_rankings_latest.json`
- FIFA men's ranking schedule JSON: `data/raw/fifa-rankings/men_ranking_schedules.json`

The raw data is intentionally stored once so model development can be reproducible.

## Next Steps

1. Parse OpenFootball `cup.txt` files into a normalized match table.
2. Normalize FIFA ranking JSON into team-level features.
3. Create training labels for win/draw/loss and score outcomes.
4. Train a baseline model and save artifacts under `models/`.
5. Build the app UI on top of the model output.
