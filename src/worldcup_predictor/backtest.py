from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import assert_data_quality, load_matches, rolling_splits, split_train_test
from .evaluation import evaluate_predictions
from .model import ScorePredictor


def run_backtest(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_data_quality(matches)
    prediction_frames: list[pd.DataFrame] = []

    for split in rolling_splits():
        train, test = split_train_test(matches, split)
        predictor = ScorePredictor().fit(train)
        predicted_scores = predictor.predict(test)
        frame = test[
            [
                "match_id",
                "tournament_year",
                "match_date",
                "stage",
                "country_a",
                "country_b",
                "goals_a_90",
                "goals_b_90",
            ]
        ].rename(
            columns={
                "tournament_year": "test_year",
                "goals_a_90": "actual_goals_a",
                "goals_b_90": "actual_goals_b",
            }
        )
        frame["train_years"] = ",".join(str(year) for year in split.train_years)
        frame = pd.concat([frame.reset_index(drop=True), predicted_scores.reset_index(drop=True)], axis=1)
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["actual_goals_a"] = predictions["actual_goals_a"].astype(int)
    predictions["actual_goals_b"] = predictions["actual_goals_b"].astype(int)
    return evaluate_predictions(predictions)


def run_backtest_from_file(
    data_path: Path | str = "data/processed/world_cup_matches.csv",
    output_dir: Path | str = "test_results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_matches(data_path)
    predictions, summary = run_backtest(matches)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "backtest_predictions.csv", index=False)
    summary.to_csv(output_path / "backtest_summary.csv", index=False)
    return predictions, summary
