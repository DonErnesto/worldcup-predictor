from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import assert_data_quality, load_matches, rolling_splits, split_train_test
from .evaluation import evaluate_predictions
from .model import (
    ModeScorePhaseSplitScorePredictor,
    MostCommonScorePredictor,
    NormalizedPointsPhaseSplitScorePredictor,
    PhaseSplitScorePredictor,
    ReducedPhaseSplitScorePredictor,
)


PREDICTORS = {
    "phase_split_poisson": PhaseSplitScorePredictor,
    "mode_score_phase_split_poisson": ModeScorePhaseSplitScorePredictor,
    "reduced_phase_split_poisson": ReducedPhaseSplitScorePredictor,
    "normalized_points_phase_split_poisson": NormalizedPointsPhaseSplitScorePredictor,
    "most_common_score": MostCommonScorePredictor,
}


def run_backtest(
    matches: pd.DataFrame,
    predictor_name: str = "phase_split_poisson",
    train_window: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_data_quality(matches)
    if predictor_name not in PREDICTORS:
        raise ValueError(f"Unknown predictor {predictor_name!r}. Choose one of {sorted(PREDICTORS)}")
    prediction_frames: list[pd.DataFrame] = []

    for split in rolling_splits(train_window=train_window):
        train, test = split_train_test(matches, split)
        predictor = PREDICTORS[predictor_name]().fit(train)
        predicted_scores = predictor.predict(test)
        if hasattr(predictor, "predict_rates"):
            predicted_scores = pd.concat(
                [predicted_scores.reset_index(drop=True), predictor.predict_rates(test).reset_index(drop=True)],
                axis=1,
            )
        frame = test[
            [
                "match_id",
                "tournament_year",
                "match_date",
                "stage",
                "is_knockout",
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
        frame["train_window"] = train_window
        frame["predictor"] = predictor_name
        frame = pd.concat([frame.reset_index(drop=True), predicted_scores.reset_index(drop=True)], axis=1)
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["actual_goals_a"] = predictions["actual_goals_a"].astype(int)
    predictions["actual_goals_b"] = predictions["actual_goals_b"].astype(int)
    evaluated, summary = evaluate_predictions(predictions)
    summary.insert(0, "train_window", train_window)
    return evaluated, summary


def run_backtest_from_file(
    data_path: Path | str = "data/processed/world_cup_matches.csv",
    output_dir: Path | str = "test_results",
    predictor_name: str = "phase_split_poisson",
    train_window: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_matches(data_path)
    predictions, summary = run_backtest(matches, predictor_name=predictor_name, train_window=train_window)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / f"{predictor_name}_backtest_predictions.csv", index=False)
    summary.to_csv(output_path / f"{predictor_name}_backtest_summary.csv", index=False)
    return predictions, summary
