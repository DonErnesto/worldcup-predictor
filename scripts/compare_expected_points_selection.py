#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.data import load_matches, rolling_splits, split_train_test
from worldcup_predictor.evaluation import evaluate_predictions
from worldcup_predictor.model import ExpectedPointsPhaseSplitScorePredictor, ModeScorePhaseSplitScorePredictor


ALPHA = 0.1
TRAIN_WINDOW = 4
EXPERIMENTS: dict[str, Callable[[], object]] = {
    "mode_score_phase_split_poisson_alpha_0_1": lambda: ModeScorePhaseSplitScorePredictor(alpha=ALPHA),
    "expected_points_phase_split_poisson_alpha_0_1": lambda: ExpectedPointsPhaseSplitScorePredictor(alpha=ALPHA),
}


def main() -> None:
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")
    output_dir = REPO_ROOT / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    summary_frames = []
    for experiment_name, predictor_factory in EXPERIMENTS.items():
        predictions, summary = run_experiment_backtest(matches, experiment_name, predictor_factory)
        prediction_frames.append(predictions)
        summary_frames.append(summary)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_by_year = pd.concat(summary_frames, ignore_index=True)
    summary_overall = (
        predictions.groupby("experiment", as_index=False)
        .agg(
            matches=("score_points", "size"),
            total_points=("score_points", "sum"),
            avg_points_per_match=("score_points", "mean"),
            zero_rate=("score_points", lambda points: float((points == 0).mean())),
            exact_score_rate=("exact_score", "mean"),
            outcome_accuracy=("outcome_correct", "mean"),
            draw_accuracy=("draw_correct", "mean"),
        )
        .sort_values("avg_points_per_match", ascending=False)
    )

    predictions.to_csv(output_dir / "expected_points_selection_predictions.csv", index=False)
    summary_by_year.to_csv(output_dir / "expected_points_selection_summary_by_year.csv", index=False)
    summary_overall.to_csv(output_dir / "expected_points_selection_comparison.csv", index=False)

    print(summary_overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBy year")
    print(
        summary_by_year[
            [
                "experiment",
                "test_year",
                "matches",
                "total_points",
                "avg_points_per_match",
                "exact_score_rate",
                "outcome_accuracy",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


def run_experiment_backtest(
    matches: pd.DataFrame,
    experiment_name: str,
    predictor_factory: Callable[[], object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    for split in rolling_splits(train_window=TRAIN_WINDOW):
        train, test = split_train_test(matches, split)
        predictor = predictor_factory().fit(train)
        predicted_scores = pd.concat(
            [
                predictor.predict(test).reset_index(drop=True),
                predictor.predict_rates(test).reset_index(drop=True),
            ],
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
        frame["train_window"] = TRAIN_WINDOW
        frame["alpha"] = ALPHA
        frame["experiment"] = experiment_name
        prediction_frames.append(pd.concat([frame.reset_index(drop=True), predicted_scores], axis=1))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["actual_goals_a"] = predictions["actual_goals_a"].astype(int)
    predictions["actual_goals_b"] = predictions["actual_goals_b"].astype(int)
    evaluated, summary = evaluate_predictions(predictions)
    summary.insert(0, "experiment", experiment_name)
    summary.insert(1, "alpha", ALPHA)
    summary.insert(2, "train_window", TRAIN_WINDOW)
    return evaluated, summary


if __name__ == "__main__":
    main()
