#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.data import load_matches, rolling_splits, split_train_test
from worldcup_predictor.evaluation import evaluate_predictions
from worldcup_predictor.model import ModeScorePhaseSplitScorePredictor


DEFAULT_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
TRAIN_WINDOW = 4
PREDICTOR_NAME = "mode_score_phase_split_poisson"


def main() -> None:
    alphas = tuple(float(value) for value in sys.argv[1:]) or DEFAULT_ALPHAS
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")
    output_dir = REPO_ROOT / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    summary_frames = []
    for alpha in alphas:
        predictions, summary = run_alpha_backtest(matches, alpha=alpha)
        prediction_frames.append(predictions)
        summary_frames.append(summary)

    predictions = _merge_existing(
        output_dir / "mode_score_poisson_alpha_tuning_predictions.csv",
        pd.concat(prediction_frames, ignore_index=True),
        alphas,
    )
    summary_by_year = _merge_existing(
        output_dir / "mode_score_poisson_alpha_tuning_summary_by_year.csv",
        pd.concat(summary_frames, ignore_index=True),
        alphas,
    )
    summary_overall = (
        predictions.groupby("alpha", as_index=False)
        .agg(
            matches=("score_points", "size"),
            total_points=("score_points", "sum"),
            avg_points_per_match=("score_points", "mean"),
            zero_rate=("score_points", lambda points: float((points == 0).mean())),
            exact_score_rate=("exact_score", "mean"),
            outcome_accuracy=("outcome_correct", "mean"),
        )
        .sort_values("alpha")
    )

    predictions.to_csv(output_dir / "mode_score_poisson_alpha_tuning_predictions.csv", index=False)
    summary_by_year.to_csv(output_dir / "mode_score_poisson_alpha_tuning_summary_by_year.csv", index=False)
    summary_overall.to_csv(output_dir / "mode_score_poisson_alpha_tuning_summary_overall.csv", index=False)

    print(summary_overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBy year")
    print(
        summary_by_year[
            [
                "alpha",
                "test_year",
                "matches",
                "total_points",
                "avg_points_per_match",
                "exact_score_rate",
                "outcome_accuracy",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


def run_alpha_backtest(matches: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    for split in rolling_splits(train_window=TRAIN_WINDOW):
        train, test = split_train_test(matches, split)
        predictor = ModeScorePhaseSplitScorePredictor(alpha=alpha).fit(train)
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
        frame["alpha"] = alpha
        frame["predictor"] = PREDICTOR_NAME
        prediction_frames.append(pd.concat([frame.reset_index(drop=True), predicted_scores], axis=1))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["actual_goals_a"] = predictions["actual_goals_a"].astype(int)
    predictions["actual_goals_b"] = predictions["actual_goals_b"].astype(int)
    evaluated, summary = evaluate_predictions(predictions)
    summary.insert(0, "alpha", alpha)
    summary.insert(1, "train_window", TRAIN_WINDOW)
    return evaluated, summary


def _merge_existing(path: Path, fresh_rows: pd.DataFrame, alphas: tuple[float, ...]) -> pd.DataFrame:
    if not path.exists():
        return fresh_rows.sort_values(["alpha", "test_year", "match_id"], ignore_index=True)
    existing = pd.read_csv(path)
    existing = existing[~existing["alpha"].isin(alphas)]
    merged = pd.concat([existing, fresh_rows], ignore_index=True)
    sort_columns = [column for column in ["alpha", "test_year", "match_id"] if column in merged.columns]
    return merged.sort_values(sort_columns, ignore_index=True)


if __name__ == "__main__":
    main()
