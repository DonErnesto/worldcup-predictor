#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.backtest import run_backtest
from worldcup_predictor.data import load_matches


PREDICTOR_NAME = "mode_score_phase_split_poisson"
TRAIN_WINDOWS = (3, 4, 5)


def main() -> None:
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")
    output_dir = REPO_ROOT / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    summary_frames = []
    for train_window in TRAIN_WINDOWS:
        predictions, summary = run_backtest(matches, predictor_name=PREDICTOR_NAME, train_window=train_window)
        prediction_frames.append(predictions)
        summary_frames.append(summary)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    overall = _aggregate(predictions)
    common_years = _common_test_years(predictions)
    common = _aggregate(predictions[predictions["test_year"].isin(common_years)])
    common.insert(1, "common_test_years", ",".join(str(year) for year in common_years))

    predictions.to_csv(output_dir / "mode_score_train_window_predictions.csv", index=False)
    summary.to_csv(output_dir / "mode_score_train_window_summary_by_year.csv", index=False)
    overall.to_csv(output_dir / "mode_score_train_window_summary_overall.csv", index=False)
    common.to_csv(output_dir / "mode_score_train_window_summary_common_years.csv", index=False)

    print("\nOverall by available test years")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nCommon test years only")
    print(common.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def _common_test_years(predictions: pd.DataFrame) -> list[int]:
    years_by_window = [
        set(rows["test_year"].unique())
        for _, rows in predictions.groupby("train_window")
    ]
    return sorted(set.intersection(*years_by_window))


def _aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("train_window", as_index=False)
        .agg(
            test_years=("test_year", lambda years: ",".join(str(year) for year in sorted(set(years)))),
            matches=("score_points", "size"),
            total_points=("score_points", "sum"),
            avg_points_per_match=("score_points", "mean"),
            zero_rate=("score_points", lambda points: float((points == 0).mean())),
            exact_score_rate=("exact_score", "mean"),
            outcome_accuracy=("outcome_correct", "mean"),
        )
        .sort_values("train_window")
    )


if __name__ == "__main__":
    main()
