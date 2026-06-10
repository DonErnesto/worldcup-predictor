from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Score:
    goals_a: int
    goals_b: int


def outcome_from_score(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "A_WIN"
    if goals_a < goals_b:
        return "B_WIN"
    return "DRAW"


def score_points(actual: Score, predicted: Score) -> int:
    actual_outcome = outcome_from_score(actual.goals_a, actual.goals_b)
    predicted_outcome = outcome_from_score(predicted.goals_a, predicted.goals_b)

    if actual.goals_a == predicted.goals_a and actual.goals_b == predicted.goals_b:
        return 4

    if actual_outcome == "DRAW":
        return 2 if predicted_outcome == "DRAW" else 0

    if (actual.goals_a - actual.goals_b) == (predicted.goals_a - predicted.goals_b):
        return 3

    return 2 if actual_outcome == predicted_outcome else 0


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "test_year",
        "actual_goals_a",
        "actual_goals_b",
        "pred_goals_a",
        "pred_goals_b",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    evaluated = predictions.copy()
    evaluated["actual_outcome"] = [
        outcome_from_score(int(a), int(b))
        for a, b in zip(evaluated["actual_goals_a"], evaluated["actual_goals_b"])
    ]
    evaluated["pred_outcome"] = [
        outcome_from_score(int(a), int(b))
        for a, b in zip(evaluated["pred_goals_a"], evaluated["pred_goals_b"])
    ]
    evaluated["score_points"] = [
        score_points(Score(int(actual_a), int(actual_b)), Score(int(pred_a), int(pred_b)))
        for actual_a, actual_b, pred_a, pred_b in zip(
            evaluated["actual_goals_a"],
            evaluated["actual_goals_b"],
            evaluated["pred_goals_a"],
            evaluated["pred_goals_b"],
        )
    ]
    evaluated["exact_score"] = (
        (evaluated["actual_goals_a"] == evaluated["pred_goals_a"])
        & (evaluated["actual_goals_b"] == evaluated["pred_goals_b"])
    )
    evaluated["outcome_correct"] = evaluated["actual_outcome"] == evaluated["pred_outcome"]
    evaluated["actual_draw"] = evaluated["actual_outcome"] == "DRAW"
    evaluated["draw_correct"] = evaluated["actual_draw"] & (evaluated["pred_outcome"] == "DRAW")

    summary = (
        evaluated.groupby("test_year", as_index=False)
        .agg(
            matches=("match_id", "count"),
            total_points=("score_points", "sum"),
            avg_points_per_match=("score_points", "mean"),
            exact_score_rate=("exact_score", "mean"),
            outcome_accuracy=("outcome_correct", "mean"),
            actual_draws=("actual_draw", "sum"),
            draw_accuracy=("draw_correct", lambda s: float(s.sum())),
        )
    )
    draw_counts = evaluated.groupby("test_year")["actual_draw"].sum().to_dict()
    summary["draw_accuracy"] = [
        row.draw_accuracy / draw_counts[row.test_year] if draw_counts[row.test_year] else 0.0
        for row in summary.itertuples(index=False)
    ]
    return evaluated, summary
