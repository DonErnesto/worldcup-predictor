#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.backtest import PREDICTORS
from worldcup_predictor.data import load_matches, rolling_splits, split_train_test
from worldcup_predictor.evaluation import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain selected backtest predictions for a Poisson predictor.")
    parser.add_argument("predictor", nargs="?", default="phase_split_poisson", choices=sorted(PREDICTORS))
    parser.add_argument("--points", type=int, default=0, help="Only explain matches with this evaluation score.")
    parser.add_argument("--limit", type=int, default=24, help="Maximum number of matches to explain.")
    parser.add_argument("--top-n", type=int, default=8, help="Top feature contributions per side.")
    parser.add_argument("--output-dir", default="test_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor_cls = PREDICTORS[args.predictor]
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluated_frames: list[pd.DataFrame] = []
    trained_predictors: dict[int, object] = {}
    test_rows_by_year: dict[int, pd.DataFrame] = {}

    for split in rolling_splits():
        train, test = split_train_test(matches, split)
        predictor = predictor_cls().fit(train)
        if not hasattr(predictor, "explain_row"):
            raise ValueError(f"Predictor {args.predictor!r} does not expose explain_row")

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
                "rank_a",
                "rank_b",
                "rank_diff",
                "ranking_points_diff",
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
        frame["predictor"] = args.predictor
        evaluated, _ = evaluate_predictions(
            pd.concat([frame.reset_index(drop=True), predicted_scores.reset_index(drop=True)], axis=1)
        )
        evaluated["expected_goal_diff"] = evaluated.get("expected_goals_a", pd.NA) - evaluated.get(
            "expected_goals_b", pd.NA
        )
        evaluated["expected_margin_abs"] = evaluated["expected_goal_diff"].abs()
        evaluated_frames.append(evaluated)
        trained_predictors[split.test_year] = predictor
        test_rows_by_year[split.test_year] = test.set_index("match_id")

    all_evaluated = pd.concat(evaluated_frames, ignore_index=True)
    selected_output = all_evaluated[all_evaluated["score_points"] == args.points].copy()
    selected_output = selected_output.sort_values(
        ["expected_margin_abs", "test_year", "match_date"], ascending=[False, True, True]
    ).head(args.limit)

    contribution_rows: list[dict[str, object]] = []
    for match in selected_output.itertuples(index=False):
        predictor = trained_predictors[match.test_year]
        test_rows_by_match_id = test_rows_by_year[match.test_year]
        explanations = predictor.explain_row(test_rows_by_match_id.loc[match.match_id], top_n=args.top_n)
        for side, side_explanation in explanations.items():
            for rank, contribution in enumerate(side_explanation.itertuples(index=False), start=1):
                contribution_rows.append(
                    {
                        "match_id": match.match_id,
                        "test_year": match.test_year,
                        "match_date": match.match_date,
                        "stage": match.stage,
                        "is_knockout": match.is_knockout,
                        "country_a": match.country_a,
                        "country_b": match.country_b,
                        "actual_score": f"{match.actual_goals_a}-{match.actual_goals_b}",
                        "pred_score": f"{match.pred_goals_a}-{match.pred_goals_b}",
                        "expected_goals_a": getattr(match, "expected_goals_a", pd.NA),
                        "expected_goals_b": getattr(match, "expected_goals_b", pd.NA),
                        "score_points": match.score_points,
                        "explained_target": side,
                        "contribution_rank": rank,
                        "feature": contribution.feature,
                        "transformed_value": contribution.transformed_value,
                        "contribution_log_rate": contribution.contribution_log_rate,
                    }
                )

    contributions_output = pd.DataFrame(contribution_rows)
    match_path = output_dir / f"{args.predictor}_score_{args.points}_matches_explained.csv"
    contribution_path = output_dir / f"{args.predictor}_score_{args.points}_feature_contributions.csv"
    selected_output.to_csv(match_path, index=False)
    contributions_output.to_csv(contribution_path, index=False)

    print(f"wrote {len(selected_output)} selected matches to {match_path.relative_to(REPO_ROOT)}")
    print(f"wrote {len(contributions_output)} feature contributions to {contribution_path.relative_to(REPO_ROOT)}")
    if not selected_output.empty:
        columns = [
            "test_year",
            "stage",
            "country_a",
            "country_b",
            "actual_goals_a",
            "actual_goals_b",
            "pred_goals_a",
            "pred_goals_b",
            "expected_goals_a",
            "expected_goals_b",
            "score_points",
        ]
        print(selected_output[columns].head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
