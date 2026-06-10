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
from worldcup_predictor.model import symmetric_score_shape_likelihoods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank symmetric score-shape likelihoods from Poisson predictions.")
    parser.add_argument("predictor", nargs="?", default="phase_split_poisson", choices=sorted(PREDICTORS))
    parser.add_argument("--max-goals", type=int, default=8, help="Maximum goals per team to include in the grid.")
    parser.add_argument("--output-dir", default="test_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor_cls = PREDICTORS[args.predictor]
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_frames: list[pd.DataFrame] = []
    for split in rolling_splits():
        train, test = split_train_test(matches, split)
        predictor = predictor_cls().fit(train)
        if not hasattr(predictor, "predict_rates"):
            raise ValueError(f"Predictor {args.predictor!r} does not expose expected-goal rates")
        rates = predictor.predict_rates(test)
        for row in rates.itertuples(index=False):
            shapes = symmetric_score_shape_likelihoods(
                row.expected_goals_a,
                row.expected_goals_b,
                max_goals=args.max_goals,
            )
            shape_frames.append(shapes[["score_shape", "probability"]])

    combined = pd.concat(shape_frames, ignore_index=True)
    ranking = (
        combined.groupby("score_shape", as_index=False)
        .agg(total_probability=("probability", "sum"), avg_probability_per_match=("probability", "mean"))
        .sort_values(["total_probability", "score_shape"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_mass = ranking["total_probability"].sum()
    ranking["probability_share"] = ranking["total_probability"] / total_mass
    ranking["cumulative_probability_share"] = ranking["probability_share"].cumsum()
    ranking.insert(0, "rank", ranking.index + 1)

    output_path = output_dir / f"{args.predictor}_symmetric_score_likelihood_ranking.csv"
    ranking.to_csv(output_path, index=False)
    print(f"wrote score-shape ranking to {output_path.relative_to(REPO_ROOT)}")
    print(ranking.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
