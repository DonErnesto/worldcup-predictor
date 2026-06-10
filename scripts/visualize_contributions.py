#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.backtest import PREDICTORS
from worldcup_predictor.data import load_matches, rolling_splits, split_train_test
from worldcup_predictor.evaluation import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML contribution review for correct and incorrect matches.")
    parser.add_argument("predictor", nargs="?", default="phase_split_poisson", choices=sorted(PREDICTORS))
    parser.add_argument("--incorrect", type=int, default=10, help="Number of zero-point predictions to include.")
    parser.add_argument("--correct", type=int, default=10, help="Number of exact-score predictions to include.")
    parser.add_argument("--top-n", type=int, default=8, help="Top contribution bars per goal model.")
    parser.add_argument("--output-dir", default="test_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor_cls = PREDICTORS[args.predictor]
    matches = load_matches(REPO_ROOT / "data/processed/world_cup_matches.csv")

    evaluated_frames: list[pd.DataFrame] = []
    trained_predictors: dict[int, object] = {}
    test_rows_by_year: dict[int, pd.DataFrame] = {}

    for split in rolling_splits():
        train, test = split_train_test(matches, split)
        predictor = predictor_cls().fit(train)
        if not hasattr(predictor, "explain_row"):
            raise ValueError(f"Predictor {args.predictor!r} does not expose explain_row")

        predicted = predictor.predict(test)
        if hasattr(predictor, "predict_rates"):
            predicted = pd.concat(
                [predicted.reset_index(drop=True), predictor.predict_rates(test).reset_index(drop=True)],
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
        evaluated, _ = evaluate_predictions(pd.concat([frame.reset_index(drop=True), predicted.reset_index(drop=True)], axis=1))
        evaluated["expected_goal_diff"] = evaluated["expected_goals_a"] - evaluated["expected_goals_b"]
        evaluated["expected_margin_abs"] = evaluated["expected_goal_diff"].abs()
        evaluated_frames.append(evaluated)
        trained_predictors[split.test_year] = predictor
        test_rows_by_year[split.test_year] = test.set_index("match_id")

    evaluated = pd.concat(evaluated_frames, ignore_index=True)
    incorrect = (
        evaluated[evaluated["score_points"] == 0]
        .sort_values(["expected_margin_abs", "test_year", "match_date"], ascending=[False, True, True])
        .head(args.incorrect)
        .copy()
    )
    correct = (
        evaluated[evaluated["score_points"] == 4]
        .sort_values(["expected_margin_abs", "test_year", "match_date"], ascending=[False, True, True])
        .head(args.correct)
        .copy()
    )

    html_report = render_report(args.predictor, incorrect, correct, trained_predictors, test_rows_by_year, args.top_n)
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.predictor}_contribution_review.html"
    output_path.write_text(html_report, encoding="utf-8")
    print(f"wrote contribution review to {output_path.relative_to(REPO_ROOT)}")
    print(f"included {len(incorrect)} incorrect predictions and {len(correct)} exact-score predictions")


def render_report(
    predictor_name: str,
    incorrect: pd.DataFrame,
    correct: pd.DataFrame,
    trained_predictors: dict[int, object],
    test_rows_by_year: dict[int, pd.DataFrame],
    top_n: int,
) -> str:
    cards = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(predictor_name)} Contribution Review</title>",
        f"<style>{stylesheet()}</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(predictor_name)} contribution review</h1>",
        "<p class=\"lede\">Ten zero-point misses and ten exact-score hits. Bars show each feature's contribution to the Poisson log goal-rate estimate. Positive bars push expected goals up; negative bars push them down.</p>",
        render_section("Incorrect predictions", "score_points = 0, sorted by strongest expected-goal margin", incorrect, trained_predictors, test_rows_by_year, top_n),
        render_section("Correct predictions", "exact score predictions, sorted by strongest expected-goal margin", correct, trained_predictors, test_rows_by_year, top_n),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(cards)


def render_section(
    title: str,
    subtitle: str,
    rows: pd.DataFrame,
    trained_predictors: dict[int, object],
    test_rows_by_year: dict[int, pd.DataFrame],
    top_n: int,
) -> str:
    parts = [f"<section><h2>{escape(title)}</h2><p class=\"section-note\">{escape(subtitle)}</p>"]
    for row in rows.itertuples(index=False):
        predictor = trained_predictors[row.test_year]
        raw_row = test_rows_by_year[row.test_year].loc[row.match_id]
        explanations = predictor.explain_row(raw_row, top_n=top_n)
        parts.append(render_match_card(row, explanations))
    parts.append("</section>")
    return "\n".join(parts)


def render_match_card(row: object, explanations: dict[str, pd.DataFrame]) -> str:
    actual = f"{int(row.actual_goals_a)}-{int(row.actual_goals_b)}"
    predicted = f"{int(row.pred_goals_a)}-{int(row.pred_goals_b)}"
    expected = f"{float(row.expected_goals_a):.2f}-{float(row.expected_goals_b):.2f}"
    outcome_class = "hit" if int(row.score_points) == 4 else "miss"
    return "\n".join(
        [
            f"<article class=\"card {outcome_class}\">",
            "<header>",
            f"<div><h3>{escape(row.country_a)} vs {escape(row.country_b)}</h3><p>{int(row.test_year)} · {escape(row.stage)} · {escape(row.match_date)}</p></div>",
            f"<div class=\"score\"><span>actual {escape(actual)}</span><span>pred {escape(predicted)}</span><span>xG {escape(expected)}</span><strong>{int(row.score_points)} pts</strong></div>",
            "</header>",
            "<dl class=\"facts\">",
            f"<div><dt>rank diff</dt><dd>{float(row.rank_diff):.0f}</dd></div>",
            f"<div><dt>points diff</dt><dd>{float(row.ranking_points_diff):.1f}</dd></div>",
            f"<div><dt>actual outcome</dt><dd>{escape(row.actual_outcome)}</dd></div>",
            f"<div><dt>pred outcome</dt><dd>{escape(row.pred_outcome)}</dd></div>",
            "</dl>",
            "<div class=\"bars-grid\">",
            render_bars(f"{escape(row.country_a)} goals model", explanations["goals_a"]),
            render_bars(f"{escape(row.country_b)} goals model", explanations["goals_b"]),
            "</div>",
            "</article>",
        ]
    )


def render_bars(title: str, rows: pd.DataFrame) -> str:
    max_abs = max(rows["contribution_log_rate"].abs().max(), 1e-9)
    parts = [f"<div class=\"panel\"><h4>{title}</h4>"]
    for item in rows.itertuples(index=False):
        value = float(item.contribution_log_rate)
        width = abs(value) / max_abs * 100
        cls = "positive" if value >= 0 else "negative"
        parts.append(
            "\n".join(
                [
                    f"<div class=\"bar-row {cls}\">",
                    f"<span class=\"feature\" title=\"{escape(item.feature)}\">{escape(item.feature)}</span>",
                    "<span class=\"track\">",
                    f"<span class=\"bar\" style=\"width:{width:.1f}%\"></span>",
                    "</span>",
                    f"<span class=\"value\">{value:+.3f}</span>",
                    "</div>",
                ]
            )
        )
    parts.append("</div>")
    return "\n".join(parts)


def stylesheet() -> str:
    return """
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #f7f8fb;
      --hit: #167a55;
      --miss: #b42318;
      --pos: #2563eb;
      --neg: #dc6803;
    }
    body {
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fff;
    }
    main {
      width: min(1180px, calc(100vw - 48px));
      margin: 32px auto 56px;
    }
    h1, h2, h3, h4, p { margin: 0; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    h2 { font-size: 22px; margin: 32px 0 4px; }
    h3 { font-size: 18px; }
    h4 { font-size: 14px; margin-bottom: 10px; }
    .lede, .section-note { color: var(--muted); }
    .card {
      border: 1px solid var(--line);
      border-left: 5px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
      background: #fff;
    }
    .card.hit { border-left-color: var(--hit); }
    .card.miss { border-left-color: var(--miss); }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    header p { color: var(--muted); margin-top: 2px; }
    .score {
      display: grid;
      grid-template-columns: repeat(4, max-content);
      gap: 8px;
      align-items: center;
      white-space: nowrap;
    }
    .score span, .score strong {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      background: var(--panel);
      font-size: 13px;
    }
    .facts {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 14px;
    }
    .facts div {
      background: var(--panel);
      border-radius: 6px;
      padding: 8px;
    }
    dt { color: var(--muted); font-size: 12px; }
    dd { margin: 2px 0 0; font-weight: 650; }
    .bars-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(130px, 210px) 1fr 60px;
      gap: 8px;
      align-items: center;
      margin: 7px 0;
    }
    .feature {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #344054;
      font-size: 12px;
    }
    .track {
      height: 12px;
      border-radius: 999px;
      background: #e5e9f0;
      overflow: hidden;
    }
    .bar {
      display: block;
      height: 100%;
      border-radius: 999px;
    }
    .positive .bar { background: var(--pos); }
    .negative .bar { background: var(--neg); }
    .value {
      font-variant-numeric: tabular-nums;
      text-align: right;
      color: #344054;
      font-size: 12px;
    }
    @media (max-width: 820px) {
      main { width: min(100vw - 24px, 1180px); }
      header, .bars-grid { grid-template-columns: 1fr; display: grid; }
      .score, .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .bar-row { grid-template-columns: minmax(90px, 150px) 1fr 54px; }
    }
    """


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
