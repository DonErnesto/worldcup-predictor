from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data import completed_matches, load_matches
from .model import ExpectedPointsPhaseSplitScorePredictor, ModeScorePhaseSplitScorePredictor


MODEL_NAME = "phase_split_poisson_alpha_0_1"
MODEL_ALPHA = 0.1
TRAIN_YEARS = tuple(range(1994, 2023, 4))
SCORE_SELECTORS = [
    {
        "id": "standard",
        "name": "Standard",
        "model_name": "mode_score_phase_split_poisson",
        "default": True,
    },
    {
        "id": "kicktipp",
        "name": "KickTipp",
        "model_name": "expected_points_phase_split_poisson",
        "default": False,
    },
]


def build_site_payload(
    data_path: Path | str = "data/processed/world_cup_matches.csv",
    ranking_schedule_path: Path | str = "data/raw/fifa-rankings/men_ranking_schedules.json",
) -> dict[str, Any]:
    matches = load_matches(data_path)
    teams = _world_cup_2026_teams(matches)
    train_rows = completed_matches(matches)
    train_rows = train_rows[train_rows["tournament_year"].isin(TRAIN_YEARS)].copy()

    prediction_rows = _prediction_feature_rows(teams)
    prediction_frames = []
    for selector in SCORE_SELECTORS:
        predictor = _selector_predictor(selector["id"]).fit(train_rows)
        selected_scores = predictor.predict(prediction_rows)
        expected_goals = predictor.predict_rates(prediction_rows)
        selector_rows = pd.concat(
            [
                prediction_rows.reset_index(drop=True),
                selected_scores.reset_index(drop=True),
                expected_goals.reset_index(drop=True),
            ],
            axis=1,
        )
        selector_rows["score_selector"] = selector["id"]
        prediction_frames.append(selector_rows)
    prediction_rows = pd.concat(prediction_frames, ignore_index=True)

    return {
        "metadata": {
            "model_name": MODEL_NAME,
            "model_alpha": MODEL_ALPHA,
            "ranking_snapshot_date": _ranking_snapshot_date(ranking_schedule_path),
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "train_years": list(TRAIN_YEARS),
        },
        "score_selectors": SCORE_SELECTORS,
        "teams": _team_records(teams),
        "predictions": _prediction_records(prediction_rows, teams),
        "head_to_head": _head_to_head_records(matches, teams),
    }


def write_site_payload(
    output_path: Path | str = "site/data/predictions.json",
    data_path: Path | str = "data/processed/world_cup_matches.csv",
    ranking_schedule_path: Path | str = "data/raw/fifa-rankings/men_ranking_schedules.json",
) -> dict[str, Any]:
    payload = build_site_payload(data_path=data_path, ranking_schedule_path=ranking_schedule_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def normalize_head_to_head_match(match: dict[str, Any], country_a_code: str, country_b_code: str) -> dict[str, Any]:
    if match["country_a_code"] == country_a_code and match["country_b_code"] == country_b_code:
        goals_a = match["goals_a_90"]
        goals_b = match["goals_b_90"]
    elif match["country_a_code"] == country_b_code and match["country_b_code"] == country_a_code:
        goals_a = match["goals_b_90"]
        goals_b = match["goals_a_90"]
    else:
        raise ValueError("Head-to-head match does not contain the requested countries")

    return {
        "match_date": match["match_date"],
        "tournament_year": match["tournament_year"],
        "stage": match["stage"],
        "country_a": country_a_code,
        "country_b": country_b_code,
        "score": f"{int(goals_a)}-{int(goals_b)}",
    }


def _selector_predictor(selector_id: str) -> ModeScorePhaseSplitScorePredictor | ExpectedPointsPhaseSplitScorePredictor:
    if selector_id == "standard":
        return ModeScorePhaseSplitScorePredictor(alpha=MODEL_ALPHA)
    if selector_id == "kicktipp":
        return ExpectedPointsPhaseSplitScorePredictor(alpha=MODEL_ALPHA)
    raise ValueError(f"Unknown score selector: {selector_id}")


def _world_cup_2026_teams(matches: pd.DataFrame) -> pd.DataFrame:
    fixtures = matches[
        (matches["tournament_year"] == 2026)
        & matches["country_a_code"].notna()
        & matches["country_b_code"].notna()
    ].copy()
    teams = pd.concat(
        [
            fixtures[
                ["country_a", "country_a_code", "rank_a", "ranking_points_a", "confederation_a"]
            ].rename(
                columns={
                    "country_a": "country",
                    "country_a_code": "code",
                    "rank_a": "rank",
                    "ranking_points_a": "ranking_points",
                    "confederation_a": "confederation",
                }
            ),
            fixtures[
                ["country_b", "country_b_code", "rank_b", "ranking_points_b", "confederation_b"]
            ].rename(
                columns={
                    "country_b": "country",
                    "country_b_code": "code",
                    "rank_b": "rank",
                    "ranking_points_b": "ranking_points",
                    "confederation_b": "confederation",
                }
            ),
        ],
        ignore_index=True,
    )
    teams = teams.drop_duplicates("code").sort_values(["rank", "country"]).reset_index(drop=True)
    if len(teams) != 48:
        raise ValueError(f"Expected 48 concrete 2026 teams, found {len(teams)}")
    return teams


def _prediction_feature_rows(teams: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, stage, is_knockout in [("group", "group", False), ("knockout", "round_of_16", True)]:
        for team_a in teams.itertuples(index=False):
            for team_b in teams.itertuples(index=False):
                if team_a.code == team_b.code:
                    continue
                rows.append(
                    {
                        "country_a": team_a.country,
                        "country_b": team_b.country,
                        "country_a_code": team_a.code,
                        "country_b_code": team_b.code,
                        "phase": phase,
                        "stage": stage,
                        "is_knockout": is_knockout,
                        "rank_a": float(team_a.rank),
                        "rank_b": float(team_b.rank),
                        "rank_diff": float(team_b.rank - team_a.rank),
                        "ranking_points_a": float(team_a.ranking_points),
                        "ranking_points_b": float(team_b.ranking_points),
                        "ranking_points_diff": float(team_a.ranking_points - team_b.ranking_points),
                        "confederation_a": team_a.confederation,
                        "confederation_b": team_b.confederation,
                        "same_confederation": team_a.confederation == team_b.confederation,
                    }
                )
    return pd.DataFrame(rows)


def _team_records(teams: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": row.country,
            "code": row.code,
            "rank": int(row.rank),
            "ranking_points": round(float(row.ranking_points), 2),
            "confederation": row.confederation,
        }
        for row in teams.itertuples(index=False)
    ]


def _prediction_records(prediction_rows: pd.DataFrame, teams: pd.DataFrame) -> dict[str, dict[str, Any]]:
    team_by_code = teams.set_index("code").to_dict("index")
    predictions: dict[str, dict[str, Any]] = {}
    for row in prediction_rows.itertuples(index=False):
        key = f"{row.country_a_code}|{row.country_b_code}|{row.phase}|{row.score_selector}"
        predictions[key] = {
            "country_a_code": row.country_a_code,
            "country_b_code": row.country_b_code,
            "phase": row.phase,
            "score_selector": row.score_selector,
            "stage": row.stage,
            "rank_a": int(row.rank_a),
            "rank_b": int(row.rank_b),
            "ranking_points_a": round(float(row.ranking_points_a), 2),
            "ranking_points_b": round(float(row.ranking_points_b), 2),
            "confederation_a": team_by_code[row.country_a_code]["confederation"],
            "confederation_b": team_by_code[row.country_b_code]["confederation"],
            "expected_goals_a": round(float(row.expected_goals_a), 2),
            "expected_goals_b": round(float(row.expected_goals_b), 2),
            "selected_goals_a": int(row.pred_goals_a),
            "selected_goals_b": int(row.pred_goals_b),
        }
    return predictions


def _head_to_head_records(matches: pd.DataFrame, teams: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    team_codes = set(teams["code"])
    played = completed_matches(matches)
    played = played[
        played["country_a_code"].isin(team_codes)
        & played["country_b_code"].isin(team_codes)
        & played["country_a_code"].notna()
        & played["country_b_code"].notna()
    ].copy()
    played["pair_key"] = played.apply(
        lambda row: _pair_key(str(row["country_a_code"]), str(row["country_b_code"])), axis=1
    )
    head_to_head: dict[str, list[dict[str, Any]]] = {}
    for pair_key, pair_rows in played.sort_values("match_date", ascending=False).groupby("pair_key", sort=False):
        head_to_head[pair_key] = [
            {
                "match_date": row.match_date,
                "tournament_year": int(row.tournament_year),
                "stage": row.stage,
                "country_a": row.country_a,
                "country_b": row.country_b,
                "country_a_code": row.country_a_code,
                "country_b_code": row.country_b_code,
                "goals_a_90": int(row.goals_a_90),
                "goals_b_90": int(row.goals_b_90),
            }
            for row in pair_rows.head(3).itertuples(index=False)
        ]
    return head_to_head


def _ranking_snapshot_date(ranking_schedule_path: Path | str) -> str:
    data = json.loads(Path(ranking_schedule_path).read_text(encoding="utf-8"))
    official_date = data["Results"][0]["OfficialDate"]
    return official_date[:10]


def _pair_key(country_a_code: str, country_b_code: str) -> str:
    return "|".join(sorted([country_a_code, country_b_code]))
