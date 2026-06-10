from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NORMALIZED_POINTS_CATEGORICAL_FEATURES,
    NORMALIZED_POINTS_FEATURE_COLUMNS,
    NORMALIZED_POINTS_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_FEATURE_COLUMNS,
    REDUCED_NUMERIC_FEATURES,
)


PERSPECTIVE_NUMERIC_FEATURES = [
    "is_knockout",
    "focal_rank",
    "opponent_rank",
    "focal_rank_diff",
    "focal_ranking_points",
    "opponent_ranking_points",
    "focal_ranking_points_diff",
    "same_confederation",
]
PERSPECTIVE_CATEGORICAL_FEATURES = [
    "focal_team_code",
    "opponent_team_code",
    "stage",
    "focal_confederation",
    "opponent_confederation",
]
PERSPECTIVE_FEATURE_COLUMNS = PERSPECTIVE_NUMERIC_FEATURES + PERSPECTIVE_CATEGORICAL_FEATURES

GOAL_AVERAGE_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "avg_goals_for_a_window",
    "avg_goals_for_b_window",
    "avg_goals_for_diff_window",
]
GOAL_AVERAGE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
GOAL_AVERAGE_FEATURE_COLUMNS = GOAL_AVERAGE_NUMERIC_FEATURES + GOAL_AVERAGE_CATEGORICAL_FEATURES


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_goal_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    alpha: float = 1.0,
) -> Pipeline:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", PoissonRegressor(alpha=alpha, max_iter=300, tol=1e-5)),
        ]
    )


class ScorePredictor:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        score_selection: str = "rounded",
        alpha: float = 1.0,
    ) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        if score_selection not in {"rounded", "mode"}:
            raise ValueError("score_selection must be 'rounded' or 'mode'")
        self.score_selection = score_selection
        self.goals_a_model = build_goal_pipeline(numeric_features, categorical_features, alpha=alpha)
        self.goals_b_model = build_goal_pipeline(numeric_features, categorical_features, alpha=alpha)

    def fit(self, train_rows: pd.DataFrame) -> "ScorePredictor":
        x_train = train_rows[self.feature_columns].copy()
        self.goals_a_model.fit(x_train, train_rows["goals_a_90"].astype(int))
        self.goals_b_model.fit(x_train, train_rows["goals_b_90"].astype(int))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        rates = self.predict_rates(rows)
        if self.score_selection == "mode":
            scores = [
                most_likely_independent_poisson_score(expected_a, expected_b)
                for expected_a, expected_b in zip(rates["expected_goals_a"], rates["expected_goals_b"])
            ]
            return pd.DataFrame(scores, columns=["pred_goals_a", "pred_goals_b"], index=rows.index)
        return pd.DataFrame(
            {
                "pred_goals_a": np.rint(np.clip(rates["expected_goals_a"], 0, None)).astype(int),
                "pred_goals_b": np.rint(np.clip(rates["expected_goals_b"], 0, None)).astype(int),
            },
            index=rows.index,
        )

    def predict_rates(self, rows: pd.DataFrame) -> pd.DataFrame:
        x_test = rows[self.feature_columns].copy()
        pred_a = self.goals_a_model.predict(x_test)
        pred_b = self.goals_b_model.predict(x_test)
        return pd.DataFrame(
            {
                "expected_goals_a": np.clip(pred_a, 0, None),
                "expected_goals_b": np.clip(pred_b, 0, None),
            },
            index=rows.index,
        )

    def explain_row(self, row: pd.Series, top_n: int = 8) -> dict[str, pd.DataFrame]:
        x_row = row.to_frame().T[self.feature_columns]
        return {
            "goals_a": _explain_goal_pipeline(self.goals_a_model, x_row, top_n),
            "goals_b": _explain_goal_pipeline(self.goals_b_model, x_row, top_n),
        }


class PhaseSplitScorePredictor:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        score_selection: str = "rounded",
        alpha: float = 1.0,
    ) -> None:
        self.group_predictor = ScorePredictor(
            feature_columns,
            numeric_features,
            categorical_features,
            score_selection,
            alpha=alpha,
        )
        self.knockout_predictor = ScorePredictor(
            feature_columns,
            numeric_features,
            categorical_features,
            score_selection,
            alpha=alpha,
        )

    def fit(self, train_rows: pd.DataFrame) -> "PhaseSplitScorePredictor":
        group_rows = train_rows[~train_rows["is_knockout"].astype(bool)]
        knockout_rows = train_rows[train_rows["is_knockout"].astype(bool)]
        if group_rows.empty:
            raise ValueError("Cannot fit phase-split predictor without group-stage rows")
        if knockout_rows.empty:
            raise ValueError("Cannot fit phase-split predictor without knockout rows")
        self.group_predictor.fit(group_rows)
        self.knockout_predictor.fit(knockout_rows)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=rows.index, columns=["pred_goals_a", "pred_goals_b"], dtype=int)
        group_mask = ~rows["is_knockout"].astype(bool)
        knockout_mask = rows["is_knockout"].astype(bool)
        if group_mask.any():
            predictions.loc[group_mask, ["pred_goals_a", "pred_goals_b"]] = self.group_predictor.predict(rows[group_mask])
        if knockout_mask.any():
            predictions.loc[knockout_mask, ["pred_goals_a", "pred_goals_b"]] = self.knockout_predictor.predict(rows[knockout_mask])
        return predictions.astype(int)

    def predict_rates(self, rows: pd.DataFrame) -> pd.DataFrame:
        rates = pd.DataFrame(index=rows.index, columns=["expected_goals_a", "expected_goals_b"], dtype=float)
        group_mask = ~rows["is_knockout"].astype(bool)
        knockout_mask = rows["is_knockout"].astype(bool)
        if group_mask.any():
            rates.loc[group_mask, ["expected_goals_a", "expected_goals_b"]] = self.group_predictor.predict_rates(
                rows[group_mask]
            )
        if knockout_mask.any():
            rates.loc[knockout_mask, ["expected_goals_a", "expected_goals_b"]] = self.knockout_predictor.predict_rates(
                rows[knockout_mask]
            )
        return rates.astype(float)

    def explain_row(self, row: pd.Series, top_n: int = 8) -> dict[str, pd.DataFrame]:
        predictor = self.knockout_predictor if bool(row["is_knockout"]) else self.group_predictor
        return predictor.explain_row(row, top_n=top_n)


class ReducedPhaseSplitScorePredictor(PhaseSplitScorePredictor):
    def __init__(self) -> None:
        super().__init__(
            feature_columns=REDUCED_FEATURE_COLUMNS,
            numeric_features=REDUCED_NUMERIC_FEATURES,
            categorical_features=REDUCED_CATEGORICAL_FEATURES,
        )


class NormalizedPointsPhaseSplitScorePredictor(PhaseSplitScorePredictor):
    def __init__(self) -> None:
        super().__init__(
            feature_columns=NORMALIZED_POINTS_FEATURE_COLUMNS,
            numeric_features=NORMALIZED_POINTS_NUMERIC_FEATURES,
            categorical_features=NORMALIZED_POINTS_CATEGORICAL_FEATURES,
        )


class ModeScorePhaseSplitScorePredictor(PhaseSplitScorePredictor):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(score_selection="mode", alpha=alpha)


def calculate_team_goal_averages(train_rows: pd.DataFrame) -> pd.Series:
    appearances = pd.concat(
        [
            pd.DataFrame(
                {
                    "team_code": train_rows["country_a_code"],
                    "goals_for_90": train_rows["goals_a_90"],
                }
            ),
            pd.DataFrame(
                {
                    "team_code": train_rows["country_b_code"],
                    "goals_for_90": train_rows["goals_b_90"],
                }
            ),
        ],
        ignore_index=True,
    )
    return appearances.groupby("team_code")["goals_for_90"].mean().astype(float)


class GoalAverageModeScorePhaseSplitScorePredictor:
    def __init__(self, alpha: float = 0.1) -> None:
        self.goal_average_by_code = pd.Series(dtype=float)
        self.global_goal_average = 0.0
        self.predictor = PhaseSplitScorePredictor(
            feature_columns=GOAL_AVERAGE_FEATURE_COLUMNS,
            numeric_features=GOAL_AVERAGE_NUMERIC_FEATURES,
            categorical_features=GOAL_AVERAGE_CATEGORICAL_FEATURES,
            score_selection="mode",
            alpha=alpha,
        )

    def fit(self, train_rows: pd.DataFrame) -> "GoalAverageModeScorePhaseSplitScorePredictor":
        self.goal_average_by_code = calculate_team_goal_averages(train_rows)
        if self.goal_average_by_code.empty:
            raise ValueError("Cannot fit goal-average predictor without completed training rows")
        self.global_goal_average = float(self.goal_average_by_code.mean())
        self.predictor.fit(self._add_goal_average_features(train_rows))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict(self._add_goal_average_features(rows))

    def predict_rates(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict_rates(self._add_goal_average_features(rows))

    def explain_row(self, row: pd.Series, top_n: int = 8) -> dict[str, pd.DataFrame]:
        enriched = self._add_goal_average_features(row.to_frame().T)
        return self.predictor.explain_row(enriched.iloc[0], top_n=top_n)

    def _add_goal_average_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        enriched = rows.copy()
        avg_a = enriched["country_a_code"].map(self.goal_average_by_code).fillna(self.global_goal_average)
        avg_b = enriched["country_b_code"].map(self.goal_average_by_code).fillna(self.global_goal_average)
        enriched["avg_goals_for_a_window"] = avg_a.astype(float)
        enriched["avg_goals_for_b_window"] = avg_b.astype(float)
        enriched["avg_goals_for_diff_window"] = (
            enriched["avg_goals_for_a_window"] - enriched["avg_goals_for_b_window"]
        )
        return enriched


def make_team_perspective_rows(rows: pd.DataFrame) -> pd.DataFrame:
    focal_a = pd.DataFrame(
        {
            "match_id": rows.get("match_id"),
            "is_knockout": rows["is_knockout"],
            "stage": rows["stage"],
            "focal_team_code": rows["country_a_code"],
            "opponent_team_code": rows["country_b_code"],
            "focal_rank": rows["rank_a"],
            "opponent_rank": rows["rank_b"],
            "focal_rank_diff": rows["rank_b"] - rows["rank_a"],
            "focal_ranking_points": rows["ranking_points_a"],
            "opponent_ranking_points": rows["ranking_points_b"],
            "focal_ranking_points_diff": rows["ranking_points_a"] - rows["ranking_points_b"],
            "focal_confederation": rows["confederation_a"],
            "opponent_confederation": rows["confederation_b"],
            "same_confederation": rows["same_confederation"],
            "goals_for_90": rows["goals_a_90"] if "goals_a_90" in rows else pd.NA,
        },
        index=rows.index,
    )
    focal_b = pd.DataFrame(
        {
            "match_id": rows.get("match_id"),
            "is_knockout": rows["is_knockout"],
            "stage": rows["stage"],
            "focal_team_code": rows["country_b_code"],
            "opponent_team_code": rows["country_a_code"],
            "focal_rank": rows["rank_b"],
            "opponent_rank": rows["rank_a"],
            "focal_rank_diff": rows["rank_a"] - rows["rank_b"],
            "focal_ranking_points": rows["ranking_points_b"],
            "opponent_ranking_points": rows["ranking_points_a"],
            "focal_ranking_points_diff": rows["ranking_points_b"] - rows["ranking_points_a"],
            "focal_confederation": rows["confederation_b"],
            "opponent_confederation": rows["confederation_a"],
            "same_confederation": rows["same_confederation"],
            "goals_for_90": rows["goals_b_90"] if "goals_b_90" in rows else pd.NA,
        },
        index=rows.index,
    )
    return pd.concat([focal_a, focal_b], ignore_index=True)


class TeamPerspectiveGoalPredictor:
    def __init__(self) -> None:
        self.model = build_goal_pipeline(PERSPECTIVE_NUMERIC_FEATURES, PERSPECTIVE_CATEGORICAL_FEATURES)

    def fit(self, train_rows: pd.DataFrame) -> "TeamPerspectiveGoalPredictor":
        perspective_rows = make_team_perspective_rows(train_rows)
        x_train = perspective_rows[PERSPECTIVE_FEATURE_COLUMNS].copy()
        y_train = perspective_rows["goals_for_90"].astype(int)
        self.model.fit(x_train, y_train)
        return self

    def predict_rates(self, rows: pd.DataFrame) -> pd.DataFrame:
        perspective_rows = make_team_perspective_rows(rows)
        focal_rows = perspective_rows.iloc[: len(rows)]
        opponent_rows = perspective_rows.iloc[len(rows) :]
        expected_a = self.model.predict(focal_rows[PERSPECTIVE_FEATURE_COLUMNS])
        expected_b = self.model.predict(opponent_rows[PERSPECTIVE_FEATURE_COLUMNS])
        return pd.DataFrame(
            {
                "expected_goals_a": np.clip(expected_a, 0, None),
                "expected_goals_b": np.clip(expected_b, 0, None),
            },
            index=rows.index,
        )

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        rates = self.predict_rates(rows)
        scores = [
            most_likely_independent_poisson_score(expected_a, expected_b)
            for expected_a, expected_b in zip(rates["expected_goals_a"], rates["expected_goals_b"])
        ]
        return pd.DataFrame(scores, columns=["pred_goals_a", "pred_goals_b"], index=rows.index)


class SymmetricModeScorePhaseSplitScorePredictor:
    def __init__(self) -> None:
        self.group_predictor = TeamPerspectiveGoalPredictor()
        self.knockout_predictor = TeamPerspectiveGoalPredictor()

    def fit(self, train_rows: pd.DataFrame) -> "SymmetricModeScorePhaseSplitScorePredictor":
        group_rows = train_rows[~train_rows["is_knockout"].astype(bool)]
        knockout_rows = train_rows[train_rows["is_knockout"].astype(bool)]
        if group_rows.empty:
            raise ValueError("Cannot fit symmetric phase-split predictor without group-stage rows")
        if knockout_rows.empty:
            raise ValueError("Cannot fit symmetric phase-split predictor without knockout rows")
        self.group_predictor.fit(group_rows)
        self.knockout_predictor.fit(knockout_rows)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=rows.index, columns=["pred_goals_a", "pred_goals_b"], dtype=int)
        group_mask = ~rows["is_knockout"].astype(bool)
        knockout_mask = rows["is_knockout"].astype(bool)
        if group_mask.any():
            predictions.loc[group_mask, ["pred_goals_a", "pred_goals_b"]] = self.group_predictor.predict(rows[group_mask])
        if knockout_mask.any():
            predictions.loc[knockout_mask, ["pred_goals_a", "pred_goals_b"]] = self.knockout_predictor.predict(
                rows[knockout_mask]
            )
        return predictions.astype(int)

    def predict_rates(self, rows: pd.DataFrame) -> pd.DataFrame:
        rates = pd.DataFrame(index=rows.index, columns=["expected_goals_a", "expected_goals_b"], dtype=float)
        group_mask = ~rows["is_knockout"].astype(bool)
        knockout_mask = rows["is_knockout"].astype(bool)
        if group_mask.any():
            rates.loc[group_mask, ["expected_goals_a", "expected_goals_b"]] = self.group_predictor.predict_rates(
                rows[group_mask]
            )
        if knockout_mask.any():
            rates.loc[knockout_mask, ["expected_goals_a", "expected_goals_b"]] = self.knockout_predictor.predict_rates(
                rows[knockout_mask]
            )
        return rates.astype(float)


class MostCommonScorePredictor:
    def __init__(self, by_phase: bool = True) -> None:
        self.by_phase = by_phase
        self.default_score: tuple[int, int] | None = None
        self.phase_scores: dict[bool, tuple[int, int]] = {}

    def fit(self, train_rows: pd.DataFrame) -> "MostCommonScorePredictor":
        self.default_score = self._most_common_score(train_rows)
        if self.by_phase:
            for is_knockout, phase_rows in train_rows.groupby(train_rows["is_knockout"].astype(bool)):
                self.phase_scores[bool(is_knockout)] = self._most_common_score(phase_rows)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.default_score is None:
            raise ValueError("Predictor must be fit before prediction")
        predicted = []
        for row in rows.itertuples(index=False):
            score = self.phase_scores.get(bool(row.is_knockout), self.default_score)
            predicted.append(score)
        return pd.DataFrame(predicted, columns=["pred_goals_a", "pred_goals_b"], index=rows.index)

    @staticmethod
    def _most_common_score(rows: pd.DataFrame) -> tuple[int, int]:
        counts = rows.groupby(["goals_a_90", "goals_b_90"]).size().sort_values(ascending=False)
        if counts.empty:
            raise ValueError("Cannot derive most common score from empty rows")
        goals_a, goals_b = counts.index[0]
        return int(goals_a), int(goals_b)


def _explain_goal_pipeline(pipeline: Pipeline, x_row: pd.DataFrame, top_n: int) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(x_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    contributions = transformed[0] * model.coef_
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "transformed_value": transformed[0],
            "contribution_log_rate": contributions,
        }
    )
    frame = frame[frame["contribution_log_rate"].abs() > 1e-12].copy()
    frame = frame.reindex(frame["contribution_log_rate"].abs().sort_values(ascending=False).index)
    return frame.head(top_n).reset_index(drop=True)


def _clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def most_likely_independent_poisson_score(
    expected_goals_a: float,
    expected_goals_b: float,
    max_goals: int = 10,
) -> tuple[int, int]:
    likelihoods = independent_poisson_score_likelihoods(expected_goals_a, expected_goals_b, max_goals=max_goals)
    best = likelihoods.sort_values(["probability", "goals_a", "goals_b"], ascending=[False, True, True]).iloc[0]
    return int(best.goals_a), int(best.goals_b)


def independent_poisson_score_likelihoods(
    expected_goals_a: float,
    expected_goals_b: float,
    max_goals: int = 10,
) -> pd.DataFrame:
    pmf_a = _poisson_pmf(float(expected_goals_a), max_goals=max_goals)
    pmf_b = _poisson_pmf(float(expected_goals_b), max_goals=max_goals)
    rows = []
    for goals_a, prob_a in enumerate(pmf_a):
        for goals_b, prob_b in enumerate(pmf_b):
            rows.append(
                {
                    "goals_a": goals_a,
                    "goals_b": goals_b,
                    "score": f"{goals_a}-{goals_b}",
                    "score_shape": _score_shape(goals_a, goals_b),
                    "probability": prob_a * prob_b,
                }
            )
    return pd.DataFrame(rows)


def symmetric_score_shape_likelihoods(
    expected_goals_a: float,
    expected_goals_b: float,
    max_goals: int = 10,
) -> pd.DataFrame:
    likelihoods = independent_poisson_score_likelihoods(expected_goals_a, expected_goals_b, max_goals=max_goals)
    ranked = (
        likelihoods.groupby("score_shape", as_index=False)
        .agg(probability=("probability", "sum"))
        .sort_values(["probability", "score_shape"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1
    ranked["cumulative_probability"] = ranked["probability"].cumsum()
    return ranked[["rank", "score_shape", "probability", "cumulative_probability"]]


def _poisson_pmf(mean: float, max_goals: int) -> list[float]:
    mean = max(mean, 0.0)
    probabilities = [math.exp(-mean)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * mean / goals)
    return probabilities


def _score_shape(goals_a: int, goals_b: int) -> str:
    high = max(goals_a, goals_b)
    low = min(goals_a, goals_b)
    return f"{high}-{low}"
