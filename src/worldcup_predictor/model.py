from __future__ import annotations

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
    NUMERIC_FEATURES,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_FEATURE_COLUMNS,
    REDUCED_NUMERIC_FEATURES,
)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_goal_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
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
            ("model", PoissonRegressor(alpha=1.0, max_iter=300, tol=1e-5)),
        ]
    )


class ScorePredictor:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.goals_a_model = build_goal_pipeline(numeric_features, categorical_features)
        self.goals_b_model = build_goal_pipeline(numeric_features, categorical_features)

    def fit(self, train_rows: pd.DataFrame) -> "ScorePredictor":
        x_train = train_rows[self.feature_columns].copy()
        self.goals_a_model.fit(x_train, train_rows["goals_a_90"].astype(int))
        self.goals_b_model.fit(x_train, train_rows["goals_b_90"].astype(int))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        x_test = rows[self.feature_columns].copy()
        pred_a = self.goals_a_model.predict(x_test)
        pred_b = self.goals_b_model.predict(x_test)
        return pd.DataFrame(
            {
                "pred_goals_a": np.rint(np.clip(pred_a, 0, None)).astype(int),
                "pred_goals_b": np.rint(np.clip(pred_b, 0, None)).astype(int),
            },
            index=rows.index,
        )


class PhaseSplitScorePredictor:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.group_predictor = ScorePredictor(feature_columns, numeric_features, categorical_features)
        self.knockout_predictor = ScorePredictor(feature_columns, numeric_features, categorical_features)

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


class ReducedPhaseSplitScorePredictor(PhaseSplitScorePredictor):
    def __init__(self) -> None:
        super().__init__(
            feature_columns=REDUCED_FEATURE_COLUMNS,
            numeric_features=REDUCED_NUMERIC_FEATURES,
            categorical_features=REDUCED_CATEGORICAL_FEATURES,
        )


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
