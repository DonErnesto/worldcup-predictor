from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_goal_pipeline() -> Pipeline:
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
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
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
    def __init__(self) -> None:
        self.goals_a_model = build_goal_pipeline()
        self.goals_b_model = build_goal_pipeline()

    def fit(self, train_rows: pd.DataFrame) -> "ScorePredictor":
        x_train = train_rows[FEATURE_COLUMNS].copy()
        self.goals_a_model.fit(x_train, train_rows["goals_a_90"].astype(int))
        self.goals_b_model.fit(x_train, train_rows["goals_b_90"].astype(int))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        x_test = rows[FEATURE_COLUMNS].copy()
        pred_a = self.goals_a_model.predict(x_test)
        pred_b = self.goals_b_model.predict(x_test)
        return pd.DataFrame(
            {
                "pred_goals_a": np.rint(np.clip(pred_a, 0, None)).astype(int),
                "pred_goals_b": np.rint(np.clip(pred_b, 0, None)).astype(int),
            },
            index=rows.index,
        )
