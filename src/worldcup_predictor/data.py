from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/world_cup_matches.csv")
BACKTEST_SPLITS = (
    ((1994, 1998, 2002, 2006), 2010),
    ((1998, 2002, 2006, 2010), 2014),
    ((2002, 2006, 2010, 2014), 2018),
    ((2006, 2010, 2014, 2018), 2022),
)
EXPECTED_PLAYED_COUNTS = {
    1930: 18,
    1934: 17,
    1938: 18,
    1950: 22,
    1954: 26,
    1958: 35,
    1962: 32,
    1966: 32,
    1970: 32,
    1974: 38,
    1978: 38,
    1982: 52,
    1986: 52,
    1990: 52,
    1994: 52,
    1998: 64,
    2002: 64,
    2006: 64,
    2010: 64,
    2014: 64,
    2018: 64,
    2022: 64,
}

NUMERIC_FEATURES = [
    "is_knockout",
    "rank_a",
    "rank_b",
    "rank_diff",
    "ranking_points_a",
    "ranking_points_b",
    "ranking_points_diff",
    "same_confederation",
]
CATEGORICAL_FEATURES = [
    "country_a_code",
    "country_b_code",
    "stage",
    "confederation_a",
    "confederation_b",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REDUCED_NUMERIC_FEATURES = [
    "is_knockout",
    "rank_a",
    "rank_b",
    "rank_diff",
]
REDUCED_CATEGORICAL_FEATURES = [
    "country_a_code",
    "country_b_code",
    "stage",
]
REDUCED_FEATURE_COLUMNS = REDUCED_NUMERIC_FEATURES + REDUCED_CATEGORICAL_FEATURES
NORMALIZED_POINTS_NUMERIC_FEATURES = [
    "is_knockout",
    "rank_a",
    "rank_b",
    "rank_diff",
    "ranking_points_diff_normalized_by_year",
]
NORMALIZED_POINTS_CATEGORICAL_FEATURES = REDUCED_CATEGORICAL_FEATURES
NORMALIZED_POINTS_FEATURE_COLUMNS = NORMALIZED_POINTS_NUMERIC_FEATURES + NORMALIZED_POINTS_CATEGORICAL_FEATURES
TARGET_COLUMNS = ["goals_a_90", "goals_b_90"]
BACKTEST_YEARS = sorted({year for train_years, test_year in BACKTEST_SPLITS for year in (*train_years, test_year)})


@dataclass(frozen=True)
class BacktestSplit:
    train_years: tuple[int, int, int, int]
    test_year: int


def load_matches(path: Path | str = DATA_PATH) -> pd.DataFrame:
    matches = pd.read_csv(path)
    for column in ["is_knockout", "same_confederation"]:
        if column in matches.columns:
            matches[column] = matches[column].astype("boolean")
    return add_normalized_ranking_points_diff(matches)


def add_normalized_ranking_points_diff(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    team_points = pd.concat(
        [
            matches[["tournament_year", "country_a_code", "ranking_points_a"]].rename(
                columns={"country_a_code": "country_code", "ranking_points_a": "ranking_points"}
            ),
            matches[["tournament_year", "country_b_code", "ranking_points_b"]].rename(
                columns={"country_b_code": "country_code", "ranking_points_b": "ranking_points"}
            ),
        ],
        ignore_index=True,
    )
    team_points = team_points.dropna(subset=["country_code", "ranking_points"]).drop_duplicates(
        ["tournament_year", "country_code"]
    )
    points_std_by_year = team_points.groupby("tournament_year")["ranking_points"].std()
    matches["ranking_points_diff_normalized_by_year"] = (
        matches["ranking_points_diff"] / matches["tournament_year"].map(points_std_by_year)
    )
    return matches


def completed_matches(matches: pd.DataFrame) -> pd.DataFrame:
    return matches[matches["outcome_90"].notna()].copy()


def rolling_splits() -> list[BacktestSplit]:
    return [BacktestSplit(tuple(train_years), test_year) for train_years, test_year in BACKTEST_SPLITS]


def split_train_test(matches: pd.DataFrame, split: BacktestSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = completed_matches(matches)
    train = completed[completed["tournament_year"].isin(split.train_years)].copy()
    test = completed[completed["tournament_year"] == split.test_year].copy()
    if train.empty:
        raise ValueError(f"No training rows for years {split.train_years}")
    if test.empty:
        raise ValueError(f"No test rows for year {split.test_year}")
    return train, test


def assert_data_quality(matches: pd.DataFrame) -> pd.DataFrame:
    completed = completed_matches(matches)
    counts = completed.groupby("tournament_year").size().to_dict()
    problems: list[dict[str, object]] = []

    for year, expected in EXPECTED_PLAYED_COUNTS.items():
        actual = int(counts.get(year, 0))
        if actual != expected:
            problems.append(
                {
                    "check": "played_match_count",
                    "tournament_year": year,
                    "expected": expected,
                    "actual": actual,
                    "details": "",
                }
            )

    backtest_rows = completed[completed["tournament_year"].isin(BACKTEST_YEARS)]
    required_columns = [
        "country_a_code",
        "country_b_code",
        "rank_a",
        "rank_b",
        "ranking_points_a",
        "ranking_points_b",
        "confederation_a",
        "confederation_b",
    ]
    for column in required_columns:
        missing = backtest_rows[backtest_rows[column].isna()]
        if len(missing):
            problems.append(
                {
                    "check": f"missing_{column}",
                    "tournament_year": "1994+",
                    "expected": 0,
                    "actual": len(missing),
                    "details": "; ".join(
                        f"{row.tournament_year}:{row.country_a}-{row.country_b}"
                        for row in missing[["tournament_year", "country_a", "country_b"]]
                        .drop_duplicates()
                        .head(10)
                        .itertuples(index=False)
                    ),
                }
            )

    report = pd.DataFrame(problems, columns=["check", "tournament_year", "expected", "actual", "details"])
    if not report.empty:
        raise ValueError(f"Data quality checks failed:\n{report.to_string(index=False)}")
    return report


def feature_target_frames(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return rows[FEATURE_COLUMNS].copy(), rows[TARGET_COLUMNS].astype(int).copy()
