import pandas as pd

from worldcup_predictor.data import (
    BACKTEST_SPLITS,
    BACKTEST_YEARS,
    EXPECTED_PLAYED_COUNTS,
    assert_data_quality,
    completed_matches,
    load_matches,
    rolling_splits,
)


def test_rolling_splits_have_four_train_years_and_later_test_year():
    for split in rolling_splits():
        assert len(split.train_years) == 4
        assert split.test_year not in split.train_years
        assert max(split.train_years) < split.test_year


def test_rolling_split_values_are_stable():
    assert [(split.train_years, split.test_year) for split in rolling_splits()] == [
        (tuple(train_years), test_year) for train_years, test_year in BACKTEST_SPLITS
    ]


def test_clean_data_has_expected_played_counts():
    matches = completed_matches(load_matches())
    counts = matches.groupby("tournament_year").size().to_dict()
    for year, expected_count in EXPECTED_PLAYED_COUNTS.items():
        assert counts[year] == expected_count


def test_backtest_rows_have_ranking_features():
    matches = load_matches()
    assert isinstance(assert_data_quality(matches), pd.DataFrame)


def test_backtest_rows_have_normalized_ranking_points_diff():
    matches = completed_matches(load_matches())
    backtest_rows = matches[matches["tournament_year"].isin(BACKTEST_YEARS)]
    assert "ranking_points_diff_normalized_by_year" in backtest_rows.columns
    assert backtest_rows["ranking_points_diff_normalized_by_year"].notna().all()
