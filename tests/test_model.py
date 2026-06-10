import pandas as pd

from worldcup_predictor.data import (
    BacktestSplit,
    NORMALIZED_POINTS_FEATURE_COLUMNS,
    REDUCED_FEATURE_COLUMNS,
    load_matches,
    split_train_test,
)
from worldcup_predictor.model import (
    ModeScorePhaseSplitScorePredictor,
    MostCommonScorePredictor,
    NormalizedPointsPhaseSplitScorePredictor,
    PhaseSplitScorePredictor,
    ReducedPhaseSplitScorePredictor,
    independent_poisson_score_likelihoods,
    most_likely_independent_poisson_score,
    symmetric_score_shape_likelihoods,
)


def test_phase_split_predictor_trains_group_and_knockout_models():
    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = PhaseSplitScorePredictor().fit(train)
    sample = pd.concat(
        [
            test[~test["is_knockout"].astype(bool)].head(2),
            test[test["is_knockout"].astype(bool)].head(2),
        ]
    )
    predictions = predictor.predict(sample)
    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert len(predictions) == 4
    assert (predictions >= 0).all().all()


def test_phase_split_predictor_exposes_rates_and_explanations():
    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = PhaseSplitScorePredictor().fit(train)
    sample = test.head(3)

    rates = predictor.predict_rates(sample)
    assert list(rates.columns) == ["expected_goals_a", "expected_goals_b"]
    assert len(rates) == 3
    assert (rates >= 0).all().all()

    explanation = predictor.explain_row(sample.iloc[0], top_n=5)
    assert set(explanation) == {"goals_a", "goals_b"}
    assert len(explanation["goals_a"]) <= 5
    assert {"feature", "transformed_value", "contribution_log_rate"}.issubset(explanation["goals_a"].columns)


def test_reduced_phase_split_predictor_uses_slim_feature_set():
    assert REDUCED_FEATURE_COLUMNS == [
        "is_knockout",
        "rank_a",
        "rank_b",
        "rank_diff",
        "country_a_code",
        "country_b_code",
        "stage",
    ]

    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = ReducedPhaseSplitScorePredictor().fit(train)
    sample = test.head(4)
    predictions = predictor.predict(sample)
    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert len(predictions) == 4
    assert (predictions >= 0).all().all()


def test_normalized_points_phase_split_predictor_uses_normalized_points_feature():
    assert NORMALIZED_POINTS_FEATURE_COLUMNS == [
        "is_knockout",
        "rank_a",
        "rank_b",
        "rank_diff",
        "ranking_points_diff_normalized_by_year",
        "country_a_code",
        "country_b_code",
        "stage",
    ]

    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = NormalizedPointsPhaseSplitScorePredictor().fit(train)
    sample = test.head(4)
    predictions = predictor.predict(sample)
    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert len(predictions) == 4
    assert (predictions >= 0).all().all()


def test_mode_score_predictor_uses_most_likely_poisson_score_not_rounded_mean():
    assert most_likely_independent_poisson_score(2.81, 0.28) == (2, 0)

    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = ModeScorePhaseSplitScorePredictor().fit(train)
    sample = test.head(4)
    predictions = predictor.predict(sample)
    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert len(predictions) == 4
    assert (predictions >= 0).all().all()


def test_symmetric_score_shape_likelihoods_fold_mirrored_scores():
    ranking = symmetric_score_shape_likelihoods(1.4, 1.4, max_goals=4)
    assert ranking.iloc[0]["score_shape"] == "1-0"
    assert ranking["cumulative_probability"].is_monotonic_increasing

    one_nil = ranking[ranking["score_shape"] == "1-0"]["probability"].iloc[0]
    raw = independent_poisson_score_likelihoods(1.4, 1.4, max_goals=4)
    raw_mirrored = raw[raw["score"].isin(["1-0", "0-1"])]["probability"].sum()
    assert one_nil == raw_mirrored


def test_most_common_score_predictor_uses_phase_specific_scores():
    rows = pd.DataFrame(
        {
            "is_knockout": [False, False, False, True, True, True],
            "goals_a_90": [1, 1, 2, 0, 0, 1],
            "goals_b_90": [0, 0, 1, 0, 0, 1],
        }
    )
    predictor = MostCommonScorePredictor(by_phase=True).fit(rows)
    predictions = predictor.predict(pd.DataFrame({"is_knockout": [False, True]}))
    assert predictions.to_dict("records") == [
        {"pred_goals_a": 1, "pred_goals_b": 0},
        {"pred_goals_a": 0, "pred_goals_b": 0},
    ]
