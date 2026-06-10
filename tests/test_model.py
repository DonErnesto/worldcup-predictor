import pandas as pd

from worldcup_predictor.data import (
    BacktestSplit,
    NORMALIZED_POINTS_FEATURE_COLUMNS,
    REDUCED_FEATURE_COLUMNS,
    load_matches,
    split_train_test,
)
from worldcup_predictor.model import (
    GOAL_AVERAGE_FEATURE_COLUMNS,
    GoalAverageModeScorePhaseSplitScorePredictor,
    ModeScorePhaseSplitScorePredictor,
    MostCommonScorePredictor,
    NormalizedPointsPhaseSplitScorePredictor,
    PhaseSplitScorePredictor,
    ReducedPhaseSplitScorePredictor,
    SymmetricModeScorePhaseSplitScorePredictor,
    calculate_team_goal_averages,
    independent_poisson_score_likelihoods,
    make_team_perspective_rows,
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


def test_mode_score_predictor_accepts_poisson_alpha():
    predictor = ModeScorePhaseSplitScorePredictor(alpha=0.3)

    assert predictor.group_predictor.goals_a_model.named_steps["model"].alpha == 0.3
    assert predictor.group_predictor.goals_b_model.named_steps["model"].alpha == 0.3
    assert predictor.knockout_predictor.goals_a_model.named_steps["model"].alpha == 0.3
    assert predictor.knockout_predictor.goals_b_model.named_steps["model"].alpha == 0.3


def test_calculate_team_goal_averages_uses_both_match_sides():
    rows = pd.DataFrame(
        {
            "country_a_code": ["AAA", "BBB", "AAA"],
            "country_b_code": ["BBB", "AAA", "CCC"],
            "goals_a_90": [2, 0, 3],
            "goals_b_90": [1, 1, 0],
        }
    )

    averages = calculate_team_goal_averages(rows)

    assert averages["AAA"] == 2.0
    assert averages["BBB"] == 0.5
    assert averages["CCC"] == 0.0


def test_goal_average_predictor_adds_window_features_and_uses_alpha_point_one():
    assert {
        "avg_goals_for_a_window",
        "avg_goals_for_b_window",
        "avg_goals_for_diff_window",
    }.issubset(GOAL_AVERAGE_FEATURE_COLUMNS)

    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = GoalAverageModeScorePhaseSplitScorePredictor().fit(train)
    sample = test.head(4)

    predictions = predictor.predict(sample)
    rates = predictor.predict_rates(sample)

    assert predictor.predictor.group_predictor.goals_a_model.named_steps["model"].alpha == 0.1
    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert list(rates.columns) == ["expected_goals_a", "expected_goals_b"]
    assert len(predictions) == 4
    assert (predictions >= 0).all().all()
    assert (rates >= 0).all().all()


def test_team_perspective_rows_double_matches_and_map_goals():
    rows = pd.DataFrame(
        {
            "match_id": [1],
            "is_knockout": [False],
            "stage": ["group"],
            "country_a_code": ["AAA"],
            "country_b_code": ["BBB"],
            "rank_a": [10],
            "rank_b": [25],
            "ranking_points_a": [1700.0],
            "ranking_points_b": [1500.0],
            "confederation_a": ["UEFA"],
            "confederation_b": ["CAF"],
            "same_confederation": [False],
            "goals_a_90": [2],
            "goals_b_90": [1],
        }
    )

    perspective = make_team_perspective_rows(rows)

    assert len(perspective) == 2
    assert perspective.iloc[0]["focal_team_code"] == "AAA"
    assert perspective.iloc[0]["opponent_team_code"] == "BBB"
    assert perspective.iloc[0]["focal_rank_diff"] == 15
    assert perspective.iloc[0]["focal_ranking_points_diff"] == 200.0
    assert perspective.iloc[0]["goals_for_90"] == 2
    assert perspective.iloc[1]["focal_team_code"] == "BBB"
    assert perspective.iloc[1]["opponent_team_code"] == "AAA"
    assert perspective.iloc[1]["focal_rank_diff"] == -15
    assert perspective.iloc[1]["focal_ranking_points_diff"] == -200.0
    assert perspective.iloc[1]["goals_for_90"] == 1


def test_symmetric_predictor_swapping_teams_swaps_predictions():
    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = SymmetricModeScorePhaseSplitScorePredictor().fit(train)
    row = test.head(1).copy()
    swapped = row.copy()
    swap_pairs = [
        ("country_a", "country_b"),
        ("country_a_code", "country_b_code"),
        ("rank_a", "rank_b"),
        ("ranking_points_a", "ranking_points_b"),
        ("confederation_a", "confederation_b"),
    ]
    for left, right in swap_pairs:
        swapped[left] = row[right].values
        swapped[right] = row[left].values
    swapped["rank_diff"] = swapped["rank_b"] - swapped["rank_a"]
    swapped["ranking_points_diff"] = swapped["ranking_points_a"] - swapped["ranking_points_b"]

    original_rates = predictor.predict_rates(row).iloc[0]
    swapped_rates = predictor.predict_rates(swapped).iloc[0]
    original_scores = predictor.predict(row).iloc[0]
    swapped_scores = predictor.predict(swapped).iloc[0]

    assert original_rates["expected_goals_a"] == swapped_rates["expected_goals_b"]
    assert original_rates["expected_goals_b"] == swapped_rates["expected_goals_a"]
    assert original_scores["pred_goals_a"] == swapped_scores["pred_goals_b"]
    assert original_scores["pred_goals_b"] == swapped_scores["pred_goals_a"]


def test_symmetric_predictor_returns_scores_and_rates():
    matches = load_matches()
    train, test = split_train_test(matches, split=BacktestSplit((2006, 2010, 2014, 2018), 2022))
    predictor = SymmetricModeScorePhaseSplitScorePredictor().fit(train)
    sample = test.head(4)

    predictions = predictor.predict(sample)
    rates = predictor.predict_rates(sample)

    assert list(predictions.columns) == ["pred_goals_a", "pred_goals_b"]
    assert list(rates.columns) == ["expected_goals_a", "expected_goals_b"]
    assert len(predictions) == 4
    assert len(rates) == 4
    assert (predictions >= 0).all().all()
    assert (rates >= 0).all().all()


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
