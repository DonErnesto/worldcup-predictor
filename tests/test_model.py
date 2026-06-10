import pandas as pd

from worldcup_predictor.data import BacktestSplit, load_matches, split_train_test
from worldcup_predictor.model import MostCommonScorePredictor, PhaseSplitScorePredictor


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
