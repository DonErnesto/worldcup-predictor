from worldcup_predictor.backtest import run_backtest
from worldcup_predictor.data import load_matches


def test_full_backtest_smoke():
    predictions, summary = run_backtest(load_matches())
    assert len(predictions) == 256
    assert set(summary["test_year"]) == {2010, 2014, 2018, 2022}
    assert {"score_points", "actual_outcome", "pred_outcome"}.issubset(predictions.columns)
    assert predictions["score_points"].between(0, 4).all()


def test_most_common_score_backtest_smoke():
    predictions, summary = run_backtest(load_matches(), predictor_name="most_common_score")
    assert len(predictions) == 256
    assert set(summary["test_year"]) == {2010, 2014, 2018, 2022}
    assert predictions["score_points"].between(0, 4).all()


def test_mode_score_backtest_supports_three_tournament_training_window():
    predictions, summary = run_backtest(
        load_matches(),
        predictor_name="mode_score_phase_split_poisson",
        train_window=3,
    )
    assert len(predictions) == 320
    assert set(summary["test_year"]) == {2006, 2010, 2014, 2018, 2022}
    assert set(summary["train_window"]) == {3}
    assert predictions["score_points"].between(0, 4).all()


def test_symmetric_mode_score_backtest_smoke():
    predictions, summary = run_backtest(
        load_matches(),
        predictor_name="symmetric_mode_score_phase_split_poisson",
    )
    assert len(predictions) == 256
    assert set(summary["test_year"]) == {2010, 2014, 2018, 2022}
    assert predictions["score_points"].between(0, 4).all()
    assert {"expected_goals_a", "expected_goals_b"}.issubset(predictions.columns)
