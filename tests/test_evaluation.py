from worldcup_predictor.evaluation import Score, score_points


def test_non_draw_exact_score_gets_best_tier_only():
    assert score_points(Score(2, 1), Score(2, 1)) == 4


def test_non_draw_correct_signed_goal_difference_gets_three():
    assert score_points(Score(3, 1), Score(2, 0)) == 3


def test_non_draw_correct_winner_only_gets_two():
    assert score_points(Score(2, 0), Score(1, 0)) == 2


def test_actual_draw_exact_score_gets_four():
    assert score_points(Score(1, 1), Score(1, 1)) == 4


def test_actual_draw_wrong_draw_score_gets_two():
    assert score_points(Score(1, 1), Score(0, 0)) == 2


def test_wrong_prediction_gets_zero():
    assert score_points(Score(2, 0), Score(0, 1)) == 0
