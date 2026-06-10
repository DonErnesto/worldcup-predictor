import pytest

from worldcup_predictor.site_data import build_site_payload, normalize_head_to_head_match


@pytest.fixture(scope="module")
def site_payload():
    return build_site_payload()


def test_site_payload_has_2026_teams_and_ordered_predictions(site_payload):
    payload = site_payload

    assert len(payload["teams"]) == 48
    assert payload["metadata"]["model_name"] == "phase_split_poisson_alpha_0_1"
    assert payload["metadata"]["model_alpha"] == 0.1
    assert payload["metadata"]["ranking_snapshot_date"] == "2026-04-01"
    assert {selector["id"] for selector in payload["score_selectors"]} == {"standard", "kicktipp"}

    predictions = payload["predictions"]
    assert len(predictions) == 48 * 47 * 2 * 2
    assert not any(key.split("|")[0] == key.split("|")[1] for key in predictions)


def test_site_predictions_include_required_fields(site_payload):
    payload = site_payload
    prediction = next(iter(payload["predictions"].values()))

    assert {
        "country_a_code",
        "country_b_code",
        "phase",
        "score_selector",
        "rank_a",
        "rank_b",
        "ranking_points_a",
        "ranking_points_b",
        "expected_goals_a",
        "expected_goals_b",
        "selected_goals_a",
        "selected_goals_b",
    }.issubset(prediction)


def test_site_predictions_include_standard_and_kicktipp_selectors(site_payload):
    payload = site_payload
    team_a, team_b = payload["teams"][0], payload["teams"][1]

    assert f"{team_a['code']}|{team_b['code']}|group|standard" in payload["predictions"]
    assert f"{team_a['code']}|{team_b['code']}|group|kicktipp" in payload["predictions"]


def test_head_to_head_score_normalization_handles_reversed_order():
    match = {
        "match_date": "2022-12-13",
        "tournament_year": 2022,
        "stage": "semi_final",
        "country_a_code": "ARG",
        "country_b_code": "CRO",
        "goals_a_90": 3,
        "goals_b_90": 0,
    }

    normalized = normalize_head_to_head_match(match, "CRO", "ARG")

    assert normalized["country_a"] == "CRO"
    assert normalized["country_b"] == "ARG"
    assert normalized["score"] == "0-3"
