"""Check that the 3D page can play a full season through the local server."""
from environment import api


def play_to_the_end(session, action=None):
    """Play all the weeks of the season and give back the last answer."""
    answer = None
    while not session.done:
        answer = session.step(action)
    return answer


def test_agent_plays_the_default_season():
    """The best trained farmer plays the normal season and gets the known result."""
    api.new_season({"mode": "watch", "model": "ppo", "seed": 0})
    answer = play_to_the_end(api.session)
    assert answer["step"]["end"] == "Harvest"
    assert round(answer["step"]["kg"]) == 1451
    assert api.session.total_spent == 51_000
    assert round(api.session.total_return, 2) == 31.17


def test_user_choices_have_consequences():
    """A user who only waits gets a much smaller harvest, and the season of the trained farmer is given to compare."""
    api.new_season({"mode": "play", "model": "ppo", "seed": 0})
    answer = play_to_the_end(api.session, action=0)
    assert answer["step"]["kg"] < 800
    assert api.session.total_spent == 0
    assert round(answer["comparison"]["kg"]) == 1451


def test_play_mode_needs_an_action():
    """A wrong action in the I play mode is refused."""
    api.new_season({"mode": "play", "model": "expert", "seed": 1})
    try:
        api.session.step(9)
    except ValueError:
        return
    raise AssertionError("The action 9 should be refused")


def test_crop_failure_gives_the_penalty_of_the_environment():
    """A season with a failed crop shows the real penalty, and a season with a harvest shows none."""
    api.new_season({"mode": "play", "model": "ppo", "seed": 5, "test": "outbreak"})
    answer = play_to_the_end(api.session, action=0)
    assert answer["step"]["end"] == "Crop failure"
    assert answer["failure_penalty"] == 10
    assert answer["comparison"]["failure_penalty"] is None

    api.new_season({"mode": "watch", "model": "ppo", "seed": 0})
    answer = play_to_the_end(api.session)
    assert "failure_penalty" not in answer
