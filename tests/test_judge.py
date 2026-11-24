"""Check the expert judge, the expert advice and the notes that the 3D page shows."""
from environment import api, judge
from environment.custom_env import MaizeSeasonEnv


def start_of_season(seed=0):
    """Start a season and give back the farm and the first 12 numbers the farmer sees."""
    env = MaizeSeasonEnv()
    obs, _ = env.reset(seed=seed)
    return env, obs


def test_expert_gives_fertilizer_first():
    """In the first week the expert gives the first dose of fertilizer, with a reason."""
    env, obs = start_of_season()
    action, reason = judge.expert_advice(obs, env)
    assert action == 4
    assert "first dose" in reason


def test_judge_marks_a_wasted_spray_as_poor():
    """A spray when no plant is known to be sick is a poor decision, and the better choice is given."""
    env, obs = start_of_season()
    verdict = judge.judge(obs, env, 2)
    assert verdict["level"] == "poor"
    assert verdict["better"] == 4
    assert verdict["better_why"]


def test_judge_marks_the_expert_choice_as_good():
    """The choice of the expert is always a good decision, and there is no better choice."""
    env, obs = start_of_season()
    verdict = judge.judge(obs, env, 4)
    assert verdict["level"] == "good"
    assert verdict["better"] is None


def test_fourth_spray_is_poor():
    """After 3 sprays the judge says that another spray does nothing."""
    env, obs = start_of_season()
    env.sprays_used = 3
    env.week = 8
    verdict = judge.judge(obs, env, 2)
    assert verdict["level"] == "poor"
    assert "3 sprays" in verdict["text"]


def test_notes_are_plain_sentences():
    """The notes are a list of short sentences, and there is always at least one."""
    env, obs = start_of_season()
    notes = judge.notices(obs, env)
    assert 1 <= len(notes) <= 3
    assert all(isinstance(note, str) and note.endswith(".") for note in notes)


def test_notes_never_give_more_than_three():
    """Even when many things need attention, the panel gets only the 3 most important notes."""
    env, obs = start_of_season()
    env.week = 9
    env.budget = 1000
    env.last_scout = -1
    obs = obs.copy()
    obs[2] = 0.1
    obs[3] = 0.9
    obs[4] = 0.2
    obs[5] = 0.3
    obs[10] = 0.8
    notes = judge.notices(obs, env)
    assert len(notes) == 3
    # The most important notes come first, and a note of rank 1 is never dropped for a lower one.
    assert "The maize is flowering and the soil is dry." in notes
    assert "The field is full of weeds." in notes
    assert "Money is running low." in notes

    for seed in range(5):
        env, obs = start_of_season(seed)
        for week in range(18):
            env.week = week
            assert len(judge.notices(obs, env)) <= 3


def test_notes_say_fine_when_nothing_stands_out():
    """When no note applies, the farmer is told that the field looks fine."""
    env, obs = start_of_season()
    obs = obs.copy()
    obs[2] = 0.5
    obs[3] = 0.5
    obs[4] = 0.6
    obs[5] = 0.0
    obs[10] = 0.0
    env.week = 2
    assert judge.notices(obs, env) == ["The field looks fine this week."]


def test_api_gives_the_notes_the_judge_and_the_counts():
    """Each week of the season has notes, a judge result and the count of good decisions."""
    api.new_season({"mode": "watch", "model": "expert", "seed": 0})
    answer = None
    while not api.session.done:
        answer = api.session.step()
        step = answer["step"]
        assert step["notes"]
        assert step["judge"]["level"] in ("good", "ok", "poor")
        assert 0 <= step["good"] <= step["judged"]
        assert "weeds_before" in step and "weeds_after" in step
    assert answer["your_judged"] == 18
    assert 0 <= answer["your_good"] <= 18


def test_api_compares_the_good_decisions_in_the_play_mode():
    """At the end of a season that the user played, the good decisions of both players are given."""
    api.new_season({"mode": "play", "model": "expert", "seed": 0})
    answer = None
    while not api.session.done:
        answer = api.session.step(0)
    assert answer["your_judged"] == 18
    assert answer["comparison"]["judged"] == 18
    assert answer["comparison"]["good"] >= answer["your_good"]
