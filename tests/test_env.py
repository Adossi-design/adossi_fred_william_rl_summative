"""Tests for the maize farm environment."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTION_COST, BUDGET0, MAX_SPRAYS, MaizeSeasonEnv


def test_check_env():
    """The environment follows the Gymnasium rules."""
    check_env(MaizeSeasonEnv())


def test_reset_obs():
    """The first observation has 12 numbers between 0 and 1."""
    obs, _ = MaizeSeasonEnv().reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.shape == (12,)
    assert obs.min() >= 0
    assert obs.max() <= 1


def test_reproducible():
    """The same seed and the same actions give the same rewards."""

    def play_season():
        env = MaizeSeasonEnv()
        env.reset(seed=7)
        actions = [1, 0, 3, 2, 6, 0, 4, 1, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        rewards = []
        for action in actions:
            result = env.step(action)
            rewards.append(result[1])
        return rewards

    assert play_season() == play_season()


def test_wait_season():
    """A season of waiting lasts 18 weeks and costs nothing."""
    env = MaizeSeasonEnv()
    env.reset(seed=1)
    weeks_played = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(0)
        weeks_played += 1
    assert weeks_played == 18
    assert env.budget == BUDGET0


def test_budget_exhaustion():
    """An action the farmer cannot afford costs no money and gets a penalty of 0.5."""
    env = MaizeSeasonEnv()
    env.reset(seed=2)
    fertilizer_cost = ACTION_COST[4]

    # Buy fertilizer until the money left is less than one more purchase.
    while env.budget >= fertilizer_cost:
        _, reward, done, _, _ = env.step(4)
        assert not done
        assert abs(reward - (-fertilizer_cost / 10_000)) < 1e-9

    # The next purchase is too expensive, so it only gets the penalty.
    budget_before = env.budget
    _, reward, _, _, _ = env.step(4)
    assert abs(reward - (-0.5)) < 1e-9
    assert env.budget == budget_before
    assert env.budget >= 0


def test_remove_only_flagged():
    """Removing plants takes out only the plants that were flagged."""
    env = MaizeSeasonEnv(max_initial_infected=8)
    env.reset(seed=3)
    # Scout first and then remove, and only the flagged plants should go.
    env.step(1)
    flagged = env.flagged.copy()
    env.step(5)
    assert (env.removed == flagged).all()


def test_eight_actions():
    """The farm has 8 actions, and the new one is weeding for 3,000 XAF."""
    env = MaizeSeasonEnv()
    assert env.action_space.n == 8
    assert len(ACTION_COST) == 8
    assert ACTION_COST[7] == 3000


def test_weeding_sets_the_weeds_to_zero():
    """After weeding, the weeds are cleared, and then they start to grow again."""
    env = MaizeSeasonEnv()
    env.reset(seed=4)
    for _ in range(3):
        env.step(0)
    assert env.weeds > 0.2
    budget_before = env.budget
    env.step(7)
    assert env.budget == budget_before - 3000
    # The weeds grow a little in the same week, so they are small but not the old size.
    assert env.weeds < 0.2


def test_fourth_spray_costs_money_but_does_nothing():
    """Only 3 sprays work in a season, and the 4th costs money without any protection."""
    env = MaizeSeasonEnv(max_initial_infected=8)
    env.reset(seed=3)
    for _ in range(MAX_SPRAYS):
        env.step(2)
    assert env.sprays_used == MAX_SPRAYS
    # Let the protection of the last spray run out.
    env.step(0)
    env.step(0)
    assert env.fungicide_weeks == 0
    budget_before = env.budget
    _, reward, _, _, _ = env.step(2)
    assert env.sprays_used == MAX_SPRAYS + 1
    assert env.budget == budget_before - ACTION_COST[2]
    assert env.fungicide_weeks == 0
    assert abs(reward - (-ACTION_COST[2] / 10_000)) < 1e-9


def test_late_fertilizer_adds_much_less():
    """A dose of fertilizer in week 10 adds much less food than a dose in week 0."""

    def food_added(week):
        env = MaizeSeasonEnv()
        env.reset(seed=6)
        env.week = week
        env.nitrogen = 0.30
        before = env.nitrogen
        env.step(4)
        # The soil loses 0.04 at the end of the week, so add it back.
        return env.nitrogen + 0.04 - before

    early = food_added(0)
    late = food_added(10)
    assert late < early / 5


def test_fertilizer_on_rich_soil_burns_the_plants():
    """A dose of fertilizer on soil above 0.85 burns the plants and slows their growth."""

    def growth_after_fertilizer(nitrogen):
        env = MaizeSeasonEnv()
        env.reset(seed=8)
        env.week = 4
        env.nitrogen = nitrogen
        env.step(4)
        return env.burn, env.biomass.sum()

    burned, biomass_burned = growth_after_fertilizer(0.90)
    not_burned, _ = growth_after_fertilizer(0.50)
    assert burned and not not_burned
    assert biomass_burned > 0
