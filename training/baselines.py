"""Measure the simple farmers on the farm and calibrate YIELD_MAX."""
import argparse

import numpy as np

import environment.custom_env as ce
from environment.custom_env import PLOT_HA, MaizeSeasonEnv


def wait_policy(obs, env):
    """Always wait."""
    return 0


def random_policy(obs, env):
    """Pick a random action every week."""
    return int(env.np_random.integers(0, 8))


def calendar_policy(obs, env):
    """Fertilize and spray on fixed weeks and never check the field, like the old habit of many farmers."""
    plan = {0: 4, 1: 2, 2: 4, 3: 2, 4: 4, 5: 2, 6: 4, 7: 2, 8: 4, 9: 2}
    return plan.get(env.week, 0)


def expert_policy(obs, env):
    """Follow the guidance of farming experts (IITA and the leaf blight guide), checked in order."""
    week = env.week
    scouted_last_week = env.last_scout == week - 1
    detected_share = obs[5]
    fungicide_left = obs[7]
    soil_moisture = obs[2]
    rain_forecast = obs[3]

    # Fertilize at planting, weed before the second dose, and give the second dose in week 4.
    if week == 0:
        return 4
    if week == 2:
        return 7
    if week == 4:
        return 4
    if week == 5:
        return 7

    # Late in the season nothing helps any more.
    if week >= 15:
        return 0

    # Spray when the drone found sick plants and there is no fungicide on the plants.
    if scouted_last_week and detected_share > 0.06 and fungicide_left == 0 and env.sprays_used < ce.MAX_SPRAYS:
        return 2

    # Irrigate when the soil is dry and no rain is coming, from week 6 to week 13.
    if 6 <= week <= 13 and soil_moisture < 0.3 and rain_forecast < 0.5:
        return 3

    # Scout when we never did or when the last scout was 2 or more weeks ago.
    if week >= 3 and (env.last_scout < 0 or week - env.last_scout >= 2):
        return 1

    return 0


def play(policy, seeds, **env_kwargs):
    """Play one season for each seed and return the average results."""
    returns = []
    yields = []
    spent = []
    biomass_ratios = []
    failures = 0

    for seed in seeds:
        env = MaizeSeasonEnv(**env_kwargs)
        obs, _ = env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
        done = False
        total_return = 0.0
        total_cost = 0

        while not done:
            action = policy(obs, env)
            obs, reward, done, _, info = env.step(action)
            total_return += reward
            total_cost += info["cost"]

        returns.append(total_return)
        yields.append(info["kg"] / PLOT_HA / 1000)
        spent.append(total_cost)
        biomass_ratios.append(env.biomass_ratio())

        # A failed crop gives a harvest of 0 kg, and this also counts a failure in the last week.
        if info["kg"] == 0.0:
            failures += 1

    return dict(
        ret=np.mean(returns),
        ret_std=np.std(returns),
        yld=np.mean(yields),
        spent=np.mean(spent),
        B=np.mean(biomass_ratios),
        failures=failures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        # Yield grows in step with YIELD_MAX, so one run is enough to find the right value.
        trial_value = ce.YIELD_MAX
        results = play(wait_policy, range(200))
        new_value = float(trial_value * 1.2 / results["yld"])
        print(
            f"Wait-only yield with YIELD_MAX={trial_value}: {results['yld']:.4f} t/ha "
            f"-> YIELD_MAX = {new_value!r}"
        )
    else:
        print("YIELD_MAX =", ce.YIELD_MAX)
        policies = [
            ("Wait only", wait_policy),
            ("Random", random_policy),
            ("Old calendar habit", calendar_policy),
            ("Expert rules", expert_policy),
        ]
        print("| Farmer | Score | t/ha | Money spent | Outbreak score | Failed fields | Failed fields (outbreak) |")
        print("|---|---|---|---|---|---|---|")
        for name, policy in policies:
            normal = play(policy, range(100))
            outbreak = play(policy, range(100), max_initial_infected=8)
            print(
                f"| {name} | {normal['ret']:.2f} | {normal['yld']:.2f} | {normal['spent']:,.0f} XAF "
                f"| {outbreak['ret']:.2f} | {normal['failures']} | {outbreak['failures']} |"
            )
