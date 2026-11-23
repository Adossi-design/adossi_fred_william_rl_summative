"""These helpers show what an agent does, with action traces, action charts and detector use."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import ACTION_NAMES, MaizeSeasonEnv

torch.set_num_threads(1)


@torch.no_grad()
def trace(policy, n=100, seed0=10_000, is_actor=True, **env_kwargs):
    """Play n seasons and return every step and a summary of every season."""
    steps = []
    seasons = []

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for season in range(n):
            env = MaizeSeasonEnv(**env_kwargs)
            obs, _ = env.reset(seed=seed0 + season)
            done = False
            total_return = 0.0
            total_cost = 0

            while not done:
                week = env.week
                if is_actor:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    data = TensorDict({"observation": obs_tensor}, batch_size=[])
                    action = int(policy(data)["action"])
                else:
                    action = policy(obs, env)

                # Each step keeps the week, the action, the detected disease and the weeks since the last scout.
                steps.append((week, action, float(obs[5]), float(obs[6])))
                obs, reward, done, _, info = env.step(action)
                total_return += reward
                total_cost += info["cost"]

            # Each season keeps if the crop failed, the return, the yield in tonnes per hectare and the money spent.
            # A failed crop gives a harvest of 0 kg, and this also counts a failure in the last week.
            crop_failed = info["kg"] == 0.0
            yield_t_ha = info["kg"] / 0.5 / 1000
            seasons.append((crop_failed, total_return, yield_t_ha, total_cost))

    return steps, seasons


def action_chart(steps, title, path):
    """Save a bar chart of how often each action is used in each week."""
    counts = np.zeros((18, len(ACTION_NAMES)))
    for week, action, *_ in steps:
        counts[week, action] += 1

    # Turn the counts into shares of the seasons.
    shares = counts / np.maximum(counts.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(18)
    for action in range(len(ACTION_NAMES)):
        ax.bar(np.arange(1, 19), shares[:, action], bottom=bottom, label=ACTION_NAMES[action])
        bottom += shares[:, action]
    ax.set_xlabel("Week of season")
    ax.set_ylabel("Share of seasons choosing action")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return shares
