"""Every training script shares this code for the environment, the evaluation and the CSV log."""
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper, RewardSum, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import PLOT_HA, MaizeSeasonEnv

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
TOTAL_FRAMES = int(os.environ.get("TOTAL_FRAMES", 200_000))

# During training the agent is tested on the seasons with seeds 5000 to 5019.
EVAL_SEEDS_START = 5_000

COLUMNS = [
    "frames",
    "episodes",
    "mean_episode_return",
    "loss",
    "entropy",
    "epsilon",
    "eval_return",
    "eval_yield_t_ha",
    "seconds",
]


def set_seed(seed):
    """Make the random numbers repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed=0, **kwargs):
    """Build the farm in the format TorchRL needs."""
    env = GymWrapper(MaizeSeasonEnv(**kwargs), categorical_action_encoding=True)
    # RewardSum adds the reward of the whole season to the data.
    env = TransformedEnv(env, RewardSum())
    env.set_seed(seed)
    return env


@torch.no_grad()
def evaluate(policy, n_seasons=20, seed_start=EVAL_SEEDS_START, return_actions=False, **env_kwargs):
    """Play seasons without exploring and return the average return, yield and money spent."""
    returns = []
    yields = []
    spent = []
    actions = []

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for season in range(n_seasons):
            env = MaizeSeasonEnv(**env_kwargs)
            obs, _ = env.reset(seed=seed_start + season)
            done = False
            total_return = 0.0
            total_cost = 0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                data = TensorDict({"observation": obs_tensor}, batch_size=[])
                action = int(policy(data)["action"])
                actions.append((env.week, action))
                obs, reward, done, _, info = env.step(action)
                total_return += reward
                total_cost += info["cost"]

            returns.append(total_return)
            yields.append(info["kg"] / PLOT_HA / 1000)
            spent.append(total_cost)

    results = (float(np.mean(returns)), float(np.mean(yields)), float(np.mean(spent)))
    if return_actions:
        return results + (float(np.std(returns)), actions)
    return results


class CsvLogger:
    """Write one row for each training step to a CSV file in the logs folder."""

    def __init__(self, algo, exp, seed=1, log_dir=LOGS):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{algo}_exp{exp}_seed{seed}.csv"
        self.start_time = time.time()
        with open(self.path, "w", newline="") as file:
            csv.writer(file).writerow(COLUMNS)

    def log(self, **values):
        """Add one row and leave empty any column that has no value."""
        row = []
        for column in COLUMNS[:-1]:
            value = values.get(column, "")
            if value is None:
                value = ""
            row.append(value)
        row.append(round(time.time() - self.start_time, 1))

        with open(self.path, "a", newline="") as file:
            csv.writer(file).writerow(row)
