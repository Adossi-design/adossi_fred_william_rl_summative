"""Tell an app which action the trained farmer would take this week."""
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from environment.custom_env import ACTION_NAMES

ROOT = Path(__file__).resolve().parent
BEST_FILE = ROOT / "logs" / "best.json"

# The best trained farmer of each method, written as the method name and the experiment number.
TRAINED_MODELS = {
    "ppo": ("ppo", 2),
    "a2c": ("a2c", 3),
    "reinforce": ("reinforce", 3),
    "dqn": ("dqn", 10),
}

# The trained farmers that are already loaded, so each one is loaded only once.
loaded_actors = {}


def best_model():
    """Find which method and experiment gave the best trained farmer."""
    if BEST_FILE.exists():
        best = json.loads(BEST_FILE.read_text())
        return best["algo"], int(best["exp"])
    return "ppo", 1


def best_seed():
    """Find which training run gave the best trained farmer."""
    if BEST_FILE.exists():
        best = json.loads(BEST_FILE.read_text())
        return int(best.get("seed", 1))
    return 1


def model_path(algo, exp, seed=1, tag=None):
    """Find the file where a trained farmer is saved."""
    if tag is None:
        tag = algo

    # The first training run is kept in the seed1 folder, and its file names have no seed.
    if seed == 1 and tag == algo:
        if algo == "dqn":
            return ROOT / "models" / "seed1" / "dqn" / f"exp{exp}.pt"
        return ROOT / "models" / "seed1" / "pg" / f"{algo}_exp{exp}.pt"

    if algo == "dqn":
        return ROOT / "models" / "dqn" / f"exp{exp}_seed{seed}.pt"
    return ROOT / "models" / "pg" / f"{tag}_exp{exp}_seed{seed}.pt"


def load_actor(algo, exp, seed=1, tag=None):
    """Load a trained farmer from its saved file."""
    from training.common import make_env

    env = make_env()
    if algo == "dqn":
        from training.dqn_training import build_actor

        actor = build_actor(env)
    else:
        from training.pg_training import TABLES, build_actor

        if algo == "reinforce":
            hidden = TABLES[algo][exp][5]
        else:
            hidden = (64, 64)
        actor = build_actor(env, hidden)

    actor.load_state_dict(torch.load(model_path(algo, exp, seed, tag)))
    actor.eval()
    return actor


def get_actor(algo, exp, seed=1):
    """Get a trained farmer, and load it from the file only the first time."""
    key = (algo, exp, seed)
    if key not in loaded_actors:
        loaded_actors[key] = load_actor(algo, exp, seed)
    return loaded_actors[key]


def pick_action(actor, observation):
    """Pick the action of the trained farmer for this week."""
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
    data = TensorDict({"observation": obs_tensor}, batch_size=[])
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        return int(actor(data)["action"])


def action_chances(actor, algo, observation):
    """Give the chance of each action for the trained farmer, or its preference for DQN."""
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
    data = TensorDict({"observation": obs_tensor}, batch_size=[])
    with torch.no_grad():
        data = actor(data)
        # The three policy methods give logits, and DQN gives the value of each action.
        if algo == "dqn":
            scores = data["action_value"]
        else:
            scores = data["logits"]
        return torch.softmax(scores, dim=-1).tolist()


def recommend(observation, algo=None, exp=None):
    """Tell the name of the action the farmer should take, from the 12 numbers the farmer sees."""
    if algo is None:
        algo, exp = best_model()
        seed = best_seed()
    else:
        seed = 1

    actor = get_actor(algo, exp, seed)
    return ACTION_NAMES[pick_action(actor, observation)]
