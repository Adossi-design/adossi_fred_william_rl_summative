"""Train a REINFORCE, PPO or A2C agent."""
import argparse
import time

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

# TorchRL renamed SyncDataCollector to Collector, so we accept both names.
try:
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector

from training.common import MODELS, TOTAL_FRAMES, CsvLogger, evaluate, make_env, set_seed

torch.set_num_threads(1)

# Test the agent every 1800 frames, like the DQN script does.
EVAL_EVERY = 1800

# Each REINFORCE experiment gives the learning rate, gamma, entropy coefficient, return normalization, seasons per update, hidden layers and what it tests.
REINFORCE = {
    1: (1e-3, 0.99, 0.01, True, 10, (64, 64), "Baseline"),
    2: (1e-4, 0.99, 0.01, True, 10, (64, 64), "Slow learning"),
    3: (5e-3, 0.99, 0.01, True, 10, (64, 64), "Fast learning"),
    4: (1e-3, 0.90, 0.01, True, 10, (64, 64), "Short horizon"),
    5: (1e-3, 0.99, 0.0, True, 10, (64, 64), "No entropy bonus"),
    6: (1e-3, 0.99, 0.05, True, 10, (64, 64), "Strong entropy bonus"),
    7: (1e-3, 0.99, 0.01, False, 10, (64, 64), "No baseline: noisier?"),
    8: (1e-3, 0.99, 0.01, True, 1, (64, 64), "Single-season updates"),
    9: (1e-3, 0.99, 0.01, True, 50, (64, 64), "Large batches"),
    10: (1e-3, 0.99, 0.01, True, 10, (128, 128), "Bigger network"),
}

# Each PPO experiment gives the learning rate, gamma, GAE lambda, clip epsilon, entropy coefficient, epochs and what it tests.
PPO = {
    1: (3e-4, 0.99, 0.95, 0.2, 0.01, 4, "Baseline"),
    2: (1e-3, 0.99, 0.95, 0.2, 0.01, 4, "Faster learning"),
    3: (1e-4, 0.99, 0.95, 0.2, 0.01, 4, "Slower learning"),
    4: (3e-4, 0.90, 0.95, 0.2, 0.01, 4, "Short horizon"),
    5: (3e-4, 0.99, 0.80, 0.2, 0.01, 4, "More critic, less noise"),
    6: (3e-4, 0.99, 1.00, 0.2, 0.01, 4, "Full returns, more noise"),
    7: (3e-4, 0.99, 0.95, 0.1, 0.01, 4, "Tighter clipping"),
    8: (3e-4, 0.99, 0.95, 0.3, 0.01, 4, "Looser clipping"),
    9: (3e-4, 0.99, 0.95, 0.2, 0.0, 4, "No entropy bonus"),
    10: (3e-4, 0.99, 0.95, 0.2, 0.01, 10, "More passes per batch"),
}

# Each A2C experiment gives the learning rate, gamma, GAE lambda, entropy coefficient, critic coefficient, seasons per update and what it tests.
A2C = {
    1: (7e-4, 0.99, 0.95, 0.01, 0.5, 10, "Baseline"),
    2: (1e-4, 0.99, 0.95, 0.01, 0.5, 10, "Slow learning"),
    3: (3e-3, 0.99, 0.95, 0.01, 0.5, 10, "Fast learning"),
    4: (7e-4, 0.90, 0.95, 0.01, 0.5, 10, "Short horizon"),
    5: (7e-4, 0.99, 1.00, 0.01, 0.5, 10, "Full returns"),
    6: (7e-4, 0.99, 0.50, 0.01, 0.5, 10, "Relies on the critic"),
    7: (7e-4, 0.99, 0.95, 0.0, 0.5, 10, "No entropy bonus"),
    8: (7e-4, 0.99, 0.95, 0.05, 0.5, 10, "Strong entropy bonus"),
    9: (7e-4, 0.99, 0.95, 0.01, 1.0, 10, "Critic weighted more"),
    10: (7e-4, 0.99, 0.95, 0.01, 0.5, 1, "Single-season updates"),
}

TABLES = {"reinforce": REINFORCE, "ppo": PPO, "a2c": A2C}


def build_actor(env, hidden=(64, 64)):
    """Build the network that gives the chance of each of the 8 actions."""
    network = MLP(in_features=12, out_features=8, num_cells=list(hidden))
    logits_module = TensorDictModule(network, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        logits_module,
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True,
        spec=env.action_spec,
    )


def build_critic(hidden=(64, 64)):
    """Build the network that guesses how good a situation is."""
    network = MLP(in_features=12, out_features=1, num_cells=list(hidden))
    return ValueOperator(network, in_keys=["observation"])


def returns_to_go(rewards, done, gamma):
    """Add up the future rewards of every step, going backwards through the season."""
    future_return = 0.0
    result = torch.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        if done[step]:
            future_return = 0.0
        future_return = rewards[step] + gamma * future_return
        result[step] = future_return
    return result


def main(algo, exp, total_frames=TOTAL_FRAMES, seed=1, out_dir=None, tag=None):
    """Train one experiment, write its CSV log and save the agent."""
    settings = TABLES[algo][exp]
    if tag is None:
        tag = algo

    set_seed(seed)
    env = make_env(seed)

    if algo == "reinforce":
        hidden = settings[5]
    else:
        hidden = (64, 64)
    actor = build_actor(env, hidden)
    if algo == "reinforce":
        critic = None
    else:
        critic = build_critic()

    if out_dir:
        logger = CsvLogger(tag, exp, seed, log_dir=out_dir)
    else:
        logger = CsvLogger(tag, exp, seed)

    # Read the settings of this algorithm and build its loss.
    if algo == "reinforce":
        learning_rate, gamma, entropy_coef, normalize_returns, seasons_per_update, _, _ = settings
        frames_per_batch = 18 * seasons_per_update
        parameters = actor.parameters()
    elif algo == "ppo":
        learning_rate, gamma, gae_lambda, clip_epsilon, entropy_coef, epochs, _ = settings
        frames_per_batch = 1800
        gae = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic)
        loss_function = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coef,
        )
        parameters = loss_function.parameters()
    else:
        learning_rate, gamma, gae_lambda, entropy_coef, critic_coef, seasons_per_update, _ = settings
        frames_per_batch = 18 * seasons_per_update
        gae = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic)
        loss_function = A2CLoss(
            actor_network=actor,
            critic_network=critic,
            entropy_coeff=entropy_coef,
            critic_coeff=critic_coef,
        )
        parameters = loss_function.parameters()

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # REINFORCE needs every batch to start with a fresh season.
    collector = SyncDataCollector(
        lambda: make_env(seed),
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        reset_at_each_iter=(algo == "reinforce"),
    )

    frames = 0
    episodes = 0
    last_eval_frames = 0
    for data in collector:
        batch_frames = data.numel()
        frames += batch_frames

        # Count the seasons that finished in this batch.
        done = data["next", "done"].squeeze(-1)
        finished_returns = data["next", "episode_reward"][data["next", "done"]]
        episodes += int(done.sum())

        losses = []
        entropies = []

        if algo == "reinforce":
            # Learn from the future rewards of every step.
            rewards = data["next", "reward"].squeeze(-1)
            returns = returns_to_go(rewards, done, gamma)
            if normalize_returns:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            distribution = actor.get_dist(data)
            log_probability = distribution.log_prob(data["action"])
            entropy = distribution.entropy()
            loss = -(log_probability * returns).mean() - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            entropies.append(entropy.mean().item())
        else:
            # Work out how much better each action was than expected.
            with torch.no_grad():
                gae(data)
            flat_data = data.reshape(-1)

            # PPO learns several times from the same batch while A2C learns only once.
            if algo == "ppo":
                number_of_passes = epochs
            else:
                number_of_passes = 1

            for _ in range(number_of_passes):
                # PPO shuffles the batch and cuts it into pieces of 256 steps while A2C uses the whole batch.
                if algo == "ppo":
                    permutation = torch.randperm(batch_frames)
                    minibatches = []
                    for start in range(0, batch_frames, 256):
                        indices = permutation[start:start + 256]
                        minibatches.append(flat_data[indices])
                else:
                    minibatches = [flat_data]

                for minibatch in minibatches:
                    loss_parts = loss_function(minibatch)

                    # Add up the loss terms that exist, because the entropy term is missing when its coefficient is 0.
                    total_loss = 0
                    for name in ("loss_objective", "loss_critic", "loss_entropy"):
                        if name in loss_parts.keys():
                            total_loss = total_loss + loss_parts[name]

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_function.parameters(), 1.0)
                    optimizer.step()
                    losses.append(total_loss.item())

                    if "entropy" in loss_parts.keys():
                        entropies.append(float(loss_parts["entropy"]))
                    else:
                        entropies.append(float(actor.get_dist(minibatch).entropy().mean()))

        row = dict(
            frames=frames,
            episodes=episodes,
            loss=float(np.mean(losses)),
            entropy=float(np.mean(entropies)),
        )
        if finished_returns.numel() > 0:
            row["mean_episode_return"] = float(finished_returns.mean())
        else:
            row["mean_episode_return"] = None

        # Test the agent on 20 seasons every 1800 frames.
        if frames - last_eval_frames >= EVAL_EVERY:
            last_eval_frames = frames
            eval_return, eval_yield, _ = evaluate(actor, 20)
            row["eval_return"] = eval_return
            row["eval_yield_t_ha"] = eval_yield

        logger.log(**row)

    collector.shutdown()
    (MODELS / "pg").mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), MODELS / "pg" / f"{tag}_exp{exp}_seed{seed}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(TABLES), required=True)
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fungicide-cost",
        type=int,
        default=None,
        help="sensitivity test: override spray cost (XAF)",
    )
    parser.add_argument("--tag", default=None, help="file-name prefix (default: the algorithm)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    # For the sensitivity test we change the price of spraying.
    if args.fungicide_cost is not None:
        import environment.custom_env as custom_env

        custom_env.ACTION_COST[2] = args.fungicide_cost

    start_time = time.time()
    main(args.algo, args.exp, args.frames, args.seed, args.out_dir, args.tag)
    minutes = (time.time() - start_time) / 60
    print(f"{args.algo} exp{args.exp} seed{args.seed} done in {minutes:.1f} min")
