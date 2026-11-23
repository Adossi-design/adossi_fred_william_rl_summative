"""Train a DQN agent."""
import argparse
import time

import numpy as np
import torch
from tensordict.nn import TensorDictSequential

# TorchRL renamed SyncDataCollector to Collector, so we accept both names.
try:
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from training.common import MODELS, TOTAL_FRAMES, CsvLogger, evaluate, make_env, set_seed

torch.set_num_threads(1)

# Each experiment gives the learning rate, gamma, buffer size, batch size, final epsilon, share of frames for epsilon, target update interval and what it tests.
EXPERIMENTS = {
    1: (1e-3, 0.99, 50_000, 64, 0.05, 0.5, 500, "Baseline"),
    2: (1e-4, 0.99, 50_000, 64, 0.05, 0.5, 500, "Slow learning"),
    3: (5e-3, 0.99, 50_000, 64, 0.05, 0.5, 500, "Fast learning, unstable?"),
    4: (1e-3, 0.90, 50_000, 64, 0.05, 0.5, 500, "Short-sighted: ignores harvest?"),
    5: (1e-3, 0.95, 50_000, 64, 0.05, 0.5, 500, "Medium horizon"),
    6: (1e-3, 0.99, 5_000, 64, 0.05, 0.5, 500, "Small replay memory"),
    7: (1e-3, 0.99, 50_000, 256, 0.05, 0.5, 500, "Large batch"),
    8: (1e-3, 0.99, 50_000, 64, 0.05, 0.1, 500, "Explores too briefly?"),
    9: (1e-3, 0.99, 50_000, 64, 0.20, 0.5, 500, "Keeps exploring a lot"),
    10: (1e-3, 0.99, 50_000, 64, 0.05, 0.5, 50, "Target network updated often"),
}


def build_actor(env):
    """Build the network that gives a score to each of the 8 actions."""
    network = MLP(in_features=12, out_features=8, num_cells=[64, 64])
    return QValueActor(network, in_keys=["observation"], spec=env.action_spec)


def main(exp, total_frames=TOTAL_FRAMES, seed=1, out_dir=None, tag="dqn"):
    """Train one experiment, write its CSV log and save the agent."""
    settings = EXPERIMENTS[exp]
    learning_rate = settings[0]
    gamma = settings[1]
    buffer_size = settings[2]
    batch_size = settings[3]
    epsilon_end = settings[4]
    decay_share = settings[5]
    target_update_interval = settings[6]

    set_seed(seed)
    env = make_env(seed)
    actor = build_actor(env)

    # Epsilon starts at 1 and falls in a straight line to its final value.
    exploration = EGreedyModule(
        env.action_spec,
        eps_init=1.0,
        eps_end=epsilon_end,
        annealing_num_steps=int(decay_share * total_frames),
    )
    policy = TensorDictSequential(actor, exploration)

    # Each batch is 180 frames, which is 10 seasons.
    collector = SyncDataCollector(
        lambda: make_env(seed),
        policy,
        frames_per_batch=180,
        total_frames=total_frames,
    )
    buffer = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)

    # The loss keeps a slow copy of the network, called the target network.
    loss_function = DQNLoss(actor, action_space=env.action_spec, delay_value=True)
    loss_function.make_value_estimator(gamma=gamma)
    target_updater = HardUpdate(loss_function, value_network_update_interval=target_update_interval)
    optimizer = torch.optim.Adam(loss_function.parameters(), lr=learning_rate)

    if out_dir:
        logger = CsvLogger(tag, exp, seed, log_dir=out_dir)
    else:
        logger = CsvLogger(tag, exp, seed)

    frames = 0
    episodes = 0
    for batch_number, data in enumerate(collector):
        frames += data.numel()
        buffer.extend(data.reshape(-1))

        # Learn 10 times from random samples of the buffer.
        losses = []
        if len(buffer) >= batch_size:
            for _ in range(10):
                sample = buffer.sample()
                loss_values = loss_function(sample)
                loss_values["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                target_updater.step()
                losses.append(loss_values["loss"].item())

        exploration.step(data.numel())

        # Count the seasons that finished in this batch.
        done = data["next", "done"]
        finished_returns = data["next", "episode_reward"][done]
        episodes += int(done.sum())

        row = dict(frames=frames, episodes=episodes, epsilon=float(exploration.eps))
        if finished_returns.numel() > 0:
            row["mean_episode_return"] = float(finished_returns.mean())
        else:
            row["mean_episode_return"] = None
        if len(losses) > 0:
            row["loss"] = float(np.mean(losses))
        else:
            row["loss"] = None

        # Every 10 batches, test the agent on 20 seasons.
        if (batch_number + 1) % 10 == 0:
            eval_return, eval_yield, _ = evaluate(actor, 20)
            row["eval_return"] = eval_return
            row["eval_yield_t_ha"] = eval_yield

        logger.log(**row)

    collector.shutdown()
    (MODELS / "dqn").mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), MODELS / "dqn" / f"exp{exp}_seed{seed}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    start_time = time.time()
    main(args.exp, args.frames, args.seed)
    minutes = (time.time() - start_time) / 60
    print(f"dqn exp{args.exp} seed{args.seed} done in {minutes:.1f} min")
