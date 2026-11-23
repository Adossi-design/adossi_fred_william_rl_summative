"""Build the result tables, test the best agents on new seasons and draw the charts."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.custom_env import ACTION_NAMES
from farmer_policy import load_actor
from training import dqn_training, pg_training
from training.analysis import action_chart, trace
from training.baselines import calendar_policy, expert_policy, random_policy, wait_policy
from training.common import LOGS, ROOT

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#1f77b4", "reinforce": "#d62728", "ppo": "#2ca02c", "a2c": "#ff7f0e"}

# The three kinds of unseen seasons used for the final test.
TEST_SETS = {
    "New seeds": dict(),
    "Drought year": dict(rain_scale=0.6),
    "Heavy outbreak": dict(max_initial_infected=8),
}
TEST_SEED_START = 10_000
NOT_CONVERGED = "not converged within budget"

# Column names of the settings of each algorithm, in the order used in the experiment tables.
SETTING_NAMES = {
    "dqn": ["Learning rate", "Gamma", "Buffer size", "Batch size", "Epsilon end", "Decay share", "Target update"],
    "reinforce": ["Learning rate", "Gamma", "Entropy coef", "Return norm", "Seasons/update", "Hidden"],
    "ppo": ["Learning rate", "Gamma", "GAE lambda", "Clip eps", "Entropy coef", "Epochs"],
    "a2c": ["Learning rate", "Gamma", "GAE lambda", "Entropy coef", "Critic coef", "Seasons/update"],
}


def seeds_for(exp):
    """Return the two seeds used for the two runs of this experiment."""
    if exp == 2:
        return [1, 3]
    return [1, 2]


def csv_path(algo, exp, seed):
    """Return the log file of one run."""
    if seed == 1:
        return LOGS / "seed1" / f"{algo}_exp{exp}.csv"
    return LOGS / f"{algo}_exp{exp}_seed{seed}.csv"


def load(algo, exp, seed):
    """Read the log of one run."""
    return pd.read_csv(csv_path(algo, exp, seed))


def mean_reward(log):
    """Return the average of the last 3 evaluation returns of one run."""
    return float(log["eval_return"].dropna().tail(3).mean())


def mean_and_std(values):
    """Return the mean and the sample standard deviation of a few numbers."""
    numbers = np.array(values, dtype=float)
    if len(numbers) > 1:
        std = numbers.std(ddof=1)
    else:
        std = 0.0
    return numbers.mean(), std


def experiment_table(algo):
    """Build the table of the 10 experiments of one algorithm."""
    if algo == "dqn":
        experiments = dqn_training.EXPERIMENTS
    else:
        experiments = pg_training.TABLES[algo]
    setting_names = SETTING_NAMES[algo]

    rows = []
    for exp, settings in experiments.items():
        row = {"Exp": exp}
        for name, value in zip(setting_names, settings[:-1]):
            row[name] = value
        row["What it tests"] = settings[-1]

        seeds = seeds_for(exp)
        rewards = []
        for seed in seeds:
            rewards.append(mean_reward(load(algo, exp, seed)))
        mean, std = mean_and_std(rewards)

        row["First run"] = round(rewards[0], 2)
        row["Second run"] = round(rewards[1], 2)
        row["Second-run seed"] = seeds[1]
        row["Mean reward (2-seed)"] = f"{mean:.2f} ± {std:.2f}"
        row["_mean"] = mean
        rows.append(row)
    return pd.DataFrame(rows)


def per_episode(log):
    """Give every finished season its own return, so the training curve is per season."""
    data = log.dropna(subset=["mean_episode_return"])
    seasons_per_row = data["episodes"].diff().fillna(data["episodes"]).astype(int).clip(lower=0)
    return np.repeat(data["mean_episode_return"].values, seasons_per_row.values)


def moving_average(values, window=50):
    """Return the moving average of the values with the given window size."""
    return pd.Series(values).rolling(window, min_periods=window).mean().values


def convergence(log, tolerance=0.10):
    """Return the frame and season after which the evaluation return stays within 10% of its final value."""
    data = log.dropna(subset=["eval_return"])
    frames = data["frames"].values
    returns = data["eval_return"].values

    # The final value is the average of the last 3 evaluations.
    final_value = returns[-3:].mean()
    far_from_final = np.where(np.abs(returns - final_value) > tolerance * abs(final_value))[0]
    if len(far_from_final) > 0:
        first_settled = far_from_final[-1] + 1
    else:
        first_settled = 0

    # The settled part must include at least the last 3 evaluations.
    if first_settled > len(returns) - 3:
        return None
    return int(frames[first_settled]), frames[first_settled] / 18


def summarise(seasons, algorithm, seed, test_set):
    """Turn the seasons of one test into one row of the results table."""
    returns = []
    yields = []
    spent = []
    crop_failures = 0
    for crop_failed, total_return, yield_t_ha, cost in seasons:
        returns.append(total_return)
        yields.append(yield_t_ha)
        spent.append(cost)
        if crop_failed:
            crop_failures += 1

    return dict(
        algorithm=algorithm,
        seed=seed,
        test_set=test_set,
        mean_return=np.mean(returns),
        std_return=np.std(returns),
        mean_yield_t_ha=np.mean(yields),
        mean_spent_XAF=np.mean(spent),
        crop_failures=crop_failures,
    )


def main():
    ASSETS.mkdir(exist_ok=True)

    # Build the experiment tables and find the best setting of each algorithm.
    tables = {}
    best = {}
    for algo in ALGOS:
        table = experiment_table(algo)
        best[algo] = int(table.loc[table["_mean"].idxmax(), "Exp"])
        table.drop(columns="_mean").to_csv(LOGS / f"table_{algo}.csv", index=False)
        tables[algo] = table
        print(f"\n=== {LABEL[algo]} (best setting by 2-seed average: exp{best[algo]}) ===")
        print(table.drop(columns="_mean").to_string(index=False))

    differences = []
    for algo in ALGOS:
        table = tables[algo]
        differences += list((table["First run"] - table["Second run"]).abs())
    print(
        f"\nSeed 1 vs seed 2 per-experiment |difference| in mean reward: "
        f"mean {np.mean(differences):.2f}, max {np.max(differences):.2f}"
    )

    # Find how many seasons each best run needs to settle.
    convergence_rows = []
    for algo in ALGOS:
        for seed in seeds_for(best[algo]):
            settled = convergence(load(algo, best[algo], seed))
            if settled:
                settled_frame = settled[0]
                settled_episode = round(settled[1])
            else:
                settled_frame = NOT_CONVERGED
                settled_episode = NOT_CONVERGED
            convergence_rows.append(dict(
                algorithm=LABEL[algo],
                exp=best[algo],
                seed=seed,
                converged_at_frame=settled_frame,
                converged_at_episode=settled_episode,
            ))
    convergence_table = pd.DataFrame(convergence_rows)
    convergence_table.to_csv(LOGS / "convergence.csv", index=False)
    print("\n=== Convergence (eval-return curve, within 10% of final) ===")
    print(convergence_table.to_string(index=False))

    # Test both runs of each best setting on 100 new seasons for each test set.
    result_rows = []
    steps_by_run = {}
    for algo in ALGOS:
        for seed in seeds_for(best[algo]):
            actor = load_actor(algo, best[algo], seed)
            for test_name, env_settings in TEST_SETS.items():
                steps, seasons = trace(actor, 100, TEST_SEED_START, **env_settings)
                name = f"{LABEL[algo]} (exp{best[algo]})"
                result_rows.append(summarise(seasons, name, seed, test_name))
                if test_name == "New seeds":
                    steps_by_run[(algo, seed)] = steps

    # The three baselines are tested on the same seasons.
    baselines = [
        ("Wait only", wait_policy),
        ("Random", random_policy),
        ("Old calendar habit", calendar_policy),
        ("Expert rules", expert_policy),
    ]
    for baseline_name, policy in baselines:
        for test_name, env_settings in TEST_SETS.items():
            steps, seasons = trace(policy, 100, TEST_SEED_START, is_actor=False, **env_settings)
            result_rows.append(summarise(seasons, baseline_name, "-", test_name))

    results = pd.DataFrame(result_rows)
    results.to_csv(LOGS / "evaluation.csv", index=False)
    print("\n=== Test results (100 seasons per cell) ===")
    print(results.round(2).to_string(index=False))

    # The winner has the best average return on new seasons and its better run is used for the demo.
    new_seeds = results[(results.test_set == "New seeds") & (results.seed != "-")]
    average_return = {}
    for algo in ALGOS:
        name = f"{LABEL[algo]} (exp{best[algo]})"
        average_return[algo] = new_seeds[new_seeds.algorithm == name].mean_return.mean()
    winner = max(average_return, key=average_return.get)

    winner_name = f"{LABEL[winner]} (exp{best[winner]})"
    winner_rows = new_seeds[new_seeds.algorithm == winner_name]
    winner_seed = int(winner_rows.loc[winner_rows.mean_return.idxmax(), "seed"])
    best_file = LOGS / "best.json"
    best_file.write_text(json.dumps({"algo": winner, "exp": best[winner], "seed": winner_seed}))

    rounded_averages = {}
    for algo, value in average_return.items():
        rounded_averages[LABEL[algo]] = round(value, 2)
    print(
        f"\nwinner: {winner} exp{best[winner]} (demo model: seed {winner_seed}); "
        f"new-seeds 2-seed averages:",
        rounded_averages,
    )

    # Draw the chart of the return during training for the best setting of each algorithm.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        for seed, line_style in zip(seeds_for(best[algo]), ["-", "--"]):
            season_returns = per_episode(load(algo, best[algo], seed))
            ax.plot(season_returns, color=COLOR[algo], alpha=0.12)
            ax.plot(moving_average(season_returns), color=COLOR[algo], lw=2, ls=line_style, label=f"seed {seed}")
        ax.set_title(f"{LABEL[algo]} (exp{best[algo]})")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode return (reward units)")
        ax.legend(fontsize=8)
    fig.suptitle("Episode return during training, best setting per algorithm, 2 seeds (bold = 50-episode moving average)")
    fig.tight_layout()
    fig.savefig(ASSETS / "rewards_best.png", dpi=150)
    plt.close(fig)

    # Draw the chart of the DQN loss for the best experiment and two others.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dqn_experiments = [best["dqn"]]
    for exp in (2, 3):
        if exp != best["dqn"] and len(dqn_experiments) < 3:
            dqn_experiments.append(exp)
    for exp in dqn_experiments:
        for seed, line_style in zip(seeds_for(exp), ["-", "--"]):
            log = load("dqn", exp, seed).dropna(subset=["loss"])
            label = f"exp{exp} seed {seed}"
            if exp == best["dqn"]:
                label = label + " (best)"
            ax.plot(
                log["frames"],
                log["loss"].rolling(10, min_periods=1).mean(),
                ls=line_style,
                color=f"C{dqn_experiments.index(exp)}",
                label=label,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Environment frames")
    ax.set_ylabel("DQN loss (smoothed, log scale)")
    ax.set_title("DQN training loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "dqn_loss.png", dpi=150)
    plt.close(fig)

    # Draw the chart of the policy entropy of the three policy gradient methods.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in ["reinforce", "ppo", "a2c"]:
        for seed, line_style in zip(seeds_for(best[algo]), ["-", "--"]):
            log = load(algo, best[algo], seed).dropna(subset=["entropy"])
            ax.plot(
                log["frames"],
                log["entropy"].rolling(10, min_periods=1).mean(),
                color=COLOR[algo],
                ls=line_style,
                label=f"{LABEL[algo]} exp{best[algo]} seed {seed}",
            )
    ax.axhline(np.log(8), ls=":", color="grey", label="max entropy (ln 8)")
    ax.set_xlabel("Environment frames")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy entropy during training")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "pg_entropy.png", dpi=150)
    plt.close(fig)

    # Draw the chart of the evaluation return, with a dot where each run has settled.
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGOS:
        for seed, line_style in zip(seeds_for(best[algo]), ["-", "--"]):
            log = load(algo, best[algo], seed)
            evaluations = log.dropna(subset=["eval_return"])
            ax.plot(
                evaluations["frames"],
                evaluations["eval_return"],
                color=COLOR[algo],
                ls=line_style,
                alpha=0.85,
                label=f"{LABEL[algo]} seed {seed}",
            )
            settled = convergence(log)
            if settled:
                settled_frame = settled[0]
                settled_return = evaluations.loc[evaluations.frames == settled_frame, "eval_return"].iloc[0]
                ax.plot(settled_frame, settled_return, "o", color=COLOR[algo], ms=9, mec="k")
    ax.set_xlabel("Environment frames")
    ax.set_ylabel("Greedy evaluation return (20 seasons)")
    ax.set_title("Convergence on the evaluation curve (dot = within 10% of final from here on)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png", dpi=150)
    plt.close(fig)

    # Draw one small chart for each algorithm, with a dot where each run has settled (the x axis is the season number).
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        for seed, line_style in zip(seeds_for(best[algo]), ["-", "--"]):
            log = load(algo, best[algo], seed)
            evaluations = log.dropna(subset=["eval_return"])
            ax.plot(
                evaluations["episodes"],
                evaluations["eval_return"],
                color=COLOR[algo],
                ls=line_style,
                label=f"run with seed {seed}",
            )
            settled = convergence(log)
            if settled:
                settled_return = evaluations.loc[evaluations.frames == settled[0], "eval_return"].iloc[0]
                ax.plot(settled[1], settled_return, "o", color=COLOR[algo], ms=9, mec="k")
        ax.set_title(f"{LABEL[algo]} (exp {best[algo]})")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Episodes to converge (dot = return stays within 10% of its final value from here on)")
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence_subplots.png", dpi=150)
    plt.close(fig)

    # Count how often each action is used in a season, by each trained farmer, over the 200 test seasons of its two runs.
    # The steps of the seasons come one after the other, and a new season starts when the week is 0 again.
    weeks_per_season = {}
    seasons_with_action = {}
    for algo in ALGOS:
        counts = np.zeros((0, len(ACTION_NAMES)))
        for seed in seeds_for(best[algo]):
            steps = steps_by_run[(algo, seed)]
            season_counts = np.zeros((sum(1 for step in steps if step[0] == 0), len(ACTION_NAMES)))
            season = -1
            for step in steps:
                if step[0] == 0:
                    season += 1
                season_counts[season, step[1]] += 1
            counts = np.vstack([counts, season_counts])
        weeks_per_season[LABEL[algo]] = counts.mean(axis=0)
        seasons_with_action[LABEL[algo]] = (counts > 0).mean(axis=0) * 100
    action_use = pd.DataFrame(weeks_per_season, index=ACTION_NAMES).round(2)
    seasons_used = pd.DataFrame(seasons_with_action, index=ACTION_NAMES).round(0).astype(int)
    action_use.index.name = "Action"
    seasons_used.index.name = "Action"
    action_use.to_csv(LOGS / "action_weeks_per_season.csv")
    seasons_used.to_csv(LOGS / "action_seasons_used.csv")
    print("\n=== Weeks per season in which each action is done (average over 200 test seasons, 2 runs) ===")
    print(action_use.to_string())
    print("\n=== Share of the test seasons (%) in which the action is used at least once ===")
    print(seasons_used.to_string())

    # Draw the chart of the yield of every agent and baseline on each test set.
    fig, ax = plt.subplots(figsize=(10, 5))
    agent_names = list(dict.fromkeys(results.algorithm))
    bar_width = 0.8 / len(agent_names)
    for position, agent_name in enumerate(agent_names):
        mean_yields = []
        below = []
        above = []
        for test_name in TEST_SETS:
            rows = results[(results.algorithm == agent_name) & (results.test_set == test_name)]
            yields = rows.mean_yield_t_ha.values
            mean_yields.append(yields.mean())
            below.append(yields.mean() - yields.min())
            above.append(yields.max() - yields.mean())
        ax.bar(
            np.arange(len(TEST_SETS)) + position * bar_width,
            mean_yields,
            bar_width,
            yerr=[below, above],
            capsize=2,
            label=agent_name,
        )
    ax.set_xticks(np.arange(len(TEST_SETS)) + 0.4 - bar_width / 2)
    ax.set_xticklabels(list(TEST_SETS))
    ax.set_ylabel("Mean yield (t/ha)")
    ax.set_title("Generalization to unseen seasons (100 per test set; bar = mean of 2 seeds, whisker = seed range)")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(ASSETS / "generalization.png", dpi=150)
    plt.close(fig)

    # Draw the chart of which actions the best agent uses in each week.
    action_chart(
        steps_by_run[(winner, winner_seed)],
        f"Actions used by the best agent ({LABEL[winner]} exp{best[winner]}, seed {winner_seed}, 100 new seasons)",
        ASSETS / "actions_used.png",
    )


if __name__ == "__main__":
    main()
