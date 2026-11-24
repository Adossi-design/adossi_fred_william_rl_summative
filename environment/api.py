"""Run the local server that lets the 3D page play real seasons of the farm."""
import functools
import http.server
import json
import threading

import numpy as np

from farmer_policy import TRAINED_MODELS, action_chances, get_actor, pick_action
from environment import judge, rendering
from environment.custom_env import (
    ACTION_COST,
    ACTION_NAMES,
    BUDGET0,
    PLOT_HA,
    PRICE,
    MaizeSeasonEnv,
)
from training.baselines import calendar_policy, expert_policy

PORT = 8000

# The two kinds of hard season that the 3D page can play.
TESTS = {
    "drought": dict(rain_scale=0.6),
    "outbreak": dict(max_initial_infected=8),
}

LABELS = {
    "ppo": "PPO",
    "a2c": "A2C",
    "reinforce": "REINFORCE",
    "dqn": "DQN",
    "expert": "Expert rules",
    "calendar": "Old calendar habit",
    "wait": "Wait only",
    "random": "Random",
}

# Only one season is played at a time, and the lock stops two requests from mixing it up.
lock = threading.Lock()
session = None


def make_policy(model, seed):
    """Get the way to choose the action of the farmer, and the name to show on the page."""
    return make_farmer(model, seed)[:2]


def make_farmer(model, seed):
    """Get the way to choose the action, the name to show, and the trained farmer with its method (if any)."""
    if model == "expert":
        return (lambda obs, env: int(expert_policy(obs, env))), LABELS[model], None, None
    if model == "calendar":
        return (lambda obs, env: int(calendar_policy(obs, env))), LABELS[model], None, None
    if model == "wait":
        return (lambda obs, env: 0), LABELS[model], None, None
    if model == "random":
        generator = np.random.default_rng(seed)
        return (lambda obs, env: int(generator.integers(0, len(ACTION_NAMES)))), LABELS[model], None, None

    # The farmer can be one of the four best ones, or a name like ppo-5-2 for method, experiment and run.
    if model in TRAINED_MODELS:
        algo, exp = TRAINED_MODELS[model]
        model_seed = 1
        label = LABELS[model]
    else:
        algo, exp, model_seed = model.split("-")
        exp = int(exp)
        model_seed = int(model_seed)
        label = f"{algo.upper()} exp {exp}"
    actor = get_actor(algo, exp, model_seed)
    return (lambda obs, env: pick_action(actor, obs)), label, actor, algo


def named_obs(obs):
    """Match each of the 12 numbers the farmer sees with its name."""
    pairs = []
    for name, value in zip(rendering.OBS_NAMES, obs):
        pairs.append([name, round(float(value), 4)])
    return pairs


def unaffordable_penalty():
    """Find the small penalty for an action that costs more than the money left."""
    env = MaizeSeasonEnv()
    env.reset(seed=0)
    env.budget = 0
    _, reward, _, _, _ = env.step(1)
    return -reward


def failure_penalty(reward, cost, was_unaffordable):
    """Find the penalty for a failed crop, taken from the score of the last week."""
    penalty = -reward - cost / 10_000
    if was_unaffordable:
        penalty -= unaffordable_penalty()
    return round(penalty, 4)


class Session:
    """One season that is played week by week, by a trained farmer or by the user."""

    def __init__(self, mode, model, seed, test):
        self.mode = mode
        self.model = model
        self.seed = seed
        self.test = test
        self.env_settings = TESTS.get(test, {})
        self.policy, self.label, self.actor, self.algo = make_farmer(model, seed)
        self.good = 0
        self.judged = 0

        # This farm saves every week, so the 3D page can show it.
        self.env = MaizeSeasonEnv(render_mode="human", **self.env_settings)
        self.obs, _ = self.env.reset(seed=seed)
        self.done = False
        self.total_return = 0.0
        self.total_spent = 0
        self.advice_cache = None
        print(f"Season started | mode {mode} | model {self.label} | seed {seed} | test set: {test or 'default'}")

    def chances_now(self):
        """Give the chance of each action for the trained farmer now, or nothing for the simple farmers."""
        if self.actor is None:
            return None
        return [round(value, 4) for value in action_chances(self.actor, self.algo, self.obs)]

    def look_now(self):
        """Give what the 3D page shows about the coming decision, which is the notes and the chances."""
        return {
            "next_notes": judge.notices(self.obs, self.env),
            "next_chances": self.chances_now(),
            "chance_kind": "preference" if self.algo == "dqn" else "chance",
            "trained": self.actor is not None,
        }

    def advice(self):
        """Tell which action the trained farmer would take now, and keep the same answer for this week."""
        if self.advice_cache is None or self.advice_cache[0] != self.env.week:
            self.advice_cache = (self.env.week, self.policy(self.obs, self.env))
        return self.advice_cache[1]

    def step(self, action=None):
        """Play one week of the farm, and let the trained farmer choose when the user gives no action."""
        if self.done:
            raise ValueError("The season is over, start a new one")
        if action is None:
            action = self.advice()
        action = int(action)
        if action < 0 or action >= len(ACTION_NAMES):
            raise ValueError("The action must be a number from 0 to 7")

        # The notes, the judge and the chances look at what the farmer knew before this week.
        notes = judge.notices(self.obs, self.env)
        verdict = judge.judge(self.obs, self.env, action)
        chances = self.chances_now()
        weeds_before = float(self.obs[10])
        model_pick = self.advice()
        if verdict["level"] == "good":
            self.good += 1
        self.judged += 1

        week = self.env.week + 1
        was_unaffordable = ACTION_COST[action] > self.env.budget
        self.obs, reward, self.done, _, info = self.env.step(action)
        self.total_return += reward
        self.total_spent += info["cost"]
        print(
            f"Week {week:02d} | moisture {self.obs[2]:.2f} | detected {self.obs[5]:.2f} | "
            f"action: {ACTION_NAMES[action]} | reward {reward:+.2f} | return {self.total_return:+.2f}"
        )

        step = dict(rendering.saved["steps"][-1])
        step["notes"] = notes
        step["judge"] = verdict
        step["good"] = self.good
        step["judged"] = self.judged
        step["chances"] = chances
        step["chance_kind"] = "preference" if self.algo == "dqn" else "chance"
        step["model_pick"] = model_pick
        step["weeds_before"] = weeds_before
        step["weeds_after"] = float(self.obs[10])
        result = {
            "step": step,
            "next_obs": named_obs(self.obs),
            "done": self.done,
        }
        if not self.done:
            result.update(self.look_now())
        if self.done:
            kg = info["kg"]
            print(
                f"\nHarvest: {kg:.0f} kg = {kg / PLOT_HA / 1000:.2f} t/ha "
                f"| spent {self.total_spent:,} XAF | total return {self.total_return:.2f}"
            )
            if kg == 0:
                print("Crop failure: too many plants were lost to blight.")
                result["failure_penalty"] = failure_penalty(reward, info["cost"], was_unaffordable)
            result["your_good"] = self.good
            result["your_judged"] = self.judged
            if self.mode == "play":
                result["comparison"] = self.model_season()
        return result

    def model_season(self):
        """Let the trained farmer play the same season from the start, so the user can compare."""
        env = MaizeSeasonEnv(**self.env_settings)
        policy, _ = make_policy(self.model, self.seed)
        obs, _ = env.reset(seed=self.seed)
        done = False
        total_return = 0.0
        spent = 0
        good = 0
        judged = 0
        while not done:
            action = policy(obs, env)
            if judge.judge(obs, env, action)["level"] == "good":
                good += 1
            judged += 1
            was_unaffordable = ACTION_COST[action] > env.budget
            obs, reward, done, _, info = env.step(action)
            total_return += reward
            spent += info["cost"]
        answer = {"kg": info["kg"], "ret": total_return, "spent": spent, "failure_penalty": None, "good": good, "judged": judged}
        if info["kg"] == 0:
            answer["failure_penalty"] = failure_penalty(reward, info["cost"], was_unaffordable)
        return answer


def new_season(body):
    """Start a new season and send the 3D page what it needs to draw it."""
    global session
    mode = body.get("mode", "watch")
    if mode not in ("watch", "play"):
        raise ValueError("The mode must be watch or play")
    test = body.get("test") or None
    if test is not None and test not in TESTS:
        raise ValueError("The test must be drought or outbreak")

    session = Session(mode, body.get("model", "ppo"), int(body.get("seed", 0)), test)
    actions = []
    for name, cost in zip(ACTION_NAMES, ACTION_COST):
        actions.append({"name": name, "cost": cost})
    return {
        "label": session.label,
        "start": rendering.saved["start"],
        "next_obs": named_obs(session.obs),
        "config": {"actions": actions, "budget": BUDGET0, "price": PRICE, "plot_ha": PLOT_HA},
        **session.look_now(),
    }


class ApiHandler(http.server.SimpleHTTPRequestHandler):
    """Send the files of the 3D page and answer its three requests."""

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, data, status=200):
        """Send an answer to the 3D page."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        """Read what the 3D page sent."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        with lock:
            try:
                if self.path == "/api/new":
                    self.send_json(new_season(self.read_body()))
                elif self.path == "/api/step":
                    if session is None:
                        raise ValueError("Start a season first with /api/new")
                    body = self.read_body()
                    action = body.get("action") if session.mode == "play" else None
                    if session.mode == "play" and action is None:
                        raise ValueError("In play mode the step needs an action")
                    self.send_json(session.step(action))
                else:
                    self.send_json({"error": "Unknown address"}, 404)
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, 400)

    def do_GET(self):
        if self.path == "/api/advice":
            with lock:
                if session is None or session.done:
                    self.send_json({"error": "There is no season to suggest an action for"}, 400)
                else:
                    action = session.advice()
                    self.send_json({"action": action, "name": ACTION_NAMES[action], "label": session.label})
        else:
            super().do_GET()


class ExclusiveServer(http.server.ThreadingHTTPServer):
    """A server that refuses a port that is already used, because Windows would let two copies share it."""

    allow_reuse_address = False


def start_server(project_root, port=PORT):
    """Start the local server and give back the address of the 3D page."""
    handler = functools.partial(ApiHandler, directory=str(project_root))
    server = ExclusiveServer(("localhost", port), handler)
    return server, f"http://localhost:{port}/environment/viewer/index.html"
