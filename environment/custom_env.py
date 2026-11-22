"""A Gymnasium environment for one maize plot in southern Chad over one rainy season."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# The plot has 12 columns and 8 rows of plants and the season lasts 18 weeks.
N_COLS = 12
N_ROWS = 8
N_PLANTS = 96
WEEKS = 18

# The money, the price and the size of the plot are estimates.
BUDGET0 = 60_000
PRICE = 250
PLOT_HA = 0.5

# This value makes a farmer who only waits get 1.2 tonnes per hectare over the seeds 0 to 199.
YIELD_MAX = 3.299545764266835

# This list holds the average rain in mm for each week, taken from the Moundou monthly averages.
RAIN_MEAN = [35, 35, 60, 60, 60, 60, 66, 66, 66, 66, 66, 47, 47, 47, 47, 13, 13, 13]

# This list shows how much of the final harvest each week can add.
STAGE_W = [.02, .03, .04, .05, .06, .07, .07, .07, .08, .09, .09, .08, .07, .06, .05, .04, .03, 0]

# These three values are estimates that differ from the first specification.
DISEASE_GROWTH = 0.30
SPORE_COEF = 0.60
DRAIN = 0.30

# These two values are placeholders for the real detector.
DETECTOR_RECALL = 0.88
DETECTOR_FALSE_ALARM = 0.04

# Spray and fertilizer cost 6,000 XAF here, which is less than in the specification.
ACTION_COST = [0, 1000, 6000, 6000, 6000, 3000, 4000, 3000]
ACTION_NAMES = [
    "Wait",
    "Drone scout",
    "Spray fungicide",
    "Irrigate",
    "Apply fertilizer",
    "Remove flagged plants",
    "Drone scout + remove",
    "Weed the field",
]

# The numbers below are new in version 2. They are estimates, unless a source is named.
# The soil food (nitrogen) at the start is low, because the first dose of fertilizer comes later.
N_START = 0.30
# The weeds start small, and they grow faster in wet weeks.
WEED_START = 0.05
WEED_GROWTH = 0.12
# Weeds take up to this share of the growth when the field is full of them early in the season.
WEED_LOSS = 0.6
# Fertilizer on soil that is already this rich burns the plants and slows the growth.
BURN_LEVEL = 0.85
BURN_FACTOR = 0.8
# Most fungicide labels allow about 3 sprays in one season.
MAX_SPRAYS = 3
# Each spray after the second works 20% less well, because the disease gets used to it.
RESISTANCE = 0.8
# The chance of an outside spore blowing in is 0, 0.01, 0.02 or 0.03 in a season. In a severe
# year an untreated field loses about 39% of its harvest, and the Delaware leaf blight guide
# says 30 to 50%.
DISEASE_PRESSURES = [0.0, 0.01, 0.02, 0.03]
DISEASE_PRESSURE_CHANCES = [0.3, 0.3, 0.25, 0.15]


def fert_uptake(week):
    """Tell how much of a fertilizer dose the maize can use in this week (timing from the IITA guide)."""
    if week <= 1 or 4 <= week <= 5:
        return 1.0
    if week <= 8:
        return 0.5
    return 0.1


def weed_weight(week):
    """Tell how much weeds hurt the maize in this week, which is most before the leaves cover the soil."""
    if week <= 6:
        return 1.0
    if week <= 10:
        return 0.5
    return 0.2


def kc(week):
    """Return how much water the maize uses in this week."""
    if week <= 3:
        return 0.4
    if week <= 8:
        return 0.8
    if week <= 13:
        return 1.15
    return 0.85


def ky(week):
    """Return how badly a dry week hurts the maize in this week."""
    if week <= 8:
        return 0.4
    if week <= 13:
        return 1.3
    return 0.5


def stage(week):
    """Return the growth stage of this week as a number from 0 to 4."""
    if week <= 3:
        return 0
    if week <= 8:
        return 1
    if week <= 11:
        return 2
    if week <= 15:
        return 3
    return 4


class MaizeSeasonEnv(gym.Env):
    """Each step of the farm is one week and the agent picks one of 8 actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, rain_scale=1.0, max_initial_infected=0, render_mode=None):
        super().__init__()
        self.rain_scale = rain_scale
        self.max_initial_infected = max_initial_infected
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
        self._init_state()

    def _init_state(self):
        """Set every value back to the start of a season."""
        self.week = 0
        self.budget = BUDGET0
        self.fungicide_weeks = 0
        self.last_scout = -1
        self.detected_disease = 0.0
        self.moisture = 0.45
        self.nitrogen = N_START
        self.forecast = 0.0
        self.severity = np.zeros(N_PLANTS)
        self.biomass = np.zeros(N_PLANTS)
        self.removed = np.zeros(N_PLANTS, dtype=bool)
        self.flagged = np.zeros(N_PLANTS, dtype=bool)
        self.infected_week = np.full(N_PLANTS, -1)
        self.total_return = 0.0
        self.last_rain = 0.0
        self.last_action = 0
        self.last_reward = 0.0
        self.last_cost = 0
        self.weeds = WEED_START
        self.sprays_used = 0
        self.fert_doses = 0
        self.burn = False
        self.fungicide_strength = 1.0
        self.disease_pressure = 0.0

    def _forecast(self):
        """Return a noisy forecast of this week's rain, between 0 and 1."""
        if self.week < WEEKS:
            week_index = self.week
        else:
            week_index = 0
        expected = RAIN_MEAN[week_index] * self.rain_scale / 70
        forecast = expected + self.np_random.normal(0, 0.1)
        return float(np.clip(forecast, 0, 1))

    def _get_obs(self):
        """Return the 12 numbers the agent can see."""
        # The soil sensor is a little noisy.
        noisy_moisture = np.clip(self.moisture + self.np_random.normal(0, 0.03), 0, 1)

        if self.last_scout < 0:
            weeks_since_scout = 1.0
        else:
            weeks_since_scout = (self.week - self.last_scout) / WEEKS

        obs = np.array([
            self.week / WEEKS,
            stage(self.week) / 4,
            noisy_moisture,
            self.forecast,
            self.nitrogen,
            self.detected_disease,
            weeks_since_scout,
            self.fungicide_weeks / 2,
            self.budget / BUDGET0,
            self.removed.sum() / N_PLANTS,
            self.weeds,
            self.sprays_used / MAX_SPRAYS,
        ], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        return obs.astype(np.float32)

    def _scout(self):
        """Run the disease detector on every plant that is still standing."""
        standing = ~self.removed

        # The detector flags sick plants often and healthy plants rarely.
        flag_chance = np.where(self.severity > 0.15, DETECTOR_RECALL, DETECTOR_FALSE_ALARM)
        random_numbers = self.np_random.random(N_PLANTS)
        self.flagged = (random_numbers < flag_chance) & standing

        standing_count = standing.sum()
        if standing_count > 0:
            self.detected_disease = float(self.flagged.sum() / standing_count)
        else:
            self.detected_disease = 0.0
        self.last_scout = self.week

    def _remove(self):
        """Pull out every flagged plant."""
        self.removed |= self.flagged
        self.severity[self.removed] = 0.0
        self.flagged[:] = False
        self.detected_disease = 0.0

    def _info(self, rain, cost, kg):
        """Return extra facts about the farm, used by the 3D view."""
        standing = ~self.removed
        diseased = (self.severity > 0) & standing
        return {
            "rain_mm": float(rain),
            "true_disease_share": float(diseased.sum() / N_PLANTS),
            "cost": int(cost),
            "kg": float(kg),
            "severity": self.severity.tolist(),
            "biomass": self.biomass.tolist(),
            "removed": self.removed.tolist(),
            "flagged": self.flagged.tolist(),
        }

    def outside_spores(self, week):
        """Tell the chance that spores from other fields infect a healthy plant in this week."""
        # Leaf blight usually arrives in the middle of the season, in warm and wet weeks.
        if week < 4:
            return 0.0
        return self.disease_pressure

    def biomass_ratio(self):
        """Return the standing biomass as a share of the best possible biomass."""
        standing_biomass = self.biomass[~self.removed].sum()
        return float(standing_biomass / (N_PLANTS * sum(STAGE_W)))

    def reset(self, seed=None, options=None):
        """Start a new season and return the first observation."""
        super().reset(seed=seed)
        self._init_state()
        rng = self.np_random

        # The soil starts a little different each season.
        self.moisture = float(np.clip(0.45 + rng.normal(0, 0.08), 0, 1))
        self.nitrogen = float(np.clip(N_START + rng.normal(0, 0.05), 0, 1))

        # Each season has its own disease pressure, and some years have almost no blight.
        self.disease_pressure = float(rng.choice(DISEASE_PRESSURES, p=DISEASE_PRESSURE_CHANCES))

        # In the outbreak test a few plants already carry blight at the start.
        initial_infected = int(rng.integers(0, self.max_initial_infected + 1))
        if initial_infected > 0:
            infected_plants = rng.choice(N_PLANTS, size=initial_infected, replace=False)
            self.severity[infected_plants] = rng.uniform(0.10, 0.20, size=initial_infected)
            self.infected_week[infected_plants] = -2

        self.forecast = self._forecast()
        obs = self._get_obs()
        info = self._info(0.0, 0, 0.0)
        if self.render_mode == "human":
            self._render_state(obs, info)
        return obs, info

    def step(self, action):
        """Play one week and return the usual Gymnasium results."""
        action = int(action)
        week = self.week
        rng = self.np_random

        # Play the action, and an action that costs too much becomes Wait.
        cost = ACTION_COST[action]
        penalty = 0.0
        if cost > self.budget:
            action_used = 0
            cost = 0
            penalty = 0.5
        else:
            action_used = action
        self.budget -= cost

        if action_used == 1 or action_used == 6:
            self._scout()

        self.burn = False
        if action_used == 2:
            # Only 3 sprays work in a season, and the 4th costs money but does nothing.
            self.sprays_used += 1
            if self.sprays_used <= MAX_SPRAYS:
                strength = RESISTANCE ** max(0, self.sprays_used - 2)
                self.fungicide_weeks = 2
                self.fungicide_strength = strength
                self.severity[self.severity > 0] -= 0.08 * strength
        elif action_used == 3:
            self.moisture += 0.30
        elif action_used == 4:
            if self.nitrogen > BURN_LEVEL:
                self.burn = True
            self.nitrogen += 0.35 * fert_uptake(week)
            self.fert_doses += 1
        elif action_used == 7:
            self.weeds = 0.0

        if action_used == 5 or action_used == 6:
            self._remove()

        self.moisture = float(np.clip(self.moisture, 0, 1))
        self.nitrogen = float(np.clip(self.nitrogen, 0, 1))
        self.severity = np.clip(self.severity, 0, 1)

        # Draw the rain, where about 15 weeks in 100 are almost dry.
        if rng.random() < 0.15:
            rain = rng.uniform(0, 3)
        else:
            rain = RAIN_MEAN[week] * rng.uniform(0.4, 1.6)
        rain = rain * self.rain_scale

        # Update the soil water with the rain, the water use and the drainage.
        new_moisture = self.moisture + rain / 110 - 0.14 * kc(week)
        new_moisture = new_moisture * (1 - DRAIN)
        self.moisture = float(np.clip(new_moisture, 0, 1))

        # Start the disease step by saving the severity of every plant.
        old_severity = self.severity.copy()
        standing = ~self.removed

        # Wet weeks help the blight to grow and fungicide slows it down.
        if rain >= 10:
            wetness = 1.0
        else:
            wetness = 0.3
        if self.fungicide_weeks > 0:
            fungicide_factor = 1 - 0.75 * self.fungicide_strength
        else:
            fungicide_factor = 1.0

        # Plants that already have blight get worse.
        infected = (old_severity > 0) & standing
        self.severity[infected] += DISEASE_GROWTH * wetness * fungicide_factor

        # Only plants infected for at least 2 weeks spread spores.
        infectious = infected & (self.infected_week <= week - 2)
        spore = np.where(infectious, old_severity, 0.0)
        spore = spore.reshape(N_ROWS, N_COLS)

        # Each plant gets spore pressure from its 4 neighbours.
        neighbours = np.zeros_like(spore)
        neighbours[1:, :] += spore[:-1, :]
        neighbours[:-1, :] += spore[1:, :]
        neighbours[:, 1:] += spore[:, :-1]
        neighbours[:, :-1] += spore[:, 1:]

        # Healthy plants may catch blight.
        catch_chance = (self.outside_spores(week) + SPORE_COEF * neighbours.reshape(-1)) * wetness * fungicide_factor
        healthy = (old_severity == 0) & standing
        random_numbers = rng.random(N_PLANTS)
        newly_infected = healthy & (random_numbers < catch_chance)
        self.severity[newly_infected] = 0.06
        self.infected_week[newly_infected] = week
        self.severity = np.clip(self.severity, 0, 1)

        # The fungicide protection loses one week.
        if self.fungicide_weeks > 0:
            self.fungicide_weeks -= 1

        # Let the plants grow, which depends on water, nitrogen and blight.
        water_supply = min(1.0, self.moisture / 0.35)
        water = max(0.0, 1 - ky(week) * (1 - water_supply))
        if self.moisture > 0.92:
            if 9 <= week <= 11:
                water = water * 0.6
            else:
                water = water * 0.85
        nutrients = 0.55 + 0.45 * self.nitrogen
        if 9 <= week <= 15:
            disease_weight = 1.0
        else:
            disease_weight = 0.3
        weed_factor = 1 - WEED_LOSS * self.weeds * weed_weight(week)
        if self.burn:
            burn_factor = BURN_FACTOR
        else:
            burn_factor = 1.0
        growth = (STAGE_W[week] * water * nutrients * weed_factor * burn_factor
                  * (1 - self.severity * disease_weight))
        self.biomass[standing] += growth[standing]

        # The weeds grow, and they grow faster when it rains.
        if rain >= 10:
            weed_rate = WEED_GROWTH
        else:
            weed_rate = WEED_GROWTH / 2
        self.weeds = float(min(1.0, self.weeds + weed_rate * (1 - self.weeds) + 0.02))

        # Move on to the next week and make a new forecast.
        self.nitrogen = float(np.clip(self.nitrogen - 0.04, 0, 1))
        self.week += 1
        self.forecast = self._forecast()

        # Work out the reward and check if the season is over.
        reward = -cost / 10_000 - penalty
        terminated = False
        kg = 0.0
        standing = ~self.removed
        healthy_share = ((self.severity < 0.5) & standing).sum() / N_PLANTS
        if healthy_share < 0.10:
            reward -= 10
            terminated = True
        elif self.week == WEEKS:
            kg = PLOT_HA * YIELD_MAX * 1000 * self.biomass_ratio()
            reward += kg * PRICE / 10_000
            terminated = True

        self.total_return += reward
        self.last_rain = rain
        self.last_action = action
        self.last_reward = reward
        self.last_cost = cost
        obs = self._get_obs()
        info = self._info(rain, cost, kg)
        if self.render_mode == "human":
            self._render_state(obs, info)
        return obs, float(reward), terminated, False, info

    def _render_state(self, obs=None, info=None):
        """Write the current farm to a file for the 3D view."""
        from environment.rendering import write_state

        if obs is None:
            obs = self._get_obs()
        if info is None:
            info = self._info(0.0, 0, 0.0)
        write_state(
            self.week,
            ACTION_NAMES[self.last_action],
            self.last_reward,
            self.total_return,
            obs,
            info,
            budget=self.budget,
            cost=self.last_cost,
        )

    def render(self):
        """Drawing is done by the 3D view, so there is nothing to do here."""
        return None
