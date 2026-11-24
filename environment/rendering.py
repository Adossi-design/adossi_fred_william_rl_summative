"""Save each week of the season in the form that the 3D page needs."""

OBS_NAMES = [
    "week",
    "growth_stage",
    "soil_moisture",
    "rain_forecast",
    "nitrogen",
    "detected_disease",
    "weeks_since_scout",
    "fungicide_left",
    "budget",
    "plants_removed",
    "weeds",
    "sprays_used",
]

# This keeps the season in memory between two weeks.
saved = {"prev": None, "start": None, "steps": []}


def _stage_name(week):
    """Find the name of the growth stage of the maize in this week."""
    if week <= 3:
        return "Emergence"
    if week <= 8:
        return "Vegetative"
    if week <= 11:
        return "Flowering"
    if week <= 15:
        return "Grain fill"
    return "Maturity"


def write_state(week, action, reward, total_return, obs, info, budget=None, cost=0):
    """Save how the farm looks after this week."""
    from environment.custom_env import ACTION_COST, ACTION_NAMES, N_PLANTS

    # Round the numbers of the plants, so the data stays small.
    now = {"sev": [], "bio": [], "removed": [], "flag": []}
    for value in info["severity"]:
        now["sev"].append(round(value, 3))
    for value in info["biomass"]:
        now["bio"].append(round(value, 4))
    for value in info["removed"]:
        now["removed"].append(bool(value))
    for value in info["flagged"]:
        now["flag"].append(bool(value))

    obs_list = []
    for value in obs:
        obs_list.append(round(float(value), 4))

    if week == 0 or saved["prev"] is None:
        # Start a new season when the week is 0.
        saved["steps"] = []
        saved["start"] = {
            "plants": now,
            "obs": obs_list,
            "moisture": obs_list[2],
            "budget": budget,
        }
        saved["prev"] = {"plants": now, "obs": obs_list, "budget": budget}
        return

    previous = saved["prev"]
    before = previous["plants"]

    # Find the number of the action from its name, and count an action the farmer could not pay for as Wait.
    if action in ACTION_NAMES:
        action_number = ACTION_NAMES.index(action)
    else:
        action_number = 0
    if cost == 0 and ACTION_COST[action_number] > 0:
        action_number = 0

    # Find the plants that were removed this week.
    newly_removed = []
    for removed_now, removed_before in zip(now["removed"], before["removed"]):
        newly_removed.append(removed_now and not removed_before)

    # Find the plants that get a yellow ring right after the action.
    if action_number == 1:
        flags_after_action = now["flag"]
    elif action_number == 5 or action_number == 6:
        flags_after_action = newly_removed
    else:
        flags_after_action = before["flag"]

    # Check if the season ended this week.
    healthy_count = 0
    for is_removed, severity in zip(now["removed"], now["sev"]):
        if not is_removed and severity < 0.5:
            healthy_count += 1
    healthy_share = healthy_count / N_PLANTS
    if healthy_share < 0.10:
        end = "Crop failure"
    elif week == 18:
        end = "Harvest"
    else:
        end = None

    saved["steps"].append({
        "week": int(week) - 1,
        "stage": _stage_name(int(week) - 1),
        "a": action_number,
        "action": ACTION_NAMES[action_number],
        "cost": int(cost),
        "reward": float(reward),
        "ret": float(total_return),
        "rain": round(float(info["rain_mm"]), 2),
        "kg": float(info["kg"]),
        "end": end,
        "obs": list(zip(OBS_NAMES, previous["obs"])),
        "moist_before": previous["obs"][2],
        "moist_after": obs_list[2],
        "budget_before": previous["budget"],
        "budget_after": budget,
        "before": before,
        "after": now,
        "flags_after_action": flags_after_action,
        "removed_now": int(sum(newly_removed)),
    })
    saved["prev"] = {"plants": now, "obs": obs_list, "budget": budget}
