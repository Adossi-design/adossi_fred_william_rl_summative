"""Tell what an expert would do each week, judge the choice of the farmer, and write plain notes."""
from environment.custom_env import BURN_LEVEL, MAX_SPRAYS


def expert_advice(obs, env):
    """Find the action an expert would take this week and the reason, from the research rules in order."""
    week = env.week
    moisture = obs[2]
    forecast = obs[3]
    detected = obs[5]
    protection = obs[7]
    weeds = obs[10]
    checked_recently = env.last_scout >= 0 and week - env.last_scout <= 1

    if week >= 15:
        return 0, "the harvest is close and nothing can help much now."
    if checked_recently and detected > 0.06 and protection == 0 and env.sprays_used < MAX_SPRAYS:
        return 2, "the drone found sick plants, and spraying now stops the disease from spreading."
    if env.fert_doses == 0 and week <= 1:
        return 4, "the young plants need their first dose of food now."
    if weeds > 0.3 and week <= 6:
        return 7, "weeds are taking water and food from the young maize, and experts weed before the second dose of fertilizer."
    if env.fert_doses <= 1 and 4 <= week <= 5:
        return 4, "this is the right time for the second dose of fertilizer."
    if moisture < 0.3 and forecast < 0.5 and 8 <= week <= 13:
        return 3, "the maize is flowering and the soil is dry, so it needs water now."
    if 4 <= week <= 14 and (env.last_scout < 0 or week - env.last_scout >= 2):
        return 1, "leaf blight can arrive in warm and wet weeks, so check the plants before it spreads."
    return 0, "nothing needs doing this week, so saving money is right."


def judge(obs, env, action):
    """Judge the choice of the farmer as good, ok or poor, with one sentence and a better choice when needed."""
    better_action, better_reason = expert_advice(obs, env)
    verdict = _judge_only(obs, env, action)
    verdict["better"] = None
    verdict["better_why"] = None
    if action == better_action:
        verdict["level"] = "good"
    elif verdict["level"] != "good":
        verdict["better"] = better_action
        verdict["better_why"] = better_reason
    return verdict


def _judge_only(obs, env, action):
    """Judge the choice from what the farmer knew this week, without comparing with the expert."""
    week = env.week
    moisture = obs[2]
    forecast = obs[3]
    food = obs[4]
    detected = obs[5]
    protection = obs[7]
    weeds = obs[10]
    checked_recently = env.last_scout >= 0 and week - env.last_scout <= 1
    sick_known = checked_recently and detected > 0.06

    def good(text):
        return {"level": "good", "text": text}

    def ok(text):
        return {"level": "ok", "text": text}

    def poor(text):
        return {"level": "poor", "text": text}

    # Late in the season nothing helps any more.
    if week >= 15:
        if action == 0:
            return good("The harvest is close and nothing can help much now, so saving money is right.")
        return poor("The harvest is close, so this money is mostly wasted.")

    if action == 2:
        if env.sprays_used >= MAX_SPRAYS:
            return poor("You already used the 3 sprays allowed this season, so this one does nothing.")
        if protection > 0:
            return poor("The last spray still protects the plants, so this one is wasted.")
        if sick_known:
            return good("The drone found sick plants, so spraying now stops the disease from spreading.")
        if not checked_recently and 6 <= week <= 10:
            return ok("Spraying before flowering can protect the harvest, but checking with the drone first would show if it is needed.")
        return poor("No plant is known to be sick, so this spray wastes money and chemicals that harm the soil.")

    if sick_known and protection == 0 and env.sprays_used < MAX_SPRAYS:
        return poor("The drone found sick plants, so spraying fungicide was the urgent job this week.")

    if action == 4:
        if food > BURN_LEVEL:
            return poor("The soil is already rich, and too much fertilizer burns the plants.")
        if week > 8:
            return poor("It is too late, because the maize takes little food at this stage.")
        if env.fert_doses == 0 and week <= 1:
            return good("The young plants need their first dose of food now.")
        if env.fert_doses <= 1 and 4 <= week <= 5:
            return good("This is the right time for the second dose of fertilizer.")
        if env.fert_doses >= 2:
            return poor("Experts give two doses per season, so this extra dose costs more than it brings.")
        return ok("Fertilizer helps, but it works best at planting and at week 5 or 6.")

    if env.fert_doses == 0 and week <= 1:
        return poor("The young plants needed their first dose of fertilizer this week.")
    if env.fert_doses <= 1 and 4 <= week <= 5 and action != 7:
        return poor("This was the right time for the second dose of fertilizer.")

    if action == 7:
        if weeds < 0.12:
            return poor("The field is already clean, so this weeding is wasted.")
        if week <= 7:
            return good("Weeds steal water and food from the growing maize, so weeding now protects the harvest.")
        return ok("Weeding still helps a little, but weeds matter most in the first 6 weeks.")

    if weeds > 0.3 and week <= 6:
        return poor("Weeds are taking water and food from the young maize, so weeding was needed.")

    if action == 3:
        if moisture > 0.6 or forecast > 0.55:
            return poor("The soil already has enough water or rain is coming, and too much water hurts the plants.")
        if moisture < 0.3:
            return good("The soil is dry, so watering keeps the maize growing.")
        return ok("The soil was not dry yet, so watering could have waited.")

    if moisture < 0.3 and forecast < 0.5 and 8 <= week <= 13:
        return poor("The maize is flowering and the soil is dry, which is when a dry spell hurts the harvest most.")

    if action == 1 or action == 6:
        if env.last_scout == week - 1:
            return poor("You checked last week, so the drone will show almost the same thing.")
        if week < 4:
            return ok("Leaf blight rarely comes this early, so the check could wait a little.")
        return good("Checking with the drone shows if leaf blight has arrived, before it spreads.")

    if action == 5:
        if not checked_recently or detected == 0:
            return poor("No plant is marked by the drone, so there is nothing to pull out.")
        return ok("Pulling out plants also removes healthy plants marked by mistake, and experts rarely do it for leaf blight.")

    if action == 0:
        if week >= 5 and (env.last_scout < 0 or week - env.last_scout >= 4):
            return ok("Nothing is urgent, but the field has not been checked for disease for a long time.")
        return good("Nothing needs doing this week, so saving money is right.")

    return ok("This choice is neither clearly good nor clearly bad.")


def notices(obs, env):
    """Write the 3 most important notes in plain words about what the farmer can see this week."""
    week = env.week
    flowering = 8 <= week <= 11
    moisture = obs[2]
    forecast = obs[3]
    nitrogen = obs[4]
    sick = obs[5]
    weeds = obs[10]
    dry = moisture < 0.3

    # Each note has a rank, where 1 is the most important, and notes with the same rank keep this order.
    ranked = []
    if env.last_scout >= 0 and sick > 0.06:
        ranked.append((1, "The drone found " + str(round(sick * 100)) + "% sick plants."))
    if dry and flowering:
        ranked.append((1, "The maize is flowering and the soil is dry."))
    elif dry:
        ranked.append((3, "The soil is dry."))
    if moisture > 0.9:
        ranked.append((2 if flowering else 5, "The soil is soaked with water."))
    if weeds > 0.5:
        ranked.append((2, "The field is full of weeds."))
    elif weeds > 0.25:
        ranked.append((4, "Weeds are growing in the field."))
    if env.budget < 6000:
        ranked.append((2, "Money is running low."))
    if nitrogen < 0.35:
        ranked.append((3, "The soil is low on food (nitrogen)."))
    elif nitrogen > 0.85:
        ranked.append((6, "The soil is rich in food."))
    if week >= 4 and env.last_scout < 0:
        ranked.append((4, "The field has never been checked with the drone."))
    elif week >= 4 and week - env.last_scout >= 3:
        ranked.append((4, "No drone check for " + str(week - env.last_scout) + " weeks."))
    if flowering and not dry:
        ranked.append((5, "The maize is flowering, its most sensitive time."))
    if forecast > 0.7:
        ranked.append((6, "Heavy rain is expected."))
    elif forecast < 0.25:
        ranked.append((6, "Little rain is expected."))
    if not ranked:
        ranked.append((9, "The field looks fine this week."))

    ranked.sort(key=lambda item: item[0])
    return [text for _, text in ranked[:3]]
