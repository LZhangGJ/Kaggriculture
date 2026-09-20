#!/usr/bin/env python3
"""One official-game check for packed candidate ranking and one-day install."""
import inspect
import json

from run_strong_ab import BOTS, load


def main():
    from kaggle_environments import make

    policy = load("agent/main.py", "candidate_oracle_policy").create_agent(0)
    rival = load(BOTS["thomas_2945"], "candidate_oracle_rival").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    env = make("kaggriculture", configuration={"seed": 2609800000}, debug=True)
    state = env.reset()
    forced = None
    calls = None
    while not env.done:
        observations = []
        for player in (0, 1):
            obs = json.loads(json.dumps(state[player].observation))
            obs["step"] = obs.get("day", 0) * 24 + obs.get("hour", 0)
            obs["player"] = player
            observations.append(obs)
        step = observations[0]["step"]
        if step == 15 * 24:
            calls = policy.dynamic.debug()["search_calls"]
            candidates = policy.dynamic.prepare_candidates(observations[0])
            assert len(candidates) > 1 and all({"index", "id", "score", "predicted", "target_crops", "target_animals", "hires", "forecast_work_1", "forecast_work_3"} <= row.keys() for row in candidates)
            forced = max(candidates, key=lambda row: row["score"])
            policy.dynamic.install_candidate(forced["index"])
        actions = [policy(observations[0], env.configuration),
                   rival(observations[1], env.configuration) if with_config else rival(observations[1])]
        state = env.step(actions)
        if step == 15 * 24:
            assert policy.dynamic.debug()["search_calls"] == calls
        if step == 16 * 24:
            assert policy.dynamic.debug()["search_calls"] == calls + 1
            break
    policy.close()
    assert forced is not None
    print(json.dumps({"status": "PASS", "forced": forced, "search_calls_before": calls}))


if __name__ == "__main__":
    main()
