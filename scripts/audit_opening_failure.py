#!/usr/bin/env python3
"""Locate opening cash/hire/land/crop failures in a Kaggle replay.

Optional fixed-opponent replay checks the current student opening on the same
seed. The opponent is an action tape in that check, not an adaptive policy.
"""

import argparse
import json
from pathlib import Path


def farm(frame, player):
    return frame[player]["observation"]["farms"][player]


def tiles(f):
    return {(y, x): tile for y, row in enumerate(f["tiles"])
            for x, tile in enumerate(row)}


def count(f, kind):
    return sum(isinstance(tile, dict) and tile.get("kind") == kind
               for tile in tiles(f).values())


def audit(replay, player, end):
    steps = replay["steps"]
    events = []
    for step in range(1, end + 1):
        before, after = farm(steps[step - 1], player), farm(steps[step], player)
        action = steps[step][player]["action"]
        hires = sum(order[0] == "HIRE" for order in action["market"] if order)
        lands = sum(order[0] == "BUY_LAND" for order in action["market"] if order)
        filled_hires = (after["hires_today"] - before["hires_today"]
                        if step % 24 else 0)
        if hires and filled_hires < hires:
            events.append({"step": step, "kind": "hire_shortfall",
                           "requested": hires, "filled": filled_hires,
                           "cash_before": before["money"],
                           "shed": steps[step - 1][player]["observation"]["private"]["shed"]})
        if lands and len(after["unlocked_quadrants"]) == len(before["unlocked_quadrants"]):
            events.append({"step": step, "kind": "land_shortfall",
                           "cash_before": before["money"],
                           "land_count": len(before["unlocked_quadrants"])})
        old = tiles(before)
        deaths = [(y, x, old[y, x].get("crop"), old[y, x].get("consecutive_unwatered"),
                   old[y, x].get("watered_today"))
                  for (y, x), tile in tiles(after).items()
                  if isinstance(tile, dict) and tile.get("kind") == "WEED"
                  and isinstance(old[y, x], dict) and old[y, x].get("kind") == "PLANT"]
        if deaths:
            events.append({"step": step, "kind": "plants_to_weeds", "tiles": deaths})
    days = []
    for step in range(0, end + 1, 24):
        f = farm(steps[step], player)
        days.append({"step": step, "cash": f["money"], "plants": count(f, "PLANT"),
                     "weeds": count(f, "WEED"), "land": len(f["unlocked_quadrants"])})
    return {"seed": replay["info"]["seed"], "player": player,
            "events": events, "days": days}


def fixed_opponent_prefix(replay, player, end):
    import fast_kaggriculture
    from agent.main import create_replay_agent, replay_deployment, sell_before_unfunded_land
    from experiments.student_action_event_agent import FundedReplayRoute

    steps = replay["steps"]
    env = fast_kaggriculture.FastEnv(seed=int(replay["info"]["seed"]))
    route = FundedReplayRoute(create_replay_agent(replay_deployment(), "opening_cash_audit"))
    days = []
    first_divergence = None
    for step in range(end):
        observation = env.observation(player)
        ours = route(observation)
        ours = sell_before_unfunded_land(observation, ours)
        recorded = steps[step + 1][player]["action"]
        if first_divergence is None and ours != recorded:
            first_divergence = {"step": step + 1,
                                "recorded_market": recorded["market"],
                                "repaired_market": ours["market"]}
        actions = [None, None]
        actions[player] = ours
        actions[1 - player] = steps[step + 1][1 - player]["action"]
        env.step_raw(actions)
        if (step + 1) % 24 == 0:
            f = env.observation(player)["farms"][player]
            days.append({"step": step + 1, "cash": f["money"],
                         "plants": count(f, "PLANT"), "weeds": count(f, "WEED"),
                         "land": len(f["unlocked_quadrants"])})
    return {"opponent": "fixed_recorded_actions",
            "first_divergence": first_divergence, "days": days}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, choices=(0, 1))
    parser.add_argument("--end", type=int, default=288)
    parser.add_argument("--fixed-opponent-prefix", action="store_true")
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    player = args.player
    if player is None:
        names = replay["info"]["TeamNames"]
        player = names.index("QQ Farming") if "QQ Farming" in names else 0
    if not 0 < args.end < len(replay["steps"]):
        parser.error("--end must be within the replay")
    result = {"recorded": audit(replay, player, args.end)}
    if args.fixed_opponent_prefix:
        result["counterfactual"] = fixed_opponent_prefix(replay, player, args.end)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
