#!/usr/bin/env python3
"""Compare R1's one-day scenario cash with a real 24-step forced-candidate transition."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from collections import Counter
from pathlib import Path

from audit_r1_greedy_regret import load, observation

ROOT = Path(__file__).resolve().parents[1]


def assets(farm):
    tiles = [tile for row in farm["tiles"] for tile in row]
    return [sum(isinstance(t, dict) and bool(t.get("crop")) for t in tiles),
            sum(isinstance(t, dict) and bool(t.get("animal")) for t in tiles)]


def asset_kinds(farm):
    tiles = [tile for row in farm["tiles"] for tile in row]
    counts = Counter((tile.get("animal") or tile.get("crop") or tile.get("kind", "EMPTY"))
                     if isinstance(tile, dict) else str(tile) for tile in tiles)
    return dict(sorted(counts.items()))


def arm(binary, bot, seed, seat, candidate_id, next_candidate_id=None,
        follow_days=0, harvest_age=False):
    os.environ["R1_BINARY_PATH"] = str(binary)
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    main = load(ROOT / "agent/main.py", f"first_day_main_{bot}_{seed}_{seat}_{candidate_id}")
    policy = main.create_agent(seat)
    rival = load(ROOT / f"opponents/{bot}/main.py",
                 f"first_day_bot_{bot}_{seed}_{seat}_{candidate_id}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    from fast_kaggriculture import Config, FastEnv
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    chosen = handoff_hash = None
    day_hire_peak = 0
    remaining_ids = [candidate_id] + ([] if next_candidate_id is None else [next_candidate_id])
    forced_path = []
    try:
        while not env.done:
            obs = [observation(state, p) for p in (0, 1)]
            own = obs[seat]
            if (chosen is None and policy.ready(own)) or (chosen is not None and
                    own["day"] > day and remaining_ids):
                rows = policy.dynamic.prepare_candidates(own)
                target_id = remaining_ids.pop(0)
                matches = [row for row in rows if row["id"] == target_id
                           and not row.get("diagnostic")]
                if len(matches) != 1:
                    raise RuntimeError(f"candidate {target_id} not uniquely available")
                chosen = matches[0]
                transition = policy.dynamic.lib.td_candidate_transition_json
                transition.argtypes = [ctypes.c_void_p, ctypes.c_int]
                transition.restype = ctypes.c_char_p
                predicted_state = json.loads(transition(policy.dynamic.handle,
                                                         chosen["index"]))
                current_hash = hashlib.sha256(json.dumps(own, sort_keys=True).encode()).hexdigest()
                if handoff_hash is None:
                    handoff_hash = current_hash
                forced_path.append({"day": own["day"], "id": target_id,
                                    "state_hash": current_hash, "score": chosen["score"]})
                policy.dynamic.install_candidate(chosen["index"])
                day = own["day"]
                day_hire_peak = own["farms"][seat]["hires_today"]
            if chosen is not None and own["day"] == day:
                day_hire_peak = max(day_hire_peak, own["farms"][seat]["hires_today"])
            if chosen is not None and own["day"] > day and not remaining_ids:
                next_candidates = policy.dynamic.prepare_candidates(own)
                public_clock = policy.dynamic.clock_debug()
                tail_items = policy.dynamic.lib.td_candidate_tail_items_json
                tail_items.argtypes = [ctypes.c_void_p, ctypes.c_int]
                tail_items.restype = ctypes.c_char_p
                next_tail_items = {
                    row["index"]: json.loads(tail_items(policy.dynamic.handle, row["index"]))
                    for row in next_candidates if not row.get("diagnostic")
                }
                next_clock = {
                    row["index"]: policy.dynamic.candidate_clock_debug(row["index"])
                    for row in next_candidates if not row.get("diagnostic")
                }
                next_harvest_age = None
                if harvest_age:
                    packed = main.policy._pack(own)
                    diagnostic = policy.dynamic.lib.td_candidate_harvest_age_json
                    diagnostic.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
                    diagnostic.restype = ctypes.c_char_p
                    next_harvest_age = {
                        str(row["id"]): json.loads(diagnostic(policy.dynamic.handle,
                                                               row["index"], packed,
                                                               len(packed)))
                        for row in next_candidates if not row.get("diagnostic")
                    }
                next_action = policy(own, {})
                next_debug = policy.dynamic.debug()
                next_actions = [None, None]
                next_actions[seat] = next_action
                other = obs[1-seat]
                next_actions[1-seat] = rival(other, {}) if with_config else rival(other)
                filled_state = env.step(next_actions)
                next_fills = list(env.last_market_fills[seat])
                filled_own = observation(filled_state, seat)
                followed = None
                follow_joint_by_day = []
                if follow_days:
                    target_day = own["day"] + follow_days
                    state = filled_state
                    while not env.done and observation(state, seat)["day"] < target_day:
                        future_obs = [observation(state, p) for p in (0, 1)]
                        future_actions = [None, None]
                        future_actions[seat] = policy(future_obs[seat], {})
                        if future_obs[seat]["hour"] == 0:
                            dbg = policy.dynamic.debug()
                            follow_joint_by_day.append({"day": future_obs[seat]["day"],
                                                        "searches": dbg.get("joint_searches"),
                                                        "switches": dbg.get("joint_switches"),
                                                        "gain": dbg.get("joint_gain"),
                                                        "last": dbg.get("joint_last")})
                        future_other = future_obs[1-seat]
                        future_actions[1-seat] = (rival(future_other, {}) if with_config
                                                  else rival(future_other))
                        state = env.step(future_actions)
                    follow_obs = observation(state, seat)
                    followed = {"day": follow_obs["day"],
                                "own_cash": follow_obs["farms"][seat]["money"],
                                "own_assets": assets(follow_obs["farms"][seat]),
                                "own_kinds": asset_kinds(follow_obs["farms"][seat]),
                                "shed_sheep": follow_obs["private"]["shed"]["SHEEP"],
                                "wool_inventory": follow_obs["market"]["inventory"]["WOOL"]}
                return {"bot": bot, "seed": seed, "seat": seat,
                        "handoff_hash": handoff_hash, "id": chosen["id"],
                        "forced_path": forced_path,
                        "score": chosen["score"],
                        "predicted_own": chosen["scenario_rollout_own_cash"],
                        "predicted_rival": chosen["scenario_rollout_rival_cash"],
                        "predicted_state": predicted_state,
                        "actual_own": own["farms"][seat]["money"],
                        "actual_rival": own["farms"][1-seat]["money"],
                        "actual_hires_today": own["farms"][seat]["hires_today"],
                        "actual_hands": len(own["farms"][seat]["hands"]),
                        "previous_day_hire_peak": day_hire_peak,
                        "actual_market_inventory": [own["market"]["inventory"][item]
                                                    for item in main.policy._ITEMS[:9]],
                        "actual_market_prices": [own["market"]["prices"][item]
                                                 for item in main.policy._ITEMS[:9]],
                        "actual_own_assets": assets(own["farms"][seat]),
                        "actual_own_kinds": asset_kinds(own["farms"][seat]),
                        "actual_own_shed": own["private"]["shed"],
                        "actual_own_seeds": own["private"]["seeds"],
                        "actual_rival_assets": assets(own["farms"][1-seat]),
                        "next_day_candidates": [{"id": row["id"], "score": row["score"],
                                                 "optimizer_stop": {key: row["optimizer"].get(key)
                                                                     for key in ("stop_pos", "stop_skipped_empty",
                                                                                 "stop_examined_empty", "stop_positive_empty",
                                                                                 "stop_best_later_gain")},
                                                 "roll_own": row["scenario_rollout_own_cash"],
                                                 "roll_rival": row["scenario_rollout_rival_cash"],
                                                 "tail": row["scenario_tail"],
                                                 "frozen_clock_tail": row["scenario_frozen_clock_tail"],
                                                 "carried_tail": row["scenario_carried_tail"],
                                                 "carried_replan_tail": row["scenario_carried_replan_tail"],
                                                 "tail_parts": row["scenario_tail_parts"],
                                                 "target_crops": row["target_crops"],
                                                 "target_animals": row["target_animals"],
                                                 "hires": row["hires"],
                                                 "forecast_work_1": row["forecast_work_1"],
                                                 "forecast_work_3": row["forecast_work_3"]}
                                                for row in next_candidates if not row.get("diagnostic")],
                        "next_day_tail_items": {
                            str(row["id"]): next_tail_items[row["index"]]
                            for row in next_candidates if not row.get("diagnostic")
                        },
                        "next_day_clock": {
                            str(row["id"]): next_clock[row["index"]]
                            for row in next_candidates if not row.get("diagnostic")
                        },
                        "public_clock": public_clock,
                        "next_day_harvest_age": next_harvest_age,
                        "next_day_first_action": next_action,
                        "next_day_market_fills": next_fills,
                        "next_step_own_shed": filled_own["private"]["shed"],
                        "after_follow_days": followed,
                        "follow_joint_by_day": follow_joint_by_day,
                        "next_day_search_choices": next_debug.get("search_choices"),
                        "next_day_joint_debug": {key: next_debug.get(key) for key in
                                                 ("joint_searches", "joint_switches",
                                                  "joint_rejected", "joint_gain", "joint_last")},
                        "actual_step": own["step"]}
            actions = [None, None]
            actions[seat] = policy(own, {})
            other = obs[1-seat]
            actions[1-seat] = rival(other, {}) if with_config else rival(other)
            state = env.step(actions)
        raise RuntimeError("game ended before next day")
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--opponent", default="thomas_2945")
    parser.add_argument("--seed", type=int, default=2609800100)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--ids", default="0,1,5")
    parser.add_argument("--next-id", type=int,
                        help="force this candidate at the next day and stop one day later")
    parser.add_argument("--follow-days", type=int, default=0,
                        help="after the checkpoint action, continue this many days with online policy")
    parser.add_argument("--harvest-age", action="store_true",
                        help="audit fixed-portfolio finite-crop harvest-age alternatives")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    from meta_agent.src import teammate_expanded_routes
    from functools import lru_cache
    teammate_expanded_routes.load_action_tapes = lru_cache(maxsize=1)(
        teammate_expanded_routes.load_action_tapes)
    rows = [arm(args.binary.resolve(), args.opponent, args.seed, args.seat,
                int(value), args.next_id, args.follow_days, args.harvest_age)
            for value in args.ids.split(",")]
    if len({row["handoff_hash"] for row in rows}) != 1:
        raise RuntimeError("counterfactuals reached different handoff states")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "same handoff, forced candidate, stop at next day",
                                       "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
