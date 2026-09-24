#!/usr/bin/env python3
"""Prefix-only audit of R1's portfolio rank heuristic at warm handoff."""
import argparse
import ctypes
import functools
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOTS = ("thomas_2945", "melon_2749")


def load(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(state, player):
    value = json.loads(json.dumps(state[player]))
    value["step"] = value.get("day", 0) * 24 + value.get("hour", 0)
    value["player"] = player
    return value


def summarize(rows):
    normal = [row for row in rows if not row.get("diagnostic")]
    slots = [slot for row in normal for slot in row["optimizer"]["slots"]]
    regrets = [slot["local_regret"] for slot in slots]
    value_regrets = [slot["value_regret"] for slot in slots]
    return {
        "proposals": len(normal),
        "slots": len(slots),
        "rank_raw_mismatches": sum(
            slot["picked_kind"] != slot["raw_kind"] for slot in slots),
        "local_regret_sum": sum(regrets),
        "local_regret_max": max(regrets, default=0),
        "value_mismatches": sum(
            slot["picked_kind"] != slot["value_kind"] for slot in slots),
        "value_regret_sum": sum(value_regrets),
        "value_regret_max": max(value_regrets, default=0),
        "budget_initial": [row["optimizer"]["budget_initial"] for row in normal],
        "budget_final": [row["optimizer"]["budget_final"] for row in normal],
        "preview_removed": sum(row["optimizer"]["preview_removed"] for row in normal),
        "swap_gain": sum(row["optimizer"]["swap_gain"] for row in normal),
        "pair_gain": sum(row["optimizer"]["pair_gain"] for row in normal),
    }


def run_case(binary, bot, seed, seat, modes=(0, 1, 2), closed_loop_top=0,
             shop_branches=False, tail_seam=False, online_days=0, online_ids=(),
             shift_item=-1, shift_units=0):
    os.environ["R1_BINARY_PATH"] = str(binary)
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    main = load(ROOT / "agent/main.py", f"greedy_main_{bot}_{seed}_{seat}")
    policy = main.create_agent(seat)
    opponent_module = load(ROOT / f"opponents/{bot}/main.py", f"greedy_bot_{bot}_{seed}_{seat}")
    opponent = opponent_module.agent
    with_config = len(inspect.signature(opponent).parameters) > 1
    extras = {}
    for mode in modes:
        if mode == 0:
            continue
        config = json.loads((ROOT / "policy/r1/config.json").read_text())
        config["portfolio_swaps"] = mode
        extras[mode] = main.policy.Agent(config=config, binary_path=binary)
    from fast_kaggriculture import Config, FastEnv
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    try:
        while not env.done:
            obs = [observation(state, player) for player in (0, 1)]
            own = obs[seat]
            if policy.ready(own):
                candidates = {0: policy.dynamic.prepare_candidates(own)} if 0 in modes else {}
                candidates.update({mode: agent.prepare_candidates(own) for mode, agent in extras.items()})
                agents = {0: policy.dynamic, **extras}
                for mode, rows in candidates.items():
                    for row in rows:
                        row["clock_audit"] = agents[mode].candidate_clock_debug(row["index"])
                    if online_days:
                        dynamic = agents[mode]
                        fn = getattr(dynamic.lib, "td_candidate_online_boundaries_json", None)
                        if fn is None:
                            raise RuntimeError("online boundaries require diagnostic binary")
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
                        fn.restype = ctypes.c_char_p
                        packed = main.policy._pack(own)
                        for row in rows:
                            if row.get("diagnostic") or row["id"] not in online_ids:
                                continue
                            value = fn(dynamic.handle, row["index"], packed,
                                       len(packed), online_days, shift_item, shift_units)
                            if not value:
                                raise RuntimeError(dynamic.debug())
                            row["online_boundaries"] = json.loads(value)
                    if shop_branches or tail_seam:
                        dynamic = agents[mode]
                        fn = getattr(dynamic.lib, "td_candidate_shop_branch_json", None)
                        if fn is None:
                            raise RuntimeError("shop branches require R2_SHOP_BRANCH_AUDIT binary")
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                       ctypes.c_int]
                        fn.restype = ctypes.c_char_p
                        packed = main.policy._pack(own)
                        for row in rows:
                            if row.get("diagnostic"):
                                continue
                            baseline = fn(dynamic.handle, row["index"], packed,
                                          len(packed), -1)
                            if not baseline:
                                raise RuntimeError(dynamic.debug())
                            row["no_shop_branch"] = json.loads(baseline)
                            row["shop_mean_q"] = row["no_shop_branch"]["q"]
                            if shop_branches:
                                row["shop_branches"] = []
                                for shop in range(8):
                                    value = fn(dynamic.handle, row["index"], packed,
                                               len(packed), shop)
                                    if not value:
                                        raise RuntimeError(dynamic.debug())
                                    row["shop_branches"].append(json.loads(value))
                    if closed_loop_top:
                        eligible = [row for row in rows if not row.get("diagnostic") or
                                    row.get("diagnostic", "").startswith("outer")]
                        normal = [row for row in eligible if not row.get("diagnostic")]
                        selected = sorted(eligible, key=lambda row: row["score"], reverse=True)[:closed_loop_top]
                        if normal:
                            selected.append(max(normal, key=lambda row: row["score"]))
                        for row in {value["index"]: value for value in selected}.values():
                            row["closed_loop_terminal"] = agents[mode].candidate_score_horizon(
                                own, row["index"], 30 - own["day"])
                return {
                    "bot": bot, "seed": seed, "seat": seat,
                    "step": own["step"],
                    "clock": policy.dynamic.clock_debug(),
                    "modes": {str(mode): {"summary": summarize(rows), "proposals": rows}
                              for mode, rows in candidates.items()},
                }
            actions = [None, None]
            actions[seat] = policy(own, {})
            for agent in extras.values():
                agent.observe_external(own, actions[seat])
            other = obs[1 - seat]
            actions[1 - seat] = opponent(other, {}) if with_config else opponent(other)
            state = env.step(actions)
        raise RuntimeError("game ended before warm handoff")
    finally:
        policy.close()
        for agent in extras.values():
            agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "work/agent-greedy-audit.so")
    parser.add_argument("--start", type=int, default=2609800000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--opponents", default=",".join(BOTS))
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--modes", default="0,1,2")
    parser.add_argument("--closed-loop-top", type=int, default=0)
    parser.add_argument("--shop-branches", action="store_true",
                        help="8 fixed next-shop branches on all normal candidates; prefix only")
    parser.add_argument("--tail-seam", action="store_true",
                        help="compare static tail plan with next-day closed-loop plan")
    parser.add_argument("--online-boundaries-days", type=int, default=0,
                        help="diagnostic same-flow online Bellman boundaries, 2..6 days")
    parser.add_argument("--online-ids", default="0,1",
                        help="candidate ids for --online-boundaries-days")
    parser.add_argument("--online-shift-item", type=int, default=-1,
                        help="fixed-portfolio one-day sale shift item; diagnostic only")
    parser.add_argument("--online-shift-units", type=int, default=0,
                        help="fixed-portfolio one-day sale shift quantity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    # Every case uses the same immutable deployment tape.  Avoid paying its
    # large zlib/JSON decode cost again for each independent prefix.
    from meta_agent.src import teammate_expanded_routes
    teammate_expanded_routes.load_action_tapes = functools.lru_cache(maxsize=1)(
        teammate_expanded_routes.load_action_tapes)
    bots = [name for name in args.opponents.split(",") if name]
    seats = [int(value) for value in args.seats.split(",") if value]
    modes = tuple(int(value) for value in args.modes.split(",") if value)
    if not modes or any(mode not in (0, 1, 2) for mode in modes):
        parser.error("--modes must contain one or more of 0,1,2")
    if args.closed_loop_top < 0:
        parser.error("--closed-loop-top must be non-negative")
    if args.online_boundaries_days and not 2 <= args.online_boundaries_days <= 6:
        parser.error("--online-boundaries-days must be 2..6")
    if (args.online_shift_item not in range(-1, 9) or args.online_shift_units < 0 or
            (args.online_shift_item == -1 and args.online_shift_units)):
        parser.error("invalid online shift item/units")
    online_ids = {int(value) for value in args.online_ids.split(",") if value}
    rows = [run_case(args.binary.resolve(), bot, seed, seat, modes,
                     args.closed_loop_top, args.shop_branches, args.tail_seam,
                     args.online_boundaries_days, online_ids,
                     args.online_shift_item, args.online_shift_units)
            for bot in bots for seed in range(args.start, args.start + args.seeds)
            for seat in seats]
    totals = {}
    for mode in modes:
        summaries = [row["modes"][str(mode)]["summary"] for row in rows]
        totals[str(mode)] = {
            key: sum(summary[key] for summary in summaries)
            for key in ("proposals", "slots", "rank_raw_mismatches", "local_regret_sum",
                        "value_mismatches", "value_regret_sum", "preview_removed",
                        "swap_gain", "pair_gain")
        }
        totals[str(mode)]["local_regret_max"] = max(
            summary["local_regret_max"] for summary in summaries)
        totals[str(mode)]["value_regret_max"] = max(
            summary["value_regret_max"] for summary in summaries)
    result = {"binary": str(args.binary), "scope": "prefix only; stops before handoff action",
              "seed_range": [args.start, args.start + args.seeds], "rows": rows,
              "totals": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
