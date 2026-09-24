#!/usr/bin/env python3
"""Screen isolated DSM prefix routes through step 288 against frozen G275."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fast_kaggriculture as fk
import orjson

from audit_dsm_prefix_native import actions, state
from audit_macro_execution_quality import _make_config
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def requested(order):
    op = order[0]
    if op in ("BUY_SEED", "BUY_ANIMAL"):
        return max(0, int(order[2])) if len(order) > 2 else 1
    return int(op in ("HIRE", "BUY_LAND"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=3300770000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"

    metadata = json.loads(args.metadata.read_text())
    selected = [entry for entry in metadata["opponent_routes"]
                if entry.get("team") == "DSM" and entry.get("selected")]
    tapes = load_action_tapes(args.actions)
    bundle = NativeTeammateBundle(Path("agent/teammate_base.py"), args.actions, args.metadata)
    manifest = json.loads(args.manifest.read_text())
    replay_paths = {int(row["episode_id"]): args.manifest.parent / row["replay"]
                    for row in manifest["replays"]}
    first = manifest["replays"][0]
    replay = orjson.loads((args.manifest.parent / first["replay"]).read_bytes())
    config = _make_config(fk, replay)
    rows = []
    for entry in selected:
        family = entry["family"]
        tape = tapes[entry["route_id"]]
        source_episode, source_seat = map(int, entry["route_id"].split(":"))
        source_steps = orjson.loads(replay_paths[source_episode].read_bytes())["steps"]
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for seat in (0, 1):
                env = fk.FastEnv(config, seed)
                actors = [None, None]
                actors[seat] = fk.NativeReplayOpponent(bundle.executor, bundle.index(family), False, 0)
                actors[1 - seat] = fk.NativeReplayOpponent(bundle.executor, bundle.index("G275"), False, 0)
                first_unit = first_market = first_unfilled = first_land_unfilled = None
                first_unit_context = first_market_context = first_unfilled_context = None
                first_state_divergence = {}
                unit_diff = market_diff = shortfall_events = 0
                order_failures = {}
                order_attempts = {}
                land_attempts = land_filled = 0
                for step in range(288):
                    actual = [actors[p].action(env, p, step) for p in (0, 1)]
                    own = actual[seat]
                    observed = env.observation(seat)
                    own_state = state(observed, seat)
                    source_state = state(source_steps[step][source_seat]["observation"], source_seat)
                    for field in ("cash", "land", "crops", "animals", "weeds"):
                        if field not in first_state_divergence and own_state[field] != source_state[field]:
                            first_state_divergence[field] = {
                                "step": step, "live": own_state[field], "source": source_state[field]}
                    own_action, source_action = actions(own), actions(tape[step])
                    if own_action[0] != source_action[0]:
                        unit_diff += 1
                        if first_unit is None:
                            first_unit = step
                            first_unit_context = {"step": step, "cash": own_state["cash"],
                                                  "land": own_state["land"],
                                                  "actual": {"farmer": own.get("farmer"),
                                                             "hands": own.get("hands")},
                                                  "source": {"farmer": tape[step].get("farmer"),
                                                             "hands": tape[step].get("hands")}}
                    if own_action[1] != source_action[1]:
                        market_diff += 1
                        if first_market is None:
                            first_market = step
                            first_market_context = {"step": step, "cash": own_state["cash"],
                                                    "land": own_state["land"],
                                                    "actual": own.get("market"),
                                                    "source": tape[step].get("market")}
                    env.step_raw(actual)
                    fills = env.last_market_fills[seat]
                    shortfalls = env.last_market_cash_shortfalls[seat]
                    for index, order in enumerate(own.get("market") or []):
                        qty = requested(order)
                        if qty == 0:
                            continue
                        op = order[0]
                        fill = int(fills[index]) if index < len(fills) else 0
                        order_attempts[op] = order_attempts.get(op, 0) + qty
                        if op == "BUY_LAND":
                            land_attempts += qty
                            land_filled += min(qty, fill)
                        if fill < qty:
                            order_failures[op] = order_failures.get(op, 0) + qty - fill
                            context = {"step": step, "op": op, "order": order,
                                       "requested": qty, "filled": fill,
                                       "cash_before": own_state["cash"],
                                       "land_before": own_state["land"],
                                       "cash_shortfall": int(shortfalls[index]) if index < len(shortfalls) else 0}
                            if first_unfilled is None:
                                first_unfilled = step
                                first_unfilled_context = context
                            if op == "BUY_LAND" and first_land_unfilled is None:
                                first_land_unfilled = context
                        if index < len(shortfalls) and shortfalls[index] > 0:
                            shortfall_events += 1
                rows.append({
                    "family": family, "route_id": entry["route_id"], "support": entry["support"],
                    "seed": seed, "seat": seat, "handoff": state(env.observation(seat), seat),
                    "source_handoff": state(source_steps[288][source_seat]["observation"], source_seat),
                    "first_state_divergence": first_state_divergence,
                    "first_unit_mismatch": first_unit_context, "unit_mismatch_steps": unit_diff,
                    "first_market_mismatch": first_market_context, "market_mismatch_steps": market_diff,
                    "first_unfilled": first_unfilled_context, "first_land_unfilled": first_land_unfilled,
                    "order_attempts": order_attempts, "order_failures": order_failures,
                    "land_attempts": land_attempts, "land_filled": land_filled,
                    "cash_shortfall_events": shortfall_events,
                })
        print(f"screened {family}: {args.seeds * 2} prefixes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(rows)} prefixes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
