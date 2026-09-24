#!/usr/bin/env python3
"""Screen G275-to-DSM opening switches with native traces, stopping at step 288."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fast_kaggriculture as fk

from audit_dsm_prefix_native import actions, state
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def requested(order):
    op = order[0]
    if op in ("BUY_SEED", "BUY_ANIMAL"):
        return max(0, int(order[2])) if len(order) > 2 else 1
    return int(op in ("HIRE", "BUY_LAND"))


def replay_trace(trace, seed, seat, switch_step, baseline=None):
    env = fk.FastEnv(fk.Config(), seed)
    first_action_divergence = None
    first_state_divergence = {}
    first_unfilled = first_land_unfilled = None
    attempts = fills_total = shortfall_events = 0
    fourth_land_step = None
    failures = {}
    states = []
    switch_state = None
    for step, pair in enumerate(trace):
        own = pair[seat]
        current = state(env.observation(seat), seat)
        states.append(current)
        if step == switch_step:
            switch_state = current
        if baseline is not None:
            if first_action_divergence is None and actions(own) != actions(baseline["trace"][step][seat]):
                first_action_divergence = step
            reference = baseline["states"][step]
            for field in ("cash", "land", "crops", "animals", "weeds"):
                if field not in first_state_divergence and current[field] != reference[field]:
                    first_state_divergence[field] = {
                        "step": step, "live": current[field], "g275": reference[field]}
        env.step_raw(pair)
        market_fills = env.last_market_fills[seat]
        shortfalls = env.last_market_cash_shortfalls[seat]
        for index, order in enumerate(own.get("market") or []):
            qty = requested(order)
            if qty == 0:
                continue
            op = order[0]
            fill = int(market_fills[index]) if index < len(market_fills) else 0
            if op == "BUY_LAND":
                attempts += qty
                fills_total += min(qty, fill)
                if fill > 0 and fourth_land_step is None and state(env.observation(seat), seat)["land"] >= 4:
                    fourth_land_step = step
            if fill < qty:
                failures[op] = failures.get(op, 0) + qty - fill
                event = {"step": step, "op": op, "order": order,
                         "cash_before": current["cash"], "filled": fill,
                         "requested": qty,
                         "cash_shortfall": int(shortfalls[index]) if index < len(shortfalls) else 0}
                if first_unfilled is None:
                    first_unfilled = event
                if op == "BUY_LAND" and first_land_unfilled is None:
                    first_land_unfilled = event
            if index < len(shortfalls) and shortfalls[index] > 0:
                shortfall_events += 1
    if len(trace) != 288:
        raise ValueError(f"native prefix trace length {len(trace)} != 288")
    return {
        "handoff": state(env.observation(seat), seat),
        "switch_state": switch_state,
        "first_action_divergence_from_g275": first_action_divergence,
        "first_state_divergence_from_g275": first_state_divergence,
        "first_unfilled": first_unfilled,
        "first_land_unfilled": first_land_unfilled,
        "order_failures": failures,
        "land_attempts": attempts,
        "land_filled": fills_total,
        "fourth_land_purchase_step": fourth_land_step,
        "cash_shortfall_events": shortfall_events,
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=3300770000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--arm-plan", type=Path,
                        help="frozen JSON with arms, seed_start and seeds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    bundle = NativeTeammateBundle(Path("agent/teammate_base.py"), args.actions, args.metadata)
    metadata = json.loads(args.metadata.read_text())
    families = [entry["family"] for entry in metadata["opponent_routes"]
                if entry.get("team") == "DSM" and entry.get("selected")]
    if args.arm_plan:
        plan = json.loads(args.arm_plan.read_text())
        if plan["seed_start"] != args.seed_start or plan["seeds"] != args.seeds:
            raise ValueError("CLI seed segment differs from preregistration")
        arms = [(entry["family"], int(entry["switch_step"])) for entry in plan["arms"]]
        if len(arms) != len(set(arms)) or any(family not in families or cp not in (144, 168)
                                                 for family, cp in arms):
            raise ValueError("invalid or duplicate preregistered arm")
    else:
        arms = [(family, cp) for family in families for cp in (-1, 144, 168)]
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    baselines = {}
    rows = []
    for seed in seeds:
        for seat in (0, 1):
            result = bundle.play("G275", "G275", seed, capture_trace=True,
                                 stop_after_steps=288)
            summary = replay_trace(result["trace"], seed, seat, -1)
            baselines[seed, seat] = {"trace": result["trace"], "states": summary.pop("states")}
            rows.append({"variant": "G275", "family": "G275", "seed": seed, "seat": seat,
                         **summary, "native_macro_unit_failures": result["macro_unit_failures"][seat],
                         "native_macro_market_failures": result["macro_market_failures"][seat]})
    for family, cp in arms:
        for seed in seeds:
            for seat in (0, 1):
                baseline = baselines[seed, seat]
                if cp < 0:
                    left, right = (family, "G275") if seat == 0 else ("G275", family)
                    result = bundle.play(left, right, seed, capture_trace=True,
                                         stop_after_steps=288)
                else:
                    result = bundle.play("G275", "G275", seed, switch_step=cp,
                                         switch_target=family, seat=seat,
                                         capture_trace=True, stop_after_steps=288)
                summary = replay_trace(result["trace"], seed, seat, cp, baseline)
                summary.pop("states")
                rows.append({"variant": "pure_d" if cp < 0 else f"switch{cp}",
                             "family": family, "seed": seed, "seat": seat,
                             **summary,
                             "native_macro_unit_failures": result["macro_unit_failures"][seat],
                             "native_macro_market_failures": result["macro_market_failures"][seat]})
        print(f"screened {family}@{cp}: {args.seeds * 2} prefixes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(rows)} prefixes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
