#!/usr/bin/env python3
"""Compare DSM route-shell actions with their source replays through step 288."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fast_kaggriculture as fk
import orjson

from audit_macro_execution_quality import _make_config
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ITEM_OPS = {"PLANT", "PLACE", "PICKUP", "DROP", "SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}


def order(value):
    value = list(value or ["PASS"])
    op = value[0]
    if op not in ITEM_OPS:
        return (op,)
    return op, value[1] if len(value) > 1 else None, value[2] if len(value) > 2 else 1


def actions(value):
    return (
        (order(value.get("farmer")), *(order(x) for x in value.get("hands") or [])),
        tuple(order(x) for x in value.get("market") or []),
    )


def state(observation, player):
    farm = observation["farms"][player]
    tiles = [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]
    return {
        "land": len(farm["unlocked_quadrants"]), "cash": farm["money"],
        "crops": sum(tile.get("kind") == "PLANT" for tile in tiles),
        "animals": sum(tile.get("animal") is not None for tile in tiles),
        "weeds": sum(tile.get("kind") == "WEED" for tile in tiles),
    }


def failures(tape, other, player, seed, config):
    metrics, _ = fk.audit_raw_tapes(
        tape if player == 0 else other,
        other if player == 0 else tape,
        seed, config,
    )
    names = list(fk.raw_tape_audit_metric_names())
    row = dict(zip(names, metrics[player].tolist()))
    return {
        "unit": row["unit_attempts"] - row["unit_valid"],
        "market": row["market_requested"] - row["market_filled"],
        "land_requested": row["land_requested"],
        "land_filled": row["land_filled"],
        "cash_shortfall_events": row["cash_shortfall_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("agent/teammate_base.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    paths = {int(row["episode_id"]): (args.manifest.parent / row["replay"]).resolve()
             for row in manifest["replays"]}
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    result = []
    for entry in json.loads(args.metadata.read_text())["opponent_routes"]:
        episode, seat = map(int, entry["route_id"].split(":"))
        replay = orjson.loads(paths[episode].read_bytes())
        seed = int(replay["info"]["seed"])
        config = _make_config(fk, replay)
        env = fk.FastEnv(config, seed)
        actor = fk.NativeReplayOpponent(bundle.executor, bundle.index(entry["family"]), False, 0)
        source_tape, other_tape, actual_tape = [], [], []
        first_unit = first_market = None
        unit_mismatch = market_mismatch = 0
        for step in range(288):
            actual = actor.action(env, seat, step)
            source = replay["steps"][step + 1][seat].get("action") or {}
            other = replay["steps"][step + 1][1 - seat].get("action") or {}
            actual_tape.append(actual)
            source_tape.append(source)
            other_tape.append(other)
            actual_orders, source_orders = actions(actual), actions(source)
            if actual_orders[0] != source_orders[0]:
                unit_mismatch += 1
                if first_unit is None:
                    first_unit = step
            if actual_orders[1] != source_orders[1]:
                market_mismatch += 1
                if first_market is None:
                    first_market = step
            env.step_raw([actual, other] if seat == 0 else [other, actual])
        source_state = replay["steps"][288][seat]["observation"]
        result.append({
            **{key: entry[key] for key in ("family", "route_id", "support", "selected", "historical_margin")},
            "seed": seed, "source": state(source_state, seat),
            "native": state(env.observation(seat), seat),
            "unit_mismatch_steps": unit_mismatch, "market_mismatch_steps": market_mismatch,
            "first_unit_mismatch": first_unit, "first_market_mismatch": first_market,
            "source_failures": failures(source_tape, other_tape, seat, seed, config),
            "native_failures": failures(actual_tape, other_tape, seat, seed, config),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
