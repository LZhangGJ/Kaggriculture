#!/usr/bin/env python3
"""GPU ablation of compatible K320 route remaps against exact Boatlee V21."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    k320_route_map_candidate_player_action_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
)


REMAPS = (
    ("identity", {}),
    ("route2_to_1", {2: 1}),
    ("route2_to_0", {2: 0}),
    ("route4_to_5", {4: 5}),
    ("route4_to_6", {4: 6}),
    ("route4_to_7", {4: 7}),
    ("route4_to_13", {4: 13}),
    ("route2_to_1_route4_to_5", {2: 1, 4: 5}),
    ("route2_to_1_route4_to_13", {2: 1, 4: 13}),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=826001)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

    bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    tables = load_tables()
    simulator = rr.make_simulator_step(tables)
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    config_count = len(REMAPS)
    expanded_seeds = np.tile(seeds, config_count)
    events = Events(
        jnp.asarray(np.tile(weed, (config_count,) + (1,) * (weed.ndim - 1))),
        jnp.asarray(np.tile(shops, (config_count,) + (1,) * (shops.ndim - 1))),
    )
    route_count = int(bank.unit_op.shape[0])
    route_maps = []
    for _, edits in REMAPS:
        route_map = np.arange(route_count, dtype=np.int32)
        for source, target in edits.items():
            route_map[source] = target
        route_maps.append(route_map)
    route_lanes = jnp.repeat(
        jnp.asarray(route_maps, dtype=jnp.int32), args.seeds, axis=0
    )
    batch = int(expanded_seeds.size)
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    candidate_policies = {}
    opponent_policies = {}
    for seat in (0, 1):
        @jax.jit
        def candidate_policy(states, carry, dynamic_map, player=seat):
            return k320_route_map_candidate_player_action_v1(
                states, tables, runtime, bank, carry, dynamic_map, player
            )

        @jax.jit
        def opponent_policy(states, carry, player=1 - seat):
            return boatlee_v21_player_action_v1(states, runtime, bank, carry, player)

        candidate_policies[seat] = candidate_policy
        opponent_policies[1 - seat] = opponent_policy

    orientations = []
    started = perf_counter()
    for candidate_seat in (0, 1):
        rival = 1 - candidate_seat
        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = initialize_high_potential_v20_carry_v1(batch)
        opponent_carry = initialize_boatlee_v21_carry_v1(batch)
        for _ in range(719):
            candidate_action, candidate_carry = candidate_policies[candidate_seat](
                states, candidate_carry, route_lanes
            )
            opponent_action, opponent_carry = opponent_policies[rival](states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if candidate_seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("unfinished game")
        if any(int(np.sum(np.asarray(value))) for value in (
            terminal.hand_cap_hits,
            terminal.market_loop_cap_hits,
            terminal.price_lut_oob,
        )):
            raise AssertionError("safety counter")
        orientations.append(
            np.asarray(terminal.money, dtype=np.int64).reshape(config_count, args.seeds, 2)
        )

    first, second = orientations
    rows = []
    for config_id, (name, edits) in enumerate(REMAPS):
        candidate_cash = np.concatenate((first[config_id, :, 0], second[config_id, :, 1]))
        opponent_cash = np.concatenate((first[config_id, :, 1], second[config_id, :, 0]))
        margins = candidate_cash - opponent_cash
        row = {
            "config_id": config_id,
            "name": name,
            "edits": edits,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "per_game": [
                {
                    "seed": int(seed),
                    "candidate_seat": int(seat),
                    "candidate_cash": int(
                        first[config_id, offset, 0]
                        if seat == 0
                        else second[config_id, offset, 1]
                    ),
                    "opponent_cash": int(
                        first[config_id, offset, 1]
                        if seat == 0
                        else second[config_id, offset, 0]
                    ),
                    "margin": int(
                        first[config_id, offset, 0] - first[config_id, offset, 1]
                        if seat == 0
                        else second[config_id, offset, 1] - second[config_id, offset, 0]
                    ),
                }
                for offset, seed in enumerate(seeds.tolist())
                for seat in (0, 1)
            ],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "schema": "kaggriculture.k320-route-map-vs-b21-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "games_per_config": int(args.seeds) * 2,
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
