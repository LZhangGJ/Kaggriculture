#!/usr/bin/env python3
"""GPU screen for public-state Moon timing parameters on FC15."""

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
    fc15_moon_parameter_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
    initialize_moon_v92_carry_v1,
    initialize_soil_v26h_carry_v1,
    prvsiyan_moon_v92_player_action_v1,
    prvsiyan_soil_v26h_player_action_v1,
)


OPPONENTS = {
    "boatlee_v21_latest": (boatlee_v21_player_action_v1, initialize_boatlee_v21_carry_v1),
    "prvsiyan_soil_v26h_latest": (prvsiyan_soil_v26h_player_action_v1, initialize_soil_v26h_carry_v1),
    "prvsiyan_moon_v92_latest": (prvsiyan_moon_v92_player_action_v1, initialize_moon_v92_carry_v1),
}


def variants() -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = [
        dict(name="embedded_control", minimum_horizon=1, start=999, floor_distance=0, action_distance=6),
        dict(name="action_d0", minimum_horizon=1, start=999, floor_distance=0, action_distance=0),
        dict(name="action_d2", minimum_horizon=1, start=999, floor_distance=0, action_distance=2),
        dict(name="action_d4", minimum_horizon=1, start=999, floor_distance=0, action_distance=4),
    ]
    for horizon in (2, 3, 4, 5):
        for distance in (2, 4, 6):
            rows.append(
                dict(
                    name=f"h{horizon}_s120_fd{distance}_ad6",
                    minimum_horizon=horizon,
                    start=120,
                    floor_distance=distance,
                    action_distance=6,
                )
            )
    for horizon in (3, 4, 5):
        for start in (96, 168):
            rows.append(
                dict(
                    name=f"h{horizon}_s{start}_fd4_ad6",
                    minimum_horizon=horizon,
                    start=start,
                    floor_distance=4,
                    action_distance=6,
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="prvsiyan_moon_v92_latest")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--variant-start", type=int, default=0)
    parser.add_argument("--variant-count", type=int, default=0)
    parser.add_argument(
        "--variant-indices",
        default="",
        help="Optional comma-separated exact variant indices; overrides shard range.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    all_configs = variants()
    if args.variant_indices:
        selected_indices = [
            int(value.strip())
            for value in args.variant_indices.split(",")
            if value.strip()
        ]
        if (
            not selected_indices
            or len(selected_indices) != len(set(selected_indices))
            or any(index < 0 or index >= len(all_configs) for index in selected_indices)
        ):
            raise ValueError("invalid variant indices")
    else:
        stop = (
            len(all_configs)
            if args.variant_count <= 0
            else min(len(all_configs), args.variant_start + args.variant_count)
        )
        if args.variant_start < 0 or args.variant_start >= stop:
            raise ValueError("invalid variant shard")
        selected_indices = list(range(args.variant_start, stop))
    configs = [
        {**all_configs[index], "variant_index": index}
        for index in selected_indices
    ]
    tables = load_tables()
    old_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz")
    latest8_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz")
    latest6_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz")
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = rr.make_simulator_step(tables)
    opponent_fn, opponent_init = OPPONENTS[args.opponent]

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded_seeds = np.tile(base_seeds, len(configs))
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    minimum_horizon = jnp.repeat(jnp.asarray([row["minimum_horizon"] for row in configs], dtype=jnp.int16), args.seeds)
    start = jnp.repeat(jnp.asarray([row["start"] for row in configs], dtype=jnp.int16), args.seeds)
    floor_distance = jnp.repeat(jnp.asarray([row["floor_distance"] for row in configs], dtype=jnp.int16), args.seeds)
    action_distance = jnp.repeat(jnp.asarray([row["action_distance"] for row in configs], dtype=jnp.int16), args.seeds)
    batch = len(expanded_seeds)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc15_moon_parameter_player_action_v1(
                states,
                tables,
                latest8_bank,
                old_bank,
                runtime,
                carry,
                minimum_horizon,
                start,
                floor_distance,
                action_distance,
                player,
            )

        return policy

    orientation_money = []
    started = perf_counter()
    for candidate_seat in (0, 1):
        rival = 1 - candidate_seat

        @jax.jit
        def opponent_policy(states, carry):
            return opponent_fn(states, runtime, latest6_bank, carry, rival)

        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = initialize_fusion_champion_moon_market_carry_v1(batch)
        opponent_carry = opponent_init(batch)
        candidate_policy = make_candidate_policy(candidate_seat)
        for _ in range(719):
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if candidate_seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("unfinished game")
        if any(
            int(np.sum(np.asarray(value)))
            for value in (
                terminal.hand_cap_hits,
                terminal.market_loop_cap_hits,
                terminal.price_lut_oob,
            )
        ):
            raise AssertionError("simulator safety counter")
        orientation_money.append(
            np.asarray(terminal.money, dtype=np.int64).reshape(len(configs), args.seeds, 2)
        )

    first, second = orientation_money
    rows = []
    margin_matrix = []
    for index, config in enumerate(configs):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival_cash = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival_cash
        margin_matrix.append(margins)
        rows.append(
            {
                **config,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_margin": float(np.mean(margins)),
                "per_game": [
                    {
                        "seed": int(base_seeds[game % args.seeds]),
                        "candidate_seat": 0 if game < args.seeds else 1,
                        "margin": int(margin),
                    }
                    for game, margin in enumerate(margins.tolist())
                ],
            }
        )
    oracle = np.max(np.stack(margin_matrix, axis=0), axis=0)
    payload = {
        "schema": "kaggriculture.fc15-moon-parameter-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "variant_start": args.variant_start,
        "variant_count": len(configs),
        "variant_indices": selected_indices,
        "total_variant_count": len(all_configs),
        "games_per_variant": args.seeds * 2,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = sorted(
        ({key: value for key, value in row.items() if key != "per_game"} for row in rows),
        key=lambda row: (row["score_rate"], row["mean_margin"]),
        reverse=True,
    )
    print(json.dumps({"status": "PASS", "top": summary[:8], "oracle": payload["oracle"], "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
