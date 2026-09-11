#!/usr/bin/env python3
"""GPU screen for the thin B21 foreign-sale forecast layered on FC15."""

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
    fc16_b21_forecast_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
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


def build_configs() -> list[dict]:
    configs = [
        {
            "name": "fc15_control",
            "enabled": False,
            "start": 120,
            "distance": 6,
            "quantity": 0,
            "minimum_price_percent": 100,
        }
    ]
    priority = (
        (120, 100, 1, 100),
        (120, 100, 2, 100),
        (120, 100, 4, 100),
        (96, 100, 2, 100),
        (168, 100, 2, 100),
        (120, 24, 2, 100),
        (120, 12, 2, 100),
        (120, 100, 2, 90),
        (120, 100, 2, 80),
    )
    seen = set()
    for start, distance, quantity, minimum in priority:
        seen.add((start, distance, quantity, minimum))
        configs.append(
            {
                "name": f"b21_s{start}_d{distance}_q{quantity}_m{minimum}",
                "enabled": True,
                "start": start,
                "distance": distance,
                "quantity": quantity,
                "minimum_price_percent": minimum,
            }
        )
    for start in (96, 120, 168):
        for distance in (12, 24, 100):
            for quantity in (1, 2, 4):
                key = (start, distance, quantity, 100)
                if key in seen:
                    continue
                seen.add(key)
                configs.append(
                    {
                        "name": f"b21_s{start}_d{distance}_q{quantity}_m100",
                        "enabled": True,
                        "start": start,
                        "distance": distance,
                        "quantity": quantity,
                        "minimum_price_percent": 100,
                    }
                )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=825001)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--config-start", type=int, default=0)
    parser.add_argument("--config-count", type=int, default=8)
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected_opponents = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if any(value not in OPPONENTS for value in selected_opponents):
        raise ValueError("invalid opponent")
    all_configs = build_configs()
    if args.seeds < 1 or args.config_count < 1 or not (0 <= args.config_start < len(all_configs)):
        raise ValueError("invalid seeds or config shard")
    valid_configs = all_configs[args.config_start : args.config_start + args.config_count]
    configs = list(valid_configs)
    while len(configs) < args.config_count:
        padding = dict(all_configs[0])
        padding["name"] = f"padding_{len(configs)}"
        configs.append(padding)

    old_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz")
    latest8_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz")
    latest6_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz")
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    tables = load_tables()
    simulator = rr.make_simulator_step(tables)
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    config_count = len(configs)
    expanded_seeds = np.tile(seeds, config_count)
    expanded_weed = np.tile(weed, (config_count,) + (1,) * (weed.ndim - 1))
    expanded_shops = np.tile(shops, (config_count,) + (1,) * (shops.ndim - 1))
    events = Events(jnp.asarray(expanded_weed), jnp.asarray(expanded_shops))
    batch = int(expanded_seeds.size)
    def lanes(field, dtype):
        return jnp.repeat(jnp.asarray([row[field] for row in configs], dtype=dtype), args.seeds)
    enabled = lanes("enabled", jnp.bool_)
    start = lanes("start", jnp.int16)
    distance = lanes("distance", jnp.int16)
    quantity = lanes("quantity", jnp.int16)
    minimum = lanes("minimum_price_percent", jnp.int16)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    candidate_policies = {}
    for seat in (0, 1):
        @jax.jit
        def candidate_policy(
            states,
            carry,
            enabled_arg,
            start_arg,
            distance_arg,
            quantity_arg,
            minimum_arg,
            player=seat,
        ):
            return fc16_b21_forecast_preempt_player_action_v1(
                states,
                tables,
                latest8_bank,
                old_bank,
                runtime,
                carry,
                enabled_arg,
                start_arg,
                distance_arg,
                quantity_arg,
                minimum_arg,
                player,
            )
        candidate_policies[seat] = candidate_policy
    opponent_policies = {}
    for name in selected_opponents:
        function, _ = OPPONENTS[name]
        for seat in (0, 1):
            def make_policy(fn, player):
                @jax.jit
                def policy(states, carry):
                    return fn(states, runtime, latest6_bank, carry, player)
                return policy
            opponent_policies[(name, seat)] = make_policy(function, seat)

    rows = []
    started = perf_counter()
    for opponent_name in selected_opponents:
        _, opponent_init = OPPONENTS[opponent_name]
        orientation_money = []
        for candidate_seat in (0, 1):
            rival = 1 - candidate_seat
            states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
            candidate_carry = initialize_fusion_champion_carry_v3(batch)
            opponent_carry = opponent_init(batch)
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policies[candidate_seat](
                    states,
                    candidate_carry,
                    enabled,
                    start,
                    distance,
                    quantity,
                    minimum,
                )
                opponent_action, opponent_carry = opponent_policies[(opponent_name, rival)](states, opponent_carry)
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if candidate_seat == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError(f"configs vs {opponent_name}: unfinished")
            if any(
                int(np.sum(np.asarray(value)))
                for value in (terminal.hand_cap_hits, terminal.market_loop_cap_hits, terminal.price_lut_oob)
            ):
                raise AssertionError(f"configs vs {opponent_name}: safety counter")
            orientation_money.append(
                np.asarray(terminal.money, dtype=np.int64).reshape(config_count, args.seeds, 2)
            )
        first, second = orientation_money
        for local_id, config in enumerate(valid_configs):
            config_id = args.config_start + local_id
            candidate_cash = np.concatenate((first[local_id, :, 0], second[local_id, :, 1]))
            opponent_cash = np.concatenate((first[local_id, :, 1], second[local_id, :, 0]))
            margins = candidate_cash - opponent_cash
            row = {
                "opponent": opponent_name,
                "config_id": config_id,
                "config": config,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    aggregate = []
    for local_id, config in enumerate(valid_configs):
        config_id = args.config_start + local_id
        selected = [row for row in rows if row["config_id"] == config_id]
        aggregate.append(
            {
                "config_id": config_id,
                "config": config,
                "min_score_rate": min(row["score_rate"] for row in selected),
                "mean_score_rate": float(np.mean([row["score_rate"] for row in selected])),
                "mean_margin": float(np.mean([row["mean_margin"] for row in selected])),
                "by_opponent": {row["opponent"]: row["score_rate"] for row in selected},
            }
        )
    aggregate.sort(
        key=lambda row: (row["min_score_rate"], row["mean_score_rate"], row["mean_margin"]),
        reverse=True,
    )
    payload = {
        "schema": "kaggriculture.fc16-b21-forecast-preempt-grid.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "total_config_count": len(all_configs),
        "config_start": int(args.config_start),
        "valid_config_count": len(valid_configs),
        "padded_config_count": config_count,
        "expanded_batch": batch,
        "opponents": selected_opponents,
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
        "aggregate": aggregate,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
