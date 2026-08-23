#!/usr/bin/env python3
"""GPU paired screen of FC15 opening wheat quantities versus Boatlee V21."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
    fc15_opening_transaction_player_action_v1,
    initialize_fusion_champion_carry_v3,
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=843001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--quantities", default="6,7,8,9,10,12")
    parser.add_argument(
        "--arms",
        default="",
        help="Optional comma-separated WHEAT_SEED:MELON_SEED:WHEAT_PRODUCT triples.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.arms:
        arms = np.asarray(
            [
                tuple(int(part) for part in value.strip().split(":"))
                for value in args.arms.split(",")
                if value.strip()
            ],
            dtype=np.int16,
        )
        if arms.ndim != 2 or arms.shape[1] != 3:
            raise ValueError("each arm must be WHEAT_SEED:MELON_SEED:WHEAT_PRODUCT")
    else:
        quantities = [
            int(value.strip()) for value in args.quantities.split(",") if value.strip()
        ]
        arms = np.asarray([(7, 12, quantity) for quantity in quantities], dtype=np.int16)
    if arms.size == 0 or len(set(map(tuple, arms.tolist()))) != arms.shape[0]:
        raise ValueError("arms must be nonempty and unique")
    if np.any((arms < 1) | (arms > 32)):
        raise ValueError("opening quantities must be within 1..32")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    tables = load_tables()
    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    latest8_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    latest6_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_bank = load_bank(old_bank_path)
    latest8_bank = load_bank(latest8_bank_path)
    latest6_bank = load_bank(latest6_bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    simulator = rr.make_simulator_step(tables)

    source_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded_seeds = np.tile(source_seeds, arms.shape[0])
    expanded_arms = np.repeat(arms, source_seeds.size, axis=0)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(expanded_seeds.size)
    opening_wheat_seeds = jnp.asarray(expanded_arms[:, 0])
    opening_melon_seeds = jnp.asarray(expanded_arms[:, 1])
    opening_wheat_products = jnp.asarray(expanded_arms[:, 2])

    orientation_money = []
    orientation_routes = []
    orientation_overlay = []
    orientation_step1_money = []
    started = perf_counter()
    for candidate_seat in (0, 1):
        player = candidate_seat
        rival = 1 - player

        @jax.jit
        def candidate_policy(states, carry):
            return fc15_opening_transaction_player_action_v1(
                states,
                tables,
                latest8_bank,
                old_bank,
                runtime,
                carry,
                opening_wheat_seeds,
                opening_melon_seeds,
                opening_wheat_products,
                player,
            )

        @jax.jit
        def opponent_policy(states, carry):
            return boatlee_v21_player_action_v1(
                states, runtime, latest6_bank, carry, rival
            )

        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = initialize_fusion_champion_carry_v3(batch)
        opponent_carry = initialize_boatlee_v21_carry_v1(batch)
        step1_money = None
        for action_index in range(719):
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if candidate_seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
            if action_index == 0:
                step1_money = np.asarray(
                    jax.device_get(states.money[:, player]), dtype=np.int32
                )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError(f"seat {candidate_seat}: unfinished")
        if any(
            int(np.sum(np.asarray(value)))
            for value in (
                terminal.hand_cap_hits,
                terminal.market_loop_cap_hits,
                terminal.price_lut_oob,
            )
        ):
            raise AssertionError(f"seat {candidate_seat}: safety counter")
        orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
        host_carry = jax.device_get(opponent_carry)
        orientation_routes.append(np.asarray(host_carry.route, dtype=np.int8))
        orientation_overlay.append(np.asarray(host_carry.market_overlay, dtype=bool))
        orientation_step1_money.append(step1_money)

    rows = []
    for arm_index, arm in enumerate(arms.tolist()):
        sl = slice(arm_index * source_seeds.size, (arm_index + 1) * source_seeds.size)
        first = orientation_money[0][sl]
        second = orientation_money[1][sl]
        candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
        opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
        margins = candidate_cash - opponent_cash
        routes = np.concatenate((orientation_routes[0][sl], orientation_routes[1][sl]))
        overlays = np.concatenate((orientation_overlay[0][sl], orientation_overlay[1][sl]))
        step1_money = np.concatenate((orientation_step1_money[0][sl], orientation_step1_money[1][sl]))
        row = {
            "opening_wheat_seed_quantity": int(arm[0]),
            "opening_melon_seed_quantity": int(arm[1]),
            "opening_wheat_product_quantity": int(arm[2]),
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "mean_candidate_cash": float(np.mean(candidate_cash)),
            "mean_opponent_cash": float(np.mean(opponent_cash)),
            "step1_money_values": sorted(set(step1_money.astype(int).tolist())),
            "b21_route_counts": {
                str(route): int(np.sum(routes == route)) for route in (0, 1, 2)
            },
            "b21_market_overlay_count": int(np.sum(overlays)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    policy_path = ROOT / "experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py"
    b21_path = ROOT / "experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py"
    payload = {
        "schema": "kaggriculture.fc15-b21-opening-transaction-screen.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "games_per_arm": int(args.seeds * 2),
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
        "sources": {
            "fc15_policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
            "b21_policy": {"path": str(b21_path), "sha256": sha256(b21_path)},
            "old_bank": {"path": str(old_bank_path), "sha256": sha256(old_bank_path)},
            "latest8_bank": {"path": str(latest8_bank_path), "sha256": sha256(latest8_bank_path)},
            "latest6_bank": {"path": str(latest6_bank_path), "sha256": sha256(latest6_bank_path)},
            "runtime": {"path": str(runtime_path), "sha256": sha256(runtime_path)},
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
