#!/usr/bin/env python3
"""Capture legal public-state features at FC15/B21-counter first divergence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

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
    boatlee_v21_horizon_counter_player_action_v1,
    fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MarketOp  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    _action_equal,
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
)


def _counts(board, size: int):
    return jnp.stack(
        [jnp.sum(board == value, axis=(1, 2)) for value in range(size)], axis=1
    ).astype(jnp.int16)


def _sales(action):
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    return jnp.stack(
        [
            jnp.sum(
                jnp.where(
                    active
                    & (action.market_op == MarketOp.SELL)
                    & (action.market_item == product),
                    jnp.maximum(action.market_amount, 0),
                    0,
                ),
                axis=1,
            )
            for product in range(9)
        ],
        axis=1,
    ).astype(jnp.int16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=823001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest8_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    latest6_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    tables = load_tables()
    simulator = rr.make_simulator_step(tables)
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(args.seeds)
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    fc15_policies = {}
    counter_policies = {}
    opponent_policies = {}
    enabled = jnp.ones((batch,), dtype=jnp.bool_)
    counter_params = (
        enabled,
        jnp.full((batch,), 2, dtype=jnp.int16),
        jnp.full((batch,), 100, dtype=jnp.int16),
        jnp.full((batch,), 12, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.int16),
        jnp.full((batch,), 100, dtype=jnp.int16),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.full((batch,), 216, dtype=jnp.int16),
    )
    for seat in (0, 1):
        @jax.jit
        def fc15_policy(states, carry, player=seat):
            return fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
                states, tables, latest8_bank, old_bank, runtime, carry, player
            )

        @jax.jit
        def counter_policy(states, carry, *dynamic, player=seat):
            return boatlee_v21_horizon_counter_player_action_v1(
                states, runtime, latest6_bank, carry, *dynamic, player
            )

        @jax.jit
        def opponent_policy(states, carry, player=1 - seat):
            return boatlee_v21_player_action_v1(
                states, runtime, latest6_bank, carry, player
            )

        fc15_policies[seat] = fc15_policy
        counter_policies[seat] = counter_policy
        opponent_policies[1 - seat] = opponent_policy

    records = []
    for seat in (0, 1):
        rival = 1 - seat
        states = jax.vmap(reset)(jnp.asarray(seeds))
        fc15_carry = initialize_fusion_champion_carry_v3(batch)
        counter_carry = initialize_boatlee_v21_carry_v1(batch)
        opponent_carry = initialize_boatlee_v21_carry_v1(batch)
        found = jnp.zeros((batch,), dtype=jnp.bool_)
        saved = {
            "step": jnp.full((batch,), -1, dtype=jnp.int16),
            "money": jnp.zeros((batch, 2), dtype=jnp.int32),
            "hires_today": jnp.zeros((batch, 2), dtype=jnp.int16),
            "town_count": jnp.zeros((batch,), dtype=jnp.int8),
            "town_shops": jnp.full((batch, 8), -1, dtype=jnp.int8),
            "market_price": jnp.zeros((batch, 9), dtype=jnp.int32),
            "market_inventory": jnp.zeros((batch, 9), dtype=jnp.int32),
            "own_shed": jnp.zeros((batch, 12), dtype=jnp.int16),
            "own_crops": jnp.zeros((batch, 5), dtype=jnp.int16),
            "rival_crops": jnp.zeros((batch, 5), dtype=jnp.int16),
            "own_animals": jnp.zeros((batch, 3), dtype=jnp.int16),
            "rival_animals": jnp.zeros((batch, 3), dtype=jnp.int16),
            "unit_count": jnp.zeros((batch, 2), dtype=jnp.int16),
            "fc15_route": jnp.full((batch,), -1, dtype=jnp.int8),
            "counter_route": jnp.full((batch,), -1, dtype=jnp.int8),
            "counter_overlay": jnp.zeros((batch,), dtype=jnp.bool_),
            "fc15_sales": jnp.zeros((batch, 9), dtype=jnp.int16),
            "counter_sales": jnp.zeros((batch, 9), dtype=jnp.int16),
        }
        for _ in range(719):
            fc15_action, fc15_carry = fc15_policies[seat](states, fc15_carry)
            counter_action, counter_carry = counter_policies[seat](
                states, counter_carry, *counter_params
            )
            opponent_action, opponent_carry = opponent_policies[rival](
                states, opponent_carry
            )
            first = (~found) & (~_action_equal(fc15_action, counter_action))
            mask = first[:, None]
            current = {
                "step": states.step.astype(jnp.int16),
                "money": states.money.astype(jnp.int32),
                "hires_today": states.hires_today.astype(jnp.int16),
                "town_count": states.town_count.astype(jnp.int8),
                "town_shops": states.town_shops.astype(jnp.int8),
                "market_price": states.market_price[:, :9].astype(jnp.int32),
                "market_inventory": states.market_inventory[:, :9].astype(jnp.int32),
                "own_shed": states.shed[:, seat].astype(jnp.int16),
                "own_crops": _counts(states.tile_crop[:, seat], 5),
                "rival_crops": _counts(states.tile_crop[:, rival], 5),
                "own_animals": _counts(states.tile_animal[:, seat], 3),
                "rival_animals": _counts(states.tile_animal[:, rival], 3),
                "unit_count": jnp.sum(states.unit_active, axis=2).astype(jnp.int16),
                "fc15_route": fc15_carry.k320.ray_route_id.astype(jnp.int8),
                "counter_route": counter_carry.route.astype(jnp.int8),
                "counter_overlay": counter_carry.market_overlay,
                "fc15_sales": _sales(fc15_action),
                "counter_sales": _sales(counter_action),
            }
            for name, value in current.items():
                choose = first.reshape((batch,) + (1,) * (value.ndim - 1))
                saved[name] = jnp.where(choose, value, saved[name])
            found = found | first
            states = (
                simulator(states, fc15_action, opponent_action, events)
                if seat == 0
                else simulator(states, opponent_action, fc15_action, events)
            )
        host = {name: np.asarray(jax.device_get(value)) for name, value in saved.items()}
        host_found = np.asarray(jax.device_get(found))
        for offset, seed in enumerate(seeds.tolist()):
            row = {
                "seed": int(seed),
                "candidate_seat": int(seat),
                "divergence_found": bool(host_found[offset]),
            }
            for name, value in host.items():
                item = value[offset]
                row[name] = item.tolist() if item.ndim else item.item()
            records.append(row)

    payload = {
        "schema": "kaggriculture.fc15-b21-first-divergence-features.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "visibility": "own private state plus opponent public board only; no opponent shed, identity, action, or future event",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "records": records,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": len(records), "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
