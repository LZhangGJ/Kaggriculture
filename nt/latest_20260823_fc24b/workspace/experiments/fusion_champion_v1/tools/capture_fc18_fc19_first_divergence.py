#!/usr/bin/env python3
"""Capture public state at FC18/FC19's first action divergence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

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
    fc19_moon_h4_wheat8_player_action_v1,
    initialize_fusion_champion_carry_v3,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MarketOp,
    TileKind,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    load_bank,
)
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    _action_equal,
    initialize_moon_v92_carry_v1,
    prvsiyan_moon_v92_player_action_v1,
)
from strategic_v5 import latest_public8_gpu as lp  # noqa: E402


OPPONENTS = ("prvsiyan_moon_v92_latest", "gold_proxy_rank14_recursion")


def _counts(board, size: int):
    return jnp.stack(
        [jnp.sum(board == value, axis=(1, 2)) for value in range(size)], axis=1
    ).astype(jnp.int16)


def _market_totals(action, op: MarketOp):
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    return jnp.stack(
        [
            jnp.sum(
                jnp.where(
                    active
                    & (action.market_op == op)
                    & (action.market_item == item),
                    jnp.maximum(action.market_amount, 0),
                    0,
                ),
                axis=1,
            )
            for item in range(12)
        ],
        axis=1,
    ).astype(jnp.int16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=860001)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not selected or len(selected) != len(set(selected)) or any(value not in OPPONENTS for value in selected):
        raise ValueError("invalid opponents")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    tables = load_tables()
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
    old_receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(encoding="utf-8")
    )
    router = build_router_arrays(old_receipt)
    resources = {
        "old_bank": old_bank,
        "latest_bank": latest8_bank,
        "runtime": runtime,
        "tables": tables,
        "router": router,
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    ids = {row["name"]: index for index, row in enumerate(rr.ROSTER)}
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(seeds.size)
    simulator = rr.make_simulator_step(tables)

    records = []
    for opponent_name in selected:
        for candidate_seat in (0, 1):
            player = candidate_seat
            rival = 1 - player

            @jax.jit
            def fc18_policy(states, carry):
                shape = states.step.shape
                return fc15_opening_transaction_player_action_v1(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    carry,
                    jnp.full(shape, 8, dtype=jnp.int16),
                    jnp.full(shape, 12, dtype=jnp.int16),
                    jnp.full(shape, 6, dtype=jnp.int16),
                    player,
                )

            @jax.jit
            def fc19_policy(states, carry):
                return fc19_moon_h4_wheat8_player_action_v1(
                    states, tables, latest8_bank, old_bank, runtime, carry, player
                )

            if opponent_name == "prvsiyan_moon_v92_latest":
                opponent_carry = initialize_moon_v92_carry_v1(batch)

                @jax.jit
                def opponent_policy(states, carry):
                    return prvsiyan_moon_v92_player_action_v1(
                        states, runtime, latest6_bank, carry, rival
                    )
            else:
                opponent_id = ids[opponent_name]
                opponent_carry = rr.initialize_agent_carry(opponent_id, batch, router)
                opponent_policy = rr.make_agent_policy(
                    opponent_id, rival, **resources
                )

            states = jax.vmap(reset)(jnp.asarray(seeds))
            fc18_carry = initialize_fusion_champion_carry_v3(batch)
            fc19_carry = initialize_fusion_champion_moon_market_carry_v1(batch)
            found = jnp.zeros((batch,), dtype=jnp.bool_)
            saved = {
                "step": jnp.full((batch,), -1, dtype=jnp.int16),
                "money": jnp.zeros((batch, 2), dtype=jnp.int32),
                "active_units": jnp.zeros((batch, 2), dtype=jnp.int16),
                "unlocked": jnp.zeros((batch, 2), dtype=jnp.int16),
                "crops_own": jnp.zeros((batch, len(CROPS)), dtype=jnp.int16),
                "crops_rival": jnp.zeros((batch, len(CROPS)), dtype=jnp.int16),
                "animals_own": jnp.zeros((batch, len(ANIMALS)), dtype=jnp.int16),
                "animals_rival": jnp.zeros((batch, len(ANIMALS)), dtype=jnp.int16),
                "yield_own": jnp.zeros((batch,), dtype=jnp.int32),
                "yield_rival": jnp.zeros((batch,), dtype=jnp.int32),
                "market_inventory": jnp.zeros((batch, 9), dtype=jnp.int32),
                "market_price": jnp.zeros((batch, 9), dtype=jnp.int32),
                "town_count": jnp.zeros((batch,), dtype=jnp.int8),
                "town_shops": jnp.full((batch, 8), -1, dtype=jnp.int8),
                "clone_distance": jnp.zeros((batch,), dtype=jnp.int16),
                "fc18_route": jnp.full((batch,), -1, dtype=jnp.int8),
                "fc18_sell": jnp.zeros((batch, 12), dtype=jnp.int16),
                "fc19_sell": jnp.zeros((batch, 12), dtype=jnp.int16),
                "fc18_buy": jnp.zeros((batch, 12), dtype=jnp.int16),
                "fc19_buy": jnp.zeros((batch, 12), dtype=jnp.int16),
            }
            for _ in range(719):
                fc18_action, fc18_carry = fc18_policy(states, fc18_carry)
                fc19_action, fc19_carry = fc19_policy(states, fc19_carry)
                opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
                first = (~found) & (~_action_equal(fc18_action, fc19_action))
                kinds = states.tile_kind
                current = {
                    "step": states.step.astype(jnp.int16),
                    "money": states.money.astype(jnp.int32),
                    "active_units": jnp.sum(states.unit_active, axis=2).astype(jnp.int16),
                    "unlocked": jnp.sum(kinds != int(TileKind.LOCKED), axis=(2, 3)).astype(jnp.int16),
                    "crops_own": _counts(states.tile_crop[:, player], len(CROPS)),
                    "crops_rival": _counts(states.tile_crop[:, rival], len(CROPS)),
                    "animals_own": _counts(states.tile_animal[:, player], len(ANIMALS)),
                    "animals_rival": _counts(states.tile_animal[:, rival], len(ANIMALS)),
                    "yield_own": jnp.sum(states.tile_yield[:, player], axis=(1, 2)).astype(jnp.int32),
                    "yield_rival": jnp.sum(states.tile_yield[:, rival], axis=(1, 2)).astype(jnp.int32),
                    "market_inventory": states.market_inventory[:, :9].astype(jnp.int32),
                    "market_price": states.market_price[:, :9].astype(jnp.int32),
                    "town_count": states.town_count.astype(jnp.int8),
                    "town_shops": states.town_shops.astype(jnp.int8),
                    "clone_distance": lp._kaito_clone_distance(states).astype(jnp.int16),
                    "fc18_route": fc18_carry.k320.ray_route_id.astype(jnp.int8),
                    "fc18_sell": _market_totals(fc18_action, MarketOp.SELL),
                    "fc19_sell": _market_totals(fc19_action, MarketOp.SELL),
                    "fc18_buy": _market_totals(fc18_action, MarketOp.BUY_PRODUCT),
                    "fc19_buy": _market_totals(fc19_action, MarketOp.BUY_PRODUCT),
                }
                for name, value in current.items():
                    choose = first.reshape((batch,) + (1,) * (value.ndim - 1))
                    saved[name] = jnp.where(choose, value, saved[name])
                found = found | first
                states = (
                    simulator(states, fc18_action, opponent_action, events)
                    if player == 0
                    else simulator(states, opponent_action, fc18_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            host = {name: np.asarray(jax.device_get(value)) for name, value in saved.items()}
            host_found = np.asarray(jax.device_get(found))
            for index, seed in enumerate(seeds.tolist()):
                row = {
                    "opponent_offline_label": opponent_name,
                    "seed": int(seed),
                    "candidate_seat": int(player),
                    "divergence_found": bool(host_found[index]),
                    "fc18_terminal_margin": int(
                        terminal.money[index, player] - terminal.money[index, rival]
                    ),
                }
                for name, value in host.items():
                    item = value[index]
                    row[name] = item.tolist() if item.ndim else item.item()
                records.append(row)

    payload = {
        "schema": "kaggriculture.fc18-fc19-first-divergence-public-features.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "visibility": "policy gate candidates are public state only; opponent label is offline diagnostic metadata",
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
