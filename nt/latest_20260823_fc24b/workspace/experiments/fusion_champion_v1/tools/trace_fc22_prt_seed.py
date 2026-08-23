#!/usr/bin/env python3
"""Step-aligned FC22 trace for one seed/seat against frozen local PRT."""

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
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc22_feed_value_guard_player_action_v1,
    initialize_fusion_champion_feed_value_carry_v1,
)
from kaggriculture_jax.constants import MarketOp, PRODUCTS, UnitOp  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def action_dict(action: dict[str, np.ndarray], step: int) -> dict:
    unit_count = int(action["unit_count"][step])
    market_count = int(action["market_count"][step])
    return {
        "units": [
            {
                "unit": unit,
                "op": UnitOp(int(action["unit_op"][step, unit])).name,
                "item": int(action["unit_item"][step, unit]),
                "amount": int(action["unit_amount"][step, unit]),
            }
            for unit in range(unit_count)
        ],
        "market": [
            {
                "slot": slot,
                "op": MarketOp(int(action["market_op"][step, slot])).name,
                "item": (
                    PRODUCTS[int(action["market_item"][step, slot])]
                    if 0 <= int(action["market_item"][step, slot]) < len(PRODUCTS)
                    else int(action["market_item"][step, slot])
                ),
                "amount": int(action["market_amount"][step, slot]),
            }
            for slot in range(market_count)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=594122)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    tables = load_tables()
    old_bank = load_bank(
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest_bank = load_bank(
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "latest_public8_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "latest_public8_runtime_tables_v1.npz"
    )
    old_receipt = json.loads(
        (
            ROOT
            / "experiments/expert_business_agent_v2/receipts/"
            "jax_full37_mixed_exact_proxy_bank_v1.json"
        ).read_text(encoding="utf-8")
    )
    router = build_router_arrays(old_receipt)
    resources = {
        "old_bank": old_bank,
        "latest_bank": latest_bank,
        "runtime": runtime,
        "tables": tables,
        "router": router,
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    opponent_id = names.index("local_prt_v6")
    player = args.candidate_seat
    rival = 1 - player
    seed_values = list(range(args.seed, args.seed + args.batch_size))
    weed, shops = build_events_v1(seed_values)
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    states = jax.vmap(reset)(jnp.asarray(seed_values, dtype=jnp.int32))
    candidate_carry = initialize_fusion_champion_feed_value_carry_v1(args.batch_size)
    opponent_carry = rr.initialize_agent_carry(opponent_id, args.batch_size, router)
    simulator = rr.make_simulator_step(tables)

    @jax.jit
    def candidate_policy(current, carry):
        return fc22_feed_value_guard_player_action_v1(
            current,
            tables,
            latest_bank,
            old_bank,
            runtime,
            carry,
            player,
        )

    opponent_policy = rr.make_agent_policy(opponent_id, rival, **resources)
    state_names = (
        "step",
        "money",
        "seeds",
        "shed",
        "market_inventory",
        "market_price",
        "unit_pos",
        "unit_active",
        "unit_inventory",
        "tile_kind",
        "tile_crop",
        "tile_animal",
        "tile_yield",
        "tile_flags",
        "tile_neglect",
        "tile_pending_care",
    )
    state_values = {name: [] for name in state_names}
    candidate_actions = {
        name: []
        for name in (
            "unit_op",
            "unit_item",
            "unit_amount",
            "unit_count",
            "market_op",
            "market_item",
            "market_amount",
            "market_count",
        )
    }
    route_values = []
    credit_values = []
    for _ in range(719):
        for name in state_names:
            value = getattr(states, name)
            state_values[name].append(value[:1])
        route_values.append(candidate_carry.base.base.k320.ray_route_id[:1])
        credit_values.append(candidate_carry.wheat_credit[:1])
        candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
        opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
        for name in candidate_actions:
            candidate_actions[name].append(getattr(candidate_action, name)[:1])
        states = (
            simulator(states, candidate_action, opponent_action, events)
            if player == 0
            else simulator(states, opponent_action, candidate_action, events)
        )
    jax.block_until_ready(states.money)
    terminal = jax.device_get(states)
    state_np = {
        name: np.asarray(jax.device_get(jnp.concatenate(values, axis=0)))
        for name, values in state_values.items()
    }
    action_np = {
        name: np.asarray(jax.device_get(jnp.concatenate(values, axis=0)))
        for name, values in candidate_actions.items()
    }
    route_np = np.asarray(jax.device_get(jnp.concatenate(route_values, axis=0)))
    credit_np = np.asarray(jax.device_get(jnp.concatenate(credit_values, axis=0)))

    rows = []
    for step in range(719):
        rows.append(
            {
                "step": step,
                "day": step // 24 + 1,
                "turn": step % 24 + 1,
                "money": state_np["money"][step].astype(int).tolist(),
                "market_inventory": state_np["market_inventory"][step].astype(int).tolist(),
                "market_price": state_np["market_price"][step].astype(int).tolist(),
                "own_seeds": state_np["seeds"][step, player].astype(int).tolist(),
                "own_shed": state_np["shed"][step, player].astype(int).tolist(),
                "own_unit_pos": state_np["unit_pos"][step, player].astype(int).tolist(),
                "own_unit_active": state_np["unit_active"][step, player].astype(bool).tolist(),
                "own_unit_inventory": state_np["unit_inventory"][step, player].astype(int).tolist(),
                "own_tile_kind": state_np["tile_kind"][step, player].astype(int).tolist(),
                "own_tile_crop": state_np["tile_crop"][step, player].astype(int).tolist(),
                "own_tile_animal": state_np["tile_animal"][step, player].astype(int).tolist(),
                "own_tile_yield": state_np["tile_yield"][step, player].astype(int).tolist(),
                "own_tile_flags": state_np["tile_flags"][step, player].astype(int).tolist(),
                "own_tile_neglect": state_np["tile_neglect"][step, player].astype(int).tolist(),
                "own_tile_pending_care": state_np["tile_pending_care"][step, player].astype(int).tolist(),
                "route_id": int(route_np[step]),
                "wheat_credit": int(credit_np[step]),
                "action": action_dict(action_np, step),
            }
        )

    payload = {
        "schema": "kaggriculture.fc22.prt_seed_trace.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "seed": args.seed,
        "candidate_seat": player,
        "terminal_money": np.asarray(terminal.money)[0].astype(int).tolist(),
        "terminal_margin": int(
            np.asarray(terminal.money)[0, player]
            - np.asarray(terminal.money)[0, rival]
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal_money": payload["terminal_money"],
                "terminal_margin": payload["terminal_margin"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
