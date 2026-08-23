#!/usr/bin/env python3
"""Step-aligned FC16/FC19 trace against the frozen Rank14 proxy."""

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
    fc16_moon_h4_player_action_v1,
    fc19_moon_h4_wheat8_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.constants import ANIMALS, CROPS, MarketOp, PRODUCTS, UnitOp  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def _stack(values):
    return np.asarray(jax.device_get(jnp.stack(values, axis=0)))


def _action_dict(action, index: int) -> dict:
    unit_count = int(action["unit_count"][index])
    market_count = int(action["market_count"][index])
    units = []
    for unit in range(unit_count):
        op = int(action["unit_op"][index, unit])
        units.append({
            "unit": unit,
            "op": UnitOp(op).name,
            "item": int(action["unit_item"][index, unit]),
            "amount": int(action["unit_amount"][index, unit]),
        })
    market = []
    for slot in range(market_count):
        op = int(action["market_op"][index, slot])
        item = int(action["market_item"][index, slot])
        market.append({
            "slot": slot,
            "op": MarketOp(op).name,
            "item": PRODUCTS[item] if 0 <= item < len(PRODUCTS) else item,
            "amount": int(action["market_amount"][index, slot]),
        })
    return {"units": units, "market": market}


def _action_equal(left: dict, right: dict, index: int) -> bool:
    return all(np.array_equal(left[name][index], right[name][index]) for name in left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=594001)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Compile/run this many consecutive seeds and trace the first one. "
        "The default intentionally reuses the established batch-64 cache.",
    )
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    tables = load_tables()
    old_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz")
    latest_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz")
    runtime = load_high_potential_runtime_tables_v1(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz")
    old_receipt = json.loads((ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(encoding="utf-8"))
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
    opponent_id = names.index("gold_proxy_rank14_recursion")
    player = args.candidate_seat
    rival = 1 - player
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    seed_values = list(range(args.seed, args.seed + args.batch_size))
    seeds = jnp.asarray(seed_values, dtype=jnp.int32)
    weed, shops = build_events_v1(seed_values)
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(tables)

    traces = {}
    for variant in ("fc16", "fc19_wheat8"):
        @jax.jit
        def candidate_policy(states, carry):
            if variant == "fc16":
                return fc16_moon_h4_player_action_v1(states, tables, latest_bank, old_bank, runtime, carry, player)
            return fc19_moon_h4_wheat8_player_action_v1(states, tables, latest_bank, old_bank, runtime, carry, player)

        opponent_policy = rr.make_agent_policy(opponent_id, rival, **resources)
        states = jax.vmap(reset)(seeds)
        candidate_carry = initialize_fusion_champion_moon_market_carry_v1(args.batch_size)
        opponent_carry = rr.initialize_agent_carry(opponent_id, args.batch_size, router)
        state_fields = {name: [] for name in (
            "money", "seeds", "shed", "market_inventory", "market_price",
            "hires_today", "unlocked_count", "active_units", "crop_counts",
            "animal_counts", "yield_sum", "unit_inventory_sum", "opponent_route",
            "unit_inventory_target", "unit_pos_target", "tile_crop_target",
            "tile_animal_target", "tile_yield_target", "tile_flags_target",
            "tile_pending_care_target",
        )}
        candidate_actions = {name: [] for name in (
            "unit_op", "unit_item", "unit_amount", "unit_count",
            "market_op", "market_item", "market_amount", "market_count",
        )}
        opponent_actions = {name: [] for name in candidate_actions}
        for _ in range(719):
            crop_counts = jnp.stack([
                jnp.sum(states.tile_crop == crop, axis=(2, 3)) for crop in range(len(CROPS))
            ], axis=2)
            animal_counts = jnp.stack([
                jnp.sum(states.tile_animal == animal, axis=(2, 3)) for animal in range(len(ANIMALS))
            ], axis=2)
            current = {
                "money": states.money,
                "seeds": states.seeds,
                "shed": states.shed,
                "market_inventory": states.market_inventory,
                "market_price": states.market_price,
                "hires_today": states.hires_today,
                "unlocked_count": states.unlocked_count,
                "active_units": jnp.sum(states.unit_active, axis=2),
                "crop_counts": crop_counts,
                "animal_counts": animal_counts,
                "yield_sum": jnp.sum(states.tile_yield, axis=(2, 3)),
                "unit_inventory_sum": jnp.sum(states.unit_inventory, axis=(2, 3)),
                "opponent_route": opponent_carry.skeleton_id,
                # Only the first seed is retained for large tensors; this
                # keeps batch-64 compilation/cache reuse without storing 64
                # complete boards for all 719 steps.
                "unit_inventory_target": states.unit_inventory[:1],
                "unit_pos_target": states.unit_pos[:1],
                "tile_crop_target": states.tile_crop[:1],
                "tile_animal_target": states.tile_animal[:1],
                "tile_yield_target": states.tile_yield[:1],
                "tile_flags_target": states.tile_flags[:1],
                "tile_pending_care_target": states.tile_pending_care[:1],
            }
            for name, value in current.items():
                state_fields[name].append(value)
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            for name in candidate_actions:
                candidate_actions[name].append(getattr(candidate_action, name))
                opponent_actions[name].append(getattr(opponent_action, name))
            states = simulator(states, candidate_action, opponent_action, events) if player == 0 else simulator(states, opponent_action, candidate_action, events)
        jax.block_until_ready(states.money)
        traces[variant] = {
            "state": {name: _stack(values)[:, 0] for name, values in state_fields.items()},
            "candidate_action": {name: _stack(values)[:, 0] for name, values in candidate_actions.items()},
            "opponent_action": {name: _stack(values)[:, 0] for name, values in opponent_actions.items()},
            "terminal_money": np.asarray(jax.device_get(states.money))[0],
        }

    base = traces["fc16"]
    trial = traces["fc19_wheat8"]
    candidate_diff = [step for step in range(719) if not _action_equal(base["candidate_action"], trial["candidate_action"], step)]
    opponent_diff = [step for step in range(719) if not _action_equal(base["opponent_action"], trial["opponent_action"], step)]
    state_diff = [step for step in range(719) if any(
        not np.array_equal(base["state"][name][step], trial["state"][name][step])
        for name in base["state"] if name != "opponent_route"
    )]
    field_first_diff = {
        name: next(
            (
                step
                for step in range(719)
                if not np.array_equal(base["state"][name][step], trial["state"][name][step])
            ),
            None,
        )
        for name in base["state"]
    }

    interesting = sorted(set(candidate_diff[:40] + opponent_diff[:40]))
    comparisons = []
    for step in interesting:
        comparisons.append({
            "step": step,
            "candidate_action_diff": step in candidate_diff,
            "opponent_action_diff": step in opponent_diff,
            "fc16_candidate_action": _action_dict(base["candidate_action"], step),
            "fc19_candidate_action": _action_dict(trial["candidate_action"], step),
            "fc16_opponent_action": _action_dict(base["opponent_action"], step),
            "fc19_opponent_action": _action_dict(trial["opponent_action"], step),
            "fc16_state": {
                "money": base["state"]["money"][step].tolist(),
                "seeds": base["state"]["seeds"][step].tolist(),
                "shed": base["state"]["shed"][step].tolist(),
                "market_inventory": base["state"]["market_inventory"][step].tolist(),
                "hires_today": base["state"]["hires_today"][step].tolist(),
                "crop_counts": base["state"]["crop_counts"][step].tolist(),
                "animal_counts": base["state"]["animal_counts"][step].tolist(),
                "yield_sum": base["state"]["yield_sum"][step].tolist(),
                "opponent_route": int(base["state"]["opponent_route"][step]),
            },
            "fc19_state": {
                "money": trial["state"]["money"][step].tolist(),
                "seeds": trial["state"]["seeds"][step].tolist(),
                "shed": trial["state"]["shed"][step].tolist(),
                "market_inventory": trial["state"]["market_inventory"][step].tolist(),
                "hires_today": trial["state"]["hires_today"][step].tolist(),
                "crop_counts": trial["state"]["crop_counts"][step].tolist(),
                "animal_counts": trial["state"]["animal_counts"][step].tolist(),
                "yield_sum": trial["state"]["yield_sum"][step].tolist(),
                "opponent_route": int(trial["state"]["opponent_route"][step]),
            },
        })

    # Raw actions may be equal while the official transaction fills differ
    # because cash or inventory has already diverged.  Record every step that
    # changes the between-variant cash delta so the first economic amplifier
    # is visible instead of being mistaken for an action-policy divergence.
    base_money = base["state"]["money"]
    trial_money = trial["state"]["money"]
    before_delta = trial_money - base_money
    terminal_delta = trial["terminal_money"] - base["terminal_money"]
    after_delta = np.concatenate((before_delta[1:], terminal_delta[None, :]), axis=0)
    economic_steps = np.where(np.any(after_delta != before_delta, axis=1))[0].tolist()
    economic_events = []
    for step in economic_steps:
        economic_events.append({
            "step": int(step),
            "cash_delta_before": before_delta[step].tolist(),
            "cash_delta_after": after_delta[step].tolist(),
            "fc16_money_before": base_money[step].tolist(),
            "fc19_money_before": trial_money[step].tolist(),
            "fc16_candidate_action": _action_dict(base["candidate_action"], step),
            "fc19_candidate_action": _action_dict(trial["candidate_action"], step),
            "fc16_opponent_action": _action_dict(base["opponent_action"], step),
            "fc19_opponent_action": _action_dict(trial["opponent_action"], step),
            "fc16_candidate_shed_before": base["state"]["shed"][step, player].tolist(),
            "fc19_candidate_shed_before": trial["state"]["shed"][step, player].tolist(),
            "fc16_candidate_seeds_before": base["state"]["seeds"][step, player].tolist(),
            "fc19_candidate_seeds_before": trial["state"]["seeds"][step, player].tolist(),
            "fc16_candidate_unit_inventory_before": base["state"]["unit_inventory_target"][step, player].tolist(),
            "fc19_candidate_unit_inventory_before": trial["state"]["unit_inventory_target"][step, player].tolist(),
            "fc16_market_inventory_before": base["state"]["market_inventory"][step].tolist(),
            "fc19_market_inventory_before": trial["state"]["market_inventory"][step].tolist(),
            "fc16_market_price_before": base["state"]["market_price"][step].tolist(),
            "fc19_market_price_before": trial["state"]["market_price"][step].tolist(),
        })

    diagnostic_window = []
    for step in range(168, 201):
        diagnostic_window.append({
            "step": step,
            "fc16_money": base["state"]["money"][step].tolist(),
            "fc19_money": trial["state"]["money"][step].tolist(),
            "fc16_candidate_seeds": base["state"]["seeds"][step, player].tolist(),
            "fc19_candidate_seeds": trial["state"]["seeds"][step, player].tolist(),
            "fc16_candidate_shed": base["state"]["shed"][step, player].tolist(),
            "fc19_candidate_shed": trial["state"]["shed"][step, player].tolist(),
            "fc16_candidate_unit_inventory": base["state"]["unit_inventory_target"][step, player].tolist(),
            "fc19_candidate_unit_inventory": trial["state"]["unit_inventory_target"][step, player].tolist(),
            "fc16_candidate_unit_pos": base["state"]["unit_pos_target"][step, player].tolist(),
            "fc19_candidate_unit_pos": trial["state"]["unit_pos_target"][step, player].tolist(),
            "fc16_candidate_action": _action_dict(base["candidate_action"], step),
            "fc19_candidate_action": _action_dict(trial["candidate_action"], step),
        })

    payload = {
        "schema": "kaggriculture.fc16-fc19-rank14-step-trace.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "candidate_seat": player,
        "terminal": {
            "fc16": base["terminal_money"].tolist(),
            "fc19": trial["terminal_money"].tolist(),
        },
        "first_state_diff": state_diff[0] if state_diff else None,
        "first_candidate_action_diff": candidate_diff[0] if candidate_diff else None,
        "first_candidate_action_diff_after_opening": next((step for step in candidate_diff if step > 0), None),
        "first_opponent_action_diff": opponent_diff[0] if opponent_diff else None,
        "state_field_first_diff": field_first_diff,
        "candidate_action_diff_steps": candidate_diff,
        "opponent_action_diff_steps": opponent_diff,
        "economic_delta_change_steps": economic_steps,
        "economic_events": economic_events,
        "diagnostic_window_168_200": diagnostic_window,
        "comparisons": comparisons,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "terminal": payload["terminal"],
        "first_candidate_action_diff_after_opening": payload["first_candidate_action_diff_after_opening"],
        "first_opponent_action_diff": payload["first_opponent_action_diff"],
        "candidate_action_diff_count": len(candidate_diff),
        "opponent_action_diff_count": len(opponent_diff),
        "output": str(output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
