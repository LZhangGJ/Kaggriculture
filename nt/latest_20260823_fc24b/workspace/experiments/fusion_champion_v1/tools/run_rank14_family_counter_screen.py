#!/usr/bin/env python3
"""GPU screen for a public-morphology-conditioned heavy-sheep market counter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    OpponentFamilyMarketScheduleV1,
    initialize_k320_heavy_sheep_counter_carry_v1,
    initialize_x562_family_counter_carry_v1,
    k320_heavy_sheep_counter_player_action_v1,
    x562_family_counter_player_action_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


# name, enabled, lead, multiplier, cap, product_group, price_percent, clean_town
# product groups: 0 milk, 1 wool, 2 milk+wool, 3 strawberry+milk+wool, 4 all.
VARIANTS = (
    ("baseline", False, 1, 1, 30, 4, 25, False),
    ("lead1_milk_wool", True, 1, 1, 30, 2, 25, False),
    ("lead2_milk_wool", True, 2, 1, 30, 2, 25, False),
    ("lead3_milk_wool", True, 3, 1, 30, 2, 25, False),
    ("lead1_premium3", True, 1, 1, 30, 3, 25, False),
    ("lead2_premium3", True, 2, 1, 30, 3, 25, False),
    ("lead1_all", True, 1, 1, 30, 4, 25, False),
    ("lead2_all", True, 2, 1, 30, 4, 25, False),
    ("lead1_milk", True, 1, 1, 30, 0, 25, False),
    ("lead1_wool", True, 1, 1, 30, 1, 25, False),
    ("lead1_milk_wool_x2", True, 1, 2, 30, 2, 25, False),
    ("lead2_milk_wool_x2", True, 2, 2, 30, 2, 25, False),
    ("lead1_milk_wool_clean", True, 1, 1, 30, 2, 25, True),
    ("lead2_milk_wool_clean", True, 2, 1, 30, 2, 25, True),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=540301)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponent", default="gold_proxy_rank14_recursion")
    parser.add_argument("--candidate-base", choices=("x562", "k320"), default="x562")
    parser.add_argument("--schedule-route-a", type=int, default=63)
    parser.add_argument("--schedule-route-b", type=int, default=64)
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant indices; keep a shard <= 8 variants.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected_indices = (
        [int(value) for value in args.variants.split(",") if value]
        if args.variants
        else list(range(len(VARIANTS)))
    )
    if not selected_indices or any(index < 0 or index >= len(VARIANTS) for index in selected_indices):
        raise ValueError("invalid --variants")
    variants = [VARIANTS[index] for index in selected_indices]

    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(old_bank_path)
    resources = {
        "old_bank": old_bank,
        "latest_bank": load_bank(latest_bank_path),
        "runtime": load_high_potential_runtime_tables_v1(runtime_path),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    # Prototype-only extraction: the policy receives generic family schedules,
    # not route IDs and never sees the opponent identity at runtime.
    if not (
        0 <= args.schedule_route_a < old_bank.market_op.shape[0]
        and 0 <= args.schedule_route_b < old_bank.market_op.shape[0]
    ):
        raise ValueError("schedule route out of range")
    schedule = OpponentFamilyMarketScheduleV1(
        cow_heavy_op=old_bank.market_op[args.schedule_route_a],
        cow_heavy_item=old_bank.market_item[args.schedule_route_a],
        cow_heavy_amount=old_bank.market_amount[args.schedule_route_a],
        cow_heavy_count=old_bank.market_count[args.schedule_route_a],
        balanced_op=old_bank.market_op[args.schedule_route_b],
        balanced_item=old_bank.market_item[args.schedule_route_b],
        balanced_amount=old_bank.market_amount[args.schedule_route_b],
        balanced_count=old_bank.market_count[args.schedule_route_b],
    )
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError("unknown opponent")
    opponent_id = names.index(args.opponent)

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded_seeds = np.tile(base_seeds, len(variants))
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    enabled = jnp.repeat(jnp.asarray([row[1] for row in variants]), args.seeds)
    lead = jnp.repeat(jnp.asarray([row[2] for row in variants], dtype=jnp.int16), args.seeds)
    multiplier = jnp.repeat(jnp.asarray([row[3] for row in variants], dtype=jnp.int16), args.seeds)
    cap = jnp.repeat(jnp.asarray([row[4] for row in variants], dtype=jnp.int16), args.seeds)
    group = jnp.repeat(jnp.asarray([row[5] for row in variants], dtype=jnp.int8), args.seeds)
    price = jnp.repeat(jnp.asarray([row[6] for row in variants], dtype=jnp.int16), args.seeds)
    clean = jnp.repeat(jnp.asarray([row[7] for row in variants]), args.seeds)
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            if args.candidate_base == "k320":
                return k320_heavy_sheep_counter_player_action_v1(
                    states,
                    resources["tables"],
                    resources["runtime"],
                    resources["latest_bank"],
                    schedule,
                    carry,
                    enabled,
                    lead,
                    multiplier,
                    cap,
                    group,
                    price,
                    clean,
                    player,
                )
            return x562_family_counter_player_action_v1(
                states,
                resources["runtime"],
                resources["latest_bank"],
                schedule,
                carry,
                enabled,
                lead,
                multiplier,
                cap,
                group,
                price,
                clean,
                player,
            )
        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    batch = len(expanded_seeds)
    orientation_money = []
    for candidate_seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = (
            initialize_k320_heavy_sheep_counter_carry_v1(batch)
            if args.candidate_base == "k320"
            else initialize_x562_family_counter_carry_v1(batch)
        )
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(candidate_seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - candidate_seat, **resources)
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
            raise AssertionError("not all games DONE")
        if (
            int(np.sum(np.asarray(terminal.hand_cap_hits)))
            or int(np.sum(np.asarray(terminal.market_loop_cap_hits)))
            or int(np.sum(np.asarray(terminal.price_lut_oob)))
        ):
            raise AssertionError("simulator safety counter hit")
        orientation_money.append(
            np.asarray(terminal.money, dtype=np.int64).reshape(len(variants), args.seeds, 2)
        )

    first, second = orientation_money
    rows = []
    margin_rows = []
    for index, variant in enumerate(variants):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival
        margin_rows.append(margins)
        rows.append({
            "variant_index": selected_indices[index],
            "name": variant[0],
            "enabled": variant[1],
            "lead_steps": variant[2],
            "quantity_multiplier": variant[3],
            "quantity_cap": variant[4],
            "product_group": variant[5],
            "minimum_price_percent": variant[6],
            "require_clean_town": variant[7],
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "per_game": [
                {
                    "seed": int(base_seeds[game % args.seeds]),
                    "candidate_seat": 0 if game < args.seeds else 1,
                    "margin": int(margin),
                }
                for game, margin in enumerate(margins.tolist())
            ],
        })
    matrix = np.stack(margin_rows, axis=0)
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.rank14_family_counter_screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped",
        "online_features": "public opponent animal counts, public shops, market quote, own inventory",
        "candidate_base": args.candidate_base,
        "prototype_schedule_sources": [
            args.schedule_route_a,
            args.schedule_route_b,
        ],
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "rows": [{key: value for key, value in row.items() if key != "per_game"} for row in rows],
        "oracle": payload["oracle"],
        "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
