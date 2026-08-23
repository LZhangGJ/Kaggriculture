#!/usr/bin/env python3
"""Paired GPU screen for FC15 X562/PRT residual-sale variants."""

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
    initialize_x562_old_suffix_carry_v1,
    x562_prt_suffix_weed_hire_guard_residual_sale_player_action_v1,
)
from kaggriculture_jax.constants import PRODUCTS  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def mask(*names: str) -> int:
    return sum(1 << PRODUCTS.index(name) for name in names)


PREMIUM = mask("STRAWBERRY", "MILK", "WOOL")
ALL_OUTPUT = mask(
    "WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY",
    "MILK", "WOOL", "EGG", "FERTILIZER",
)

# name, enabled, start, clone distance, cap, floor%, product mask, seat1 only
VARIANTS = (
    ("source_fc15", False, 120, 6, 1, 100, PREMIUM, False),
    ("premium_d0_s120_p100_c1", True, 120, 0, 1, 100, PREMIUM, False),
    ("premium_d3_s120_p100_c1", True, 120, 3, 1, 100, PREMIUM, False),
    ("premium_d6_s120_p100_c1", True, 120, 6, 1, 100, PREMIUM, False),
    ("premium_d6_s168_p100_c1", True, 168, 6, 1, 100, PREMIUM, False),
    ("strawberry_d6_s120_p100_c1", True, 120, 6, 1, 100, mask("STRAWBERRY"), False),
    ("milk_d6_s120_p100_c1", True, 120, 6, 1, 100, mask("MILK"), False),
    ("wool_d6_s120_p100_c1", True, 120, 6, 1, 100, mask("WOOL"), False),
    ("premium_fertilizer_d6_s120_p100_c1", True, 120, 6, 1, 100, PREMIUM | mask("FERTILIZER"), False),
    ("all_output_d6_s120_p100_c1", True, 120, 6, 1, 100, ALL_OUTPUT, False),
    ("premium_d6_s120_p125_c1", True, 120, 6, 1, 125, PREMIUM, False),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    old_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"),
        "latest_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"),
        "runtime": load_high_potential_runtime_tables_v1(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError(f"unknown opponent: {args.opponent}")
    opponent_id = names.index(args.opponent)

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(VARIANTS))
    repeat = lambda values, dtype=None: jnp.repeat(jnp.asarray(values, dtype=dtype), args.seeds)
    enabled = repeat([row[1] for row in VARIANTS])
    start = repeat([row[2] for row in VARIANTS], jnp.int16)
    distance = repeat([row[3] for row in VARIANTS], jnp.int16)
    cap = repeat([row[4] for row in VARIANTS], jnp.int16)
    floor = repeat([row[5] for row in VARIANTS], jnp.int16)
    product_mask = repeat([row[6] for row in VARIANTS], jnp.int16)
    seat1_only = repeat([row[7] for row in VARIANTS])
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(resources["tables"])

    def make_policy(player: int):
        @jax.jit
        def policy(states, carry):
            batch = states.step.shape[0]
            return x562_prt_suffix_weed_hire_guard_residual_sale_player_action_v1(
                states,
                resources["runtime"],
                resources["latest_bank"],
                resources["old_bank"],
                carry,
                jnp.ones((batch,), dtype=jnp.bool_),
                jnp.full((batch,), 4, dtype=jnp.int16),
                jnp.full((batch,), 2, dtype=jnp.int16),
                jnp.full((batch,), 5, dtype=jnp.int16),
                jnp.ones((batch,), dtype=jnp.int16),
                enabled,
                start,
                distance,
                cap,
                floor,
                product_mask,
                seat1_only,
                player,
            )
        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    money = []
    batch = expanded.size
    for seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded))
        carry = initialize_x562_old_suffix_carry_v1(batch)
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        policy = make_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        for _ in range(719):
            action, carry = policy(states, carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = simulator(states, action, opponent_action, events) if seat == 0 else simulator(states, opponent_action, action, events)
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        safety = {name: int(np.sum(np.asarray(getattr(terminal, name)))) for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")}
        if any(safety.values()):
            raise AssertionError(f"simulator safety counter hit: {safety}")
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2))

    rows, margins_all = [], []
    for index, variant in enumerate(VARIANTS):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margin = own - rival
        margins_all.append(margin)
        rows.append({
            "variant": variant[0],
            "sale_enabled": variant[1],
            "sale_start_step": variant[2],
            "clone_distance_limit": variant[3],
            "sale_quantity_cap": variant[4],
            "minimum_price_percent": variant[5],
            "product_mask": variant[6],
            "seat1_only": variant[7],
            "games": int(margin.size),
            "wins": int(np.sum(margin > 0)),
            "ties": int(np.sum(margin == 0)),
            "losses": int(np.sum(margin < 0)),
            "score_rate": float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margin)),
            "per_game": [
                {
                    "seed": int(base_seeds[i % args.seeds]),
                    "candidate_seat": 0 if i < args.seeds else 1,
                    "candidate_cash": int(own[i]),
                    "opponent_cash": int(rival[i]),
                    "margin": int(margin[i]),
                }
                for i in range(margin.size)
            ],
        })
    source = margins_all[0]
    paired = []
    for index, row in enumerate(rows[1:], start=1):
        candidate = margins_all[index]
        paired.append({
            "variant": row["variant"],
            "improved_games": int(np.sum(candidate > source)),
            "unchanged_games": int(np.sum(candidate == source)),
            "regressed_games": int(np.sum(candidate < source)),
            "source_losses_rescued": int(np.sum((source < 0) & (candidate > 0))),
            "source_wins_harmed": int(np.sum((source > 0) & (candidate < 0))),
            "mean_margin_delta": float(np.mean(candidate - source)),
        })
    matrix = np.stack(margins_all, axis=0)
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.x562-residual-sale-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "paired_delta": paired,
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
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "rows": [{k: v for k, v in row.items() if k != "per_game"} for row in rows],
        "paired_delta": paired,
        "oracle": payload["oracle"],
        "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
