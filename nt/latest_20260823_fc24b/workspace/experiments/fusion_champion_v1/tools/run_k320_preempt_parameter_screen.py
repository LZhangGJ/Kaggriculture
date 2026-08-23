#!/usr/bin/env python3
"""Screen rule-first K320 premium-sale timing variants against a mirror."""

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
from fusion_champion_v1.policy_gpu import k320_preempt_parameter_player_action_v1  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)


# name, source, horizon, distance, cap, minimum future quantity,
# minimum price percent, start step, product bitmask.
VARIANTS = [
    ("source", 1, 0, 0, 0, 0, 0, 0, 0),
    ("disabled", 0, 0, 100, 0, 4, 0, 120, 0),
]
ALL_PREMIUM_MASK = sum(1 << product for product in (3, 4, 6, 7))
for distance in (6, 12, 100):
    for horizon in (1, 2, 3, 4):
        VARIANTS.append(
            (f"h{horizon}_d{distance}_c12_q4", 0, horizon, distance, 12, 4, 0, 120, ALL_PREMIUM_MASK)
        )
for cap in (4, 8, 16, 24):
    VARIANTS.append((f"h2_d6_c{cap}_q4", 0, 2, 6, cap, 4, 0, 120, ALL_PREMIUM_MASK))
for minimum in (1, 2, 6, 8):
    VARIANTS.append((f"h2_d6_c12_q{minimum}", 0, 2, 6, 12, minimum, 0, 120, ALL_PREMIUM_MASK))
for price in (50, 75, 100, 125):
    VARIANTS.append((f"h2_d6_c12_q4_p{price}", 0, 2, 6, 12, 4, price, 120, ALL_PREMIUM_MASK))
for start in (72, 96, 144):
    VARIANTS.append((f"h2_d6_c12_q4_s{start}", 0, 2, 6, 12, 4, 0, start, ALL_PREMIUM_MASK))

PRODUCT_LABELS = ((3, "S"), (4, "M"), (6, "K"), (7, "W"))
for horizon in (3, 4):
    for compact_mask in range(1, 16):
        mask = sum(
            1 << product
            for bit, (product, _) in enumerate(PRODUCT_LABELS)
            if compact_mask & (1 << bit)
        )
        label = "".join(
            name for bit, (_, name) in enumerate(PRODUCT_LABELS)
            if compact_mask & (1 << bit)
        )
        VARIANTS.append(
            (f"h{horizon}_mask_{label}", 0, horizon, 6, 12, 4, 0, 120, mask)
        )

SKW_MASK = (1 << 3) | (1 << 6) | (1 << 7)
WOOL_MASK = 1 << 7
for cap in (2, 4, 6, 8, 10, 16, 20, 24, 32, 40, 48, 64, 100):
    VARIANTS.append((f"h4_skw_cap{cap}", 0, 4, 6, cap, 4, 0, 120, SKW_MASK))
for minimum in (1, 2, 3, 5, 6, 8, 10):
    VARIANTS.append((f"h4_skw_q{minimum}", 0, 4, 6, 12, minimum, 0, 120, SKW_MASK))
for start in (72, 96, 144, 168, 192):
    VARIANTS.append((f"h4_skw_start{start}", 0, 4, 6, 12, 4, 0, start, SKW_MASK))
for price in (25, 50, 75, 100, 125, 150):
    VARIANTS.append((f"h4_skw_price{price}", 0, 4, 6, 12, 4, price, 120, SKW_MASK))
for distance in (2, 4, 8, 12, 20, 100):
    VARIANTS.append((f"h4_skw_distance{distance}", 0, 4, distance, 12, 4, 0, 120, SKW_MASK))
for cap in (20, 32, 64):
    VARIANTS.extend(
        (
            (f"h4_all_d100_cap{cap}", 0, 4, 100, cap, 4, 0, 120, ALL_PREMIUM_MASK),
            (f"h4_skw_d100_cap{cap}", 0, 4, 100, cap, 4, 0, 120, SKW_MASK),
            (f"h4_wool_d100_cap{cap}", 0, 4, 100, cap, 4, 0, 120, WOOL_MASK),
        )
    )
VARIANTS = tuple(VARIANTS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--opponent", default="rayk_k320_adaptive_rank1")
    parser.add_argument("--variants", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    requested = set(value for value in args.variants.split(",") if value)
    variants = tuple(row for row in VARIANTS if not requested or row[0] in requested)
    if not variants or (requested and len(variants) != len(requested)):
        raise ValueError("invalid --variants")

    old_receipt = json.loads((
        ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    ).read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"),
        "latest_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError("unknown opponent")
    opponent_id = names.index(args.opponent)

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(variants))
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    arrays = [
        jnp.repeat(jnp.asarray([row[column] for row in variants], dtype=dtype), args.seeds)
        for column, dtype in (
            (1, jnp.bool_),
            (2, jnp.int16),
            (3, jnp.int16),
            (4, jnp.int16),
            (5, jnp.int16),
            (6, jnp.int16),
            (7, jnp.int16),
            (8, jnp.int16),
        )
    ]
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry, *parameters):
            return k320_preempt_parameter_player_action_v1(
                states,
                resources["tables"],
                resources["runtime"],
                resources["latest_bank"],
                carry,
                *parameters,
                player,
            )
        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    money = []
    batch = len(expanded)
    for seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded))
        candidate_carry = initialize_high_potential_v20_carry_v1(batch)
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        for _ in range(719):
            candidate_action, candidate_carry = candidate_policy(
                states, candidate_carry, *arrays
            )
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        if any(int(np.sum(np.asarray(getattr(terminal, name)))) for name in (
            "hand_cap_hits", "market_loop_cap_hits", "price_lut_oob"
        )):
            raise AssertionError("simulator safety counter hit")
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(variants), args.seeds, 2))

    rows, margins_all = [], []
    for index, variant in enumerate(variants):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margins = own - rival
        margins_all.append(margins)
        rows.append({
            "variant": variant[0],
            "parameters": {
                "source_preempt": bool(variant[1]), "horizon": variant[2],
                "distance_limit": variant[3], "quantity_cap": variant[4],
                "minimum_future_quantity": variant[5],
                "minimum_price_percent": variant[6], "start_step": variant[7],
                "product_mask": variant[8],
            },
            "games": int(len(margins)), "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)), "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)), "mean_margin": float(np.mean(margins)),
            "per_game": [
                {"seed": int(base_seeds[i % args.seeds]), "candidate_seat": 0 if i < args.seeds else 1, "margin": int(value)}
                for i, value in enumerate(margins.tolist())
            ],
        })
    oracle = np.max(np.stack(margins_all), axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.k320_preempt_parameter_screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "backend": jax.default_backend(),
        "official_package_version": "1.32.7", "opponent": args.opponent,
        "seed_start": args.seed_start, "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped", "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)), "ties": int(np.sum(oracle == 0)),
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
        "best": sorted(
            [{k: v for k, v in row.items() if k != "per_game"} for row in rows],
            key=lambda row: (row["score_rate"], row["mean_margin"]), reverse=True,
        )[:8],
        "oracle": payload["oracle"], "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
