#!/usr/bin/env python3
"""GPU screen for a close-mirror residual premium-sale capability."""

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
    k320_mass_hire_guard_residual_sale_player_action_v1,
)
from kaggriculture_jax.constants import PRODUCTS  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)


PREMIUM_MASK = sum(1 << PRODUCTS.index(item) for item in ("STRAWBERRY", "MILK", "WOOL"))
# name, enabled, start, distance, cap, floor percent, seat1 only
VARIANTS = (
    ("baseline", False, 120, 0, 1, 0, True),
    ("seat1_d0_c1_p0", True, 120, 0, 1, 0, True),
    ("seat1_d6_c1_p0", True, 120, 6, 1, 0, True),
    ("seat1_d6_c1_p50", True, 120, 6, 1, 50, True),
    ("seat1_d6_c1_p100", True, 120, 6, 1, 100, True),
    ("seat1_d6_c4_p50", True, 120, 6, 4, 50, True),
    ("both_d0_c1_p50", True, 120, 0, 1, 50, False),
    ("both_d6_c1_p50", True, 120, 6, 1, 50, False),
    ("both_d6_c1_p75", True, 120, 6, 1, 75, False),
    ("both_d6_c1_p100", True, 120, 6, 1, 100, False),
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

    old_receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(encoding="utf-8")
    )
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
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    enabled = jnp.repeat(jnp.asarray([row[1] for row in VARIANTS]), args.seeds)
    start = jnp.repeat(jnp.asarray([row[2] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    distance = jnp.repeat(jnp.asarray([row[3] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    cap = jnp.repeat(jnp.asarray([row[4] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    floor = jnp.repeat(jnp.asarray([row[5] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    seat1_only = jnp.repeat(jnp.asarray([row[6] for row in VARIANTS]), args.seeds)
    product_mask = jnp.full((expanded.size,), PREMIUM_MASK, dtype=jnp.int16)
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return k320_mass_hire_guard_residual_sale_player_action_v1(
                states,
                resources["tables"],
                resources["runtime"],
                resources["latest_bank"],
                carry,
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
    safety = {"hand_cap_hits": 0, "market_loop_cap_hits": 0, "price_lut_oob": 0}
    for seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded))
        candidate_carry = initialize_high_potential_v20_carry_v1(batch)
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        for _ in range(719):
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = simulator(states, candidate_action, opponent_action, events) if seat == 0 else simulator(states, opponent_action, candidate_action, events)
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        for name in safety:
            safety[name] += int(np.sum(np.asarray(getattr(terminal, name))))
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2))
    if any(safety.values()):
        raise AssertionError(f"simulator safety counter hit: {safety}")

    first, second = money
    rows = []
    margins_by_variant = []
    for index, variant in enumerate(VARIANTS):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival
        margins_by_variant.append(margins)
        rows.append({
            "variant": variant[0],
            "enabled": variant[1],
            "start_step": variant[2],
            "distance_limit": variant[3],
            "quantity_cap": variant[4],
            "minimum_price_percent": variant[5],
            "seat1_only": variant[6],
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "seat0_wins": int(np.sum((first[index, :, 0] - first[index, :, 1]) > 0)),
            "seat1_wins": int(np.sum((second[index, :, 1] - second[index, :, 0]) > 0)),
            "per_game": [
                {"seed": int(base_seeds[game % args.seeds]), "candidate_seat": 0 if game < args.seeds else 1, "margin": int(margin)}
                for game, margin in enumerate(margins.tolist())
            ],
        })
    matrix = np.stack(margins_by_variant, axis=0)
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.k320-residual-sale-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped",
        "product_mask": ["STRAWBERRY", "MILK", "WOOL"],
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "safety": safety,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": [{k: v for k, v in row.items() if k != "per_game"} for row in rows], "oracle": payload["oracle"], "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
