#!/usr/bin/env python3
"""Counterfactual screen for handing FC16 control to Boatlee V21 online.

Both controllers are advanced from reset on the same *actual* public state.
The selected arm therefore changes only the action owner after a public
decision boundary; it never initializes a suffix from hindsight state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import NamedTuple

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
    fc15_moon_parameter_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    BoatleeV21CarryV1,
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
)
import strategic_v5.latest_public8_gpu as lp  # noqa: E402


# name, switch step, maximum public farm distance.  99 means unconditional.
VARIANTS = (
    ("fc16_control", 999, 99),
    ("b21_s120_all", 120, 99),
    ("b21_s168_all", 168, 99),
    ("b21_s216_all", 216, 99),
    ("b21_s120_d2", 120, 2),
    ("b21_s120_d4", 120, 4),
    ("b21_s168_d2", 168, 2),
    ("b21_s168_d4", 168, 4),
)


class HybridCarry(NamedTuple):
    fc16: object
    b21: BoatleeV21CarryV1


def choose(mask, left, right):
    return type(left)(
        *(
            jnp.where(mask.reshape((mask.shape[0],) + (1,) * (lhs.ndim - 1)), lhs, rhs)
            for lhs, rhs in zip(left, right, strict=True)
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    tables = load_tables()
    old_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz")
    latest8_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz")
    latest6_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz")
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = rr.make_simulator_step(tables)

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(VARIANTS))
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    switch_step = jnp.repeat(jnp.asarray([row[1] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    max_distance = jnp.repeat(jnp.asarray([row[2] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    batch = len(expanded)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    def make_candidate(player: int):
        @jax.jit
        def policy(states, carry):
            shape = states.step.shape
            fc16_action, fc16 = fc15_moon_parameter_player_action_v1(
                states,
                tables,
                latest8_bank,
                old_bank,
                runtime,
                carry.fc16,
                jnp.full(shape, 4, dtype=jnp.int16),
                jnp.full(shape, 120, dtype=jnp.int16),
                jnp.full(shape, 2, dtype=jnp.int16),
                jnp.full(shape, 6, dtype=jnp.int16),
                player,
            )
            b21_action, b21 = boatlee_v21_player_action_v1(
                states, runtime, latest6_bank, carry.b21, player
            )
            distance = lp._kaito_clone_distance(states)
            use_b21 = (
                (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
                & (distance <= max_distance.astype(jnp.int32))
            )
            return choose(use_b21, b21_action, fc16_action), HybridCarry(fc16, b21)

        return policy

    orientation_money = []
    started = perf_counter()
    for seat in (0, 1):
        rival = 1 - seat

        @jax.jit
        def opponent(states, carry):
            return boatlee_v21_player_action_v1(states, runtime, latest6_bank, carry, rival)

        states = jax.vmap(reset)(jnp.asarray(expanded))
        candidate_carry = HybridCarry(
            initialize_fusion_champion_moon_market_carry_v1(batch),
            initialize_boatlee_v21_carry_v1(batch),
        )
        opponent_carry = initialize_boatlee_v21_carry_v1(batch)
        candidate = make_candidate(seat)
        for _ in range(719):
            own_action, candidate_carry = candidate(states, candidate_carry)
            rival_action, opponent_carry = opponent(states, opponent_carry)
            states = (
                simulator(states, own_action, rival_action, events)
                if seat == 0
                else simulator(states, rival_action, own_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("unfinished game")
        if any(int(np.sum(np.asarray(x))) for x in (terminal.hand_cap_hits, terminal.market_loop_cap_hits, terminal.price_lut_oob)):
            raise AssertionError("simulator safety counter")
        orientation_money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2))

    first, second = orientation_money
    rows = []
    margins_all = []
    for index, (name, step, distance) in enumerate(VARIANTS):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival
        margins_all.append(margins)
        rows.append({
            "variant": name,
            "switch_step": step,
            "max_public_farm_distance": distance,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_margin": float(np.mean(margins)),
            "per_game": [
                {"seed": int(base_seeds[i % args.seeds]), "candidate_seat": 0 if i < args.seeds else 1, "margin": int(m)}
                for i, m in enumerate(margins.tolist())
            ],
        })
    oracle = np.max(np.stack(margins_all), axis=0)
    payload = {
        "schema": "kaggriculture.fc16-b21-policy-switch-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "games_per_variant": args.seeds * 2,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "elapsed_seconds": perf_counter() - started,
        "truth_boundary": "Only current public state and step gates are used; the oracle is diagnostic only.",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = [{k: v for k, v in row.items() if k != "per_game"} for row in rows]
    print(json.dumps({"status": "PASS", "rows": compact, "oracle": payload["oracle"], "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
