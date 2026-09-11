#!/usr/bin/env python3
"""Screen a later public heavy-sheep trigger for FC2B's shadow recovery plan."""

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
    fc11e_fc2b_late_sheep_pressure_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


# Source plus a hypothesis-driven grid.  The thresholds are public morphology,
# not opponent IDs: five sheep is an early warning, seven is a committed sheep
# business, and nine is a high-volume business.  Start steps span the period in
# which PRT's public structure becomes visible while leaving time to recover.
VARIANTS = (
    ("source", False, 32767, 127, -1),
    ("s96_s5_c4", True, 96, 5, 4),
    ("s120_s5_c4", True, 120, 5, 4),
    ("s144_s5_c4", True, 144, 5, 4),
    ("s120_s7_c6", True, 120, 7, 6),
    ("s144_s7_c6", True, 144, 7, 6),
    ("s168_s7_c6", True, 168, 7, 6),
    ("s192_s7_c6", True, 192, 7, 6),
    ("s144_s9_c6", True, 144, 9, 6),
    ("s192_s9_c6", True, 192, 9, 6),
    ("s144_s7_c8", True, 144, 7, 8),
    ("s192_s7_c8", True, 192, 7, 8),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed-batches", type=int, default=1)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--variants", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1 or args.seed_batches < 1:
        raise ValueError("seed counts must be positive")

    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    variants = tuple(row for row in VARIANTS if not requested or row[0] in requested)
    if not variants or (requested and len(variants) != len(requested)):
        raise ValueError("invalid variants")

    old_receipt = json.loads(
        (
            ROOT
            / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
        ).read_text(encoding="utf-8")
    )
    resources = {
        "old_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
        ),
        "latest_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
        ),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError(f"unknown opponent: {args.opponent}")
    opponent_id = names.index(args.opponent)

    segment_seeds = [
        np.arange(
            args.seed_start + batch_index * args.seeds,
            args.seed_start + (batch_index + 1) * args.seeds,
            dtype=np.int32,
        )
        for batch_index in range(args.seed_batches)
    ]
    base_seeds = np.concatenate(segment_seeds)
    enabled = jnp.repeat(
        jnp.asarray([row[1] for row in variants], dtype=jnp.bool_), args.seeds
    )
    start_step = jnp.repeat(
        jnp.asarray([row[2] for row in variants], dtype=jnp.int16), args.seeds
    )
    minimum_sheep = jnp.repeat(
        jnp.asarray([row[3] for row in variants], dtype=jnp.int8), args.seeds
    )
    maximum_cows = jnp.repeat(
        jnp.asarray([row[4] for row in variants], dtype=jnp.int8), args.seeds
    )
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc11e_fc2b_late_sheep_pressure_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["old_bank"],
                resources["runtime"],
                carry,
                enabled,
                start_step,
                minimum_sheep,
                maximum_cows,
                player,
            )

        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    orientation_money: list[list[np.ndarray]] = [[], []]
    orientation_pressure: list[list[np.ndarray]] = [[], []]
    for segment in segment_seeds:
        expanded = np.tile(segment, len(variants))
        weed, shops = build_events_v1(expanded.tolist())
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        batch = len(expanded)
        for seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded))
            candidate_carry = initialize_fusion_champion_carry_v3(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            candidate_policy = make_candidate_policy(seat)
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - seat, **resources
            )
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if seat == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            if any(
                int(np.sum(np.asarray(getattr(terminal, name))))
                for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            ):
                raise AssertionError("simulator safety counter hit")
            orientation_money[seat].append(
                np.asarray(terminal.money, dtype=np.int64).reshape(
                    len(variants), args.seeds, 2
                )
            )
            orientation_pressure[seat].append(
                np.asarray(jax.device_get(candidate_carry.sheep_pressure), dtype=bool).reshape(
                    len(variants), args.seeds
                )
            )

    money = [np.concatenate(values, axis=1) for values in orientation_money]
    pressure = [np.concatenate(values, axis=1) for values in orientation_pressure]
    rows = []
    all_margins = []
    for index, variant in enumerate(variants):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margins = own - rival
        all_margins.append(margins)
        pressure_values = np.concatenate((pressure[0][index], pressure[1][index]))
        rows.append(
            {
                "variant": variant[0],
                "late_enabled": bool(variant[1]),
                "start_step": int(variant[2]),
                "minimum_rival_sheep": int(variant[3]),
                "maximum_rival_cows": int(variant[4]),
                "games": int(len(margins)),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                "mean_candidate_cash": float(np.mean(own)),
                "mean_opponent_cash": float(np.mean(rival)),
                "mean_margin": float(np.mean(margins)),
                "pressure_trigger_rate": float(np.mean(pressure_values)),
                "per_game": [
                    {
                        "seed": int(base_seeds[game % len(base_seeds)]),
                        "candidate_seat": 0 if game < len(base_seeds) else 1,
                        "margin": int(value),
                        "pressure_triggered": bool(pressure_values[game]),
                    }
                    for game, value in enumerate(margins.tolist())
                ],
            }
        )

    oracle = np.max(np.stack(all_margins), axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-late-sheep-pressure-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "policy_inputs": "current public rival cow/sheep board counts and current step only",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": int(len(base_seeds)),
        "seed_batch_size": args.seeds,
        "seed_batches": args.seed_batches,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(
                np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)
            ),
            "mean_margin": float(np.mean(oracle)),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": [
                    {key: value for key, value in row.items() if key != "per_game"}
                    for row in rows
                ],
                "oracle": payload["oracle"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
