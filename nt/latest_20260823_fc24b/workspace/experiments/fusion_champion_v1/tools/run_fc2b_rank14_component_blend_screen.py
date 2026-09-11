#!/usr/bin/env python3
"""Ablate Rank14 suffix market and unit components after an FC2B prefix."""

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
    fc11i_fc2b_shadow_suffix_component_player_action_v1,
    initialize_fusion_champion_carry_v6,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


RANK14_ROUTE = 63
VARIANTS = (
    ("source", 0, 32767),
    ("market_s192", 1, 192),
    ("market_s240", 1, 240),
    ("market_s288", 1, 288),
    ("market_s360", 1, 360),
    ("units_s192", 2, 192),
    ("units_s240", 2, 240),
    ("units_s288", 2, 288),
    ("units_s360", 2, 360),
    ("full_s192", 3, 192),
    ("full_s240", 3, 240),
    ("full_s288", 3, 288),
    ("full_s360", 3, 360),
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
    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    variants = tuple(row for row in VARIANTS if not requested or row[0] in requested)
    if not variants or (requested and len(variants) != len(requested)):
        raise ValueError("invalid variants")
    if args.seeds < 1 or args.seed_batches < 1:
        raise ValueError("seed counts must be positive")

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

    segments = [
        np.arange(
            args.seed_start + index * args.seeds,
            args.seed_start + (index + 1) * args.seeds,
            dtype=np.int32,
        )
        for index in range(args.seed_batches)
    ]
    base_seeds = np.concatenate(segments)
    modes = jnp.repeat(
        jnp.asarray([row[1] for row in variants], dtype=jnp.int8), args.seeds
    )
    switch_steps = jnp.repeat(
        jnp.asarray([row[2] for row in variants], dtype=jnp.int16), args.seeds
    )
    suffix_routes = jnp.full(
        (len(variants) * args.seeds,), RANK14_ROUTE, dtype=jnp.int16
    )
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc11i_fc2b_shadow_suffix_component_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["old_bank"],
                resources["runtime"],
                carry,
                switch_steps,
                suffix_routes,
                modes,
                player,
            )

        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    orientation: list[list[np.ndarray]] = [[], []]
    for segment in segments:
        expanded = np.tile(segment, len(variants))
        weed, shops = build_events_v1(expanded.tolist())
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        batch = len(expanded)
        for seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded))
            candidate_carry = initialize_fusion_champion_carry_v6(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            candidate_policy = make_candidate_policy(seat)
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - seat, **resources
            )
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
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
            if any(
                int(np.sum(np.asarray(getattr(terminal, name))))
                for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            ):
                raise AssertionError("simulator safety counter hit")
            orientation[seat].append(
                np.asarray(terminal.money, dtype=np.int64).reshape(
                    len(variants), args.seeds, 2
                )
            )

    money = [np.concatenate(values, axis=1) for values in orientation]
    rows = []
    matrices = []
    for index, (name, mode, switch_step) in enumerate(variants):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margins = own - rival
        matrices.append(margins)
        rows.append(
            {
                "variant": name,
                "blend_mode": mode,
                "switch_step": switch_step,
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
                "per_game": [
                    {
                        "seed": int(base_seeds[game % len(base_seeds)]),
                        "candidate_seat": 0 if game < len(base_seeds) else 1,
                        "margin": int(value),
                    }
                    for game, value in enumerate(margins.tolist())
                ],
            }
        )

    oracle = np.max(np.stack(matrices), axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-rank14-component-blend-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "suffix_route": RANK14_ROUTE,
        "seed_start": args.seed_start,
        "seed_count": int(len(base_seeds)),
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
