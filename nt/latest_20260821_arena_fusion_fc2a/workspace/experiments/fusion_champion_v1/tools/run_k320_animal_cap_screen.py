#!/usr/bin/env python3
"""Screen legal cow/sheep investment caps on top of the complete K320 policy."""

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
    _cap_animal_purchases,
    k320_heavy_sheep_animal_substitution_player_action_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MODE_RAY_K320,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)


VARIANTS = (
    ("source", 99, 99, False),
    ("6c4s", 6, 4, True),
    ("7c4s", 7, 4, True),
    ("8c4s", 8, 4, True),
    ("9c4s", 9, 4, True),
    ("10c4s", 10, 4, True),
    ("8c2s", 8, 2, True),
    ("8c3s", 8, 3, True),
    ("8c5s", 8, 5, True),
    ("8c6s", 8, 6, True),
    ("10c2s", 10, 2, True),
    ("10c3s", 10, 3, True),
    ("10c5s", 10, 5, True),
    ("10c6s", 10, 6, True),
)

SUBSTITUTION_VARIANTS = (
    ("source", 0, 192),
    ("drop_sheep_s192", 1, 192),
    ("half_sheep_s192", 2, 192),
    ("cow_same_s192", 3, 192),
    ("cow_half_s192", 4, 192),
    ("drop_sheep_s216", 1, 216),
    ("half_sheep_s216", 2, 216),
    ("cow_same_s216", 3, 216),
    ("cow_half_s216", 4, 216),
    ("drop_sheep_s240", 1, 240),
    ("half_sheep_s240", 2, 240),
    ("cow_same_s240", 3, 240),
    ("cow_half_s240", 4, 240),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--substitution", action="store_true")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant names in substitution mode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

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
        raise ValueError(f"unknown opponent: {args.opponent}")
    opponent_id = names.index(args.opponent)

    if args.substitution:
        requested = {name for name in args.variants.split(",") if name}
        variants = tuple(
            row for row in SUBSTITUTION_VARIANTS if not requested or row[0] in requested
        )
        if not variants or (requested and len(variants) != len(requested)):
            raise ValueError("invalid substitution variants")
    else:
        variants = VARIANTS

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(variants))
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    if args.substitution:
        substitution_mode = jnp.repeat(
            jnp.asarray([row[1] for row in variants], dtype=jnp.int8), args.seeds
        )
        substitution_start = jnp.repeat(
            jnp.asarray([row[2] for row in variants], dtype=jnp.int16), args.seeds
        )
        cow_cap = sheep_cap = enabled = None
    else:
        cow_cap = jnp.repeat(jnp.asarray([row[1] for row in variants], dtype=jnp.int16), args.seeds)
        sheep_cap = jnp.repeat(jnp.asarray([row[2] for row in variants], dtype=jnp.int16), args.seeds)
        enabled = jnp.repeat(jnp.asarray([row[3] for row in variants]), args.seeds)
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            if args.substitution:
                return k320_heavy_sheep_animal_substitution_player_action_v1(
                    states,
                    resources["tables"],
                    resources["runtime"],
                    resources["latest_bank"],
                    carry,
                    substitution_mode,
                    substitution_start,
                    player,
                )
            action, carry = high_potential_v20_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["runtime"],
                carry,
                player,
                MODE_RAY_K320,
            )
            action = _cap_animal_purchases(
                states, action, cow_cap, sheep_cap, enabled, player
            )
            return action, carry

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
        if any(int(np.sum(np.asarray(getattr(terminal, name)))) for name in (
            "hand_cap_hits", "market_loop_cap_hits", "price_lut_oob"
        )):
            raise AssertionError("simulator safety counter hit")
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(variants), args.seeds, 2))

    rows = []
    margins_all = []
    for index, variant in enumerate(variants):
        name = variant[0]
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margins = own - rival
        margins_all.append(margins)
        row = {
            "variant": name,
            "games": int(len(margins)),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "per_game": [
                {"seed": int(base_seeds[i % args.seeds]), "candidate_seat": 0 if i < args.seeds else 1, "margin": int(value)}
                for i, value in enumerate(margins.tolist())
            ],
        }
        if args.substitution:
            row.update(
                substitution_mode=int(variant[1]),
                start_step=int(variant[2]),
            )
        else:
            row.update(
                cap_enabled=bool(variant[3]),
                cow_cap=int(variant[1]),
                sheep_cap=int(variant[2]),
            )
        rows.append(row)

    oracle = np.max(np.stack(margins_all), axis=0)
    payload = {
        "schema": (
            "kaggriculture.fusion_champion.k320-animal-substitution-screen.v1"
            if args.substitution
            else "kaggriculture.fusion_champion.k320-animal-cap-screen.v1"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "rows": [{key: value for key, value in row.items() if key != "per_game"} for row in rows],
        "oracle": payload["oracle"],
        "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
