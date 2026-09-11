#!/usr/bin/env python3
"""Screen prefix-compatible K320 suffix choices in FC16 versus Boatlee V21."""

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
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc16_moon_h4_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
from run_fc15_latest_public6_panel import OPPONENTS  # noqa: E402


# name, source route, target route, public decision step
VARIANTS = (
    ("fc16_control", -1, -1, 999),
    # Routes 0, 1 and 2 share an exact raw prefix through step 215.  These
    # arms ask whether the no/late-Yarn economy should be locked to another
    # production mix once three shops are public.
    ("non_yarn_to_r0_s216", -2, 0, 216),
    ("non_yarn_to_r1_s216", -2, 1, 216),
    ("non_yarn_to_r2_s216", -2, 2, 216),
    ("r2_to_r0_s216", 2, 0, 216),
    ("r2_to_r1_s216", 2, 1, 216),
    ("r0_to_r1_s264", 0, 1, 264),
    ("r1_to_r0_s264", 1, 0, 264),
    # K320 route 4 and X562-high route 13 have an exact raw-action
    # common prefix through step 187.  This is the first safe decision
    # boundary; unlike the earlier route-family probes it does not replay an
    # incompatible opening before adopting the X562 production suffix.
    ("r4_to_x562high_s188", 4, 13, 188),
    ("r4_to_x562high_s216", 4, 13, 216),
    ("r4_to_x562high_s240", 4, 13, 240),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="boatlee_v21_latest")
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
    expanded_seeds = np.tile(base_seeds, len(VARIANTS))
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    source_route = jnp.repeat(jnp.asarray([row[1] for row in VARIANTS], dtype=jnp.int8), args.seeds)
    target_route = jnp.repeat(jnp.asarray([row[2] for row in VARIANTS], dtype=jnp.int8), args.seeds)
    switch_step = jnp.repeat(jnp.asarray([row[3] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    batch = len(expanded_seeds)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            k320 = carry.base.k320
            step = states.step.astype(jnp.int32)
            apply = (
                (target_route >= 0)
                & (step >= switch_step.astype(jnp.int32))
                & (
                    (k320.ray_route_id == source_route)
                    | ((source_route == -2) & (k320.ray_route_id <= 2))
                )
                & (~carry.base.sheep_pressure)
            )
            k320 = k320._replace(
                ray_route_locked=k320.ray_route_locked | apply,
                ray_route_id=jnp.where(apply, target_route, k320.ray_route_id).astype(jnp.int8),
            )
            carry = carry._replace(base=carry.base._replace(k320=k320))
            return fc16_moon_h4_player_action_v1(
                states, tables, latest8_bank, old_bank, runtime, carry, player
            )

        return policy

    orientation_money = []
    started = perf_counter()
    opponent_fn, opponent_init = OPPONENTS[args.opponent]
    for candidate_seat in (0, 1):
        rival = 1 - candidate_seat

        @jax.jit
        def opponent_policy(states, carry):
            return opponent_fn(
                states, runtime, latest6_bank, carry, rival
            )

        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = initialize_fusion_champion_moon_market_carry_v1(batch)
        opponent_carry = opponent_init(batch)
        candidate_policy = make_candidate_policy(candidate_seat)
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
            raise AssertionError("unfinished game")
        if any(
            int(np.sum(np.asarray(value)))
            for value in (
                terminal.hand_cap_hits,
                terminal.market_loop_cap_hits,
                terminal.price_lut_oob,
            )
        ):
            raise AssertionError("simulator safety counter")
        orientation_money.append(
            np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2)
        )

    first, second = orientation_money
    rows = []
    all_margins = []
    for index, (name, source, target, switch) in enumerate(VARIANTS):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival_cash = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival_cash
        all_margins.append(margins)
        rows.append(
            {
                "variant": name,
                "source_route": source,
                "target_route": target,
                "switch_step": switch,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_margin": float(np.mean(margins)),
                "per_game": [
                    {
                        "seed": int(base_seeds[game % args.seeds]),
                        "candidate_seat": 0 if game < args.seeds else 1,
                        "margin": int(margin),
                    }
                    for game, margin in enumerate(margins.tolist())
                ],
            }
        )
    oracle = np.max(np.stack(all_margins, axis=0), axis=0)
    payload = {
        "schema": "kaggriculture.fc16-b21-prefix-compatible-route-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
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
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = [{key: value for key, value in row.items() if key != "per_game"} for row in rows]
    print(json.dumps({"status": "PASS", "rows": compact, "oracle": payload["oracle"], "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
