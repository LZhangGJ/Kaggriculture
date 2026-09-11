#!/usr/bin/env python3
"""Build paired FC16-vs-ueddy counterfactual labels on responsive opponents.

For each seed and seat, the control and suffix arm see independent copies of
the same responsive opponent.  Their states are identical through step 191;
the public/own-private step-192 snapshot is therefore a valid selector input.
"""

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
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc16_shadow_old_suffix_player_action_v1,
    initialize_fusion_champion_moon_suffix_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_fc15_latest_public6_panel import (  # noqa: E402
    OPPONENTS,
    compact_decision_features,
)
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def arm_metrics(margins: np.ndarray) -> dict:
    return {
        "games": int(margins.size),
        "wins": int(np.sum(margins > 0)),
        "ties": int(np.sum(margins == 0)),
        "losses": int(np.sum(margins < 0)),
        "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=837001)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--switch-step", type=int, default=192)
    parser.add_argument("--suffix-route", type=int, default=30)
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if len(selected) != len(set(selected)) or any(value not in OPPONENTS for value in selected):
        raise ValueError("invalid or duplicate opponent")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")

    tables = load_tables()
    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest8_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    latest6_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = rr.make_simulator_step(tables)
    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, 2)
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(expanded.size)
    enable = jnp.repeat(jnp.asarray([False, True]), args.seeds)
    switch_step = jnp.full((batch,), args.switch_step, dtype=jnp.int16)
    suffix_route = jnp.full((batch,), args.suffix_route, dtype=jnp.int16)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    rows = []
    labels = []

    for opponent_name in selected:
        opponent_fn, opponent_init = OPPONENTS[opponent_name]
        seat_money = []
        seat_features = []
        for candidate_seat in (0, 1):
            player = candidate_seat
            rival = 1 - player

            @jax.jit
            def candidate_policy(states, carry, enabled_values):
                return fc16_shadow_old_suffix_player_action_v1(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    carry,
                    enabled_values,
                    switch_step,
                    suffix_route,
                    player,
                )

            @jax.jit
            def opponent_policy(states, carry):
                return opponent_fn(states, runtime, latest6_bank, carry, rival)

            states = jax.vmap(reset)(jnp.asarray(expanded))
            candidate_carry = initialize_fusion_champion_moon_suffix_carry_v1(batch)
            opponent_carry = opponent_init(batch)
            decision_features = None
            for action_index in range(719):
                if action_index == args.switch_step:
                    decision_features = compact_decision_features(
                        jax.tree_util.tree_map(lambda value: value[: args.seeds], states),
                        player,
                    )
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry, enable
                )
                opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if player == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError(f"{opponent_name} seat{player}: unfinished")
            if any(
                int(np.sum(np.asarray(value)))
                for value in (
                    terminal.hand_cap_hits,
                    terminal.market_loop_cap_hits,
                    terminal.price_lut_oob,
                )
            ):
                raise AssertionError(f"{opponent_name} seat{player}: safety counter")
            if decision_features is None:
                raise AssertionError("decision features were not captured")
            seat_money.append(
                np.asarray(terminal.money, dtype=np.int64).reshape(2, args.seeds, 2)
            )
            seat_features.append(decision_features)

        arm_margins = []
        for arm in range(2):
            seat0 = seat_money[0][arm]
            seat1 = seat_money[1][arm]
            arm_margins.append(
                np.concatenate((seat0[:, 0] - seat0[:, 1], seat1[:, 1] - seat1[:, 0]))
            )
        control_margin, suffix_margin = arm_margins
        delta = suffix_margin - control_margin
        row = {
            "opponent": opponent_name,
            "control": arm_metrics(control_margin),
            "suffix": arm_metrics(suffix_margin),
            "suffix_better_games": int(np.sum(delta > 0)),
            "suffix_equal_games": int(np.sum(delta == 0)),
            "suffix_worse_games": int(np.sum(delta < 0)),
            "mean_margin_delta": float(np.mean(delta)),
            "oracle": arm_metrics(np.maximum(control_margin, suffix_margin)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

        for seat in (0, 1):
            offset = seat * args.seeds
            for index, seed in enumerate(base_seeds.tolist()):
                labels.append(
                    {
                        "opponent": opponent_name,
                        "seed": int(seed),
                        "candidate_seat": seat,
                        "control_margin": int(control_margin[offset + index]),
                        "suffix_margin": int(suffix_margin[offset + index]),
                        "margin_delta": int(delta[offset + index]),
                        "suffix_better": bool(delta[offset + index] > 0),
                        "control_win": bool(control_margin[offset + index] > 0),
                        "suffix_win": bool(suffix_margin[offset + index] > 0),
                        "decision_features": seat_features[seat][index],
                    }
                )

    payload = {
        "schema": "kaggriculture.fc17-ueddy-gate-dataset.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "switch_step": args.switch_step,
        "suffix_route": args.suffix_route,
        "seat_protocol": "paired control/suffix on same seeds with seats swapped",
        "rows": rows,
        "labels": labels,
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "labels": len(labels), "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
