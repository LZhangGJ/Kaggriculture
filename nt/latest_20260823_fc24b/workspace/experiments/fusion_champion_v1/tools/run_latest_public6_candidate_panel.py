#!/usr/bin/env python3
"""Evaluate one frozen latest-public-six controller against the same pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
    initialize_kaito_v39_carry_v1,
    initialize_latest_e284_carry_v1,
    initialize_moon_v92_carry_v1,
    initialize_soil_v26h_carry_v1,
    kaito_v39_history_gate_player_action_v1,
    prvsiyan_moon_v92_player_action_v1,
    prvsiyan_soil_v26h_player_action_v1,
    salem_harvestforge_x_player_action_v1,
    steven_e284_hadouken_player_action_v1,
)
from strategic_v5.latest_public8_gpu import initialize_x562_carry_v1  # noqa: E402


AGENTS = {
    "boatlee_v21_latest": (boatlee_v21_player_action_v1, initialize_boatlee_v21_carry_v1),
    "prvsiyan_soil_v26h_latest": (prvsiyan_soil_v26h_player_action_v1, initialize_soil_v26h_carry_v1),
    "prvsiyan_moon_v92_latest": (prvsiyan_moon_v92_player_action_v1, initialize_moon_v92_carry_v1),
    "kaito_v39_history_gate_latest": (kaito_v39_history_gate_player_action_v1, initialize_kaito_v39_carry_v1),
    "steven_e284_hadouken_latest": (steven_e284_hadouken_player_action_v1, initialize_latest_e284_carry_v1),
    "salem_harvestforge_x_latest": (salem_harvestforge_x_player_action_v1, initialize_x562_carry_v1),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(AGENTS), required=True)
    parser.add_argument("--opponents", default=",".join(AGENTS))
    parser.add_argument("--seed-start", type=int, default=823001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if len(selected) != len(set(selected)) or any(value not in AGENTS for value in selected):
        raise ValueError("invalid or duplicate opponent")

    tables = load_tables()
    bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    bank = load_bank(bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    simulator = rr.make_simulator_step(tables)
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = len(seeds)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    candidate_fn, candidate_init = AGENTS[args.candidate]
    rows = []
    started = perf_counter()
    for opponent_name in selected:
        opponent_fn, opponent_init = AGENTS[opponent_name]
        orientation_money = []
        for candidate_seat in (0, 1):
            rival = 1 - candidate_seat

            @jax.jit
            def candidate_policy(states, carry):
                return candidate_fn(states, runtime, bank, carry, candidate_seat)

            @jax.jit
            def opponent_policy(states, carry):
                return opponent_fn(states, runtime, bank, carry, rival)

            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = candidate_init(batch)
            opponent_carry = opponent_init(batch)
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
                raise AssertionError(f"{opponent_name}: unfinished")
            if any(
                int(np.sum(np.asarray(value)))
                for value in (
                    terminal.hand_cap_hits,
                    terminal.market_loop_cap_hits,
                    terminal.price_lut_oob,
                )
            ):
                raise AssertionError(f"{opponent_name}: safety counter")
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))

        first, second = orientation_money
        own = np.concatenate((first[:, 0], second[:, 1]))
        rival_cash = np.concatenate((first[:, 1], second[:, 0]))
        margins = own - rival_cash
        row = {
            "opponent": opponent_name,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival_cash)),
            "mean_margin": float(np.mean(margins)),
            "candidate_seat0_wins": int(np.sum(first[:, 0] > first[:, 1])),
            "candidate_seat1_wins": int(np.sum(second[:, 1] > second[:, 0])),
            "per_game": [
                {
                    "seed": int(seed),
                    "candidate_seat": seat,
                    "candidate_cash": int(
                        first[index, 0] if seat == 0 else second[index, 1]
                    ),
                    "opponent_cash": int(
                        first[index, 1] if seat == 0 else second[index, 0]
                    ),
                    "margin": int(
                        (first[index, 0] - first[index, 1])
                        if seat == 0
                        else (second[index, 1] - second[index, 0])
                    ),
                }
                for index, seed in enumerate(seeds.tolist())
                for seat in (0, 1)
            ],
        }
        rows.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "per_game"}),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.latest-public6-candidate-panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "candidate": args.candidate,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "games_per_opponent": args.seeds * 2,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "min_score_rate": min(row["score_rate"] for row in rows),
        "mean_score_rate": float(np.mean([row["score_rate"] for row in rows])),
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
