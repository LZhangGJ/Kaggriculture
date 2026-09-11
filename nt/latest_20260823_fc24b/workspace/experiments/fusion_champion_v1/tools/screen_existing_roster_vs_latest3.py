#!/usr/bin/env python3
"""Screen the frozen exact-JAX roster against the three FC15 shortfalls.

This is a direction screen, not a promotion result.  Every pair uses the same
event seeds in both seats.  Candidate routes are selected by their own public
logic only; the arena never passes opponent identity into a policy.
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
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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
    initialize_moon_v92_carry_v1,
    initialize_soil_v26h_carry_v1,
    prvsiyan_moon_v92_player_action_v1,
    prvsiyan_soil_v26h_player_action_v1,
)


OPPONENTS = {
    "boatlee_v21_latest": (
        boatlee_v21_player_action_v1,
        initialize_boatlee_v21_carry_v1,
    ),
    "prvsiyan_soil_v26h_latest": (
        prvsiyan_soil_v26h_player_action_v1,
        initialize_soil_v26h_carry_v1,
    ),
    "prvsiyan_moon_v92_latest": (
        prvsiyan_moon_v92_player_action_v1,
        initialize_moon_v92_carry_v1,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=824001)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--candidate-ids",
        default=",".join(str(value) for value in range(len(rr.ROSTER))),
        help="Comma-separated frozen roster IDs; order is preserved.",
    )
    parser.add_argument(
        "--opponents",
        default=",".join(OPPONENTS),
        help="Comma-separated opponent names; order is preserved.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    candidate_ids = [
        int(value.strip())
        for value in args.candidate_ids.split(",")
        if value.strip()
    ]
    if (
        not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or any(value < 0 or value >= len(rr.ROSTER) for value in candidate_ids)
    ):
        raise ValueError("invalid or duplicate candidate ID")
    selected_opponents = [
        value.strip() for value in args.opponents.split(",") if value.strip()
    ]
    if (
        not selected_opponents
        or len(selected_opponents) != len(set(selected_opponents))
        or any(value not in OPPONENTS for value in selected_opponents)
    ):
        raise ValueError("invalid or duplicate opponent")

    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest8_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    latest6_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(old_bank_path)
    latest8_bank = load_bank(latest8_bank_path)
    latest6_bank = load_bank(latest6_bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    router = rr.build_router_arrays(old_receipt)
    boatlee_trace = rr.load_boatlee_trace_v1()
    tables = load_tables()
    simulator = rr.make_simulator_step(tables)

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(seeds.size)
    # Reuse the frozen roster's already-validated batch-shape executables.
    cache = ROOT / ".jax_cache/all_exact_round_robin"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    candidate_policies = {
        (agent_id, seat): rr.make_agent_policy(
            agent_id,
            seat,
            old_bank,
            latest8_bank,
            runtime,
            tables,
            router,
            boatlee_trace,
        )
        for agent_id in candidate_ids
        for seat in (0, 1)
    }
    opponent_policies = {}
    for name in selected_opponents:
        function, _ = OPPONENTS[name]
        for seat in (0, 1):
            def make_policy(fn, player):
                @jax.jit
                def policy(states, carry):
                    return fn(states, runtime, latest6_bank, carry, player)
                return policy
            opponent_policies[(name, seat)] = make_policy(function, seat)

    rows = []
    started = perf_counter()
    for opponent_name in selected_opponents:
        _, opponent_init = OPPONENTS[opponent_name]
        for agent_id in candidate_ids:
            agent = rr.ROSTER[agent_id]
            orientation_money = []
            for candidate_seat in (0, 1):
                rival = 1 - candidate_seat
                states = jax.vmap(reset)(jnp.asarray(seeds))
                candidate_carry = rr.initialize_agent_carry(agent_id, batch, router)
                opponent_carry = opponent_init(batch)
                candidate_policy = candidate_policies[(agent_id, candidate_seat)]
                opponent_policy = opponent_policies[(opponent_name, rival)]
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
                    raise AssertionError(f"{agent['name']} vs {opponent_name}: unfinished")
                if any(
                    int(np.sum(np.asarray(value)))
                    for value in (
                        terminal.hand_cap_hits,
                        terminal.market_loop_cap_hits,
                        terminal.price_lut_oob,
                    )
                ):
                    raise AssertionError(f"{agent['name']} vs {opponent_name}: safety counter")
                orientation_money.append(np.asarray(terminal.money, dtype=np.int64))

            first, second = orientation_money
            candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
            opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
            margins = candidate_cash - opponent_cash
            row = {
                "opponent": opponent_name,
                "candidate_id": agent_id,
                "candidate": agent["name"],
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "seat0_wins": int(np.sum(first[:, 0] > first[:, 1])),
                "seat1_wins": int(np.sum(second[:, 1] > second[:, 0])),
                "per_game": [
                    {
                        "seed": int(seed),
                        "candidate_seat": int(seat),
                        "candidate_cash": int(
                            first[offset, 0] if seat == 0 else second[offset, 1]
                        ),
                        "opponent_cash": int(
                            first[offset, 1] if seat == 0 else second[offset, 0]
                        ),
                        "margin": int(
                            first[offset, 0] - first[offset, 1]
                            if seat == 0
                            else second[offset, 1] - second[offset, 0]
                        ),
                    }
                    for offset, seed in enumerate(seeds.tolist())
                    for seat in (0, 1)
                ],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    leaders = {}
    for opponent_name in selected_opponents:
        leaders[opponent_name] = sorted(
            (row for row in rows if row["opponent"] == opponent_name),
            key=lambda row: (row["score_rate"], row["mean_margin"]),
            reverse=True,
        )[:10]
    payload = {
        "schema": "kaggriculture.existing-roster-vs-latest3-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": int(args.seed_start),
        "seed_count": batch,
        "games_per_pair": batch * 2,
        "candidate_ids": candidate_ids,
        "seat_protocol": "same seeds with seats swapped",
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
        "leaders": leaders,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
