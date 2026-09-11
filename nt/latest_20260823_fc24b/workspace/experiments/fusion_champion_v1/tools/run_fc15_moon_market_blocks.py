#!/usr/bin/env python3
"""Compile FC15+Moon once, then evaluate multiple same-shape event blocks."""

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
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc15_moon_embedded_market_observer_player_action_v1,
    fc15_moon_market_observer_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
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
    parser.add_argument("--seed-start", type=int, default=823001)
    parser.add_argument("--seeds-per-block", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument(
        "--candidate",
        choices=("fc15_moon_market", "fc15_moon_embedded_market"),
        default="fc15_moon_market",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds_per_block < 1 or args.blocks < 1:
        raise ValueError("positive seeds-per-block and blocks required")
    opponent_names = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not opponent_names or any(value not in OPPONENTS for value in opponent_names):
        raise ValueError("invalid opponent")

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
    batch = args.seeds_per_block
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    candidate_fn = (
        fc15_moon_embedded_market_observer_player_action_v1
        if args.candidate == "fc15_moon_embedded_market"
        else fc15_moon_market_observer_player_action_v1
    )

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return candidate_fn(
                states,
                tables,
                latest8_bank,
                old_bank,
                runtime,
                carry,
                player,
            )

        return policy

    candidate_policies = {seat: make_candidate_policy(seat) for seat in (0, 1)}
    opponent_policies = {}
    for name in opponent_names:
        fn, _ = OPPONENTS[name]
        for seat in (0, 1):
            def make_opponent_policy(function, player):
                @jax.jit
                def policy(states, carry):
                    return function(states, runtime, latest6_bank, carry, player)

                return policy

            opponent_policies[(name, seat)] = make_opponent_policy(fn, seat)

    rows = []
    started = perf_counter()
    for opponent_name in opponent_names:
        _, opponent_init = OPPONENTS[opponent_name]
        all_candidate_cash = []
        all_opponent_cash = []
        per_game = []
        for block in range(args.blocks):
            block_start = args.seed_start + block * batch
            seeds = np.arange(block_start, block_start + batch, dtype=np.int32)
            weed, shops = build_events_v1(seeds.tolist())
            events = Events(jnp.asarray(weed), jnp.asarray(shops))
            orientation_money = []
            for candidate_seat in (0, 1):
                rival = 1 - candidate_seat
                states = jax.vmap(reset)(jnp.asarray(seeds))
                candidate_carry = initialize_fusion_champion_moon_market_carry_v1(batch)
                opponent_carry = opponent_init(batch)
                for _ in range(719):
                    candidate_action, candidate_carry = candidate_policies[candidate_seat](
                        states, candidate_carry
                    )
                    opponent_action, opponent_carry = opponent_policies[
                        (opponent_name, rival)
                    ](states, opponent_carry)
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
                    raise AssertionError("safety counter")
                orientation_money.append(np.asarray(terminal.money, dtype=np.int64))

            first, second = orientation_money
            candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
            opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
            all_candidate_cash.append(candidate_cash)
            all_opponent_cash.append(opponent_cash)
            for offset, seed in enumerate(seeds.tolist()):
                for seat in (0, 1):
                    own = int(first[offset, 0] if seat == 0 else second[offset, 1])
                    rival = int(first[offset, 1] if seat == 0 else second[offset, 0])
                    per_game.append(
                        {
                            "seed": int(seed),
                            "candidate_seat": seat,
                            "candidate_cash": own,
                            "opponent_cash": rival,
                            "margin": own - rival,
                        }
                    )
            margins = candidate_cash - opponent_cash
            print(
                json.dumps(
                    {
                        "opponent": opponent_name,
                        "block": block,
                        "seed_start": block_start,
                        "wins": int(np.sum(margins > 0)),
                        "ties": int(np.sum(margins == 0)),
                        "losses": int(np.sum(margins < 0)),
                    }
                ),
                flush=True,
            )

        candidate_cash = np.concatenate(all_candidate_cash)
        opponent_cash = np.concatenate(all_opponent_cash)
        margins = candidate_cash - opponent_cash
        rows.append(
            {
                "opponent": opponent_name,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "per_game": per_game,
            }
        )

    payload = {
        "schema": "kaggriculture.fc15-moon-market-block-panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "candidate": args.candidate,
        "seed_start": args.seed_start,
        "seeds_per_block": batch,
        "blocks": args.blocks,
        "games_per_opponent": batch * args.blocks * 2,
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
