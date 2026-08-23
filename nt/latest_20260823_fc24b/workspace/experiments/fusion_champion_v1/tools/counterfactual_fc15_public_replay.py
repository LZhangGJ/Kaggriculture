#!/usr/bin/env python3
"""Same-seed JAX counterfactuals against a frozen public Replay opponent.

The recorded-vs-recorded arm must reproduce the official terminal cash before
any candidate result is accepted.  Candidate arms replace only FC15's seat;
the opponent requests remain the public Replay requests, so results are local
causal probes rather than claims about a fully responsive opponent.
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
from run_fc15_latest_public6_panel import OPPONENTS  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
    fc15_opening_hedge_player_action_v1,
    fc15_moon_parameter_player_action_v1,
    fc17_opening_hedge_player_action_v1,
    initialize_fusion_champion_carry_v3,
    initialize_fusion_champion_base_suffix_carry_v1,
    initialize_fusion_champion_moon_market_carry_v1,
    initialize_fusion_champion_moon_suffix_carry_v1,
)
from kaggriculture_jax.codec import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
)


def take(action: Action, step: int, player: int, batch: int = 1) -> Action:
    return Action(
        *(
            jnp.repeat(value[step, player][None, ...], batch, axis=0)
            for value in action
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--own-team", default="QQ Farming")
    parser.add_argument(
        "--mode",
        choices=("basic", "route0", "public6", "full", "gate"),
        default="basic",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    replay = json.loads(args.episode.read_text(encoding="utf-8"))
    names = list((replay.get("info") or {}).get("TeamNames") or [])
    if args.own_team not in names or len(names) != 2:
        raise ValueError(f"cannot identify own seat from {names}")
    own_seat = names.index(args.own_team)
    rival_seat = 1 - own_seat
    seed = int((replay.get("info") or {})["seed"])
    recorded = stack_actions(
        [
            encode_actions([
                (replay["steps"][step + 1][seat].get("action") or {})
                for seat in (0, 1)
            ])
            for step in range(719)
        ]
    )

    tables = load_tables()
    old_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz")
    latest8_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz")
    latest6_bank = load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz")
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = jax.jit(rr.make_simulator_step(tables))
    weed, shops = build_events_v1([seed])
    replay_events = Events(jnp.asarray(weed), jnp.asarray(shops))

    # Hard truth gate: JAX must replay both official request tapes exactly.
    state = jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))
    for step in range(719):
        state = simulator(state, take(recorded, step, 0), take(recorded, step, 1), replay_events)
    jax.block_until_ready(state.money)
    replay_money = np.asarray(jax.device_get(state.money[0]), dtype=np.int64)
    official_money = np.asarray(replay["rewards"], dtype=np.int64)
    if not np.array_equal(replay_money, official_money):
        raise AssertionError(f"recorded Replay parity failed: JAX={replay_money}, official={official_money}")
    print(json.dumps({"stage": "recorded_replay_parity", "money": replay_money.tolist()}), flush=True)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    batch = args.batch
    repeated_seeds = np.full((batch,), seed, dtype=np.int32)
    candidate_weed, candidate_shops = build_events_v1(repeated_seeds.tolist())
    candidate_events = Events(jnp.asarray(candidate_weed), jnp.asarray(candidate_shops))

    def run(name: str, target_route: int | None = None) -> dict:
        state = jax.vmap(reset)(jnp.asarray(repeated_seeds))
        if name == "fc15":
            carry = initialize_fusion_champion_carry_v3(batch)

            @jax.jit
            def policy(states, policy_carry):
                return fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
                    states, tables, latest8_bank, old_bank, runtime, policy_carry, own_seat
                )
        elif name.startswith("fc16"):
            carry = initialize_fusion_champion_moon_market_carry_v1(batch)

            @jax.jit
            def policy(states, policy_carry):
                if target_route is not None:
                    k320 = policy_carry.base.k320
                    apply = (
                        (states.step.astype(jnp.int32) >= 216)
                        & (k320.ray_route_id <= 2)
                        & (~policy_carry.base.sheep_pressure)
                    )
                    k320 = k320._replace(
                        ray_route_locked=k320.ray_route_locked | apply,
                        ray_route_id=jnp.where(apply, target_route, k320.ray_route_id).astype(jnp.int8),
                    )
                    policy_carry = policy_carry._replace(
                        base=policy_carry.base._replace(k320=k320)
                    )
                shape = states.step.shape
                return fc15_moon_parameter_player_action_v1(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    policy_carry,
                    jnp.full(shape, 4, dtype=jnp.int16),
                    jnp.full(shape, 120, dtype=jnp.int16),
                    jnp.full(shape, 2, dtype=jnp.int16),
                    jnp.full(shape, 6, dtype=jnp.int16),
                    own_seat,
                )
        elif name == "fc15_opening_hedge":
            carry = initialize_fusion_champion_base_suffix_carry_v1(batch)

            @jax.jit
            def policy(states, policy_carry):
                return fc15_opening_hedge_player_action_v1(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    policy_carry,
                    own_seat,
                )
        elif name == "fc17_opening_hedge":
            carry = initialize_fusion_champion_moon_suffix_carry_v1(batch)

            @jax.jit
            def policy(states, policy_carry):
                return fc17_opening_hedge_player_action_v1(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    policy_carry,
                    own_seat,
                )
        elif name == "boatlee_v21":
            carry = initialize_boatlee_v21_carry_v1(batch)

            @jax.jit
            def policy(states, policy_carry):
                return boatlee_v21_player_action_v1(
                    states, runtime, latest6_bank, policy_carry, own_seat
                )
        elif name in OPPONENTS:
            public_fn, public_init = OPPONENTS[name]
            carry = public_init(batch)

            @jax.jit
            def policy(states, policy_carry):
                return public_fn(states, runtime, latest6_bank, policy_carry, own_seat)
        else:
            raise ValueError(name)

        for step in range(719):
            own_action, carry = policy(state, carry)
            rival_action = take(recorded, step, rival_seat, batch)
            state = (
                simulator(state, own_action, rival_action, candidate_events)
                if own_seat == 0
                else simulator(state, rival_action, own_action, candidate_events)
            )
        jax.block_until_ready(state.money)
        all_money = np.asarray(jax.device_get(state.money), dtype=np.int64)
        if not np.all(all_money == all_money[0]):
            raise AssertionError("repeated same-seed candidate batch diverged")
        money = all_money[0]
        result = {
            "candidate": name,
            "target_route": target_route,
            "own_cash": int(money[own_seat]),
            "recorded_opponent_cash_under_counterfactual": int(money[rival_seat]),
            "margin": int(money[own_seat] - money[rival_seat]),
        }
        if name in ("fc15_opening_hedge", "fc17_opening_hedge"):
            host_carry = jax.device_get(carry)
            result["opening_match"] = bool(np.asarray(host_carry.opening_match)[0])
            result["suffix_switched"] = bool(np.asarray(host_carry.switched)[0])
        return result

    started = perf_counter()
    rows = []
    for name, target in (("fc15", None), ("fc16", None)):
        row = run(name, target)
        rows.append(row)
        print(json.dumps({"stage": "candidate", **row}), flush=True)
    if args.mode == "gate":
        row = run("fc15_opening_hedge")
        rows.append(row)
        print(json.dumps({"stage": "candidate", **row}), flush=True)
        row = run("fc17_opening_hedge")
        rows.append(row)
        print(json.dumps({"stage": "candidate", **row}), flush=True)
    if args.mode in ("route0", "full"):
        routes = (0,) if args.mode == "route0" else (0, 1, 2)
        for route in routes:
            row = run(f"fc16_force_r{route}", route)
            rows.append(row)
            print(json.dumps({"stage": "candidate", **row}), flush=True)
    if args.mode in ("public6", "full"):
        for name in OPPONENTS:
            row = run(name)
            rows.append(row)
            print(json.dumps({"stage": "candidate", **row}), flush=True)
    elif args.mode != "gate":
        row = run("boatlee_v21")
        rows.append(row)
        print(json.dumps({"stage": "candidate", **row}), flush=True)
    payload = {
        "schema": "kaggriculture.fc15-public-replay-counterfactual.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "episode_id": int((replay.get("info") or {}).get("EpisodeId")),
        "seed": seed,
        "team_names": names,
        "own_seat": own_seat,
        "repeated_candidate_batch": batch,
        "recorded_replay_parity": {
            "official_money": official_money.tolist(),
            "jax_money": replay_money.tolist(),
            "exact": True,
        },
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
        "truth_boundary": (
            "The opponent action requests are frozen from the public Replay. "
            "This is a same-seed local causal probe, not a responsive-opponent Arena estimate."
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "parity": payload["recorded_replay_parity"], "rows": rows, "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
