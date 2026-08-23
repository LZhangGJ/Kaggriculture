#!/usr/bin/env python3
"""Screen every old-bank suffix after the FC16 opening on one public Replay.

The rival request tape is frozen, so this is a same-seed capability/causal
screen rather than a responsive-opponent Arena estimate.  Both FC16 and every
candidate suffix are advanced from step zero on the same real states; takeover
therefore does not cold-start route ledgers.
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
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1 import policy_gpu as fg  # noqa: E402
from kaggriculture_jax.codec import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5 import high_potential_v20_gpu as hp  # noqa: E402
from strategic_v5 import latest_public8_gpu as lp  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


class Carry(NamedTuple):
    base: fg.FusionChampionMoonMarketCarryV1
    suffix: hp.HighPotentialV20CarryV1
    switched: jax.Array


def initialize_carry(batch: int) -> Carry:
    return Carry(
        base=fg.initialize_fusion_champion_moon_market_carry_v1(batch),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch),
        switched=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def take(action: Action, step: int, player: int, batch: int) -> Action:
    return Action(
        *(jnp.repeat(value[step, player][None, ...], batch, axis=0) for value in action)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--own-team", default="QQ Farming")
    parser.add_argument("--routes", default="0-86")
    parser.add_argument("--switch-steps", default="72,96,120,144,168,192,216,240,288")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    def parse_routes(value: str) -> list[int]:
        result: list[int] = []
        for part in value.split(","):
            if "-" in part:
                start, stop = (int(item) for item in part.split("-", 1))
                result.extend(range(start, stop + 1))
            else:
                result.append(int(part))
        return sorted(set(result))

    routes = parse_routes(args.routes)
    switch_steps = sorted(set(int(value) for value in args.switch_steps.split(",")))
    variants = [(-1, 999)] + [(route, step) for step in switch_steps for route in routes]

    replay = json.loads(args.episode.read_text(encoding="utf-8"))
    names = list((replay.get("info") or {}).get("TeamNames") or [])
    if args.own_team not in names or len(names) != 2:
        raise ValueError(f"cannot identify own seat from {names}")
    own_seat = names.index(args.own_team)
    rival_seat = 1 - own_seat
    seed = int((replay.get("info") or {})["seed"])
    recorded = stack_actions(
        [
            encode_actions(
                [
                    replay["steps"][step + 1][seat].get("action") or {}
                    for seat in (0, 1)
                ]
            )
            for step in range(719)
        ]
    )

    tables = load_tables()
    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest8_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = jax.jit(rr.make_simulator_step(tables))

    # Official/JAX truth gate before accepting any candidate result.
    weed, shops = build_events_v1([seed])
    one_events = Events(jnp.asarray(weed), jnp.asarray(shops))
    state = jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))
    for step in range(719):
        state = simulator(
            state, take(recorded, step, 0, 1), take(recorded, step, 1, 1), one_events
        )
    jax.block_until_ready(state.money)
    replay_money = np.asarray(jax.device_get(state.money[0]), dtype=np.int64)
    official_money = np.asarray(replay["rewards"], dtype=np.int64)
    if not np.array_equal(replay_money, official_money):
        raise AssertionError(
            f"recorded Replay parity failed: JAX={replay_money}, official={official_money}"
        )

    batch = len(variants)
    repeated_seed = np.full((batch,), seed, dtype=np.int32)
    weed, shops = build_events_v1(repeated_seed.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    route_arg = jnp.asarray([max(route, 0) for route, _ in variants], dtype=jnp.int16)
    switch_arg = jnp.asarray([step for _, step in variants], dtype=jnp.int16)
    enabled_arg = jnp.asarray([route >= 0 for route, _ in variants], dtype=jnp.bool_)

    @jax.jit
    def policy(
        states,
        carry: Carry,
        route_values: jax.Array,
        switch_values: jax.Array,
        enabled_values: jax.Array,
    ):
        shape = states.step.shape
        base_action, base = fg.fc15_moon_parameter_player_action_v1(
            states,
            tables,
            latest8_bank,
            old_bank,
            runtime,
            carry.base,
            jnp.full(shape, 4, dtype=jnp.int16),
            jnp.full(shape, 120, dtype=jnp.int16),
            jnp.full(shape, 2, dtype=jnp.int16),
            jnp.full(shape, 6, dtype=jnp.int16),
            own_seat,
        )
        suffix_action, suffix = hp._raw_with_weed(
            states, old_bank, route_values.astype(jnp.int32), carry.suffix, own_seat
        )
        suffix_action, suffix = hp._room_evac(states, suffix_action, suffix, own_seat)
        suffix_action, suffix = hp._repay(
            suffix_action, suffix, states.step.astype(jnp.int32), hp.MODE_BOATLEE
        )
        suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
        suffix_action = lp._room_guard(states, suffix_action, own_seat)
        suffix_action = lp._x562_seed_budget_guard(states, suffix_action, own_seat)
        suffix_action = lp._terminal_liquidation(states, suffix_action, own_seat)
        suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, own_seat)
        switched = carry.switched | (
            enabled_values
            & (states.step.astype(jnp.int32) >= switch_values.astype(jnp.int32))
        )
        action = fg._select_action(switched, suffix_action, base_action)
        return action, Carry(base=base, suffix=suffix, switched=switched)

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    state = jax.vmap(reset)(jnp.asarray(repeated_seed))
    carry = initialize_carry(batch)
    for step in range(719):
        own_action, carry = policy(
            state, carry, route_arg, switch_arg, enabled_arg
        )
        rival_action = take(recorded, step, rival_seat, batch)
        state = (
            simulator(state, own_action, rival_action, events)
            if own_seat == 0
            else simulator(state, rival_action, own_action, events)
        )
    jax.block_until_ready(state.money)
    terminal = jax.device_get(state)
    if not bool(np.all(np.asarray(terminal.done))):
        raise AssertionError("not all games DONE")
    if any(
        int(np.sum(np.asarray(value)))
        for value in (
            terminal.hand_cap_hits,
            terminal.market_loop_cap_hits,
            terminal.price_lut_oob,
        )
    ):
        raise AssertionError("simulator safety counter hit")
    money = np.asarray(terminal.money, dtype=np.int64)

    receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(
            encoding="utf-8"
        )
    )
    route_names = {
        int(row["skeleton_id"]): f'{row["opponent"]}::{row["route"]}'
        for row in receipt["skeletons"]
    }
    rows = []
    for index, (route, step) in enumerate(variants):
        own_cash = int(money[index, own_seat])
        rival_cash = int(money[index, rival_seat])
        rows.append(
            {
                "suffix_route": route,
                "route_name": "FC16_CONTROL" if route < 0 else route_names.get(route, "UNKNOWN"),
                "switch_step": step,
                "own_cash": own_cash,
                "recorded_opponent_cash_under_counterfactual": rival_cash,
                "margin": own_cash - rival_cash,
            }
        )
    ranked = sorted(rows, key=lambda row: (row["margin"], row["own_cash"]), reverse=True)
    payload = {
        "schema": "kaggriculture.fc16-old-suffix-public-replay-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "episode_id": int((replay.get("info") or {}).get("EpisodeId")),
        "seed": seed,
        "team_names": names,
        "own_seat": own_seat,
        "recorded_replay_parity": {
            "official_money": official_money.tolist(),
            "jax_money": replay_money.tolist(),
            "exact": True,
        },
        "candidate_count": batch,
        "routes": routes,
        "switch_steps": switch_steps,
        "control": rows[0],
        "top20": ranked[:20],
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
        "truth_boundary": (
            "The opponent requests are frozen from the public Replay. Results are same-seed "
            "capability probes, not responsive-opponent win-rate estimates."
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "control": rows[0], "top10": ranked[:10], "output": str(output.resolve())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
