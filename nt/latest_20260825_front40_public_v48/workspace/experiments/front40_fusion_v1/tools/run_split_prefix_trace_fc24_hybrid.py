"""Causally evaluate a trace opening with public-shop trace/FC24B suffixes.

Both suffixes start from the exact same materialized prefix state.  They are
rolled out separately and merged only by the already-visible first shop, so no
future event or Replay identity is used by the deployed policy definition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from hasegawa_jax_v2.agent import _pair  # noqa: E402
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_PREFIX_LOCK  # noqa: E402
from kaggriculture_jax.constants import SHOP_NAMES  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def make_trace_block(tables, latest_bank, old_bank, runtime, trace_player: int, length: int):
    opponent = 1 - trace_player

    @jax.jit
    def rollout(
        states,
        trace_carry,
        opponent_carry,
        events,
        trace_bank,
        forced_route_id,
        active_steps,
    ):
        batch = states.step.shape[0]

        def body(value, block_step):
            def advance(current_value):
                current, trace_value, opponent_value = current_value
                opponent_action, next_opponent = fc24_terminal_crop_salvage_player_action_v1(
                    current, tables, latest_bank, old_bank, runtime, opponent_value, opponent
                )
                forced = jnp.full((batch,), forced_route_id, dtype=jnp.int16)
                current, trace_value, _, _ = hasegawa_step_with_external_v3(
                    current,
                    trace_value,
                    trace_bank,
                    opponent_action,
                    trace_player,
                    events,
                    tables,
                    ROUTER_PREFIX_LOCK,
                    None,
                    forced_route=forced,
                    forced_lock=jnp.ones((batch,), dtype=jnp.bool_),
                )
                return current, trace_value, next_opponent

            value = jax.lax.cond(
                block_step < active_steps, advance, lambda current_value: current_value, value
            )
            return value, None

        return jax.lax.scan(
            body,
            (states, trace_carry, opponent_carry),
            jnp.arange(length, dtype=jnp.int8),
        )[0]

    return rollout


def make_fc_block(tables, latest_bank, old_bank, runtime, candidate: int, length: int):
    opponent = 1 - candidate

    @jax.jit
    def rollout(states, candidate_carry, opponent_carry, events, active_steps):
        def body(value, block_step):
            def advance(current_value):
                current, candidate_value, opponent_value = current_value
                candidate_action, candidate_value = fc24_terminal_crop_salvage_player_action_v1(
                    current, tables, latest_bank, old_bank, runtime, candidate_value, candidate
                )
                opponent_action, opponent_value = fc24_terminal_crop_salvage_player_action_v1(
                    current, tables, latest_bank, old_bank, runtime, opponent_value, opponent
                )
                actions = _pair(candidate_action, opponent_action, candidate)
                current = batched_step_sync(current, actions, events, tables)
                return current, candidate_value, opponent_value

            value = jax.lax.cond(
                block_step < active_steps, advance, lambda current_value: current_value, value
            )
            return value, None

        return jax.lax.scan(
            body,
            (states, candidate_carry, opponent_carry),
            jnp.arange(length, dtype=jnp.int8),
        )[0]

    return rollout


def run_trace_blocks(
    block24,
    states,
    trace_carry,
    opponent_carry,
    events,
    trace_bank,
    forced_route_id: int,
    length: int,
):
    route = jnp.asarray(forced_route_id, dtype=jnp.int16)
    value = (states, trace_carry, opponent_carry)
    remaining = length
    while remaining > 0:
        active = min(24, remaining)
        value = block24(
            *value,
            events,
            trace_bank,
            route,
            jnp.asarray(active, dtype=jnp.int8),
        )
        remaining -= active
    return value


def run_fc_blocks(block24, states, opponent_carry, events, length: int):
    batch = states.step.shape[0]
    value = (
        states,
        initialize_fusion_champion_terminal_salvage_carry_v1(batch),
        opponent_carry,
    )
    remaining = length
    while remaining > 0:
        active = min(24, remaining)
        value = block24(*value, events, jnp.asarray(active, dtype=jnp.int8))
        remaining -= active
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace-shop-id", type=int, default=7)
    parser.add_argument("--decision-step", type=int, default=72)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=268001)
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.trace_shop_id < 8:
        raise ValueError("trace shop must be in [0, 7]")
    if not 0 < args.decision_step < 719:
        raise ValueError("decision step must be inside the episode")

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    if int(trace_bank.unit_op.shape[0]) != 2:
        raise ValueError("split hybrid expects compact bank [opening route, trace suffix route]")
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    suffix_length = 719 - args.decision_step

    rows = []
    timings = []
    for seat in (0, 1):
        trace_block24 = make_trace_block(
            tables, latest_bank, old_bank, runtime, seat, 24
        )
        batch = initial.step.shape[0]
        trace_carry = initialize_hasegawa_carry_v3(
            batch, trace_bank.bootstrap_route_id
        )
        opponent_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
        tick = time.perf_counter()
        prefix = run_trace_blocks(
            trace_block24,
            initial,
            trace_carry,
            opponent_carry,
            events,
            trace_bank,
            0,
            args.decision_step,
        )
        jax.block_until_ready(prefix[0].money)
        prefix_seconds = time.perf_counter() - tick
        prefix_state, trace_carry, opponent_carry = prefix
        first_shop = np.asarray(jax.device_get(prefix_state.town_shops[:, 0]), dtype=np.int8)
        town_count = np.asarray(jax.device_get(prefix_state.town_count), dtype=np.int8)
        if not np.all(town_count > 0):
            raise RuntimeError("first shop not public at decision step")
        print(json.dumps({"seat": seat, "stage": "prefix", "seconds": prefix_seconds}), flush=True)

        tick = time.perf_counter()
        trace_terminal, trace_final_carry, _ = run_trace_blocks(
            trace_block24,
            prefix_state,
            trace_carry,
            opponent_carry,
            events,
            trace_bank,
            1,
            suffix_length,
        )
        jax.block_until_ready(trace_terminal.money)
        trace_seconds = time.perf_counter() - tick
        print(json.dumps({"seat": seat, "stage": "trace_suffix", "seconds": trace_seconds}), flush=True)

        fc_block24 = make_fc_block(
            tables, latest_bank, old_bank, runtime, seat, 24
        )
        tick = time.perf_counter()
        fc_terminal, _, _ = run_fc_blocks(
            fc_block24,
            prefix_state,
            opponent_carry,
            events,
            suffix_length,
        )
        jax.block_until_ready(fc_terminal.money)
        fc_seconds = time.perf_counter() - tick
        print(json.dumps({"seat": seat, "stage": "fc_suffix", "seconds": fc_seconds}), flush=True)

        trace_terminal, trace_final_carry, fc_terminal = jax.device_get(
            (trace_terminal, trace_final_carry, fc_terminal)
        )
        use_trace = first_shop == args.trace_shop_id
        trace_own = np.asarray(trace_terminal.money[:, seat], dtype=np.int64)
        trace_rival = np.asarray(trace_terminal.money[:, 1 - seat], dtype=np.int64)
        fc_own = np.asarray(fc_terminal.money[:, seat], dtype=np.int64)
        fc_rival = np.asarray(fc_terminal.money[:, 1 - seat], dtype=np.int64)
        own = np.where(use_trace, trace_own, fc_own)
        rival = np.where(use_trace, trace_rival, fc_rival)
        margin = own - rival
        rows.append(
            {
                "candidate_seat": seat,
                "games": args.batch,
                "wins": int(np.sum(margin > 0)),
                "win_rate": float(np.mean(margin > 0)),
                "mean_margin": float(np.mean(margin)),
                "trace_games": int(np.sum(use_trace)),
                "trace_wins": int(np.sum((margin > 0) & use_trace)),
                "trace_win_rate": float(np.mean(margin[use_trace] > 0)) if np.any(use_trace) else None,
                "fallback_games": int(np.sum(~use_trace)),
                "fallback_wins": int(np.sum((margin > 0) & ~use_trace)),
                "fallback_win_rate": float(np.mean(margin[~use_trace] > 0)) if np.any(~use_trace) else None,
                "all_done": bool(
                    np.all(np.where(use_trace, trace_terminal.done, fc_terminal.done))
                ),
                "trace_hard_counter_total": int(
                    np.sum(np.asarray(trace_final_carry.hard_counter_total)[use_trace])
                ),
                "trace_invalid_intent_per_game": float(
                    np.mean(np.asarray(trace_final_carry.invalid_intent_total)[use_trace])
                ) if np.any(use_trace) else None,
            }
        )
        timings.append(
            {
                "seat": seat,
                "prefix_seconds": prefix_seconds,
                "trace_suffix_seconds": trace_seconds,
                "fc_suffix_seconds": fc_seconds,
            }
        )

    games = sum(row["games"] for row in rows)
    payload = {
        "schema": "kaggriculture.front40_fusion.split-prefix-trace-fc24-hybrid.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(
            row["all_done"] and row["trace_hard_counter_total"] == 0 for row in rows
        ) else "FAIL",
        "device": str(jax.devices()[0]),
        "trace_bank": str(args.trace_bank),
        "decision_step": args.decision_step,
        "trace_shop_id": args.trace_shop_id,
        "trace_shop": SHOP_NAMES[args.trace_shop_id],
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "runtime_inputs": ["current_state", "own_private_inventory", "public_first_shop"],
        "aggregate": {
            "games": games,
            "wins": sum(row["wins"] for row in rows),
            "win_rate": sum(row["wins"] for row in rows) / games,
            "mean_margin": sum(row["mean_margin"] * row["games"] for row in rows) / games,
        },
        "timings": timings,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
