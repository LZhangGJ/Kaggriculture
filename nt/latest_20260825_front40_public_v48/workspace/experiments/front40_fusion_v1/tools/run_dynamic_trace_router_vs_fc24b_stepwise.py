"""Evaluate a Replay bank with observable-state rerouting against frozen FC24B.

This is intentionally a diagnostic runner, not a deployable submission.  It
tests whether a player's collection of coherent Replay programs can recover
strength when the route is selected from currently visible state rather than
from one fixed full-season trace.
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
    ROOT / "experiments/hasegawa_jax_v2/tools",
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
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import (  # noqa: E402
    ROUTER_PREFIX_COMPATIBLE,
    ROUTER_PREFIX_LOCK,
    ROUTER_PREFIX_PUBLIC_STATE,
    ROUTER_STEP_PUBLIC_STATE,
    ROUTER_DAY_PUBLIC_STATE,
)
from hasegawa_jax_v3.trace_bank import HasegawaTraceBankV3  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


ROUTER_MODES = {
    "public_state": ROUTER_PREFIX_PUBLIC_STATE,
    "compatible": ROUTER_PREFIX_COMPATIBLE,
    "prefix_lock": ROUTER_PREFIX_LOCK,
    "step_public_state": ROUTER_STEP_PUBLIC_STATE,
    "day_public_state": ROUTER_DAY_PUBLIC_STATE,
}


def pad_bank(bank: HasegawaTraceBankV3, capacity: int) -> HasegawaTraceBankV3:
    route_count = int(bank.unit_op.shape[0])
    if route_count > capacity:
        raise ValueError(f"route count {route_count} exceeds capacity {capacity}")
    if route_count == capacity:
        return bank
    values = []
    for field, value in zip(HasegawaTraceBankV3._fields, bank):
        if field == "bootstrap_route_id":
            values.append(value)
        else:
            values.append(
                jnp.concatenate(
                    (value, jnp.repeat(value[:1], capacity - route_count, axis=0)),
                    axis=0,
                )
            )
    return HasegawaTraceBankV3(*values)


def make_step(
    trace_player: int,
    tables,
    latest_bank,
    old_bank,
    runtime,
    router_mode: int,
    compatibility_threshold: int,
    compatibility_slack: int,
    lock_shop_count: int,
):
    fc_player = 1 - trace_player

    @jax.jit
    def one_step(states, trace_carry, fc_carry, events, trace_bank):
        fc_action, fc_carry = fc24_terminal_crop_salvage_player_action_v1(
            states, tables, latest_bank, old_bank, runtime, fc_carry, fc_player
        )
        states, trace_carry, _, _ = hasegawa_step_with_external_v3(
            states,
            trace_carry,
            trace_bank,
            fc_action,
            trace_player,
            events,
            tables,
            router_mode,
            None,
            compatibility_threshold,
            compatibility_slack,
            lock_shop_count,
        )
        return states, trace_carry, fc_carry

    return one_step


def rollout(step_fn, initial, events, bank, execution_mode: str):
    batch = int(initial.step.shape[0])
    states = initial
    trace_carry = initialize_hasegawa_carry_v3(batch, bank.bootstrap_route_id)
    fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
    if execution_mode == "scan":
        @jax.jit
        def scan_rollout(states, trace_carry, fc_carry, events, bank):
            def body(value, _):
                state_value, trace_value, fc_value = value
                state_value, trace_value, fc_value = step_fn(
                    state_value, trace_value, fc_value, events, bank
                )
                return (state_value, trace_value, fc_value), None

            return jax.lax.scan(
                body, (states, trace_carry, fc_carry), None, length=719
            )[0]

        states, trace_carry, fc_carry = scan_rollout(
            states, trace_carry, fc_carry, events, bank
        )
    else:
        for _ in range(719):
            states, trace_carry, fc_carry = step_fn(
                states, trace_carry, fc_carry, events, bank
            )
    jax.block_until_ready(states.money)
    return jax.device_get((states.money, states.done, trace_carry))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--router-mode", choices=tuple(ROUTER_MODES), default="public_state")
    parser.add_argument("--route-capacity", type=int, default=128)
    parser.add_argument("--compatibility-threshold", type=int, default=15_000)
    parser.add_argument("--compatibility-slack", type=int, default=5_000)
    parser.add_argument("--lock-shop-count", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--execution-mode", choices=("python", "scan"), default="python")
    parser.add_argument("--seed-start", type=int, default=220_001)
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    bank = pad_bank(load_hasegawa_trace_bank_v3(args.trace_bank), args.route_capacity)
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    rows = []
    for trace_player in (0, 1):
        step_fn = make_step(
            trace_player,
            tables,
            latest_bank,
            old_bank,
            runtime,
            ROUTER_MODES[args.router_mode],
            args.compatibility_threshold,
            args.compatibility_slack,
            args.lock_shop_count,
        )
        start = time.perf_counter()
        money, done, carry = rollout(step_fn, initial, events, bank, args.execution_mode)
        elapsed = time.perf_counter() - start
        fc_player = 1 - trace_player
        candidate_cash = money[:, trace_player].astype(np.int64)
        fc_cash = money[:, fc_player].astype(np.int64)
        wins = candidate_cash > fc_cash
        row = {
            "candidate_seat": trace_player,
            "games": args.batch,
            "wins": int(np.sum(wins)),
            "win_rate": float(np.mean(wins)),
            "candidate_cash_mean": float(np.mean(candidate_cash)),
            "fc24b_cash_mean": float(np.mean(fc_cash)),
            "mean_margin": float(np.mean(candidate_cash - fc_cash)),
            "all_done": bool(np.all(done)),
            "route_switch_total": int(np.sum(carry.route_switch_total)),
            "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
            "resync_total": int(np.sum(carry.resync_total)),
            "market_trim_total": int(np.sum(carry.market_trim_total)),
            "hard_counter_total": int(np.sum(carry.hard_counter_total)),
            "elapsed_seconds": elapsed,
            "transitions_per_second": args.batch * 719 / elapsed,
            "candidate_cash": candidate_cash.tolist(),
            "fc24b_cash": fc_cash.tolist(),
            "final_route": np.asarray(carry.branch_id, dtype=np.int64).tolist(),
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}, ensure_ascii=False), flush=True)

    games = sum(row["games"] for row in rows)
    payload = {
        "schema": "kaggriculture.front40_fusion.dynamic-trace-router-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows) else "FAIL",
        "device": str(jax.devices()[0]),
        "candidate": args.candidate_name,
        "trace_bank": str(args.trace_bank),
        "router_mode": args.router_mode,
        "route_capacity": args.route_capacity,
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "execution_mode": args.execution_mode,
        "parameters": {
            "compatibility_threshold": args.compatibility_threshold,
            "compatibility_slack": args.compatibility_slack,
            "lock_shop_count": args.lock_shop_count,
        },
        "aggregate": {
            "games": games,
            "wins": sum(row["wins"] for row in rows),
            "win_rate": sum(row["wins"] for row in rows) / games,
            "mean_margin": sum(row["mean_margin"] * row["games"] for row in rows) / games,
            "invalid_intent_per_game": sum(row["invalid_intent_total"] for row in rows) / games,
            "resync_per_game": sum(row["resync_total"] for row in rows) / games,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
