from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments" / "expert_business_agent_v2" / "tools",
    ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "gpu_sim" / "src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, MarketOp, UnitOp  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MODE_BOATLEE,
    MODE_RAY_K320,
    MODE_TETSUTANI,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    initialize_public_g02_carry_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import public_v27_player_action_exact_v1  # noqa: E402


TARGETS = (
    ("boatlee_v20_multi_route", "v20", MODE_BOATLEE, -1),
    ("rayk_k320_adaptive_rank1", "v20", MODE_RAY_K320, -1),
    ("tetsutani_adaptive_premium_queue", "v20", MODE_TETSUTANI, -1),
    ("kaito_v27_midgame_reset", "v25", -1, 8),
    ("flexonafft_v59_multi_route", "flex", -1, 9),
)


def null_action(states, player: int) -> Action:
    batch = states.step.shape[0]
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    return Action(
        jnp.full((batch, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        jnp.full((batch, MAX_UNITS), -1, dtype=jnp.int8),
        jnp.ones((batch, MAX_UNITS), dtype=jnp.int32),
        count,
        jnp.full((batch, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int32),
        jnp.zeros((batch,), dtype=jnp.int8),
    )


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(bank, runtime, tables, kind: str, mode: int, route_id: int):
    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), route_id, dtype=jnp.int32)

        def body(value, _):
            states, carry = value
            if kind == "v20":
                action, carry = high_potential_v20_player_action_v1(
                    states, tables, bank, runtime, carry, 0, mode
                )
            elif kind == "v25":
                action, carry = public_v27_player_action_exact_v1(
                    states, runtime, bank, route_ids, carry, 0
                )
            else:
                action, carry = public_rc5_weed_player_action_v1(
                    states, bank, route_ids, carry, 0
                )
            states = batched_step_sync(states, pair(action, null_action(states, 1)), events, tables)
            return (states, carry), None

        initial_carry = (
            initialize_high_potential_v20_carry_v1(batch)
            if kind == "v20"
            else initialize_public_g02_carry_v1(batch)
        )
        (terminal, _), _ = jax.lax.scan(
            body, (initial, initial_carry), xs=None, length=719
        )
        return terminal.money, terminal.done

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed-start", type=int, default=98401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    bank = load_bank(args.bank.resolve())
    runtime = load_high_potential_runtime_tables_v1(args.runtime.resolve())
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    rows = []
    for name, kind, mode, route_id in TARGETS:
        rollout = make_rollout(bank, runtime, tables, kind, mode, route_id)
        start = time.perf_counter()
        first_money, first_done = rollout(initial, events)
        jax.block_until_ready(first_money)
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        money, done = rollout(initial, events)
        jax.block_until_ready(money)
        steady_seconds = time.perf_counter() - start
        host_money, host_done = jax.device_get((money, done))
        row = {
            "agent": name,
            "batch": args.batch,
            "compile_and_first_seconds": compile_seconds,
            "steady_seconds": steady_seconds,
            "steady_games_per_second": args.batch / steady_seconds,
            "steady_transitions_per_second": args.batch * 719 / steady_seconds,
            "all_done": bool(np.all(host_done)),
            "cash_mean": float(np.mean(host_money[:, 0])),
            "cash_min": int(np.min(host_money[:, 0])),
            "cash_max": int(np.max(host_money[:, 0])),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    output = {
        "schema": "kaggriculture.high_potential_exact5_gpu_benchmark.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "batch": args.batch,
        "steps": 719,
        "all_done": all(row["all_done"] for row in rows),
        "results": rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS" if output["all_done"] else "FAIL", "output": str(output_path)}))
    return 0 if output["all_done"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
