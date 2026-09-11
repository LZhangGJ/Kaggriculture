"""Run Hasegawa V1 against passive or the local Exact5 JAX opponents."""

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
    ROOT / "experiments" / "hasegawa_jax_v1" / "src",
    ROOT / "experiments" / "general_project_planner_v1" / "src",
    ROOT / "experiments" / "kaggriculture_execution_core" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "experiments" / "expert_business_agent_v2" / "tools",
    ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools",
    ROOT / "gpu_sim" / "src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v1 import (  # noqa: E402
    hasegawa_step_with_external_v1,
    initialize_hasegawa_carry_v1,
    load_hasegawa_plan_bank_v1,
)
from kaggriculture_jax.constants import (  # noqa: E402
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MarketOp,
    UnitOp,
)
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


TARGETS = {
    "passive": ("passive", -1, -1),
    "boatlee_v20": ("v20", MODE_BOATLEE, -1),
    "ray_k320": ("v20", MODE_RAY_K320, -1),
    "tetsutani": ("v20", MODE_TETSUTANI, -1),
    "kaito_v27": ("v25", -1, 8),
    "flex_v59": ("flex", -1, 9),
}


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


def make_rollout(hbank, exact_bank, runtime, tables, kind: str, mode: int, route_id: int, h_player: int):
    opponent = 1 - h_player

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), route_id, dtype=jnp.int32)

        def body(value, _):
            states, hcarry, opponent_carry, hard_total = value
            if kind == "v20":
                action, opponent_carry = high_potential_v20_player_action_v1(
                    states, tables, exact_bank, runtime, opponent_carry, opponent, mode
                )
            elif kind == "v25":
                action, opponent_carry = public_v27_player_action_exact_v1(
                    states, runtime, exact_bank, route_ids, opponent_carry, opponent
                )
            elif kind == "flex":
                action, opponent_carry = public_rc5_weed_player_action_v1(
                    states, exact_bank, route_ids, opponent_carry, opponent
                )
            else:
                action = null_action(states, opponent)
            states, hcarry, diagnostics, _ = hasegawa_step_with_external_v1(
                states, hcarry, hbank, action, h_player, events, tables
            )
            return (
                states,
                hcarry,
                opponent_carry,
                hard_total + diagnostics.hard_error_count,
            ), None

        hcarry = initialize_hasegawa_carry_v1(batch)
        opponent_carry = (
            initialize_high_potential_v20_carry_v1(batch)
            if kind == "v20"
            else initialize_public_g02_carry_v1(batch)
        )
        hard = jnp.zeros((batch,), dtype=jnp.int32)
        (terminal, hcarry, _, hard), _ = jax.lax.scan(
            body, (initial, hcarry, opponent_carry, hard), xs=None, length=719
        )
        return (
            terminal.money,
            terminal.reward,
            terminal.done,
            hard,
            hcarry.route_switch_count,
            hcarry.selected_route,
        )

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--exact-bank", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument(
        "--opponents",
        default="passive",
        help="Comma list from passive,boatlee_v20,ray_k320,tetsutani,kaito_v27,flex_v59 or all5",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=140001)
    parser.add_argument("--seats", default="0,1", help="Hasegawa seats, e.g. 0 or 0,1")
    parser.add_argument("--backend", choices=("any", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.backend == "gpu" and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    hbank = load_hasegawa_plan_bank_v1(args.hasegawa_bank.resolve())
    exact_bank = None
    runtime = None
    requested = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if requested == ["all5"]:
        requested = [name for name in TARGETS if name != "passive"]
    invalid = sorted(set(requested) - set(TARGETS))
    if invalid:
        raise ValueError(f"Unknown opponents: {invalid}")
    seats = [int(value.strip()) for value in args.seats.split(",") if value.strip()]
    if not seats or any(value not in (0, 1) for value in seats):
        raise ValueError("--seats must contain only 0 and/or 1")
    if any(name != "passive" for name in requested):
        if args.exact_bank is None or args.runtime is None:
            raise ValueError("--exact-bank and --runtime are required for Exact5 opponents")
        exact_bank = load_bank(args.exact_bank.resolve())
        runtime = load_high_potential_runtime_tables_v1(args.runtime.resolve())
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    rows = []
    for name in requested:
        kind, mode, route_id = TARGETS[name]
        for h_player in seats:
            rollout = make_rollout(
                hbank, exact_bank, runtime, tables, kind, mode, route_id, h_player
            )
            start = time.perf_counter()
            first = rollout(initial, events)
            jax.block_until_ready(first[0])
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
            result = rollout(initial, events)
            jax.block_until_ready(result[0])
            steady_seconds = time.perf_counter() - start
            money, reward, done, hard, switches, routes = jax.device_get(result)
            other = 1 - h_player
            h_money = money[:, h_player]
            o_money = money[:, other]
            wins = h_money > o_money
            row = {
                "opponent": name,
                "hasegawa_seat": h_player,
                "batch": args.batch,
                "compile_and_first_seconds": compile_seconds,
                "steady_seconds": steady_seconds,
                "steady_transitions_per_second": args.batch * 719 / steady_seconds,
                "all_done": bool(np.all(done)),
                "hard_error_total": int(np.sum(hard)),
                "hasegawa_win_rate": float(np.mean(wins)),
                "hasegawa_cash_mean": float(np.mean(h_money)),
                "opponent_cash_mean": float(np.mean(o_money)),
                "mean_margin": float(np.mean(h_money - o_money)),
                "route_switch_mean": float(np.mean(switches)),
                "final_unique_routes": int(np.unique(routes).size),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "schema": "kaggriculture.hasegawa_jax_arena.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "steps": 719,
        "seed_start": args.seed_start,
        "results": rows,
        "status": "PASS"
        if rows and all(row["all_done"] and row["hard_error_total"] == 0 for row in rows)
        else "FAIL",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
