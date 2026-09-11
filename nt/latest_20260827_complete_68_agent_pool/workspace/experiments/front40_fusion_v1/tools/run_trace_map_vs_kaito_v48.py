"""Dual-seat GPU Arena between one trace-map candidate and exact Kaito V48."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/front40_fusion_v1/tools",
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_FIRST_SHOP_MAP  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.recent_public_20260825_gpu import (  # noqa: E402
    initialize_kaito_v48_carry_v1,
    kaito_v48_player_action_v1,
)


K48_BANK = ROOT / "experiments/public_recent_20260825/artifacts/recent_v48_route_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"


def make_rollout(tables, trace_bank, route_map, k48_bank, runtime, trace_seat: int):
    k48_seat = 1 - trace_seat

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]

        def body(value, _):
            states, trace_carry, k48_carry = value
            k48_action, k48_carry = kaito_v48_player_action_v1(
                states, runtime, k48_bank, k48_carry, k48_seat
            )
            states, trace_carry, _diagnostics, _joint = hasegawa_step_with_external_v3(
                states,
                trace_carry,
                trace_bank,
                k48_action,
                trace_seat,
                events,
                tables,
                ROUTER_FIRST_SHOP_MAP,
                route_map,
                15_000,
                5_000,
                2,
            )
            return (states, trace_carry, k48_carry), None

        initial_value = (
            initial,
            initialize_hasegawa_carry_v3(batch, trace_bank.bootstrap_route_id),
            initialize_kaito_v48_carry_v1(batch),
        )
        return jax.lax.scan(body, initial_value, None, length=719)[0]

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--route-map", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=1_515_001)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    route_ids = np.asarray(
        json.loads(args.route_map.read_text(encoding="utf-8"))["route_ids"], np.int16
    )
    if route_ids.shape != (8,) or np.any(route_ids < 0) or np.any(route_ids >= trace_bank.source_reward.shape[0]):
        raise ValueError(f"invalid route map: {route_ids.tolist()}")
    route_map = jnp.asarray(route_ids)
    k48_bank = load_bank(K48_BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    rows = []
    for trace_seat in (0, 1):
        rollout = make_rollout(
            tables, trace_bank, route_map, k48_bank, runtime, trace_seat
        )
        started = perf_counter()
        terminal, trace_carry, k48_carry = jax.device_get(rollout(initial, events))
        elapsed = perf_counter() - started
        own = np.asarray(terminal.money[:, trace_seat], np.int64)
        rival = np.asarray(terminal.money[:, 1 - trace_seat], np.int64)
        margin = own - rival
        row = {
            "candidate_seat": trace_seat,
            "games": args.batch,
            "wins": int(np.sum(margin > 0)),
            "ties": int(np.sum(margin == 0)),
            "losses": int(np.sum(margin < 0)),
            "win_rate": float(np.mean(margin > 0)),
            "mean_margin": float(np.mean(margin)),
            "candidate_cash_mean": float(np.mean(own)),
            "kaito_v48_cash_mean": float(np.mean(rival)),
            "all_done": bool(np.all(terminal.done)),
            "hard_counts": {
                "hand_cap_hits": int(np.sum(terminal.hand_cap_hits)),
                "market_loop_cap_hits": int(np.sum(terminal.market_loop_cap_hits)),
                "price_lut_oob": int(np.sum(terminal.price_lut_oob)),
                "trace_hard_counter": int(np.sum(trace_carry.hard_counter_total)),
            },
            "invalid_intent_total": int(np.sum(trace_carry.invalid_intent_total)),
            "resync_total": int(np.sum(trace_carry.resync_total)),
            "kaito_route_counts": {
                str(route): int(np.sum(np.asarray(k48_carry.route_id) == route))
                for route in np.unique(np.asarray(k48_carry.route_id)).tolist()
            },
            "elapsed_seconds": elapsed,
            "transitions_per_second": args.batch * 719 / elapsed,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    games = 2 * args.batch
    hard_clean = all(
        row["all_done"] and all(value == 0 for value in row["hard_counts"].values())
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-map-vs-kaito-v48.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if hard_clean else "FAIL",
        "device": str(jax.devices()[0]),
        "candidate": args.candidate_name,
        "trace_bank": str(args.trace_bank.resolve()),
        "route_map": str(args.route_map.resolve()),
        "opponent": "public_kaito_v48_exact_jax",
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "aggregate": {
            "games": games,
            "wins": sum(row["wins"] for row in rows),
            "ties": sum(row["ties"] for row in rows),
            "losses": sum(row["losses"] for row in rows),
            "win_rate": sum(row["wins"] for row in rows) / games,
            "mean_margin": sum(row["mean_margin"] for row in rows) / 2,
            "invalid_intent_per_game": sum(row["invalid_intent_total"] for row in rows) / games,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"]}, ensure_ascii=False))
    return 0 if hard_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
