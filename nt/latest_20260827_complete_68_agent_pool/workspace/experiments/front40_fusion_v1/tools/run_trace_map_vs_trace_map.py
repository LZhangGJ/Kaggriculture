"""Dual-seat full-season GPU Arena between observable trace-map policies.

Each controller independently chooses and repairs its action from the same
pre-step public state.  The two actions are then paired and resolved exactly
once by the frozen JAX simulator.  This avoids the common error of advancing
one controller before asking the other controller to act.
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

from hasegawa_jax_v2.agent import _pair  # noqa: E402
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
)
from hasegawa_jax_v3.agent import ROUTER_TWO_SHOP_COMPATIBLE_MAP  # noqa: E402
from kaggriculture_jax.constants import (  # noqa: E402
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_trace_backbones_vs_old28_batched import load_candidate  # noqa: E402


def _pass_action(batch: int) -> Action:
    return Action(
        unit_op=jnp.full((batch, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        unit_item=jnp.full((batch, MAX_UNITS), -1, dtype=jnp.int8),
        unit_amount=jnp.ones((batch, MAX_UNITS), dtype=jnp.int32),
        unit_count=jnp.ones((batch,), dtype=jnp.int8),
        market_op=jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int8),
        market_item=jnp.full((batch, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        market_amount=jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int32),
        market_count=jnp.zeros((batch,), dtype=jnp.int8),
    )


def _controlled_action(joint: Action, player: int) -> Action:
    return Action(*(field[:, player] for field in joint))


def make_rollout(
    tables,
    candidate,
    opponent,
    candidate_seat: int,
):
    candidate_bank, candidate_first, candidate_second, candidate_compact = candidate
    opponent_bank, opponent_first, opponent_second, opponent_compact = opponent
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        dummy = _pass_action(batch)

        def body(value, _):
            states, candidate_carry, opponent_carry = value
            _, candidate_carry, candidate_diag, candidate_joint = (
                hasegawa_step_with_external_v3(
                    states,
                    candidate_carry,
                    candidate_bank,
                    dummy,
                    candidate_seat,
                    events,
                    tables,
                    ROUTER_TWO_SHOP_COMPATIBLE_MAP,
                    candidate_first,
                    15_000,
                    5_000,
                    2,
                    second_shop_route_map=candidate_second,
                    second_shop_map_is_compact=candidate_compact,
                )
            )
            _, opponent_carry, opponent_diag, opponent_joint = (
                hasegawa_step_with_external_v3(
                    states,
                    opponent_carry,
                    opponent_bank,
                    dummy,
                    opponent_seat,
                    events,
                    tables,
                    ROUTER_TWO_SHOP_COMPATIBLE_MAP,
                    opponent_first,
                    15_000,
                    5_000,
                    2,
                    second_shop_route_map=opponent_second,
                    second_shop_map_is_compact=opponent_compact,
                )
            )
            candidate_action = _controlled_action(candidate_joint, candidate_seat)
            opponent_action = _controlled_action(opponent_joint, opponent_seat)
            actual_joint = _pair(candidate_action, opponent_action, candidate_seat)
            states = batched_step_sync(states, actual_joint, events, tables)
            diagnostics = (
                candidate_diag.route_id,
                opponent_diag.route_id,
                candidate_diag.invalid_unit_intent_count,
                opponent_diag.invalid_unit_intent_count,
            )
            return (states, candidate_carry, opponent_carry), diagnostics

        initial_value = (
            initial,
            initialize_hasegawa_carry_v3(batch, candidate_bank.bootstrap_route_id),
            initialize_hasegawa_carry_v3(batch, opponent_bank.bootstrap_route_id),
        )
        return jax.lax.scan(body, initial_value, None, length=719)

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--opponent-names", required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=1_516_001)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = {row["name"]: row for row in manifest["candidates"]}
    opponent_names = [value.strip() for value in args.opponent_names.split(",") if value.strip()]
    missing = {args.candidate_name, *opponent_names} - set(specs)
    if missing:
        raise ValueError(f"unknown candidates: {sorted(missing)}")

    candidate_loaded = load_candidate(specs[args.candidate_name], args.route_capacity)
    candidate = candidate_loaded[:4]
    opponents = {
        name: load_candidate(specs[name], args.route_capacity)[:4]
        for name in opponent_names
    }
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    rows = []
    for opponent_name, opponent in opponents.items():
        for candidate_seat in (0, 1):
            rollout = make_rollout(tables, candidate, opponent, candidate_seat)
            started = perf_counter()
            (terminal, candidate_carry, opponent_carry), diagnostics = jax.device_get(
                rollout(initial, events)
            )
            elapsed = perf_counter() - started
            own = np.asarray(terminal.money[:, candidate_seat], dtype=np.int64)
            rival = np.asarray(terminal.money[:, 1 - candidate_seat], dtype=np.int64)
            margin = own - rival
            row = {
                "opponent": opponent_name,
                "candidate_seat": candidate_seat,
                "games": args.batch,
                "wins": int(np.sum(margin > 0)),
                "ties": int(np.sum(margin == 0)),
                "losses": int(np.sum(margin < 0)),
                "win_rate": float(np.mean(margin > 0)),
                "mean_margin": float(np.mean(margin)),
                "candidate_cash_mean": float(np.mean(own)),
                "opponent_cash_mean": float(np.mean(rival)),
                "all_done": bool(np.all(terminal.done)),
                "hard_counts": {
                    "hand_cap_hits": int(np.sum(terminal.hand_cap_hits)),
                    "market_loop_cap_hits": int(np.sum(terminal.market_loop_cap_hits)),
                    "price_lut_oob": int(np.sum(terminal.price_lut_oob)),
                },
                "candidate_invalid_intent_total": int(np.sum(diagnostics[2])),
                "opponent_invalid_intent_total": int(np.sum(diagnostics[3])),
                "candidate_resync_total": int(np.sum(candidate_carry.resync_total)),
                "opponent_resync_total": int(np.sum(opponent_carry.resync_total)),
                "candidate_final_route_counts": {
                    str(route): int(np.sum(np.asarray(diagnostics[0][-1]) == route))
                    for route in np.unique(np.asarray(diagnostics[0][-1])).tolist()
                },
                "opponent_final_route_counts": {
                    str(route): int(np.sum(np.asarray(diagnostics[1][-1]) == route))
                    for route in np.unique(np.asarray(diagnostics[1][-1])).tolist()
                },
                "elapsed_seconds": elapsed,
                "transitions_per_second": args.batch * 719 / elapsed,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    aggregates = []
    for opponent_name in opponent_names:
        selected = [row for row in rows if row["opponent"] == opponent_name]
        games = sum(row["games"] for row in selected)
        aggregates.append(
            {
                "opponent": opponent_name,
                "games": games,
                "wins": sum(row["wins"] for row in selected),
                "ties": sum(row["ties"] for row in selected),
                "losses": sum(row["losses"] for row in selected),
                "win_rate": sum(row["wins"] for row in selected) / games,
                "mean_margin": float(np.mean([row["mean_margin"] for row in selected])),
            }
        )
    hard_clean = all(
        row["all_done"] and all(value == 0 for value in row["hard_counts"].values())
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-map-vs-trace-map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if hard_clean else "FAIL",
        "device": str(jax.devices()[0]),
        "candidate": args.candidate_name,
        "opponents": opponent_names,
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "route_capacity": args.route_capacity,
        "aggregates": aggregates,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregates": aggregates}, ensure_ascii=False))
    return 0 if hard_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
