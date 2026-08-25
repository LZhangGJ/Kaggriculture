"""Reusable one-step-JIT Arena for many padded trace candidates vs FC24B.

The policy/environment transition is compiled once per seat.  A Python loop
dispatches the compiled one-step kernel 719 times, avoiding a very large XLA
``scan`` that took several minutes to compile for every differently sized
Replay bank.
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
from hasegawa_jax_v3.agent import ROUTER_TWO_SHOP_COMPATIBLE_MAP  # noqa: E402
from hasegawa_jax_v3.trace_bank import HasegawaTraceBankV3  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def pad_bank(bank: HasegawaTraceBankV3, route_capacity: int) -> HasegawaTraceBankV3:
    route_count = int(bank.unit_op.shape[0])
    if route_count > route_capacity:
        raise ValueError(f"route count {route_count} exceeds capacity {route_capacity}")
    if route_count == route_capacity:
        return bank
    padded = []
    for field, value in zip(HasegawaTraceBankV3._fields, bank):
        if field == "bootstrap_route_id":
            padded.append(value)
            continue
        pad = route_capacity - route_count
        # Repeat route zero for unused entries.  No route map can select them;
        # repeating valid values avoids introducing invalid enum sentinels.
        padded.append(jnp.concatenate((value, jnp.repeat(value[:1], pad, axis=0)), axis=0))
    return HasegawaTraceBankV3(*padded)


def make_step(trace_player, tables, latest_bank, old_bank, runtime):
    fc_player = 1 - trace_player

    @jax.jit
    def one_step(
        states, trace_carry, fc_carry, events, trace_bank, route_map,
        second_route_map
    ):
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
            ROUTER_TWO_SHOP_COMPATIBLE_MAP,
            route_map,
            15000,
            5000,
            2,
            second_shop_route_map=second_route_map,
        )
        return states, trace_carry, fc_carry

    return one_step


def rollout(step_fn, initial, events, bank, route_map, second_route_map):
    batch = initial.step.shape[0]
    states = initial
    trace_carry = initialize_hasegawa_carry_v3(batch, bank.bootstrap_route_id)
    fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
    for _ in range(719):
        states, trace_carry, fc_carry = step_fn(
            states, trace_carry, fc_carry, events, bank, route_map,
            second_route_map
        )
    jax.block_until_ready(states.money)
    return jax.device_get((states.money, states.done, trace_carry))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=161001)
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

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    tables = load_tables()
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    step_fns = {seat: make_step(seat, tables, latest_bank, old_bank, runtime) for seat in (0, 1)}

    rows = []
    compile_seconds_by_seat: dict[str, float] = {}
    for candidate_index, spec in enumerate(candidates):
        bank = pad_bank(load_hasegawa_trace_bank_v3(ROOT / spec["bank"]), args.route_capacity)
        mapping = json.loads((ROOT / spec["route_map"]).read_text(encoding="utf-8"))
        route_map = jnp.asarray(mapping["route_ids"], dtype=jnp.int16)
        if route_map.shape not in ((8,), (2, 8)):
            raise ValueError(f"{spec['name']} route map shape {route_map.shape}")
        if spec.get("second_route_map"):
            second_payload = json.loads(
                (ROOT / spec["second_route_map"]).read_text(encoding="utf-8")
            )
            second_route_map = jnp.asarray(
                second_payload[
                    "second_route_ids_by_first_shop"
                    if "second_route_ids_by_first_shop" in second_payload
                    else "second_route_ids"
                ],
                dtype=jnp.int16,
            )
        else:
            second_route_map = jnp.repeat(
                jnp.arange(args.route_capacity, dtype=jnp.int16)[:, None], 8, axis=1
            )
        if second_route_map.shape not in ((args.route_capacity, 8), (8, 8)):
            raise ValueError(
                f"{spec['name']} second route map shape {second_route_map.shape}"
            )
        for trace_player in (0, 1):
            start = time.perf_counter()
            money, done, carry = rollout(
                step_fns[trace_player], initial, events, bank, route_map,
                second_route_map
            )
            elapsed = time.perf_counter() - start
            if candidate_index == 0:
                compile_seconds_by_seat[str(trace_player)] = elapsed
            fc_player = 1 - trace_player
            candidate_cash = money[:, trace_player]
            fc_cash = money[:, fc_player]
            wins = candidate_cash > fc_cash
            row = {
                "candidate": spec["name"],
                "family": spec["family"],
                "candidate_seat": trace_player,
                "games": args.batch,
                "wins": int(np.sum(wins)),
                "win_rate": float(np.mean(wins)),
                "candidate_cash_mean": float(np.mean(candidate_cash)),
                "fc24b_cash_mean": float(np.mean(fc_cash)),
                "mean_margin": float(np.mean(candidate_cash - fc_cash)),
                "all_done": bool(np.all(done)),
                "route_switch_total": int(np.sum(carry.route_switch_total)),
                "hard_counter_total": int(np.sum(carry.hard_counter_total)),
                "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
                "resync_total": int(np.sum(carry.resync_total)),
                "elapsed_seconds": elapsed,
                "transitions_per_second": args.batch * 719 / elapsed,
                "candidate_cash": candidate_cash.astype(int).tolist(),
                "fc24b_cash": fc_cash.astype(int).tolist(),
                "final_route": carry.branch_id.astype(int).tolist(),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in {"candidate_cash", "fc24b_cash", "final_route"}},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    aggregates = []
    for spec in candidates:
        subset = [row for row in rows if row["candidate"] == spec["name"]]
        games = sum(row["games"] for row in subset)
        aggregates.append(
            {
                "candidate": spec["name"],
                "family": spec["family"],
                "games": games,
                "wins": sum(row["wins"] for row in subset),
                "win_rate": sum(row["wins"] for row in subset) / games,
                "mean_margin": sum(row["mean_margin"] * row["games"] for row in subset) / games,
                "invalid_intent_per_game": sum(row["invalid_intent_total"] for row in subset) / games,
                "passed_direction_gate": (
                    sum(row["wins"] for row in subset) / games > 0.5
                    and all(row["all_done"] and row["hard_counter_total"] == 0 for row in subset)
                ),
            }
        )
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-candidates-vs-fc24b-stepwise.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "manifest": str(args.manifest),
        "route_capacity": args.route_capacity,
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "compilation_cache": str(args.compilation_cache.resolve()),
        "compile_plus_first_candidate_seconds_by_seat": compile_seconds_by_seat,
        "aggregates": aggregates,
        "results": rows,
        "status": "PASS" if all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "aggregates": aggregates}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
