"""Screen observable daily Replay-router weights in one batched GPU rollout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
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
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_DAY_PUBLIC_STATE  # noqa: E402
from hasegawa_jax_v3.trace_bank import HasegawaTraceBankV3  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


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
                    (value, jnp.repeat(value[:1], capacity - route_count, axis=0)), axis=0
                )
            )
    return HasegawaTraceBankV3(*values)


def make_rollout(tables, latest_bank, old_bank, runtime, trace_player: int):
    fc_player = 1 - trace_player

    @jax.jit
    def rollout(initial, events, trace_bank, prefix_scale, reward_scale):
        batch = initial.step.shape[0]

        def body(value, _):
            states, trace_carry, fc_carry = value
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
                ROUTER_DAY_PUBLIC_STATE,
                None,
                prefix_score_scale=prefix_scale,
                source_reward_scale=reward_scale,
            )
            return (states, trace_carry, fc_carry), None

        trace_carry = initialize_hasegawa_carry_v3(batch, trace_bank.bootstrap_route_id)
        fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
        (terminal, trace_carry, _), _ = jax.lax.scan(
            body, (initial, trace_carry, fc_carry), None, length=719
        )
        return terminal.money, terminal.done, trace_carry

    return rollout


def parse_ints(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value]
    if not values:
        raise ValueError("weight grid cannot be empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--seeds-per-config", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=264001)
    parser.add_argument("--prefix-scales", default="0,100,1000,10000,100000,1000000")
    parser.add_argument("--reward-scales", default="0,1")
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

    configs = list(product(parse_ints(args.prefix_scales), parse_ints(args.reward_scales)))
    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds_per_config, dtype=np.int32)
    expanded_seeds = np.tile(base_seeds, len(configs))
    prefix_scale = np.repeat(np.asarray([value[0] for value in configs], np.int32), args.seeds_per_config)
    reward_scale = np.repeat(np.asarray([value[1] for value in configs], np.int32), args.seeds_per_config)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(expanded_seeds))

    trace_bank = pad_bank(load_hasegawa_trace_bank_v3(args.trace_bank), args.route_capacity)
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()

    rows = []
    timings = []
    for seat in (0, 1):
        rollout = make_rollout(tables, latest_bank, old_bank, runtime, seat)
        tick = time.perf_counter()
        result = rollout(
            initial,
            events,
            trace_bank,
            jnp.asarray(prefix_scale),
            jnp.asarray(reward_scale),
        )
        jax.block_until_ready(result[0])
        elapsed = time.perf_counter() - tick
        money, done, carry = jax.device_get(result)
        timings.append({"seat": seat, "seconds": elapsed})
        if not bool(np.all(done)):
            raise RuntimeError(f"seat {seat} did not finish all games")
        for config_index, (prefix_value, reward_value) in enumerate(configs):
            start = config_index * args.seeds_per_config
            end = start + args.seeds_per_config
            candidate_cash = money[start:end, seat].astype(np.int64)
            fc_cash = money[start:end, 1 - seat].astype(np.int64)
            margin = candidate_cash - fc_cash
            rows.append(
                {
                    "prefix_score_scale": prefix_value,
                    "source_reward_scale": reward_value,
                    "candidate_seat": seat,
                    "games": args.seeds_per_config,
                    "wins": int(np.sum(margin > 0)),
                    "win_rate": float(np.mean(margin > 0)),
                    "mean_margin": float(np.mean(margin)),
                    "invalid_intent_per_game": float(np.mean(carry.invalid_intent_total[start:end])),
                    "route_switch_per_game": float(np.mean(carry.route_switch_total[start:end])),
                    "hard_counter_total": int(np.sum(carry.hard_counter_total[start:end])),
                }
            )
        print(json.dumps({"seat": seat, "seconds": elapsed, "done": True}), flush=True)

    aggregates = []
    for prefix_value, reward_value in configs:
        subset = [
            row
            for row in rows
            if row["prefix_score_scale"] == prefix_value
            and row["source_reward_scale"] == reward_value
        ]
        games = sum(row["games"] for row in subset)
        aggregates.append(
            {
                "prefix_score_scale": prefix_value,
                "source_reward_scale": reward_value,
                "games": games,
                "wins": sum(row["wins"] for row in subset),
                "win_rate": sum(row["wins"] for row in subset) / games,
                "mean_margin": sum(row["mean_margin"] * row["games"] for row in subset) / games,
                "invalid_intent_per_game": sum(
                    row["invalid_intent_per_game"] * row["games"] for row in subset
                ) / games,
                "route_switch_per_game": sum(
                    row["route_switch_per_game"] * row["games"] for row in subset
                ) / games,
                "hard_counter_total": sum(row["hard_counter_total"] for row in subset),
            }
        )
    aggregates.sort(key=lambda row: (-row["win_rate"], -row["mean_margin"]))
    payload = {
        "schema": "kaggriculture.front40_fusion.day-router-weight-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["hard_counter_total"] == 0 for row in rows) else "FAIL",
        "device": str(jax.devices()[0]),
        "trace_bank": str(args.trace_bank),
        "seed_start": args.seed_start,
        "seeds_per_config": args.seeds_per_config,
        "configs": len(configs),
        "games": len(configs) * args.seeds_per_config * 2,
        "timings": timings,
        "aggregates": aggregates,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "top": aggregates[:5]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
