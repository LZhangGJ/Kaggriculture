#!/usr/bin/env python3
"""Vectorized FC22 feed-value parameter screen on exact JAX opponents.

Every parameter variant receives the same seeds, town shops, weeds, opponent,
and both seat orientations.  Variants are stacked into one environment batch,
so the policy and simulator compile once per opponent rather than once per
parameter choice.
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
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc22_feed_value_parameter_player_action_v1,
    initialize_fusion_champion_feed_value_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    load_bank,
)
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


DEFAULT_GRID = (
    "control:30:1:1,"
    "d12_r1:12:1:1,d11_r1:11:1:1,d10_r1:10:1:1,"
    "d9_r1:9:1:1,d8_r1:8:1:1,"
    "d10_r1p25:10:5:4,d9_r1p25:9:5:4,d8_r1p25:8:5:4,"
    "d10_r1p5:10:3:2,d9_r1p5:9:3:2,d8_r1p5:8:3:2,"
    "d10_r2:10:2:1"
)


def parse_grid(value: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for token in value.split(","):
        name, day, numerator, denominator = token.strip().split(":")
        row = {
            "name": name,
            "day_start": int(day),
            "value_numerator": int(numerator),
            "value_denominator": int(denominator),
        }
        if (
            not name
            or int(day) < 0
            or int(numerator) < 1
            or int(denominator) < 1
        ):
            raise ValueError(f"invalid grid row: {token}")
        rows.append(row)
    if not rows or len({str(row["name"]) for row in rows}) != len(rows):
        raise ValueError("grid names must be non-empty and unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=594001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--opponents", default="local_prt_v6")
    parser.add_argument("--grid", default=DEFAULT_GRID)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "experiments/fusion_champion_v1/receipts/"
            "fc22_feed_parameter_grid_seed594001_n128x2_v1.json"
        ),
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

    grid = parse_grid(args.grid)
    opponents = [value.strip() for value in args.opponents.split(",") if value.strip()]
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    variant_count = len(grid)
    seed_count = int(seeds.size)
    expanded_seeds = np.tile(seeds, variant_count)
    total_batch = int(expanded_seeds.size)
    day_start = jnp.asarray(
        np.repeat([int(row["day_start"]) for row in grid], seed_count),
        dtype=jnp.int16,
    )
    value_numerator = jnp.asarray(
        np.repeat([int(row["value_numerator"]) for row in grid], seed_count),
        dtype=jnp.int16,
    )
    value_denominator = jnp.asarray(
        np.repeat([int(row["value_denominator"]) for row in grid], seed_count),
        dtype=jnp.int16,
    )

    old_bank_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    old_receipt_path = (
        ROOT
        / "experiments/expert_business_agent_v2/receipts/"
        "jax_full37_mixed_exact_proxy_bank_v1.json"
    )
    latest_bank_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "latest_public8_route_bank_v1.npz"
    )
    runtime_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/"
        "latest_public8_runtime_tables_v1.npz"
    )
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(old_bank_path)
    latest_bank = load_bank(latest_bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    tables = load_tables()
    router = build_router_arrays(old_receipt)
    resources = {
        "old_bank": old_bank,
        "latest_bank": latest_bank,
        "runtime": runtime,
        "tables": tables,
        "router": router,
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    roster_names = [row["name"] for row in rr.ROSTER]
    roster_ids = {name: index for index, name in enumerate(roster_names)}
    if any(name not in roster_ids for name in opponents):
        raise ValueError("unknown opponent in grid panel")

    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(tables)
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc22_feed_value_parameter_player_action_v1(
                states,
                tables,
                latest_bank,
                old_bank,
                runtime,
                carry,
                day_start,
                value_numerator,
                value_denominator,
                player,
            )

        return policy

    candidate_policies = (make_candidate_policy(0), make_candidate_policy(1))
    result_rows: list[dict[str, object]] = []
    started = perf_counter()
    for opponent_name in opponents:
        opponent_id = roster_ids[opponent_name]
        orientation_money: list[np.ndarray] = []
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
            candidate_carry = initialize_fusion_champion_feed_value_carry_v1(
                total_batch
            )
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, total_batch, router
            )
            candidate_policy = candidate_policies[candidate_seat]
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                if candidate_seat == 0:
                    states = simulator(states, candidate_action, opponent_action, events)
                else:
                    states = simulator(states, opponent_action, candidate_action, events)
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError(f"{opponent_name}: not all games DONE")
            if (
                int(np.sum(np.asarray(terminal.hand_cap_hits)))
                or int(np.sum(np.asarray(terminal.market_loop_cap_hits)))
                or int(np.sum(np.asarray(terminal.price_lut_oob)))
            ):
                raise AssertionError(f"{opponent_name}: simulator safety counter hit")
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))

        first, second = orientation_money
        for variant_index, variant in enumerate(grid):
            start = variant_index * seed_count
            stop = start + seed_count
            candidate_cash = np.concatenate(
                (first[start:stop, 0], second[start:stop, 1])
            )
            opponent_cash = np.concatenate(
                (first[start:stop, 1], second[start:stop, 0])
            )
            margins = candidate_cash - opponent_cash
            per_game = []
            for seed_index, seed in enumerate(seeds.tolist()):
                for candidate_seat, money in ((0, first), (1, second)):
                    row_index = start + seed_index
                    candidate_value = int(money[row_index, candidate_seat])
                    opponent_value = int(money[row_index, 1 - candidate_seat])
                    margin = candidate_value - opponent_value
                    per_game.append(
                        {
                            "seed": int(seed),
                            "candidate_seat": candidate_seat,
                            "candidate_cash": candidate_value,
                            "opponent_cash": opponent_value,
                            "margin": margin,
                            "result": (
                                "win" if margin > 0 else "tie" if margin == 0 else "loss"
                            ),
                        }
                    )
            row = {
                "opponent": opponent_name,
                "variant": variant,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                "mean_candidate_cash": float(np.mean(candidate_cash)),
                "mean_opponent_cash": float(np.mean(opponent_cash)),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "candidate_seat0_wins": int(
                    np.sum((first[start:stop, 0] - first[start:stop, 1]) > 0)
                ),
                "candidate_seat1_wins": int(
                    np.sum((second[start:stop, 1] - second[start:stop, 0]) > 0)
                ),
                "per_game": per_game,
            }
            result_rows.append(row)
            print(
                json.dumps({key: value for key, value in row.items() if key != "per_game"}),
                flush=True,
            )

    payload = {
        "schema": "kaggriculture.fc22.feed_parameter_grid.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "seed_start": int(seeds[0]),
        "seed_count": seed_count,
        "games_per_variant_opponent": seed_count * 2,
        "seat_protocol": "same event seed in both candidate seats",
        "grid": grid,
        "opponents": opponents,
        "elapsed_seconds": perf_counter() - started,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
