#!/usr/bin/env python3
"""Independent, dual-seat GPU gate for public Kaito V48 versus frozen FC24B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.recent_public_20260825_gpu import (  # noqa: E402
    K48_BAKERY_CAPITAL,
    K48_DEFAULT,
    K48_FARM_FAST,
    K48_YARN_FAST,
    K48_YARN_SECOND,
    K48_YARN_THIRD,
    initialize_kaito_v48_carry_v1,
    kaito_v48_player_action_v1,
)


K48_BANK = ROOT / "experiments/public_recent_20260825/artifacts/recent_v48_route_bank_v1.npz"
LATEST_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
OLD_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"

ROUTE_NAMES = {
    K48_DEFAULT: "default",
    K48_YARN_FAST: "yarn_fast",
    K48_FARM_FAST: "farm_fast",
    K48_YARN_SECOND: "yarn_second",
    K48_YARN_THIRD: "yarn_third",
    K48_BAKERY_CAPITAL: "bakery_capital",
}


def pair(left: Action, right: Action) -> Action:
    return Action(
        *(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True))
    )


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054):
    if games <= 0:
        return [0.0, 1.0]
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games))
        / denominator
    )
    return [center - radius, center + radius]


def make_rollout(tables, k48_bank, latest_bank, old_bank, runtime, candidate_seat: int):
    fc_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events):
        batch_size = initial.step.shape[0]

        def body(value, _):
            states, k48_carry, fc_carry = value
            k48_action, k48_carry = kaito_v48_player_action_v1(
                states, runtime, k48_bank, k48_carry, candidate_seat
            )
            fc_action, fc_carry = fc24_terminal_crop_salvage_player_action_v1(
                states,
                tables,
                latest_bank,
                old_bank,
                runtime,
                fc_carry,
                fc_seat,
            )
            actions = (
                pair(k48_action, fc_action)
                if candidate_seat == 0
                else pair(fc_action, k48_action)
            )
            states = batched_step_sync(states, actions, events, tables)
            return (states, k48_carry, fc_carry), None

        initial_carry = (
            initial,
            initialize_kaito_v48_carry_v1(batch_size),
            initialize_fusion_champion_terminal_salvage_carry_v1(batch_size),
        )
        return jax.lax.scan(body, initial_carry, None, length=719)[0]

    return rollout


def summarize_route(route_id, candidate_cash, fc_cash):
    result = []
    for route, name in ROUTE_NAMES.items():
        mask = route_id == route
        games = int(np.sum(mask))
        if not games:
            continue
        wins = int(np.sum(candidate_cash[mask] > fc_cash[mask]))
        result.append(
            {
                "route_id": int(route),
                "route": name,
                "games": games,
                "wins": wins,
                "ties": int(np.sum(candidate_cash[mask] == fc_cash[mask])),
                "win_rate": wins / games,
                "wilson_95": wilson_interval(wins, games),
                "candidate_cash_mean": float(np.mean(candidate_cash[mask])),
                "fc24b_cash_mean": float(np.mean(fc_cash[mask])),
                "mean_margin": float(np.mean(candidate_cash[mask] - fc_cash[mask])),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=1_304_001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.batch <= 0:
        raise ValueError("batch must be positive")

    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = time.perf_counter()
    tables = load_tables()
    k48_bank = load_bank(K48_BANK)
    latest_bank = load_bank(LATEST_BANK)
    old_bank = load_bank(OLD_BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)

    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    rows = []

    for candidate_seat in (0, 1):
        rollout = make_rollout(
            tables, k48_bank, latest_bank, old_bank, runtime, candidate_seat
        )
        before = time.perf_counter()
        first = rollout(initial, events)
        jax.block_until_ready(first[0].money)
        compile_and_first = time.perf_counter() - before
        before = time.perf_counter()
        terminal, k48_carry, _fc_carry = jax.device_get(rollout(initial, events))
        steady = time.perf_counter() - before

        fc_seat = 1 - candidate_seat
        candidate_cash = np.asarray(terminal.money[:, candidate_seat], dtype=np.int64)
        fc_cash = np.asarray(terminal.money[:, fc_seat], dtype=np.int64)
        route_id = np.asarray(k48_carry.route_id, dtype=np.int32)
        wins = candidate_cash > fc_cash
        losses = candidate_cash < fc_cash
        ties = candidate_cash == fc_cash
        hard_counts = {
            "hand_cap_hits": int(np.sum(terminal.hand_cap_hits)),
            "market_loop_cap_hits": int(np.sum(terminal.market_loop_cap_hits)),
            "price_lut_oob": int(np.sum(terminal.price_lut_oob)),
        }
        loss_index = np.flatnonzero(losses)
        row = {
            "candidate_seat": candidate_seat,
            "games": args.batch,
            "wins": int(np.sum(wins)),
            "losses": int(np.sum(losses)),
            "ties": int(np.sum(ties)),
            "win_rate": float(np.mean(wins)),
            "wilson_95": wilson_interval(int(np.sum(wins)), args.batch),
            "candidate_cash_mean": float(np.mean(candidate_cash)),
            "fc24b_cash_mean": float(np.mean(fc_cash)),
            "mean_margin": float(np.mean(candidate_cash - fc_cash)),
            "all_done": bool(np.all(terminal.done)),
            "hard_counts": hard_counts,
            "compile_and_first_seconds": compile_and_first,
            "steady_seconds": steady,
            "transitions_per_second": args.batch * 719 / steady,
            "routes": summarize_route(route_id, candidate_cash, fc_cash),
            "losses_sample": [
                {
                    "seed": int(seeds[index]),
                    "route_id": int(route_id[index]),
                    "route": ROUTE_NAMES.get(int(route_id[index]), "unknown"),
                    "candidate_cash": int(candidate_cash[index]),
                    "fc24b_cash": int(fc_cash[index]),
                    "margin": int(candidate_cash[index] - fc_cash[index]),
                }
                for index in loss_index[:64]
            ],
        }
        rows.append(row)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "losses_sample"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    games = args.batch * 2
    wins = sum(row["wins"] for row in rows)
    losses = sum(row["losses"] for row in rows)
    aggregate = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": games - wins - losses,
        "win_rate": wins / games,
        "wilson_95": wilson_interval(wins, games),
        "mean_margin": sum(row["mean_margin"] for row in rows) / 2.0,
    }
    clean = all(
        row["all_done"] and all(value == 0 for value in row["hard_counts"].values())
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.public-recent-20260825.kaito-v48-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if clean else "FAIL",
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "independent_from_parity_and_small_screen": True,
        "aggregate": aggregate,
        "results": rows,
        "wall_seconds": time.perf_counter() - started,
        "strength_gate": {
            "threshold": 0.50,
            "point_estimate_pass": aggregate["win_rate"] > 0.50,
            "wilson_lower_bound_pass": aggregate["wilson_95"][0] > 0.50,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": payload["status"], "aggregate": aggregate, "output": str(args.output)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
