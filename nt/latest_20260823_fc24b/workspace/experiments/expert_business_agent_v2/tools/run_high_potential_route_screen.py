from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "experiments" / "route_playbook_v1" / "src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.constants import (  # noqa: E402
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from route_playbook_v1.trace_core import (  # noqa: E402
    CalibrationTraceV1,
    initialize_trace_player_carry_v1,
    skeleton_player_action_v1,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path: Path) -> CalibrationTraceV1:
    with np.load(path, allow_pickle=False) as data:
        return CalibrationTraceV1(
            *(jnp.asarray(data[field]) for field in CalibrationTraceV1._fields)
        )


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def null_action(states, player: int) -> Action:
    batch_size = states.step.shape[0]
    unit_count = jnp.sum(states.unit_active[:, player], axis=-1).astype(jnp.int8)
    return Action(
        unit_op=jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        unit_item=jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8),
        unit_amount=jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32),
        unit_count=unit_count,
        market_op=jnp.full(
            (batch_size, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
        ),
        market_item=jnp.full(
            (batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((batch_size,), dtype=jnp.int8),
    )


def make_rollout(bank: CalibrationTraceV1, tables, candidate_player: int):
    opponent_player = 1 - candidate_player

    @jax.jit
    def rollout(initial, events, route_ids):
        batch_size = initial.step.shape[0]

        def body(value, _):
            states, carry = value
            candidate, carry = skeleton_player_action_v1(
                states, tables, bank, route_ids, carry, candidate_player
            )
            passive = null_action(states, opponent_player)
            actions = (
                pair(candidate, passive)
                if candidate_player == 0
                else pair(passive, candidate)
            )
            states = batched_step_sync(states, actions, events, tables)
            return (states, carry), None

        result, _ = jax.lax.scan(
            body,
            (initial, initialize_trace_player_carry_v1(batch_size)),
            xs=None,
            length=719,
        )
        states, _ = result
        return states.money, states.done

    return rollout


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU passive-opponent screen of the deduplicated high-potential routes."
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=99001)
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays-output", type=Path, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU is required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path = args.bank.resolve()
    bank_receipt_path = args.bank_receipt.resolve()
    bank_receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    if sha256(bank_path) != bank_receipt["bank_sha256"]:
        raise RuntimeError("route bank hash does not match its receipt")
    route_count = int(bank_receipt["unique_routes"])
    seeds = np.arange(
        args.seed_start, args.seed_start + args.seed_count, dtype=np.int32
    )
    game_seeds = np.tile(seeds, route_count)
    route_ids = np.repeat(np.arange(route_count, dtype=np.int32), len(seeds))
    weed, shops = build_events_v1(game_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(game_seeds))
    bank = load_bank(bank_path)
    tables = load_tables()

    money_by_seat = np.zeros((2, route_count, len(seeds), 2), dtype=np.int64)
    done_by_seat = np.zeros((2, route_count, len(seeds)), dtype=np.bool_)
    timings = []
    for seat in (0, 1):
        rollout = make_rollout(bank, tables, seat)
        started = perf_counter()
        first_money, first_done = rollout(initial, events, jnp.asarray(route_ids))
        jax.block_until_ready(first_money)
        first_seconds = perf_counter() - started
        started = perf_counter()
        second_money, second_done = rollout(initial, events, jnp.asarray(route_ids))
        jax.block_until_ready(second_money)
        steady_seconds = perf_counter() - started
        if not np.array_equal(np.asarray(first_money), np.asarray(second_money)):
            raise RuntimeError(f"seat {seat}: repeated GPU rollout changed terminal money")
        if not np.array_equal(np.asarray(first_done), np.asarray(second_done)):
            raise RuntimeError(f"seat {seat}: repeated GPU rollout changed done flags")
        money_by_seat[seat] = np.asarray(second_money, dtype=np.int64).reshape(
            route_count, len(seeds), 2
        )
        done_by_seat[seat] = np.asarray(second_done, dtype=np.bool_).reshape(
            route_count, len(seeds)
        )
        games = route_count * len(seeds)
        timings.append(
            {
                "candidate_seat": seat,
                "games": games,
                "compile_and_first_seconds": first_seconds,
                "steady_seconds": steady_seconds,
                "steady_games_per_second": games / steady_seconds,
                "steady_transitions_per_second": games * 719 / steady_seconds,
            }
        )
        print(json.dumps(timings[-1], separators=(",", ":")), flush=True)

    aliases_by_route: dict[int, list[str]] = {index: [] for index in range(route_count)}
    for alias in bank_receipt["aliases"]:
        aliases_by_route[int(alias["route_id"])].append(
            f"{alias['family']}/{alias['route_name']}"
        )
    route_rows = []
    for route in bank_receipt["routes"]:
        route_id = int(route["route_id"])
        cash = np.concatenate(
            (money_by_seat[0, route_id, :, 0], money_by_seat[1, route_id, :, 1])
        )
        passive_cash = np.concatenate(
            (money_by_seat[0, route_id, :, 1], money_by_seat[1, route_id, :, 0])
        )
        route_rows.append(
            {
                "route_id": route_id,
                "canonical_name": route["canonical_name"],
                "action_sha256": route["action_sha256"],
                "aliases": aliases_by_route[route_id],
                "games": int(cash.size),
                "cash": stats(cash),
                "passive_cash_unique": sorted(set(passive_cash.astype(int).tolist())),
                "all_done": bool(np.all(done_by_seat[:, route_id])),
            }
        )
    ranking = sorted(
        route_rows, key=lambda row: (row["cash"]["mean"], row["cash"]["median"]), reverse=True
    )
    for rank, row in enumerate(ranking, start=1):
        row["passive_potential_rank"] = rank

    arrays_path = args.arrays_output.resolve()
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        seeds=seeds,
        money_by_seat=money_by_seat,
        done_by_seat=done_by_seat,
    )
    result = {
        "schema": "kaggriculture.high_potential_route_screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "routes": route_count,
        "games": int(2 * route_count * len(seeds)),
        "all_done": bool(np.all(done_by_seat)),
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "timings": timings,
        "truth_boundary": (
            "GPU passive-opponent route-potential screen. Raw production tapes are exact, "
            "but the shared screening controller does not yet reproduce every source-specific "
            "adaptive overlay; official Python remains the referee."
        ),
        "ranking": ranking,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if result["all_done"] else "FAIL",
                "games": result["games"],
                "top_routes": [
                    {
                        "rank": row["passive_potential_rank"],
                        "route": row["canonical_name"],
                        "mean": row["cash"]["mean"],
                        "max": row["cash"]["max"],
                    }
                    for row in ranking[:5]
                ],
            },
            ensure_ascii=True,
        )
    )
    return 0 if result["all_done"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
