#!/usr/bin/env python3
"""Batched GPU route screen against the compiled dynamic opponent bank.

This is an outcome screen, not a same-state counterfactual feature exporter.
Candidate routes are packed into the environment batch so one compiled rollout
evaluates several routes at once. Official Python 1.32.7 remains the referee.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_TOOLS = ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools"
sys.path.insert(0, str(SOURCE_TOOLS))

from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    Events,
    build_events_v1,
    build_router_arrays,
    load_bank,
    make_kind_rollout,
    make_rollout,
    reset,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, default=8)
    parser.add_argument(
        "--grouped-by-kind",
        action="store_true",
        help="Compile each opponent controller kind separately instead of one giant dispatcher.",
    )
    parser.add_argument(
        "--candidate-ids",
        type=int,
        nargs="+",
        help="Optional source skeleton ids to screen; defaults to every compiled candidate.",
    )
    parser.add_argument(
        "--opponent-ids",
        type=int,
        nargs="+",
        help=(
            "Optional compiled opponent ids to screen; defaults to every opponent. "
            "Use this to avoid compiling unrelated heavy controllers into a panel."
        ),
    )
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seed_count < 1:
        raise ValueError("seed-count must be positive")
    if not 1 <= args.candidate_chunk_size <= 32:
        raise ValueError("candidate-chunk-size must be between 1 and 32")

    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path = args.bank.resolve()
    receipt_path = args.bank_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bank = load_bank(bank_path)
    router = build_router_arrays(receipt)
    from kaggriculture_jax.state import load_tables  # imported after project paths

    tables = load_tables()
    compiled_candidate_ids = [int(value) for value in receipt["candidate_ids"]]
    selected_candidate_ids = (
        [int(value) for value in args.candidate_ids]
        if args.candidate_ids is not None
        else compiled_candidate_ids
    )
    unknown = sorted(set(selected_candidate_ids) - set(compiled_candidate_ids))
    if unknown:
        raise ValueError(f"candidate ids are not compiled candidates: {unknown}")
    if len(selected_candidate_ids) != len(set(selected_candidate_ids)):
        raise ValueError("candidate ids must be unique")
    candidate_source_ids = np.asarray(selected_candidate_ids, dtype=np.int32)
    compiled_opponent_ids = list(range(len(receipt["opponents"])))
    selected_opponent_ids = (
        [int(value) for value in args.opponent_ids]
        if args.opponent_ids is not None
        else compiled_opponent_ids
    )
    unknown_opponents = sorted(set(selected_opponent_ids) - set(compiled_opponent_ids))
    if unknown_opponents:
        raise ValueError(f"opponent ids are not compiled opponents: {unknown_opponents}")
    if len(selected_opponent_ids) != len(set(selected_opponent_ids)):
        raise ValueError("opponent ids must be unique")
    opponent_values = np.asarray(selected_opponent_ids, dtype=np.int16)
    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int32)
    candidate_count = len(candidate_source_ids)
    opponent_count = len(opponent_values)
    chunk_size = args.candidate_chunk_size

    weed_by_seed, shops_by_seed = build_events_v1(seeds.tolist())
    if args.grouped_by_kind:
        kind_values = np.asarray(jax.device_get(router.kind), dtype=np.int16)
        selected_kinds = kind_values[opponent_values]
        groups = [
            (int(kind_id), np.flatnonzero(selected_kinds == kind_id))
            for kind_id in np.unique(selected_kinds)
        ]
        grouped_rollouts = {
            (seat, kind_id): make_kind_rollout(bank, tables, router, seat, kind_id)
            for seat in (0, 1)
            for kind_id, _ in groups
        }
        rollouts = None
        initial = None
        events = None
        opponent_ids = None
        base_batch_size = 0
    else:
        groups = []
        grouped_rollouts = {}
        base_batch_seeds = np.tile(seeds, opponent_count)
        base_batch_opponents = np.repeat(opponent_values, len(seeds))
        base_batch_size = len(base_batch_seeds)
        batch_seeds = np.tile(base_batch_seeds, chunk_size)
        batch_opponents = np.tile(base_batch_opponents, chunk_size)
        events = Events(
            jnp.asarray(np.tile(weed_by_seed, (opponent_count * chunk_size, 1, 1))),
            jnp.asarray(np.tile(shops_by_seed, (opponent_count * chunk_size, 1, 1))),
        )
        initial = jax.vmap(reset)(jnp.asarray(batch_seeds))
        opponent_ids = jnp.asarray(batch_opponents, dtype=jnp.int32)
        rollouts = [make_rollout(bank, tables, router, seat) for seat in (0, 1)]

    wins = np.zeros((candidate_count, 2, 1, opponent_count, len(seeds)), dtype=np.bool_)
    margins = np.zeros_like(wins, dtype=np.int32)
    timings: list[dict[str, object]] = []
    all_done = True

    for start in range(0, candidate_count, chunk_size):
        actual = min(chunk_size, candidate_count - start)
        source_chunk = candidate_source_ids[start : start + actual]
        if actual < chunk_size:
            source_chunk = np.pad(source_chunk, (0, chunk_size - actual), mode="edge")
        if args.grouped_by_kind:
            for kind_id, group_positions in groups:
                group_opponents = opponent_values[group_positions]
                group_base_seeds = np.tile(seeds, len(group_positions))
                group_base_opponents = np.repeat(group_opponents, len(seeds))
                group_base_size = len(group_base_seeds)
                group_batch_seeds = np.tile(group_base_seeds, chunk_size)
                group_batch_opponents = np.tile(group_base_opponents, chunk_size)
                group_events = Events(
                    jnp.asarray(
                        np.tile(
                            weed_by_seed,
                            (len(group_positions) * chunk_size, 1, 1),
                        )
                    ),
                    jnp.asarray(
                        np.tile(
                            shops_by_seed,
                            (len(group_positions) * chunk_size, 1, 1),
                        )
                    ),
                )
                group_initial = jax.vmap(reset)(jnp.asarray(group_batch_seeds))
                group_opponent_ids = jnp.asarray(
                    group_batch_opponents, dtype=jnp.int32
                )
                group_candidate_ids = jnp.asarray(
                    np.repeat(source_chunk, group_base_size), dtype=jnp.int32
                )
                for seat in (0, 1):
                    print(
                        json.dumps(
                            {
                                "event": "group_start",
                                "candidate_start": start,
                                "candidate_count": actual,
                                "seat": seat,
                                "controller_kind": kind_id,
                                "opponent_ids": group_opponents.astype(int).tolist(),
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    rollout = grouped_rollouts[(seat, kind_id)]
                    started = perf_counter()
                    money, done, *_ = rollout(
                        group_initial,
                        group_events,
                        group_candidate_ids,
                        group_opponent_ids,
                    )
                    jax.block_until_ready(money)
                    elapsed = perf_counter() - started
                    money_np = np.asarray(money, dtype=np.int64).reshape(
                        chunk_size, len(group_positions), len(seeds), 2
                    )
                    done_np = np.asarray(done, dtype=bool).reshape(
                        chunk_size, len(group_positions), len(seeds)
                    )
                    own = money_np[:actual, :, :, seat]
                    other = money_np[:actual, :, :, 1 - seat]
                    for local_index, global_position in enumerate(group_positions):
                        wins[
                            start : start + actual,
                            seat,
                            0,
                            global_position,
                            :,
                        ] = own[:, local_index, :] > other[:, local_index, :]
                        margins[
                            start : start + actual,
                            seat,
                            0,
                            global_position,
                            :,
                        ] = own[:, local_index, :] - other[:, local_index, :]
                    all_done &= bool(np.all(done_np[:actual]))
                    timings.append(
                        {
                            "candidate_start": start,
                            "candidate_count": actual,
                            "padded_chunk_size": chunk_size,
                            "seat": seat,
                            "controller_kind": kind_id,
                            "opponent_ids": group_opponents.astype(int).tolist(),
                            "games": int(actual * group_base_size),
                            "executed_games_with_padding": int(
                                chunk_size * group_base_size
                            ),
                            "seconds": elapsed,
                            "useful_transitions_per_second": (
                                actual * group_base_size * 719 / elapsed
                            ),
                            "executed_transitions_per_second": (
                                chunk_size * group_base_size * 719 / elapsed
                            ),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "event": "group_complete",
                                "candidate_start": start,
                                "candidate_count": actual,
                                "seat": seat,
                                "controller_kind": kind_id,
                                "seconds": elapsed,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
        else:
            assert rollouts is not None
            assert initial is not None and events is not None and opponent_ids is not None
            candidate_ids = jnp.asarray(
                np.repeat(source_chunk, base_batch_size),
                dtype=jnp.int32,
            )
            for seat, rollout in enumerate(rollouts):
                started = perf_counter()
                money, done, *_ = rollout(initial, events, candidate_ids, opponent_ids)
                jax.block_until_ready(money)
                elapsed = perf_counter() - started
                money_np = np.asarray(money, dtype=np.int64).reshape(
                    chunk_size, opponent_count, len(seeds), 2
                )
                done_np = np.asarray(done, dtype=bool).reshape(
                    chunk_size, opponent_count, len(seeds)
                )
                own = money_np[:actual, :, :, seat]
                other = money_np[:actual, :, :, 1 - seat]
                wins[start : start + actual, seat, 0] = own > other
                margins[start : start + actual, seat, 0] = own - other
                all_done &= bool(np.all(done_np[:actual]))
                timings.append(
                    {
                        "candidate_start": start,
                        "candidate_count": actual,
                        "padded_chunk_size": chunk_size,
                        "seat": seat,
                        "controller_kind": "monolithic",
                        "games": int(actual * base_batch_size),
                        "executed_games_with_padding": int(chunk_size * base_batch_size),
                        "seconds": elapsed,
                        "useful_transitions_per_second": actual * base_batch_size * 719 / elapsed,
                        "executed_transitions_per_second": chunk_size * base_batch_size * 719 / elapsed,
                    }
                )

    output = args.outcomes_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        candidate_ids=candidate_source_ids.astype(np.int16),
        opponent_ids=opponent_values,
        seeds=seeds,
        wins=wins,
        margins=margins,
    )
    total_useful_games = sum(int(row["games"]) for row in timings)
    total_executed_games = sum(int(row["executed_games_with_padding"]) for row in timings)
    total_seconds = sum(float(row["seconds"]) for row in timings)
    result = {
        "schema": "kaggriculture-jax-dynamic-route-panel-batched-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_GPU_SCREEN" if all_done else "FAIL",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "seed_start": int(seeds[0]),
        "seed_count": len(seeds),
        "candidate_count": candidate_count,
        "opponent_count": opponent_count,
        "selected_opponent_ids": selected_opponent_ids,
        "selected_opponent_names": [
            str(receipt["opponents"][index]["name"])
            for index in selected_opponent_ids
        ],
        "dispatch_mode": "grouped_by_kind" if args.grouped_by_kind else "monolithic",
        "candidate_chunk_size": chunk_size,
        "game_count": int(wins.size),
        "all_done": all_done,
        "timings": timings,
        "compile_inclusive_useful_transitions_per_second": total_useful_games * 719 / total_seconds,
        "compile_inclusive_executed_transitions_per_second": total_executed_games * 719 / total_seconds,
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "bank_receipt": str(receipt_path),
        "bank_receipt_sha256": sha256(receipt_path),
        "outcomes_output": str(output),
        "outcomes_sha256": sha256(output),
        "truth_boundary": (
            "JAX GPU route screening only. Candidate routes are independent "
            "full-game rollouts, not same-state counterfactual continuations. "
            "Official Python 1.32.7 seat-swapped validation is mandatory."
        ),
    }
    receipt_output = args.receipt_output.resolve()
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_done else 2


if __name__ == "__main__":
    raise SystemExit(main())
