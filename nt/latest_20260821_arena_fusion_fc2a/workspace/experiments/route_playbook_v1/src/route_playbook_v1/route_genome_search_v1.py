"""Reusable fixed-batch evaluator and diversity selector for Route Genome V1."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from time import perf_counter
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import ANIMAL_PRODUCT
from kaggriculture_jax.state import reset

from .event_bank import select_route_events_v1
from .route_genome_full_core_rollout_v1 import (
    initialize_route_genome_full_core_carry_v1,
)
from .route_genome_v1 import RouteGenomeSpecV1, stack_route_genomes_v1


_SCALAR_METRICS = (
    "cash",
    "productive_days",
    "target_sold_units",
    "min_cash",
    "max_hands",
    "max_land",
    "first_sale_step",
    "action_hash_a",
    "action_hash_b",
    "terminal_products",
    "placed_target_animals",
    "hard_failures",
    "hard_invalid_raw",
    "hard_resource_conflict",
    "hard_unexpected_noop",
    "hard_effect_mismatch",
    "hard_owner_inactive",
    "hard_deadline_missed",
    "hard_cross_episode",
    "hard_nonfinite",
    "done",
)


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bucket(value: float, edges: list[int]) -> int:
    return int(
        np.clip(
            np.searchsorted(edges, value, side="right") - 1,
            0,
            len(edges) - 2,
        )
    )


def evaluate_specs_v1(
    *,
    specs: list[RouteGenomeSpecV1],
    seeds: np.ndarray,
    chunk_candidates: int,
    event_seeds: np.ndarray,
    event_bank,
    tables,
    trace_bank,
    rollout,
    progress: Callable[[int, int, float, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Evaluate specs in one fixed episode-batch shape, padding only the last chunk."""

    if not specs:
        raise ValueError("specs must not be empty")
    seed_count = int(len(seeds))
    episode_batch = int(chunk_candidates * seed_count)
    count = len(specs)
    shape = (count, seed_count)
    arrays: dict[str, np.ndarray] = {
        "cash": np.zeros(shape, dtype=np.int32),
        "productive_days": np.zeros(shape, dtype=np.int32),
        "target_sold_units": np.zeros(shape, dtype=np.int32),
        "min_cash": np.zeros(shape, dtype=np.int32),
        "max_hands": np.zeros(shape, dtype=np.int16),
        "max_land": np.zeros(shape, dtype=np.int8),
        "first_sale_step": np.full(shape, -1, dtype=np.int16),
        "action_hash_a": np.zeros(shape, dtype=np.uint32),
        "action_hash_b": np.zeros(shape, dtype=np.uint32),
        "terminal_products": np.zeros(shape, dtype=np.int32),
        "placed_target_animals": np.zeros(shape, dtype=np.int16),
        "hard_failures": np.zeros(shape, dtype=np.int32),
        "hard_invalid_raw": np.zeros(shape, dtype=np.int32),
        "hard_resource_conflict": np.zeros(shape, dtype=np.int32),
        "hard_unexpected_noop": np.zeros(shape, dtype=np.int32),
        "hard_effect_mismatch": np.zeros(shape, dtype=np.int32),
        "hard_owner_inactive": np.zeros(shape, dtype=np.int32),
        "hard_deadline_missed": np.zeros(shape, dtype=np.int32),
        "hard_cross_episode": np.zeros(shape, dtype=np.int32),
        "hard_nonfinite": np.zeros(shape, dtype=np.int32),
        "done": np.zeros(shape, dtype=np.bool_),
    }
    elapsed: list[float] = []
    cache_sizes: list[int] = []
    for start in range(0, count, chunk_candidates):
        actual = min(chunk_candidates, count - start)
        chunk = list(specs[start : start + actual])
        if actual < chunk_candidates:
            chunk.extend(
                [replace(chunk[0], candidate_id=-1)]
                * (chunk_candidates - actual)
            )
        repeated = [row for row in chunk for _ in range(seed_count)]
        genome = stack_route_genomes_v1(repeated)
        episode_seeds = np.tile(seeds, chunk_candidates)
        events = select_route_events_v1(
            episode_seeds.tolist(), event_seeds, event_bank
        )
        states = jax.vmap(reset)(jnp.asarray(episode_seeds))
        carry = initialize_route_genome_full_core_carry_v1(states, events)
        started = perf_counter()
        result = rollout(carry, tables, trace_bank, genome)
        jax.block_until_ready(result.final_state.money)
        elapsed.append(perf_counter() - started)
        cache_sizes.append(int(rollout._cache_size()))
        if progress is not None:
            progress(
                min(start + actual, count),
                count,
                elapsed[-1],
                cache_sizes[-1],
            )

        metric = result.metrics
        flat = np.arange(episode_batch)
        target_product = np.asarray(ANIMAL_PRODUCT, dtype=np.int8)[
            np.asarray(genome.target_animal_id, dtype=np.int32)
        ]
        target_animal = np.asarray(genome.target_animal_id, dtype=np.int8)
        values = {
            "cash": np.asarray(result.final_state.money[:, 0]),
            "productive_days": np.asarray(metric.productive_animal_days),
            "target_sold_units": np.asarray(metric.sold_units)[flat, target_product],
            "min_cash": np.asarray(metric.min_cash),
            "max_hands": np.asarray(metric.max_hands),
            "max_land": np.asarray(metric.max_land),
            "first_sale_step": np.asarray(metric.first_target_sale_step),
            "action_hash_a": np.asarray(metric.action_hash_a),
            "action_hash_b": np.asarray(metric.action_hash_b),
            "terminal_products": np.asarray(result.terminal_product_inventory),
            "placed_target_animals": np.sum(
                np.asarray(result.final_state.tile_animal[:, 0])
                == target_animal[:, None, None],
                axis=(1, 2),
                dtype=np.int16,
            ),
            "hard_failures": np.asarray(result.hard_failures),
            "hard_invalid_raw": np.asarray(
                result.rule_carry.diagnostics.invalid_raw_action_count[:, 0]
            ),
            "hard_resource_conflict": np.asarray(
                result.rule_carry.diagnostics.internal_resource_conflict_count[:, 0]
            ),
            "hard_unexpected_noop": np.asarray(
                result.rule_carry.diagnostics.unexpected_silent_noop_count[:, 0]
            ),
            "hard_effect_mismatch": np.asarray(
                result.rule_carry.diagnostics.effect_mismatch_count[:, 0]
            ),
            "hard_owner_inactive": np.asarray(
                result.rule_carry.diagnostics.owner_inactive_count[:, 0]
            ),
            "hard_deadline_missed": np.asarray(
                result.rule_carry.diagnostics.deadline_missed_count[:, 0]
            ),
            "hard_cross_episode": np.asarray(
                result.rule_carry.diagnostics.cross_episode_task_contamination_count[:, 0]
            ),
            "hard_nonfinite": np.asarray(
                result.rule_carry.diagnostics.nan_or_inf_count[:, 0]
            ),
            "done": np.asarray(result.final_state.done),
        }
        rows = slice(start, start + actual)
        chunk_shape = (chunk_candidates, seed_count)
        for name in _SCALAR_METRICS:
            value = values[name]
            arrays[name][rows] = value.reshape(chunk_shape)[:actual]

    transitions = count * seed_count * 719
    seconds = float(sum(elapsed))
    perf = {
        "candidate_count": count,
        "seed_count": seed_count,
        "games": count * seed_count,
        "transitions": transitions,
        "chunk_candidates": chunk_candidates,
        "episode_batch": episode_batch,
        "chunk_seconds": elapsed,
        "seconds": seconds,
        "transitions_per_second": transitions / seconds,
        "jit_cache_sizes": cache_sizes,
        "jit_cache_stable_one": bool(
            cache_sizes and cache_sizes[0] == 1 and len(set(cache_sizes)) == 1
        ),
    }
    return arrays, perf


def summarize_specs_v1(
    specs: list[RouteGenomeSpecV1],
    arrays: dict[str, np.ndarray],
    productive_edges: list[int],
    sold_edges: list[int],
) -> list[dict[str, object]]:
    """Build stable, JSON-serializable per-candidate results."""

    rows: list[dict[str, object]] = []
    hard_names = (
        "hard_invalid_raw",
        "hard_resource_conflict",
        "hard_unexpected_noop",
        "hard_effect_mismatch",
        "hard_owner_inactive",
        "hard_deadline_missed",
        "hard_cross_episode",
        "hard_nonfinite",
    )
    for index, spec in enumerate(specs):
        behavior_vector = {
            "productive_days": arrays["productive_days"][index].astype(int).tolist(),
            "sold_units": arrays["target_sold_units"][index].astype(int).tolist(),
            "placed_animals": arrays["placed_target_animals"][index].astype(int).tolist(),
            "max_land": arrays["max_land"][index].astype(int).tolist(),
            "max_hands": arrays["max_hands"][index].astype(int).tolist(),
            "first_sale_day": (arrays["first_sale_step"][index] // 24).astype(int).tolist(),
            "min_cash_bucket_250": (arrays["min_cash"][index] // 250).astype(int).tolist(),
            "action_hash_a": arrays["action_hash_a"][index].astype(np.uint64).astype(int).tolist(),
            "action_hash_b": arrays["action_hash_b"][index].astype(np.uint64).astype(int).tolist(),
        }
        behavior_signature = _json_hash(behavior_vector)
        hard_zero = bool(np.all(arrays["hard_failures"][index] == 0))
        done_rate = float(np.mean(arrays["done"][index]))
        closure_rate = float(np.mean(arrays["terminal_products"][index] == 0))
        productive_median = float(np.median(arrays["productive_days"][index]))
        sold_median = float(np.median(arrays["target_sold_units"][index]))
        valid = bool(
            hard_zero
            and done_rate == 1.0
            and closure_rate == 1.0
            and productive_median > 0
            and sold_median > 0
            and np.min(arrays["placed_target_animals"][index]) >= 2
        )
        rows.append(
            {
                **spec.receipt(),
                "cash": _stats(arrays["cash"][index]),
                "per_seed_cash": arrays["cash"][index].astype(int).tolist(),
                "productive_animal_days": _stats(arrays["productive_days"][index]),
                "target_sold_units": _stats(arrays["target_sold_units"][index]),
                "placed_target_animals": _stats(arrays["placed_target_animals"][index]),
                "hard_failure_zero": hard_zero,
                "hard_failure_components": {
                    name.removeprefix("hard_"): arrays[name][index].astype(int).tolist()
                    for name in hard_names
                },
                "done_rate": done_rate,
                "terminal_product_closure_rate": closure_rate,
                "valid": valid,
                "productive_intensity_bucket": _bucket(productive_median, productive_edges),
                "sold_intensity_bucket": _bucket(sold_median, sold_edges),
                "behavior_signature": behavior_signature,
                "behavior_vector": behavior_vector,
            }
        )
    return rows


def _rank(row: dict[str, object]) -> tuple[float, float, float, int]:
    cash = row["cash"]
    assert isinstance(cash, dict)
    return (
        -float(cash["median"]),
        -float(cash["p10"]),
        -float(cash["p95"]),
        int(row["candidate_id"]),
    )


def select_diverse_survivors_v1(
    rows: Iterable[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    """Select unique valid rows while protecting source and intensity diversity."""

    valid = [row for row in rows if bool(row["valid"])]
    unique: dict[str, dict[str, object]] = {}
    for row in valid:
        key = str(row["behavior_signature"])
        old = unique.get(key)
        if old is None or _rank(row) < _rank(old):
            unique[key] = row
    candidates = list(unique.values())
    if len(candidates) < limit:
        return sorted(candidates, key=_rank)

    selected: list[dict[str, object]] = []
    selected_ids: set[tuple[str, int]] = set()

    def add(row: dict[str, object]) -> None:
        key = (str(row["family"]), int(row["candidate_id"]))
        if key not in selected_ids and len(selected) < limit:
            selected.append(row)
            selected_ids.add(key)

    # Protect both proposal directions first.
    for direction in ("GOLD_CONVERSION", "NATIVE_EXPANSION"):
        options = [row for row in candidates if row["direction"] == direction]
        if options:
            add(min(options, key=_rank))

    # Protect at least three strong but behaviorally different intensity cells.
    cells: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in candidates:
        cell = (
            int(row["productive_intensity_bucket"]),
            int(row["sold_intensity_bucket"]),
        )
        cells.setdefault(cell, []).append(row)
    cell_heads = sorted((min(group, key=_rank) for group in cells.values()), key=_rank)
    represented = {
        (int(row["productive_intensity_bucket"]), int(row["sold_intensity_bucket"]))
        for row in selected
    }
    for row in cell_heads:
        cell = (
            int(row["productive_intensity_bucket"]),
            int(row["sold_intensity_bucket"]),
        )
        if cell not in represented:
            add(row)
            represented.add(cell)
        if len(represented) >= min(3, len(cells)):
            break

    # Round-robin through direction × intensity strata, strongest strata first.
    strata: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in candidates:
        key = (
            str(row["direction"]),
            int(row["productive_intensity_bucket"]),
            int(row["sold_intensity_bucket"]),
        )
        strata.setdefault(key, []).append(row)
    for group in strata.values():
        group.sort(key=_rank)
    ordered_keys = sorted(strata, key=lambda key: _rank(strata[key][0]))
    depth = 0
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            group = strata[key]
            if depth < len(group):
                before = len(selected)
                add(group[depth])
                progressed |= len(selected) > before
                if len(selected) >= limit:
                    break
        if not progressed and all(depth >= len(group) - 1 for group in strata.values()):
            break
        depth += 1

    if len(selected) < limit:
        for row in sorted(candidates, key=_rank):
            add(row)
            if len(selected) >= limit:
                break
    return selected
