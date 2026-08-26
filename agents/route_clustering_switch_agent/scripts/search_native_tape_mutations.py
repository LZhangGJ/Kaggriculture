#!/usr/bin/env python3
"""Screen genuine intent-tape mutations against the current switch policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from fast_kaggriculture import native_tree_predict
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if separator:
        return tuple(range(int(start), int(stop)))
    return tuple(int(part) for part in value.split(",") if part.strip())


def _quantity(order: Sequence[Any]) -> int:
    if len(order) < 3:
        return 1
    try:
        return max(1, int(order[2]))
    except (TypeError, ValueError):
        return 1


def _matches(
    order: Sequence[Any], operation: str, item: str, start: int, stop: int, step: int
) -> bool:
    return (
        start <= step < stop
        and bool(order)
        and str(order[0]) == operation
        and (str(order[1]) if len(order) >= 2 else "") == item
    )


def _scale_quantity(
    tape: Sequence[Mapping[str, Any]], gene: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tape)
    numerator = int(gene["numerator"])
    denominator = int(gene["denominator"])
    for step, action in enumerate(result):
        for order in action.get("market", []) or []:
            if not _matches(
                order, str(gene["operation"]), str(gene["item"]),
                int(gene["start"]), int(gene["stop"]), step,
            ):
                continue
            quantity = max(1, int(round(_quantity(order) * numerator / denominator)))
            while len(order) < 3:
                order.append(1)
            order[2] = quantity
    return result


def _shift_market(
    tape: Sequence[Mapping[str, Any]], gene: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tape)
    delta = int(gene["delta"])
    moves: list[tuple[int, list[Any]]] = []
    for step, action in enumerate(result):
        market = action.get("market", []) or []
        kept = []
        for order in market:
            if _matches(
                order, str(gene["operation"]), str(gene["item"]),
                int(gene["start"]), int(gene["stop"]), step,
            ):
                target = min(len(result) - 1, max(0, step + delta))
                # Market shifts remain inside the source day.  This changes
                # within-day timing without silently changing the macro phase.
                target = min(step // 24 * 24 + 23, max(step // 24 * 24, target))
                moves.append((target, list(order)))
            else:
                kept.append(order)
        action["market"] = kept
    for target, order in moves:
        candidates = sorted(
            range(target // 24 * 24, min(len(result), target // 24 * 24 + 24)),
            key=lambda value: (abs(value - target), value),
        )
        destination = next(
            (value for value in candidates if len(result[value].get("market", []) or []) < 10),
            None,
        )
        if destination is not None:
            result[destination].setdefault("market", []).append(order)
    return result


def _reduce_hires(
    tape: Sequence[Mapping[str, Any]], gene: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tape)
    remove = int(gene["remove_per_day"])
    for day_start in range(int(gene["start"]), int(gene["stop"]), 24):
        locations = []
        for step in range(day_start, min(day_start + 24, len(result))):
            for index, order in enumerate(result[step].get("market", []) or []):
                if order and str(order[0]) == "HIRE":
                    locations.append((step, index))
        for step, index in sorted(locations[-remove:], reverse=True):
            del result[step]["market"][index]
    return result


def _market_profile(
    tape: Sequence[Mapping[str, Any]], gene: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply independently evolvable timing/quantity values for each day."""

    result = copy.deepcopy(tape)
    shifts = {int(key): int(value) for key, value in gene.get("daily_shift", {}).items()}
    quantity_deltas = {
        int(key): int(value) for key, value in gene.get("daily_quantity_delta", {}).items()
    }
    operation, item = str(gene["operation"]), str(gene["item"])
    moves: list[tuple[int, list[Any]]] = []
    for step, action in enumerate(result):
        day = step // 24
        shift = shifts.get(day, 0)
        quantity_delta = quantity_deltas.get(day, 0)
        if shift == 0 and quantity_delta == 0:
            continue
        kept = []
        for order in action.get("market", []) or []:
            order_item = str(order[1]) if len(order) >= 2 else ""
            if not order or str(order[0]) != operation or order_item != item:
                kept.append(order)
                continue
            changed = list(order)
            if quantity_delta:
                while len(changed) < 3:
                    changed.append(1)
                changed[2] = max(1, _quantity(changed) + quantity_delta)
            if shift:
                start = day * 24
                target = min(start + 23, max(start, step + shift))
                moves.append((target, changed))
            else:
                kept.append(changed)
        action["market"] = kept
    for target, order in moves:
        day_start = target // 24 * 24
        candidates = sorted(
            range(day_start, min(len(result), day_start + 24)),
            key=lambda value: (abs(value - target), value),
        )
        destination = next(
            (value for value in candidates if len(result[value].get("market", []) or []) < 10),
            None,
        )
        if destination is not None:
            result[destination].setdefault("market", []).append(order)
    return result


def _apply_genes(
    tape: Sequence[Mapping[str, Any]], genes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tape)
    for gene in genes:
        operator = str(gene["operator"])
        if operator == "scale_quantity":
            result = _scale_quantity(result, gene)
        elif operator == "shift_market":
            result = _shift_market(result, gene)
        elif operator == "reduce_hires":
            result = _reduce_hires(result, gene)
        elif operator == "market_profile":
            result = _market_profile(result, gene)
        else:
            raise ValueError(f"unknown operator: {operator}")
    return result


def _candidate_genes(tape: Sequence[Mapping[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    yield []
    phases = tuple((start, min(len(tape), start + 144)) for start in range(0, 719, 144))
    pairs = sorted({
        (str(order[0]), str(order[1]) if len(order) >= 2 else "")
        for action in tape
        for order in action.get("market", []) or []
        if order and str(order[0]) in {"BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "SELL"}
    })
    for start, stop in phases:
        for operation, item in pairs:
            if not any(
                _matches(order, operation, item, start, stop, step)
                for step, action in enumerate(tape)
                for order in action.get("market", []) or []
            ):
                continue
            for numerator, denominator in ((3, 4), (5, 4), (1, 2), (3, 2)):
                yield [{
                    "operator": "scale_quantity", "operation": operation,
                    "item": item, "start": start, "stop": stop,
                    "numerator": numerator, "denominator": denominator,
                }]
            for delta in (-6, -2, -1, 1, 2, 6):
                yield [{
                    "operator": "shift_market", "operation": operation,
                    "item": item, "start": start, "stop": stop, "delta": delta,
                }]
    for start, stop in phases:
        for remove in (1, 2):
            yield [{
                "operator": "reduce_hires", "start": start, "stop": stop,
                "remove_per_day": remove,
            }]


def _predict(tree: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    leaf_class = np.argmax(np.asarray(tree["value"]), axis=1).astype(np.int32)
    indices = np.asarray(native_tree_predict(
        np.asarray(tree["left"], dtype=np.int32),
        np.asarray(tree["right"], dtype=np.int32),
        np.asarray(tree["feature"], dtype=np.int32),
        np.asarray(tree["threshold"], dtype=np.float64),
        leaf_class,
        np.ascontiguousarray(features, dtype=np.float32),
    ))
    return np.asarray(tree["classes"], dtype=str)[indices]


def _evaluate(
    bundle: NativeTeammateBundle,
    families: Sequence[str],
    seeds: Sequence[int],
    policy: Mapping[str, Any],
    opening: str,
) -> list[dict[str, Any]]:
    nodes = sorted(
        (
            value["selected"] for value in policy["nodes"]
            if value["selected"].get("enabled", True)
            and str(value["selected"]["opening"]) == opening
        ),
        key=lambda value: int(value["checkpoint"]),
    )
    opening_index = bundle.index(opening)
    samples = [
        (bundle.index(family), int(seed), seat)
        for family in families for seed in seeds for seat in (0, 1)
    ]
    switch_step = np.full(len(samples), -1, dtype=np.int64)
    switch_target = np.full(len(samples), opening_index, dtype=np.int64)
    for node in nodes:
        checkpoint = int(node["checkpoint"])
        tasks = np.empty((len(samples), 6), dtype=np.int64)
        for row, (candidate, seed, seat) in enumerate(samples):
            tasks[row] = (
                (candidate, opening_index, seed, checkpoint, 1, opening_index)
                if seat == 0 else
                (opening_index, candidate, seed, checkpoint, 0, opening_index)
            )
        predictions = _predict(node["tree"], np.asarray(bundle.executor.features_batch(tasks)))
        for row, prediction in enumerate(predictions):
            if switch_step[row] < 0 and prediction != opening:
                switch_step[row] = checkpoint
                switch_target[row] = bundle.index(str(prediction))
    tasks = np.empty((len(samples), 7), dtype=np.int64)
    for row, (candidate, seed, seat) in enumerate(samples):
        tasks[row] = (
            (candidate, opening_index, seed, -1, -1, switch_step[row], switch_target[row])
            if seat == 0 else
            (opening_index, candidate, seed, switch_step[row], switch_target[row], -1, -1)
        )
    rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
    block = len(seeds) * 2
    rows = []
    for index, family in enumerate(families):
        own = np.empty(block, dtype=np.float64)
        other = np.empty(block, dtype=np.float64)
        base = index * block
        for offset, (_, _, seat) in enumerate(samples[base:base + block]):
            own[offset] = rewards[base + offset, seat]
            other[offset] = rewards[base + offset, 1 - seat]
        margins = own - other
        scores = (margins > 0).astype(np.float64) + .5 * (margins == 0)
        paired_margins = margins.reshape(len(seeds), 2).mean(axis=1)
        paired_scores = scores.reshape(len(seeds), 2).mean(axis=1)
        score_se = (
            float(np.std(paired_scores, ddof=1) / math.sqrt(len(seeds)))
            if len(seeds) > 1 else 0.0
        )
        score_lower = (
            float(np.mean(paired_scores) - student_t.ppf(.95, len(seeds) - 1) * score_se)
            if len(seeds) > 1 else float(paired_scores[0])
        )
        margin_se = (
            float(np.std(paired_margins, ddof=1) / math.sqrt(len(seeds)))
            if len(seeds) > 1 else 0.0
        )
        lower = (
            float(np.mean(paired_margins) - student_t.ppf(.95, len(seeds) - 1) * margin_se)
            if len(seeds) > 1 else float(paired_margins[0])
        )
        targets = [bundle.families[int(value)] for value in switch_target[base:base + block]]
        rows.append({
            "family": family,
            "mean_score": float(np.mean(scores)),
            "one_sided_95pct_score_lower": score_lower,
            "paired_score_standard_error": score_se,
            "mean_margin": float(np.mean(margins)),
            "one_sided_95pct_margin_lower": lower,
            "minimum_paired_seed_margin": float(np.min(paired_margins)),
            "mean_reward": float(np.mean(own)),
            "paired_seed_scores": paired_scores.tolist(),
            "paired_seed_margins": paired_margins.tolist(),
            "policy_target_counts": dict(sorted(Counter(targets).items())),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent", default="G006")
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--genes-from", type=Path)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parent_entry = next(
        value for value in metadata["opponent_routes"]
        if str(value["family"]) == args.parent
    )
    parent_tape = load_action_tapes(args.actions)[str(parent_entry["route_id"])]
    if args.genes_from:
        source_payload = json.loads(args.genes_from.read_text(encoding="utf-8"))
        source_rows = list(source_payload["ranking"])
        if args.top > 0:
            source_rows = source_rows[:args.top]
        genes = [list(value.get("genes") or []) for value in source_rows]
        if [] not in genes:
            genes.append([])
    else:
        genes = list(_candidate_genes(parent_tape))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    ranking: list[dict[str, Any]] = []
    for batch_start in range(0, len(genes), args.batch_size):
        batch_genes = genes[batch_start:batch_start + args.batch_size]
        families = [f"M{index:04d}" for index in range(batch_start, batch_start + len(batch_genes))]
        routes = {
            family: _apply_genes(parent_tape, value)
            for family, value in zip(families, batch_genes)
        }
        bundle = NativeTeammateBundle(
            args.source, args.actions, args.metadata, additional_routes=routes
        )
        rows = _evaluate(bundle, families, args.seeds, policy, args.opening)
        for row, value in zip(rows, batch_genes):
            row["genes"] = value
        ranking.extend(rows)
        print(json.dumps({
            "evaluated": len(ranking), "candidate_count": len(genes),
            "elapsed_seconds": time.perf_counter() - started,
            "batch_best": max(rows, key=lambda row: (
                row["mean_score"], row["one_sided_95pct_margin_lower"], row["mean_margin"]
            )),
        }), flush=True)
    ranking.sort(key=lambda row: (
        -float(row["mean_score"]),
        -float(row["one_sided_95pct_margin_lower"]),
        -float(row["mean_margin"]),
        str(row["family"]),
    ))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    payload = {
        "schema": "native-intent-tape-sensitivity-v1",
        "parent": args.parent,
        "opening": args.opening,
        "seeds": list(args.seeds),
        "candidate_count": len(ranking),
        "games": len(ranking) * len(args.seeds) * 2,
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": ranking,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "games": payload["games"],
        "elapsed_seconds": payload["elapsed_seconds"], "top10": ranking[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
