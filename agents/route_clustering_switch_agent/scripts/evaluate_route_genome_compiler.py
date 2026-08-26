#!/usr/bin/env python3
"""Evaluate parent reconstruction and stratified RouteGenome mutations."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meta_agent.src.route_compiler import (  # noqa: E402
    CompiledRouteAgent,
    TapeAgent,
    compile_route_genome,
)
from meta_agent.src.route_plan import CarrierRoute  # noqa: E402


CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
ANIMALS = {"GOOSE", "COW", "SHEEP"}


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _observation(state: Any) -> Any:
    return _get(state, "observation", {}) or {}


def _action(state: Any) -> Mapping[str, Any]:
    return _get(state, "action", {}) or {}


def _farm(observation: Any, player: int) -> Any:
    farms = list(_get(observation, "farms", []) or [])
    return farms[player] if player < len(farms) else {}


def _tile(observation: Any, player: int, x: int, y: int) -> Any:
    try:
        return list(_get(_farm(observation, player), "tiles", []) or [])[y][x]
    except (IndexError, TypeError):
        return "LOCKED"


def _kind(order: Sequence[Any]) -> str | None:
    if not order:
        return None
    operation = str(order[0])
    item = str(order[1]) if len(order) >= 2 else ""
    if operation == "PLANT" and item in CROPS:
        return item
    if operation == "BUILD_COOP":
        return "COOP"
    if operation == "BUILD_PASTURE":
        return "PASTURE"
    if operation == "PLACE" and item in ANIMALS:
        return item
    return None


def _tile_matches(tile: Any, kind: str) -> bool:
    if not isinstance(tile, Mapping):
        return False
    if kind in CROPS:
        return str(tile.get("kind") or "") == "PLANT" and tile.get("crop") == kind
    if kind in ANIMALS:
        return tile.get("animal") == kind
    return str(tile.get("kind") or "") == kind


def _placements(value: Mapping[str, Any]) -> set[tuple[int, int, str]]:
    return {
        (int(row["x"]), int(row["y"]), str(row["kind"]))
        for row in value.get("placements", []) or []
    }


def _layout_score(
    actual: set[tuple[int, int, str]], target: set[tuple[int, int, str]]
) -> dict[str, float | int]:
    hits = len(actual & target)
    union = len(actual | target)
    return {
        "hits": hits,
        "target": len(target),
        "actual": len(actual),
        "recall": hits / len(target) if target else 1.0,
        "jaccard": hits / union if union else 1.0,
        "exact": int(actual == target),
    }


def _audit_trace(
    steps: Sequence[Sequence[Any]], player: int, target_payload: Mapping[str, Any]
) -> dict[str, Any]:
    intended: dict[tuple[int, int], str] = {}
    realized: dict[tuple[int, int], str] = {}
    targets = {
        int(value["step"]): _placements(value)
        for value in target_payload.get("anchor_targets", []) or []
    }
    intended_scores, realized_scores = [], []
    production_attempts = production_successes = 0
    last = min(len(steps) - 1, 719)
    for turn in range(last):
        previous = _observation(steps[turn][player])
        current = _observation(steps[turn + 1][player])
        farm = _farm(previous, player)
        positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
        action = _action(steps[turn + 1][player])
        orders = [action.get("farmer"), *(action.get("hands", []) or [])]
        for actor, raw_order in enumerate(orders):
            order = list(raw_order or [])
            kind = _kind(order)
            if kind is None or actor >= len(positions) or positions[actor] is None:
                continue
            x, y = int(positions[actor][0]), int(positions[actor][1])
            intended[(x, y)] = kind
            production_attempts += 1
            if _tile_matches(_tile(current, player, x, y), kind):
                realized[(x, y)] = kind
                production_successes += 1
        recorded_step = turn + 1
        if recorded_step in targets:
            intended_set = {(x, y, kind) for (x, y), kind in intended.items()}
            realized_set = {(x, y, kind) for (x, y), kind in realized.items()}
            intended_scores.append({
                "step": recorded_step,
                **_layout_score(intended_set, targets[recorded_step]),
            })
            realized_scores.append({
                "step": recorded_step,
                **_layout_score(realized_set, targets[recorded_step]),
            })
    return {
        "turns": last,
        "production_attempts": production_attempts,
        "production_successes": production_successes,
        "production_failure_rate": (
            (production_attempts - production_successes) / production_attempts
            if production_attempts else 0.0
        ),
        "intended_anchor_scores": intended_scores,
        "realized_anchor_scores": realized_scores,
        "mean_intended_anchor_recall": (
            sum(float(value["recall"]) for value in intended_scores) / len(intended_scores)
            if intended_scores else 0.0
        ),
        "mean_realized_anchor_recall": (
            sum(float(value["recall"]) for value in realized_scores) / len(realized_scores)
            if realized_scores else 0.0
        ),
        "intended_exact_anchors": sum(int(value["exact"]) for value in intended_scores),
        "realized_exact_anchors": sum(int(value["exact"]) for value in realized_scores),
    }


def _run(
    carrier: CarrierRoute,
    parent: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    from kaggle_environments import make

    plan = compile_route_genome(carrier, parent, target)
    compiled = CompiledRouteAgent(plan)
    opponent = TapeAgent(carrier.opponent_actions)
    agents = [compiled, opponent] if carrier.player_index == 0 else [opponent, compiled]
    environment = make(
        "kaggriculture",
        configuration=dict(carrier.configuration),
        info=dict(carrier.info),
        debug=True,
    )
    environment.run(agents)
    rewards = [float(_get(state, "reward", 0.0) or 0.0) for state in environment.state]
    payload = target.get("genetic_payload") or target
    return {
        "compile": plan.report.to_dict(),
        "runtime": dict(compiled.runtime),
        "reward": rewards[carrier.player_index],
        "opponent_reward": rewards[1 - carrier.player_index],
        "win_score": (
            1.0 if rewards[carrier.player_index] > rewards[1 - carrier.player_index]
            else 0.5 if rewards[carrier.player_index] == rewards[1 - carrier.player_index]
            else 0.0
        ),
        "trace": _audit_trace(environment.steps, carrier.player_index, payload),
        "completed": len(environment.steps) == 720,
    }


def _category(row: Mapping[str, Any]) -> str:
    operation = str((row.get("experiment") or {}).get("operator") or "")
    if operation in ("event_time_shift", "market_quantity"):
        return "market_or_time"
    return operation


def _stratified(
    candidates: Sequence[Mapping[str, Any]], per_category: int
) -> list[Mapping[str, Any]]:
    wanted = (
        "market_or_time", "layout_relocate", "crop_substitute",
        "compound_macro_depth_4", "compound_macro_depth_12",
        "compound_macro_depth_32",
    )
    selected = []

    def round_robin(
        values: Sequence[Mapping[str, Any]], count: int
    ) -> list[Mapping[str, Any]]:
        output = []
        by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for value in values:
            by_parent[str(value["experiment"]["parent_id"])].append(value)
        parent_ids = sorted(by_parent)
        cursor = 0
        while len(output) < count:
            progressed = False
            for parent_id in parent_ids:
                rows = by_parent[parent_id]
                if cursor < len(rows):
                    output.append(rows[cursor])
                    progressed = True
                    if len(output) >= count:
                        break
            if not progressed:
                break
            cursor += 1
        return output

    for category in wanted:
        rows = [row for row in candidates if _category(row) == category]
        if category == "market_or_time":
            time_count = per_category // 2
            selected.extend(round_robin([
                row for row in rows
                if row["experiment"]["operator"] == "event_time_shift"
            ], time_count))
            selected.extend(round_robin([
                row for row in rows
                if row["experiment"]["operator"] == "market_quantity"
            ], per_category - time_count))
        else:
            selected.extend(round_robin(rows, per_category))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--parents-only", action="store_true")
    args = parser.parse_args()

    archive = {str(row["genome_id"]): row for row in _json_lines(args.groups.resolve())}
    panel = json.loads(args.parents.resolve().read_text(encoding="utf-8"))
    parents = {str(row["parent_id"]): row for row in panel["parents"]}
    candidates = list(_json_lines(args.mutations.resolve()))
    chosen = [] if args.parents_only else _stratified(candidates, args.per_category)
    needed_parent_ids = sorted({
        str(row["experiment"]["parent_id"]) for row in chosen
    })
    if args.parents_only:
        needed_parent_ids = [
            str(row["parent_id"]) for row in panel["parents"]
            if row.get("role") == "focus_family_nearest"
        ]

    carriers: dict[str, CarrierRoute] = {}
    baselines: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for parent_id in needed_parent_ids:
        parent = parents[parent_id]
        carriers[parent_id] = CarrierRoute.from_replay(
            parent["carrier"]["replay_path"], parent["carrier"]["player_index"]
        )
        parent_genome = archive[str(parent["genome_id"])]
        baseline_target = {
            **parent_genome,
            "experiment": {"operator": "parent_reconstruction", "detail": ""},
        }
        baselines[parent_id] = _run(carriers[parent_id], parent_genome, baseline_target)
        print(
            f"baseline {parent_id}: reward={baselines[parent_id]['reward']:.0f} "
            f"intended={baselines[parent_id]['trace']['mean_intended_anchor_recall']:.3f}",
            flush=True,
        )

    rows = []
    for index, candidate in enumerate(chosen, 1):
        experiment = candidate["experiment"]
        parent_id = str(experiment["parent_id"])
        parent = parents[parent_id]
        parent_genome = archive[str(parent["genome_id"])]
        result = _run(carriers[parent_id], parent_genome, candidate)
        baseline = baselines[parent_id]
        row = {
            "genome_id": candidate["genome_id"],
            "parent_id": parent_id,
            "parent_genome_id": parent["genome_id"],
            "family": parent["family"],
            "operator": experiment["operator"],
            "category": _category(candidate),
            "detail": experiment["detail"],
            "macro_distance_to_parent": experiment.get("macro_distance_to_parent"),
            "new_family_candidate": experiment.get("new_family_candidate"),
            "baseline_reward": baseline["reward"],
            "reward_delta": result["reward"] - baseline["reward"],
            **result,
        }
        rows.append(row)
        print(
            f"candidate {index}/{len(chosen)} {row['operator']} {parent_id}: "
            f"status={row['compile']['status']} reward_delta={row['reward_delta']:.0f} "
            f"intended={row['trace']['mean_intended_anchor_recall']:.3f} "
            f"realized={row['trace']['mean_realized_anchor_recall']:.3f}",
            flush=True,
        )

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output.resolve(), "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    category_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows[str(row["category"])].append(row)

    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "compile_status": dict(Counter(row["compile"]["status"] for row in values)),
            "completed": sum(bool(row["completed"]) for row in values),
            "mean_reward_delta": sum(float(row["reward_delta"]) for row in values) / len(values),
            "positive_reward_delta": sum(float(row["reward_delta"]) > 0 for row in values),
            "mean_intended_anchor_recall": sum(
                float(row["trace"]["mean_intended_anchor_recall"]) for row in values
            ) / len(values),
            "mean_realized_anchor_recall": sum(
                float(row["trace"]["mean_realized_anchor_recall"]) for row in values
            ) / len(values),
            "mean_production_failure_rate": sum(
                float(row["trace"]["production_failure_rate"]) for row in values
            ) / len(values),
            "completed_dynamic_tasks": sum(
                len(row["runtime"]["completed_tasks"]) for row in values
            ),
            "failed_dynamic_tasks": sum(
                len(row["runtime"]["failed_tasks"]) for row in values
            ),
        }

    baseline_rows = list(baselines.values())
    summary = {
        "schema_version": 1,
        "kind": "route_genome_compiler_experiment",
        "elapsed_seconds": time.perf_counter() - started,
        "parents": len(needed_parent_ids),
        "candidates": len(rows),
        "selection": {
            "per_category": args.per_category,
            "category_counts": dict(Counter(_category(row) for row in chosen)),
        },
        "parent_reconstruction": {
            "completed": sum(bool(row["completed"]) for row in baseline_rows),
            "exact_recorded_rewards": sum(
                row["reward"] == carriers[parent_id].rewards[carriers[parent_id].player_index]
                for parent_id, row in baselines.items()
            ),
            "mean_intended_anchor_recall": (
                sum(row["trace"]["mean_intended_anchor_recall"] for row in baseline_rows)
                / len(baseline_rows) if baseline_rows else 0.0
            ),
            "mean_realized_anchor_recall": (
                sum(row["trace"]["mean_realized_anchor_recall"] for row in baseline_rows)
                / len(baseline_rows) if baseline_rows else 0.0
            ),
            "rows": baselines,
        },
        "overall": summarize(rows),
        "by_category": {
            category: summarize(values) for category, values in sorted(category_rows.items())
        },
        "output": str(args.output.resolve()),
    }
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "parents": summary["parent_reconstruction"],
        "overall": summary["overall"],
        "by_category": summary["by_category"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
