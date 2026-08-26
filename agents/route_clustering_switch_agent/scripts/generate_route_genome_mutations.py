#!/usr/bin/env python3
"""Generate and audit plan-level RouteGenome mutations before compilation."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.route_genome import RouteGenome


ANCHORS = (168, 288, 432, 576, 719)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
CATEGORIES = (*CROPS, "GOOSE", "COW", "SHEEP", "COOP", "PASTURE")
SCHEDULE_KEYS = (
    *CROPS, "BUILD_COOP", "BUILD_PASTURE", "GOOSE", "COW", "SHEEP",
    "BUY_LAND", "HIRE",
)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _recount(payload: dict[str, Any]) -> None:
    for target in payload["anchor_targets"]:
        target["placements"].sort(key=lambda value: (int(value["y"]), int(value["x"])))
        counts = Counter(str(value["kind"]) for value in target["placements"])
        target["counts"] = {kind: int(counts.get(kind, 0)) for kind in CATEGORIES}


def _macro_key(event: Mapping[str, Any]) -> str | None:
    operation = str(event.get("operation", ""))
    item = str(event.get("item") or "")
    if operation in ("BUILD_COOP", "BUILD_PASTURE", "BUY_LAND", "HIRE"):
        return operation
    if operation == "PLACE" and item in ("GOOSE", "COW", "SHEEP"):
        return item
    return None


def _shift_event(payload: dict[str, Any], rng: random.Random) -> str | None:
    events = payload["structural_events"]
    if not events:
        return None
    index = rng.randrange(len(events))
    event = events[index]
    old_step = int(event["step"])
    deltas = [value for value in (-24, -12, 12, 24) if 0 <= old_step + value <= 718]
    if not deltas:
        return None
    new_step = old_step + rng.choice(deltas)
    event["step"] = new_step
    key = _macro_key(event)
    old_phase, new_phase = min(9, old_step // 72), min(9, new_step // 72)
    if key and old_phase != new_phase:
        phases = payload["phase_macro_counts"]
        old_counts = phases[old_phase]["counts"]
        if int(old_counts.get(key, 0)) > 0:
            old_counts[key] = int(old_counts[key]) - 1
            new_counts = phases[new_phase]["counts"]
            new_counts[key] = int(new_counts.get(key, 0)) + 1
    events.sort(key=lambda value: (
        int(value["step"]), str(value.get("operation", "")),
        int(value.get("y", -1)), int(value.get("x", -1)),
    ))
    return f"event[{index}] step {old_step}->{new_step}"


def _market_quantity(payload: dict[str, Any], rng: random.Random) -> str | None:
    rows = [
        row for row in payload["market_actions"]
        if row.get("item") is not None and int(row.get("quantity", 0)) > 0
    ]
    if not rows:
        return None
    row = rng.choice(rows)
    old_quantity = int(row["quantity"])
    delta = rng.choice((-max(1, old_quantity // 10), max(1, old_quantity // 10)))
    new_quantity = max(1, old_quantity + delta)
    if new_quantity == old_quantity:
        return None
    row["quantity"] = new_quantity
    return (
        f"{row['operation']}:{row.get('item')} quantity "
        f"{old_quantity}->{new_quantity}"
    )


def _layout_relocate(payload: dict[str, Any], rng: random.Random) -> str | None:
    final = payload["anchor_targets"][-1]
    if not final["placements"]:
        return None
    candidates = list(final["placements"])
    rng.shuffle(candidates)
    for selected in candidates:
        old_x, old_y = int(selected["x"]), int(selected["y"])
        kind = str(selected["kind"])
        alternatives = [
            (old_x + dx, old_y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= old_x + dx < 10 and 0 <= old_y + dy < 10
        ]
        rng.shuffle(alternatives)
        for new_x, new_y in alternatives:
            applicable = []
            valid = True
            for target in payload["anchor_targets"]:
                old_match = next((
                    value for value in target["placements"]
                    if int(value["x"]) == old_x and int(value["y"]) == old_y
                    and str(value["kind"]) == kind
                ), None)
                if old_match is None:
                    continue
                occupied = any(
                    int(value["x"]) == new_x and int(value["y"]) == new_y
                    for value in target["placements"]
                )
                if occupied:
                    valid = False
                    break
                applicable.append(old_match)
            if not valid or not applicable:
                continue
            for placement in applicable:
                placement["x"], placement["y"] = new_x, new_y
            for event in payload["structural_events"]:
                if int(event.get("x", -1)) != old_x or int(event.get("y", -1)) != old_y:
                    continue
                event_kind = (
                    "COOP" if event.get("operation") == "BUILD_COOP"
                    else "PASTURE" if event.get("operation") == "BUILD_PASTURE"
                    else str(event.get("item") or "")
                )
                if event_kind == kind:
                    event["x"], event["y"] = new_x, new_y
            _recount(payload)
            return f"relocate {kind} ({old_x},{old_y})->({new_x},{new_y})"
    return None


def _crop_substitute(payload: dict[str, Any], rng: random.Random) -> str | None:
    final_crops = [
        value for value in payload["anchor_targets"][-1]["placements"]
        if str(value["kind"]) in CROPS
    ]
    if not final_crops:
        return None
    selected = rng.choice(final_crops)
    x, y, old_crop = int(selected["x"]), int(selected["y"]), str(selected["kind"])
    new_crop = rng.choice([value for value in CROPS if value != old_crop])
    changed = 0
    for target in payload["anchor_targets"]:
        for placement in target["placements"]:
            if (
                int(placement["x"]) == x and int(placement["y"]) == y
                and str(placement["kind"]) == old_crop
            ):
                placement["kind"] = new_crop
                changed += 1
    if not changed:
        return None
    for phase in payload["phase_macro_counts"]:
        counts = phase["counts"]
        if int(counts.get(old_crop, 0)) > 0:
            counts[old_crop] = int(counts[old_crop]) - 1
            counts[new_crop] = int(counts.get(new_crop, 0)) + 1
            break
    _recount(payload)
    return f"substitute ({x},{y}) {old_crop}->{new_crop} across {changed} anchors"


MUTATORS: dict[str, Callable[[dict[str, Any], random.Random], str | None]] = {
    "event_time_shift": _shift_event,
    "market_quantity": _market_quantity,
    "layout_relocate": _layout_relocate,
    "crop_substitute": _crop_substitute,
}


def _features(payloads: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    category_id = {name: index + 1 for index, name in enumerate(CATEGORIES)}
    layouts = np.zeros((len(payloads), 5, 100), dtype=np.int8)
    schedules = np.zeros((len(payloads), 10, 12), dtype=np.int16)
    for row_index, payload in enumerate(payloads):
        targets = {int(value["step"]): value for value in payload["anchor_targets"]}
        for anchor_index, anchor in enumerate(ANCHORS):
            for placement in targets[anchor]["placements"]:
                layouts[
                    row_index, anchor_index,
                    int(placement["y"]) * 10 + int(placement["x"]),
                ] = category_id[str(placement["kind"])]
        phases = {int(value["phase"]): value for value in payload["phase_macro_counts"]}
        for phase in range(10):
            counts = phases[phase]["counts"]
            schedules[row_index, phase] = [int(counts.get(key, 0)) for key in SCHEDULE_KEYS]
    return layouts, schedules


def _cross_distances(
    left_layouts: np.ndarray,
    left_schedules: np.ndarray,
    right_layouts: np.ndarray,
    right_schedules: np.ndarray,
) -> np.ndarray:
    left_counts = np.stack(
        [(left_layouts == value).sum(axis=2) for value in range(1, 11)], axis=2
    ).astype(np.int16)
    right_counts = np.stack(
        [(right_layouts == value).sum(axis=2) for value in range(1, 11)], axis=2
    ).astype(np.int16)
    left_totals = left_counts.sum(axis=2, dtype=np.int32)
    right_totals = right_counts.sum(axis=2, dtype=np.int32)
    left_cumulative = np.cumsum(left_schedules.astype(np.int32), axis=1)
    right_cumulative = np.cumsum(right_schedules.astype(np.int32), axis=1)
    left_schedule_totals = left_cumulative.sum(axis=2, dtype=np.int32)
    right_schedule_totals = right_cumulative.sum(axis=2, dtype=np.int32)
    output = np.zeros((len(left_layouts), len(right_layouts)), dtype=np.float32)
    for start in range(0, len(left_layouts), 16):
        stop = min(len(left_layouts), start + 16)
        shape = (stop - start, len(right_layouts))
        composition = np.zeros(shape, dtype=np.float32)
        layout = np.zeros(shape, dtype=np.float32)
        schedule = np.zeros(shape, dtype=np.float32)
        for anchor in range(5):
            delta = np.abs(
                left_counts[start:stop, None, anchor, :].astype(np.int32)
                - right_counts[None, :, anchor, :].astype(np.int32)
            ).sum(axis=2)
            denominator = np.maximum(
                8,
                np.maximum(
                    left_totals[start:stop, None, anchor],
                    right_totals[None, :, anchor],
                ),
            )
            composition += delta / denominator
            left = left_layouts[start:stop, anchor]
            right = right_layouts[:, anchor]
            active = (left[:, None, :] != 0) | (right[None, :, :] != 0)
            active_count = active.sum(axis=2)
            mismatch = ((left[:, None, :] != right[None, :, :]) & active).sum(axis=2)
            layout += np.divide(
                mismatch, active_count, out=np.zeros(shape), where=active_count > 0
            )
        for phase in range(10):
            delta = np.abs(
                left_cumulative[start:stop, None, phase, :]
                - right_cumulative[None, :, phase, :]
            ).sum(axis=2)
            denominator = np.maximum(
                6,
                np.maximum(
                    left_schedule_totals[start:stop, None, phase],
                    right_schedule_totals[None, :, phase],
                ),
            )
            schedule += delta / denominator
        output[start:stop] = (
            0.45 * composition / 5 + 0.35 * layout / 5 + 0.20 * schedule / 10
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--per-operator", type=int, default=3)
    parser.add_argument("--compound-depths", default="4,12,32")
    parser.add_argument("--compound-per-depth", type=int, default=2)
    parser.add_argument("--family-threshold", type=float, default=0.12)
    args = parser.parse_args()

    if args.per_operator <= 0:
        parser.error("--per-operator must be positive")
    if args.compound_per_depth < 0:
        parser.error("--compound-per-depth must be non-negative")
    compound_depths = sorted({
        int(value) for value in args.compound_depths.split(",") if value.strip()
    })
    if any(value <= 1 for value in compound_depths):
        parser.error("--compound-depths values must exceed one")
    archive_rows = list(_json_lines(args.groups.resolve()))
    archive = {str(row["genome_id"]): row for row in archive_rows}
    mappings = {str(row["genome_id"]): row for row in _json_lines(args.family_map.resolve())}
    parent_payload = json.loads(args.parents.resolve().read_text(encoding="utf-8"))
    parents = list(parent_payload["parents"])
    rng = random.Random(args.seed)
    candidates = []
    generated_ids: set[str] = set()
    attempts: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    def append_candidate(
        parent: Mapping[str, Any],
        parent_genome_id: str,
        payload: dict[str, Any],
        operator: str,
        detail: str,
    ) -> bool:
        _recount(payload)
        try:
            genome = RouteGenome.create(
                source={
                    "source_id": f"synthetic:{len(candidates) + 1}",
                    "episode_id": 0,
                    "player_index": 0,
                    "team_name": "synthetic_mutation",
                    "opponent_team_name": "",
                    "final_reward": 0.0,
                    "opponent_reward": 0.0,
                    "result": "not_evaluated",
                    "parent_genome_id": parent_genome_id,
                    "parent_id": str(parent["parent_id"]),
                    "mutation_operator": operator,
                    "mutation_detail": detail,
                },
                anchor_targets=payload["anchor_targets"],
                phase_macro_counts=payload["phase_macro_counts"],
                structural_events=payload["structural_events"],
                market_profile=payload["market_actions"],
                cash_profile={"status": "not_evaluated"},
            )
        except (TypeError, ValueError):
            return False
        if genome.genome_id == parent_genome_id or genome.genome_id in generated_ids:
            return False
        generated_ids.add(genome.genome_id)
        candidates.append({
            **genome.to_dict(),
            "experiment": {
                "kind": "plan_only_mutation",
                "parent_id": str(parent["parent_id"]),
                "parent_genome_id": parent_genome_id,
                "parent_family": str(parent["family"]),
                "operator": operator,
                "detail": detail,
            },
        })
        return True

    for parent in parents:
        parent_genome_id = str(parent["genome_id"])
        source_payload = archive[parent_genome_id]["genetic_payload"]
        for operator, mutate in MUTATORS.items():
            made = 0
            for _ in range(args.per_operator * 20):
                if made >= args.per_operator:
                    break
                attempts[operator] += 1
                payload = copy.deepcopy(source_payload)
                detail = mutate(payload, rng)
                if detail is None:
                    failures[operator] += 1
                    continue
                if not append_candidate(
                    parent, parent_genome_id, payload, operator, detail
                ):
                    failures[operator] += 1
                    continue
                made += 1

        macro_mutators = (
            ("layout_relocate", _layout_relocate),
            ("crop_substitute", _crop_substitute),
        )
        for depth in compound_depths:
            operator = f"compound_macro_depth_{depth}"
            made = 0
            for _ in range(max(1, args.compound_per_depth * 30)):
                if made >= args.compound_per_depth:
                    break
                attempts[operator] += 1
                payload = copy.deepcopy(source_payload)
                details = []
                for _mutation_index in range(depth):
                    name, mutate = rng.choice(macro_mutators)
                    detail = mutate(payload, rng)
                    if detail is not None:
                        details.append(f"{name}:{detail}")
                if len(details) < max(2, depth // 2):
                    failures[operator] += 1
                    continue
                if not append_candidate(
                    parent,
                    parent_genome_id,
                    payload,
                    operator,
                    " | ".join(details),
                ):
                    failures[operator] += 1
                    continue
                made += 1

    archive_payloads = [row["genetic_payload"] for row in archive_rows]
    candidate_payloads = [RouteGenome.from_dict(row).genetic_payload() for row in candidates]
    archive_layouts, archive_schedules = _features(archive_payloads)
    candidate_layouts, candidate_schedules = _features(candidate_payloads)
    distances = _cross_distances(
        candidate_layouts, candidate_schedules, archive_layouts, archive_schedules
    )
    archive_index = {str(row["genome_id"]): index for index, row in enumerate(archive_rows)}
    operator_stats: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {"count": 0, "nearest": [], "parent": []}
    )
    for index, candidate in enumerate(candidates):
        experiment = candidate["experiment"]
        parent_genome_id = str(experiment["parent_genome_id"])
        nearest_index = int(np.argmin(distances[index]))
        nearest_genome_id = str(archive_rows[nearest_index]["genome_id"])
        nearest_distance = float(distances[index, nearest_index])
        parent_distance = float(distances[index, archive_index[parent_genome_id]])
        nearest_family = str(mappings[nearest_genome_id]["family"])
        experiment.update({
            "exact_novel": str(candidate["genome_id"]) not in archive,
            "macro_distance_to_parent": parent_distance,
            "nearest_archive_genome_id": nearest_genome_id,
            "nearest_archive_distance": nearest_distance,
            "nearest_archive_family": nearest_family,
            "same_family_as_parent": nearest_family == str(experiment["parent_family"]),
            "new_family_candidate": nearest_distance > args.family_threshold,
            "execution_status": "not_compiled_or_evaluated",
        })
        stats = operator_stats[str(experiment["operator"])]
        stats["count"] = int(stats["count"]) + 1
        stats["nearest"].append(nearest_distance)
        stats["parent"].append(parent_distance)

    output = args.output.resolve()
    summary_path = (
        args.summary or output.with_name(output.name + ".summary.json")
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output, "w") as handle:
        for candidate in candidates:
            handle.write(json.dumps(
                candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")

    def values_summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    summary = {
        "schema_version": 1,
        "kind": "route_genome_plan_mutation_experiment",
        "seed": args.seed,
        "parents": len(parents),
        "operators": sorted(MUTATORS),
        "requested_per_parent_operator": args.per_operator,
        "compound_depths": compound_depths,
        "compound_per_parent_depth": args.compound_per_depth,
        "attempts": dict(sorted(attempts.items())),
        "failed_or_duplicate_attempts": dict(sorted(failures.items())),
        "generated_unique_candidates": len(candidates),
        "exact_novel_candidates": sum(
            bool(row["experiment"]["exact_novel"]) for row in candidates
        ),
        "macro_zero_distance_candidates": sum(
            float(row["experiment"]["nearest_archive_distance"]) <= 1e-8
            for row in candidates
        ),
        "same_family_as_parent": sum(
            bool(row["experiment"]["same_family_as_parent"]) for row in candidates
        ),
        "new_family_candidates": sum(
            bool(row["experiment"]["new_family_candidate"]) for row in candidates
        ),
        "family_threshold": args.family_threshold,
        "operator_stats": {
            operator: {
                "count": int(values["count"]),
                "nearest_archive_distance": values_summary(values["nearest"]),
                "parent_distance": values_summary(values["parent"]),
            }
            for operator, values in sorted(operator_stats.items())
        },
        "execution_status": (
            "plan-only dry run; candidates require a state-aware compiler before fitness evaluation"
        ),
        "output": str(output),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
