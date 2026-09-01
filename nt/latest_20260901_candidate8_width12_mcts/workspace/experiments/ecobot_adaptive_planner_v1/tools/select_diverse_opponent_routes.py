#!/usr/bin/env python3
"""Select diverse, executable frozen routes for generic planner state coverage.

Selection is based only on route action statistics (six five-day segments),
source execution health, and support.  It deliberately does not use match
outcomes against EcoBot or any opponent-specific win-rate label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


UNIT_ACTIONS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DIG", "PLANT",
    "WATER", "FERTILIZE", "HARVEST", "PICKUP", "DROP", "PLACE",
    "FEED", "CARE", "COLLECT_FERTILIZER", "BUILD_PASTURE",
    "BUILD_COOP",
]
MARKET_ACTIONS = [
    "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL",
]
PRODUCTS = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
    "WOOL", "FERTILIZER",
]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SEGMENTS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_actions(path: Path) -> dict[str, list[dict[str, object]]]:
    return json.loads(zlib.decompress(path.read_bytes()))


def route_features(tape: list[dict[str, object]]) -> np.ndarray:
    unit_index = {name: index for index, name in enumerate(UNIT_ACTIONS)}
    market_index = {name: index for index, name in enumerate(MARKET_ACTIONS)}
    product_index = {name: index for index, name in enumerate(PRODUCTS)}
    animal_index = {name: index for index, name in enumerate(ANIMALS)}
    width = (
        len(UNIT_ACTIONS) + len(MARKET_ACTIONS) + len(PRODUCTS)
        + len(ANIMALS) + 3
    )
    features = np.zeros((SEGMENTS, width), dtype=np.float64)
    for step, action in enumerate(tape):
        day = min(29, step // 24)
        segment = min(SEGMENTS - 1, day // 5)
        cursor = 0
        units = [action.get("farmer") or ["PASS"]]
        units.extend(action.get("hands") or [])
        for unit_action in units:
            verb = str((unit_action or ["PASS"])[0])
            if verb in unit_index:
                features[segment, cursor + unit_index[verb]] += 1.0
        cursor += len(UNIT_ACTIONS)
        for order in action.get("market") or []:
            if not order:
                continue
            verb = str(order[0])
            if verb in market_index:
                features[segment, cursor + market_index[verb]] += 1.0
            if len(order) >= 2:
                target = str(order[1])
                product_cursor = cursor + len(MARKET_ACTIONS)
                animal_cursor = product_cursor + len(PRODUCTS)
                quantity = float(order[2]) if len(order) >= 3 else 1.0
                if target in product_index:
                    features[segment, product_cursor + product_index[target]] += quantity
                if target in animal_index:
                    features[segment, animal_cursor + animal_index[target]] += quantity
        cursor += len(MARKET_ACTIONS) + len(PRODUCTS) + len(ANIMALS)
        features[segment, cursor] += len(action.get("hands") or [])
        features[segment, cursor + 1] += len(action.get("market") or [])
        features[segment, cursor + 2] += 1.0
    # Convert the last three accumulated fields to per-step averages.
    for segment in range(SEGMENTS):
        steps = max(1.0, features[segment, -1])
        features[segment, -3:-1] /= steps
        features[segment, -1] = 1.0
    return features.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-hard-failures", type=int, default=0)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actions = load_actions(args.actions)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    excluded = {value.upper() for value in args.exclude}
    included = [value.upper() for value in args.include]
    entries = []
    vectors = []
    for entry in metadata["opponent_routes"]:
        family = str(entry["family"]).upper()
        failures = entry.get("source_execution_hard_failures")
        if family in excluded or not entry.get("selected", False):
            continue
        if failures is None or int(failures) > args.max_hard_failures:
            continue
        route_id = str(entry["route_id"])
        if route_id not in actions or len(actions[route_id]) != 719:
            continue
        entries.append(entry)
        vectors.append(route_features(actions[route_id]))
    if len(entries) < args.count:
        raise ValueError(f"only {len(entries)} eligible routes for count={args.count}")
    family_to_index = {
        str(entry["family"]).upper(): index for index, entry in enumerate(entries)
    }
    missing = [family for family in included if family not in family_to_index]
    if missing:
        raise ValueError(f"included routes are not eligible: {missing}")

    matrix = np.asarray(vectors, dtype=np.float64)
    scale = np.std(matrix, axis=0)
    active = scale > 1e-9
    standardized = (matrix[:, active] - np.mean(matrix[:, active], axis=0)) / scale[active]
    support = np.asarray([
        math.log1p(max(0, int(entry.get("support", 0)))) for entry in entries
    ])
    support = (support - support.min()) / max(1e-9, support.max() - support.min())

    selected: list[int] = []
    for family in included:
        index = family_to_index[family]
        if index not in selected:
            selected.append(index)
    if not selected:
        selected.append(int(np.argmax(support)))
    while len(selected) < args.count:
        distance = np.full(len(entries), np.inf, dtype=np.float64)
        for selected_index in selected:
            candidate_distance = np.sqrt(np.mean(
                np.square(standardized - standardized[selected_index]), axis=1
            ))
            distance = np.minimum(distance, candidate_distance)
        distance[selected] = -np.inf
        finite = np.isfinite(distance)
        normalized_distance = np.zeros_like(distance)
        maximum = np.max(distance[finite])
        if maximum > 0:
            normalized_distance[finite] = distance[finite] / maximum
        score = 0.85 * normalized_distance + 0.15 * support
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))

    chosen = []
    for order, index in enumerate(selected):
        nearest = np.inf
        if order:
            nearest = min(
                float(np.sqrt(np.mean(np.square(
                    standardized[index] - standardized[other]
                ))))
                for other in selected[:order]
            )
        entry = entries[index]
        chosen.append({
            "selection_order": order,
            "family": entry["family"],
            "route_id": entry["route_id"],
            "alias": entry.get("alias"),
            "team": entry.get("team"),
            "support": int(entry.get("support", 0)),
            "source_execution_hard_failures": int(
                entry.get("source_execution_hard_failures", 0)
            ),
            "nearest_prior_standardized_distance": (
                None if not np.isfinite(nearest) else nearest
            ),
        })
    payload = {
        "schema": "kaggriculture.diverse-opponent-route-selection.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "uses_ecobot_match_outcome": False,
            "uses_opponent_identity_for_runtime_routing": False,
            "uses_action_trace_statistics": True,
            "segments": SEGMENTS,
            "diversity_weight": 0.85,
            "support_weight": 0.15,
            "max_source_execution_hard_failures": args.max_hard_failures,
        },
        "source": {
            "actions": str(args.actions),
            "actions_sha256": sha256(args.actions),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256(args.metadata),
        },
        "eligible_routes": len(entries),
        "excluded_families": sorted(excluded),
        "forced_included_families": included,
        "selected_routes": chosen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "eligible_routes": len(entries),
        "selected_families": [row["family"] for row in chosen],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
