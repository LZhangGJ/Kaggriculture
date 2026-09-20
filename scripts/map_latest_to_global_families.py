#!/usr/bin/env python3
"""Map fresh replay sides onto the stable 56-family macro taxonomy."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=True) as cached:
        return list(cached["rows"])


def _global_ids(rows: list[dict[str, Any]], labels: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    groups = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        median_reward = float(np.median([float(rows[index]["reward"]) for index in indices]))
        groups.append((int(label), indices, median_reward))
    groups.sort(key=lambda value: (-len(value[1]), -value[2]))
    raw_to_global = {raw: index + 1 for index, (raw, _, _) in enumerate(groups)}
    mapped = np.asarray([raw_to_global[int(value)] for value in labels], dtype=np.int16)
    summary = [
        {
            "family": f"G{index + 1}",
            "reference_support": len(indices),
            "reference_median_reward": median_reward,
        }
        for index, (_, indices, median_reward) in enumerate(groups)
    ]
    return mapped, summary


def _write_cpp_input(path: Path, rows: list[dict[str, Any]]) -> None:
    production = np.ascontiguousarray(
        np.stack([row["production"] for row in rows]), dtype=np.int16
    )
    layouts = np.ascontiguousarray(
        np.stack([row["layouts"] for row in rows]), dtype=np.int8
    )
    schedule = np.ascontiguousarray(
        np.stack([row["schedule"] for row in rows]), dtype=np.int16
    )
    with path.open("wb") as handle:
        np.asarray([len(rows)], dtype=np.int32).tofile(handle)
        production.tofile(handle)
        layouts.tofile(handle)
        schedule.tofile(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--reference-clusters", type=Path, required=True)
    parser.add_argument("--latest-cache", type=Path, required=True)
    parser.add_argument("--latest-manifest", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    reference = _load_rows(args.reference_cache)
    latest = _load_rows(args.latest_cache)
    with np.load(args.reference_clusters) as cached:
        reference_labels = cached["labels"].astype(int)
    reference_global, families = _global_ids(reference, reference_labels)
    if len(families) != 56:
        raise ValueError(f"expected stable 56-family taxonomy, got {len(families)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = reference + latest
    binary_input = args.output.parent / "reference-latest-distance-input.bin"
    binary_output = args.output.parent / "reference-latest-distance.f32"
    _write_cpp_input(binary_input, combined)
    subprocess.run(
        [str(args.cpp_executable.resolve()), str(binary_input), str(binary_output)],
        check=True,
    )
    count = len(combined)
    distances = np.memmap(binary_output, mode="r", dtype=np.float32, shape=(count, count))
    cross = distances[len(reference) :, : len(reference)]
    nearest_reference = np.argmin(cross, axis=1).astype(np.int32)
    nearest_distance = cross[np.arange(len(latest)), nearest_reference].astype(np.float32)
    latest_global = reference_global[nearest_reference]
    np.savez_compressed(
        args.output,
        family=latest_global,
        nearest_reference=nearest_reference,
        nearest_distance=nearest_distance,
    )

    manifest = json.loads(args.latest_manifest.read_text(encoding="utf-8"))
    rewards = {
        int(replay["episode_id"]): list(replay.get("rewards") or [])
        for replay in manifest.get("replays", [])
    }
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, family in enumerate(latest_global.tolist()):
        grouped[int(family)].append(index)
    for family in families:
        number = int(family["family"][1:])
        indices = grouped.get(number, [])
        wins = losses = ties = 0
        route_rows = []
        for index in indices:
            row = latest[index]
            episode_rewards = rewards.get(int(row["episode_id"]), [])
            player = int(row["player_index"])
            if len(episode_rewards) == 2:
                margin = float(episode_rewards[player]) - float(episode_rewards[1 - player])
                wins += int(margin > 0)
                losses += int(margin < 0)
                ties += int(margin == 0)
            route_rows.append(
                {
                    "route_id": f"{int(row['episode_id'])}:{player}",
                    "team": str(row["team"]),
                    "reward": float(row["reward"]),
                    "mapping_distance": float(nearest_distance[index]),
                }
            )
        route_rows.sort(key=lambda value: (value["mapping_distance"], -value["reward"]))
        family.update(
            latest_support=len(indices),
            within_012=sum(float(nearest_distance[index]) <= 0.12 for index in indices),
            outcome={"games": wins + losses + ties, "wins": wins, "losses": losses, "ties": ties},
            nearest_routes=route_rows[:10],
        )
    payload = {
        "schema_version": 1,
        "reference_sides": len(reference),
        "latest_sides": len(latest),
        "families": families,
        "mapping_distance": {
            "median": float(np.median(nearest_distance)),
            "p90": float(np.quantile(nearest_distance, 0.9)),
            "within_012": int(np.count_nonzero(nearest_distance <= 0.12)),
            "within_020": int(np.count_nonzero(nearest_distance <= 0.20)),
        },
        "latest_family_sizes": dict(Counter(f"G{int(value)}" for value in latest_global)),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["mapping_distance"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
