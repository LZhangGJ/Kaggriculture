#!/usr/bin/env python3
"""Select typical macro-family routes and audit their switch compatibility."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_macro_route_library import _distance
from meta_agent.src.fingerprints import fingerprint_distance


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--opening-mixture", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--major-families", type=int, default=8)
    parser.add_argument("--drop-zero-win-rare", action="store_true")
    return parser.parse_args()


def _family_groups(
    rows: list[dict[str, Any]], labels: np.ndarray
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped = []
    for label in sorted(set(labels.tolist())):
        subset = [row for row, value in zip(rows, labels) if int(value) == int(label)]
        grouped.append((
            str(label), subset,
            float(np.median([float(row["reward"]) for row in subset])),
        ))
    grouped.sort(key=lambda value: (-len(value[1]), -value[2]))
    return [(f"G{index + 1}", value[1]) for index, value in enumerate(grouped)]


def _mode_layout(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    result = np.empty(flat.shape[1], dtype=np.int8)
    for index in range(flat.shape[1]):
        result[index] = int(np.bincount(flat[:, index].astype(int), minlength=11).argmax())
    return result.reshape(values.shape[1:])


def _prototype(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "production": np.rint(np.median(np.stack([row["production"] for row in rows]), axis=0)).astype(np.int16),
        "layouts": _mode_layout(np.stack([row["layouts"] for row in rows])),
        "schedule": np.rint(np.median(np.stack([row["schedule"] for row in rows]), axis=0)).astype(np.int16),
    }


def _representative(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prototype = _prototype(rows)
    ranked = sorted(
        ((_distance(row, prototype), row) for row in rows),
        key=lambda value: (value[0], -float(value[1]["reward"]), int(value[1]["episode_id"])),
    )
    distance, row = ranked[0]
    return {
        "family": family,
        "route_id": f"{int(row['episode_id'])}:{int(row['player_index'])}",
        "team": str(row["team"]),
        "episode_id": int(row["episode_id"]),
        "player_index": int(row["player_index"]),
        "support": len(rows),
        "prototype_distance": round(float(distance), 6),
        "historical_reward": float(row["reward"]),
        "family_median_reward": float(np.median([float(value["reward"]) for value in rows])),
        "kind": "macro_medoid",
    }


def _outcome(rows: list[dict[str, Any]], outcomes: dict[str, Any]) -> dict[str, int]:
    wins = losses = ties = 0
    for row in rows:
        episode = outcomes.get(str(int(row["episode_id"]))) or {}
        rewards = list(episode.get("rewards") or [])
        player = int(row["player_index"])
        if len(rewards) != 2 or player not in (0, 1):
            continue
        margin = float(rewards[player]) - float(rewards[1 - player])
        wins += int(margin > 0)
        losses += int(margin < 0)
        ties += int(margin == 0)
    return {"games": wins + losses + ties, "wins": wins, "losses": losses, "ties": ties}


def _runtime_rows(runtime: Path, checkpoints: list[int]) -> dict[int, dict[str, tuple[Any, ...]]]:
    result = {}
    for checkpoint in checkpoints:
        with (runtime / f"checkpoint-{checkpoint:03d}.pkl").open("rb") as handle:
            rows = pickle.load(handle)
        result[checkpoint] = {str(row[0]): row for row in rows}
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.global_clusters) as cached:
        labels = cached["labels"].astype(int)
    if len(rows) != len(labels):
        raise ValueError("global labels do not align with macro cache")
    groups = _family_groups(rows, labels)
    outcomes = (
        json.loads(args.outcomes.read_text(encoding="utf-8"))
        if args.outcomes is not None else {}
    )
    opponent_representatives = []
    for family, values in groups:
        representative = _representative(family, values)
        representative["outcome"] = _outcome(values, outcomes)
        opponent_representatives.append(representative)
    candidates = []
    dropped = []
    for value in opponent_representatives[: args.major_families]:
        number = int(value["family"][1:])
        result = value["outcome"]
        if (
            args.drop_zero_win_rare and number > 8
            and result["games"] > 0 and result["wins"] == 0
        ):
            dropped.append(dict(value))
            continue
        candidates.append(dict(value))
    mixture = json.loads(args.opening_mixture.read_text(encoding="utf-8"))
    known_ids = {value["route_id"] for value in candidates}
    for row in mixture.get("support", []):
        route_id = str(row["route_id"])
        if route_id in known_ids:
            continue
        candidates.append({
            "family": str(row.get("source_agent") or "DEPLOYED"),
            "route_id": route_id,
            "team": str(row.get("source_agent") or ""),
            "support": 1,
            "prototype_distance": None,
            "historical_reward": None,
            "family_median_reward": None,
            "kind": "deployed_opening_control",
            "opening_weight": float(row["weight"]),
        })
        known_ids.add(route_id)

    with (args.runtime / "manifest.pkl").open("rb") as handle:
        runtime_manifest = pickle.load(handle)
    checkpoints = [int(value) for value in runtime_manifest["checkpoints"]]
    runtime_ids = {str(value) for value in runtime_manifest["route_offsets"]}
    missing = [value["route_id"] for value in candidates if value["route_id"] not in runtime_ids]
    if missing:
        raise ValueError(f"selected candidates missing from runtime: {missing}")
    runtime_rows = _runtime_rows(args.runtime, checkpoints)
    compatibility = {}
    for checkpoint in checkpoints:
        current = runtime_rows[checkpoint]
        checkpoint_rows = {}
        for source in candidates:
            source_row = current[source["route_id"]]
            values = []
            for target in candidates:
                target_row = current[target["route_id"]]
                distance = fingerprint_distance(source_row[4], target_row[4])
                values.append({
                    "route_id": target["route_id"],
                    "family": target["family"],
                    "exact_prefix": str(source_row[2]) == str(target_row[2]),
                    "state_distance": {key: round(float(value), 6) for key, value in distance.items()},
                })
            values.sort(key=lambda value: (
                not value["exact_prefix"], value["state_distance"]["total"], value["route_id"]
            ))
            checkpoint_rows[source["route_id"]] = values
        compatibility[str(checkpoint)] = checkpoint_rows
    return {
        "schema_version": 1,
        "major_family_count": args.major_families,
        "candidates": candidates,
        "dropped_zero_win_rare": dropped,
        "opponent_representatives": opponent_representatives,
        "checkpoints": checkpoints,
        "compatibility": compatibility,
        "distance_policy": {
            "exact_prefix": "always legal",
            "state_reanchor": "not assumed legal; candidates are ordered by causal own-state distance and must pass counterfactual simulation",
        },
    }


def main() -> None:
    args = arguments()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    exact_counts = {}
    for checkpoint, sources in payload["compatibility"].items():
        exact_counts[checkpoint] = {
            source: sum(value["exact_prefix"] for value in values)
            for source, values in sources.items()
        }
    print(json.dumps({
        "output": str(args.output), "candidates": len(payload["candidates"]),
        "opponent_families": len(payload["opponent_representatives"]),
        "candidate_ids": [value["route_id"] for value in payload["candidates"]],
        "exact_prefix_peer_counts": exact_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
