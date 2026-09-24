#!/usr/bin/env python3
"""Export DSM opening representatives without changing the deployed route library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import orjson
from sklearn.cluster import AgglomerativeClustering

from analyze_macro_route_library import _distance
from export_teammate_route_library import _actions
from select_major_route_representatives import _prototype
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--base-actions", type=Path)
    parser.add_argument("--base-metadata", type=Path)
    args = parser.parse_args()
    if bool(args.base_actions) != bool(args.base_metadata):
        parser.error("--base-actions and --base-metadata must be supplied together")

    manifest = json.loads(args.manifest.read_text())
    with np.load(args.features, allow_pickle=True) as saved:
        rows = list(saved["rows"])
    with np.load(args.audit) as saved:
        keys = saved["keys"]
        layouts = saved["planned_layouts"]
        reward_abs_error = saved["reward_abs_error"]
    expected = np.asarray([(int(r["episode_id"]), int(r["player_index"])) for r in rows])
    if not np.array_equal(keys, expected):
        raise ValueError("feature and audit rows are not aligned")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    replays = {}
    seeds = {}
    rewards = {}
    compatible = dict(manifest)
    compatible["replays"] = []
    for record in manifest["replays"]:
        path = (args.manifest.parent / record["replay"]).resolve()
        episode = int(record["episode_id"])
        if not path.is_file():
            raise FileNotFoundError(path)
        known_rewards = record.get("rewards")
        if known_rewards is None:
            replay = orjson.loads(path.read_bytes())
            if len(replay["steps"]) != 720 or len(replay.get("rewards") or []) != 2:
                raise ValueError(f"incomplete replay: {path}")
            if int(replay["info"]["EpisodeId"]) != episode:
                raise ValueError(f"episode mismatch: {path}")
            known_rewards = replay["rewards"]
            seeds[episode] = int(replay["info"]["seed"])
        elif len(known_rewards) != 2:
            raise ValueError(f"missing rewards: {path}")
        if record.get("seed") is not None:
            seeds[episode] = int(record["seed"])
        replays[episode] = path
        rewards[episode] = tuple(float(x) for x in known_rewards)
        compatible["replays"].append({
            **record,
            "replay": os.path.relpath(path, output),
            "rewards": known_rewards,
        })
    if set(expected[:, 0]) != set(replays):
        raise ValueError("manifest and feature episodes differ")
    (output / "manifest.compat.json").write_text(json.dumps(compatible, ensure_ascii=False, indent=2) + "\n")

    # At day 12 the worker array has reset; use intended tile layouts and the
    # first four 72-turn schedule phases to cluster only the opening.
    opening = [
        dict(row, production=row["production"][:12], layouts=layouts[i, :2],
             schedule=row["schedule"][:4])
        for i, row in enumerate(rows)
    ]
    distances = np.asarray(
        [[_distance(left, right) for right in opening] for left in opening],
        dtype=np.float32,
    )
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=args.threshold,
    ).fit_predict(distances)
    np.savez_compressed(output / "prefix-clusters.npz", keys=keys, labels=labels, distances=distances)

    groups = []
    for label in set(labels.tolist()):
        members = np.flatnonzero(labels == label)
        prototype = _prototype([opening[i] for i in members])
        winners = [i for i in members if reward_abs_error[i] == 0 and
                   rewards[int(rows[i]["episode_id"])][int(rows[i]["player_index"])]
                   > rewards[int(rows[i]["episode_id"])][1 - int(rows[i]["player_index"])]]
        verified = [i for i in members if reward_abs_error[i] == 0]
        pool = winners or verified or members
        representative = min(
            pool,
            key=lambda i: (_distance(opening[i], prototype), -float(rows[i]["reward"]),
                           int(rows[i]["episode_id"])),
        )
        groups.append((members, int(representative), len(winners),
                       float(_distance(opening[representative], prototype))))
    groups.sort(key=lambda item: (-len(item[0]), int(rows[item[1]]["episode_id"])))

    full_tapes = {}
    prefix_tapes = {}
    entries = []
    for rank, (members, representative, wins, prototype_distance) in enumerate(groups, 1):
        row = rows[representative]
        episode, player = int(row["episode_id"]), int(row["player_index"])
        route_id = f"{episode}:{player}"
        tape = _actions(replays[episode], player)
        if episode not in seeds:
            seeds[episode] = int(orjson.loads(replays[episode].read_bytes())["info"]["seed"])
        full_tapes[route_id] = tape
        prefix_tapes[route_id] = tape[:288]
        entries.append({
            "family": f"D{rank:03d}", "route_id": route_id, "team": "DSM",
            "support": len(members), "selected": len(members) >= 2 and wins > 0 and
            int(row["production"][11, 10]) == 4,
            "historical_wins": wins,
            "reward_verified_support": int(sum(reward_abs_error[i] == 0 for i in members)),
            "historical_margin": rewards[episode][player] - rewards[episode][1 - player],
            "historical_reward": rewards[episode][player],
            "prototype_distance": prototype_distance,
            "episode_ids": sorted(int(rows[i]["episode_id"]) for i in members),
            "seed": seeds[episode],
            "prefix_sha256": hashlib.sha256(orjson.dumps(tape[:288])).hexdigest(),
            "step288_land": int(row["production"][11, 10]),
        })
    packed = zlib.compress(orjson.dumps(full_tapes), level=9)
    (output / "route-actions-719.json.zlib").write_bytes(packed)
    (output / "prefix-actions-288.json.zlib").write_bytes(
        zlib.compress(orjson.dumps(prefix_tapes), level=9)
    )
    metadata = {
        "schema_version": 1, "taxonomy": "DSM step0..287 intended-prefix medoids",
        "carrier_role": "full 719-step source tapes for native lookahead; handoff at step 288",
        "threshold": args.threshold, "opponent_routes": entries,
        "selected": [entry for entry in entries if entry["selected"]],
        "actions_file": "route-actions-719.json.zlib",
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    (output / "route-library.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.base_actions:
        base_tapes = load_action_tapes(args.base_actions)
        base_metadata = json.loads(args.base_metadata.read_text())
        if base_tapes.keys() & full_tapes.keys():
            raise ValueError("DSM tape ID collides with base library")
        base_families = {entry["family"] for entry in base_metadata["opponent_routes"]}
        if base_families & {entry["family"] for entry in entries}:
            raise ValueError("DSM family collides with base library")
        combined = output / "combined"
        combined.mkdir(exist_ok=True)
        combined_tapes = zlib.compress(orjson.dumps({**base_tapes, **full_tapes}), level=9)
        (combined / "route-actions.json.zlib").write_bytes(combined_tapes)
        combined_metadata = {
            **base_metadata,
            "opponent_routes": [*base_metadata["opponent_routes"], *entries],
            "selected": [*base_metadata.get("selected", []), *metadata["selected"]],
            "actions_file": "route-actions.json.zlib",
            "actions_sha256": hashlib.sha256(combined_tapes).hexdigest(),
        }
        (combined / "route-library.json").write_text(
            json.dumps(combined_metadata, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({"replays": len(replays), "families": len(entries),
                      "support": [e["support"] for e in entries],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
