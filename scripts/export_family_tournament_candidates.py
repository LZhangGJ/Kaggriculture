#!/usr/bin/env python3
"""Export complete replay tapes for within-family native tournaments."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np


def _actions(path: Path, player: int) -> list[dict]:
    import orjson

    replay = orjson.loads(path.read_bytes())
    return [
        {
            "farmer": list((replay["steps"][step][player].get("action") or {}).get("farmer") or ["PASS"]),
            "hands": [
                list(value or ["PASS"])
                for value in ((replay["steps"][step][player].get("action") or {}).get("hands") or [])
            ],
            "market": [
                list(value)
                for value in ((replay["steps"][step][player].get("action") or {}).get("market") or [])
            ],
        }
        for step in range(1, 720)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--distance", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--top-families", type=int, default=56)
    parser.add_argument("--max-candidates", type=int, default=16)
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.families) as cached:
        labels = cached["labels"].astype(int)
        complete = cached["complete_mask"].astype(bool)
        has_core = cached["family_has_complete_core"].astype(bool)
    size = len(rows)
    distances = np.memmap(args.distance, mode="r", dtype=np.float32, shape=(size, size))
    replay_paths = {}
    for root in args.replay_root:
        for path in root.glob("episode-*-replay.json"):
            replay_paths.setdefault(int(path.name.split("-")[1]), path)

    ranked_families = []
    for label in np.flatnonzero(has_core):
        members = np.flatnonzero(labels == label)
        ranked_families.append((
            len(members),
            float(np.median([float(rows[index]["reward"]) for index in members])),
            int(label), members,
        ))
    ranked_families.sort(key=lambda value: (-value[0], -value[1], value[2]))
    ranked_families = ranked_families[:args.top_families]

    action_tapes = {}
    entries = []
    family_records = []
    for family_rank, (support, median_reward, label, members) in enumerate(ranked_families, 1):
        family = f"G{family_rank:02d}"
        candidates = members[complete[members]]
        within = np.asarray(distances[np.ix_(candidates, candidates)])
        medoid_score = within.mean(axis=1)
        rewards = np.asarray([float(rows[index]["reward"]) for index in candidates])
        min_money = np.asarray([float(rows[index]["min_money"]) for index in candidates])
        leaderboard = np.asarray([float(rows[index]["score"]) for index in candidates])

        selected_positions = []
        for ordering in (
            np.argsort(medoid_score), np.argsort(-rewards),
            np.argsort(-min_money), np.argsort(-leaderboard),
        ):
            for position in ordering[:4]:
                if int(position) not in selected_positions:
                    selected_positions.append(int(position))
        if len(selected_positions) < min(args.max_candidates, len(candidates)):
            ranks = np.empty((4, len(candidates)), dtype=np.float64)
            for row_index, values in enumerate((medoid_score, -rewards, -min_money, -leaderboard)):
                order = np.argsort(values)
                ranks[row_index, order] = np.arange(len(candidates))
            for position in np.argsort(ranks.mean(axis=0)):
                if int(position) not in selected_positions:
                    selected_positions.append(int(position))
        selected_positions = selected_positions[:args.max_candidates]

        candidate_labels = []
        for candidate_rank, position in enumerate(selected_positions, 1):
            index = int(candidates[position])
            row = rows[index]
            route_id = f"{int(row['episode_id'])}:{int(row['player_index'])}"
            candidate_label = f"{family}-C{candidate_rank:02d}"
            tape = _actions(replay_paths[int(row["episode_id"])], int(row["player_index"]))
            action_tapes[route_id] = tape
            encoded = json.dumps(tape, sort_keys=True, separators=(",", ":")).encode()
            record = {
                "family": candidate_label,
                "macro_family": family,
                "source_cluster_label": label,
                "route_id": route_id,
                "episode_id": int(row["episode_id"]),
                "player_index": int(row["player_index"]),
                "team": str(row["team"]),
                "historical_reward": float(row["reward"]),
                "leaderboard_score": float(row["score"]),
                "min_money": float(row["min_money"]),
                "core_medoid_distance": float(medoid_score[position]),
                "action_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            entries.append(record)
            candidate_labels.append(candidate_label)
        family_records.append({
            "family": family,
            "source_cluster_label": label,
            "support": support,
            "complete_candidates_available": len(candidates),
            "tournament_candidates": candidate_labels,
            "historical_median_reward": median_reward,
        })

    packed = zlib.compress(
        json.dumps(action_tapes, ensure_ascii=False, separators=(",", ":")).encode(),
        level=9,
    )
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    args.actions.write_bytes(packed)
    payload = {
        "schema_version": 1,
        "taxonomy": "complete-core intended macro families",
        "selection": "union of core medoid, reward, capital-slack, leaderboard ranks",
        "families": family_records,
        "opponent_routes": entries,
        "candidate_count": len(entries),
        "actions_file": str(args.actions),
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "families": len(family_records), "candidates": len(entries),
        "compressed_bytes": len(packed),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
