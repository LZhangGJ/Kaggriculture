#!/usr/bin/env python3
"""Export winning macro-family tapes for the teammate execution stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_macro_route_library import _distance
from select_major_route_representatives import _family_groups, _prototype


def _margin(row: dict[str, Any], rewards: dict[int, list[float]]) -> float | None:
    values = rewards.get(int(row["episode_id"]), [])
    player = int(row["player_index"])
    if len(values) != 2:
        return None
    return float(values[player]) - float(values[1 - player])


def _representative(rows: list[dict[str, Any]], rewards: dict[int, list[float]]) -> tuple[dict[str, Any], float]:
    prototype = _prototype(rows)
    victorious = [row for row in rows if (_margin(row, rewards) or 0.0) > 0.0]
    pool = victorious or rows
    ranked = sorted(
        ((_distance(row, prototype), row) for row in pool),
        key=lambda value: (value[0], -float(value[1]["reward"]), int(value[1]["episode_id"])),
    )
    return ranked[0][1], float(ranked[0][0])


def _actions(replay_path: Path, player: int) -> list[dict[str, Any]]:
    import orjson

    replay = orjson.loads(replay_path.read_bytes())
    steps = list(replay.get("steps") or [])
    result = []
    for step in range(719):
        action = steps[step + 1][player].get("action") or {}
        result.append(
            {
                "farmer": list(action.get("farmer") or ["PASS"]),
                "hands": [list(value or ["PASS"]) for value in action.get("hands", [])],
                "market": [list(value) for value in action.get("market", [])],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--families", type=int, default=56)
    parser.add_argument("--rare-after", type=int, default=8)
    args = parser.parse_args()

    with np.load(args.macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.global_clusters) as cached:
        labels = cached["labels"].astype(int)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rewards = {
        int(replay["episode_id"]): [float(value) for value in replay.get("rewards", [])]
        for replay in manifest.get("replays", [])
    }
    replay_paths = {
        int(replay["episode_id"]): args.manifest.parent / str(replay["replay"])
        for replay in manifest.get("replays", [])
    }

    groups = _family_groups(rows, labels)
    selected = []
    dropped = []
    opponent_routes = []
    action_tapes: dict[str, list[dict[str, Any]]] = {}
    tape_hash_to_route: dict[str, str] = {}
    for index, (family, values) in enumerate(groups[: args.families], start=1):
        margins = [value for row in values if (value := _margin(row, rewards)) is not None]
        wins = sum(value > 0 for value in margins)
        losses = sum(value < 0 for value in margins)
        ties = sum(value == 0 for value in margins)
        outcome = {"games": len(margins), "wins": wins, "losses": losses, "ties": ties}
        record = {
            "family": family,
            "support": len(values),
            "outcome": outcome,
            "median_reward": float(np.median([float(row["reward"]) for row in values])),
        }
        representative, prototype_distance = _representative(values, rewards)
        route_id = f"{int(representative['episode_id'])}:{int(representative['player_index'])}"
        tape = _actions(
            replay_paths[int(representative["episode_id"])],
            int(representative["player_index"]),
        )
        encoded = json.dumps(tape, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        duplicate_of = tape_hash_to_route.get(digest)
        if duplicate_of is None:
            tape_hash_to_route[digest] = route_id
            action_tapes[route_id] = tape
        record.update(
            route_id=route_id,
            team=str(representative["team"]),
            historical_reward=float(representative["reward"]),
            prototype_distance=prototype_distance,
            action_sha256=digest,
            duplicate_of=duplicate_of,
        )
        opponent_routes.append(record)
        if index > args.rare_after and margins and wins == 0:
            dropped.append(record)
        else:
            selected.append(record)

    import orjson

    packed = zlib.compress(orjson.dumps(action_tapes), level=9)
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    args.actions.write_bytes(packed)
    payload = {
        "schema_version": 1,
        "taxonomy": f"top {args.families} fresh global clusters at macro distance 0.12",
        "zero_win_policy": f"drop G>{args.rare_after} when observed games>0 and wins=0",
        "selected": selected,
        "opponent_routes": opponent_routes,
        "dropped_zero_win_rare": dropped,
        "unique_action_tapes": len(action_tapes),
        "actions_file": str(args.actions),
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "dropped": len(dropped),
                "unique_tapes": len(action_tapes),
                "compressed_bytes": len(packed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
