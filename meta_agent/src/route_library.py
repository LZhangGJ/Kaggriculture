#!/usr/bin/env python3
"""Compile replay targets into a compact trajectory library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

try:
    from .fingerprints import LandStateTracker, observation_fingerprint
except ImportError:  # Direct ``python meta_agent/src/route_library.py`` execution.
    from fingerprints import LandStateTracker, observation_fingerprint


QUADRANTS = ("NW", "NE", "SW", "SE")


def canonical_action(action: Any) -> dict[str, Any]:
    value = action if isinstance(action, dict) else {}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (value.get("hands") or [])],
        "market": [list(item or ["PASS"]) for item in (value.get("market") or [])],
    }


def semantic_action(action: dict[str, Any]) -> dict[str, Any]:
    def branch(value: Any, fallback: str) -> tuple[str, str]:
        row = list(value or [fallback])
        return str(row[0]), str(row[1]) if len(row) >= 2 else ""

    return {
        "farmer": branch(action.get("farmer"), "PASS"),
        "hands": [branch(value, "PASS") for value in action.get("hands", [])],
        "market": [branch(value, "PASS") for value in action.get("market", [])],
    }


def action_hash(actions: list[dict[str, Any]], *, semantic: bool = False) -> str:
    values = [semantic_action(action) for action in actions] if semantic else actions
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def action_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_semantic = semantic_action(left)
    right_semantic = semantic_action(right)
    pairs = [(left_semantic["farmer"], right_semantic["farmer"])]
    for key in ("hands", "market"):
        left_rows = left_semantic[key]
        right_rows = right_semantic[key]
        size = max(len(left_rows), len(right_rows))
        fallback = ("PASS", "") if key == "hands" else ("NONE", "")
        pairs.extend(
            (
                left_rows[index] if index < len(left_rows) else fallback,
                right_rows[index] if index < len(right_rows) else fallback,
            )
            for index in range(size)
        )
    return sum(left_value == right_value for left_value, right_value in pairs) / max(1, len(pairs))


def active_action_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Compare only slots where either route requests a non-idle operation."""
    left_semantic = semantic_action(left)
    right_semantic = semantic_action(right)
    pairs = [(left_semantic["farmer"], right_semantic["farmer"])]
    for key in ("hands", "market"):
        left_rows = left_semantic[key]
        right_rows = right_semantic[key]
        size = max(len(left_rows), len(right_rows))
        fallback = ("PASS", "") if key == "hands" else ("NONE", "")
        pairs.extend(
            (
                left_rows[index] if index < len(left_rows) else fallback,
                right_rows[index] if index < len(right_rows) else fallback,
            )
            for index in range(size)
        )
    active = [pair for pair in pairs if pair[0][0] not in {"PASS", "NONE"} or pair[1][0] not in {"PASS", "NONE"}]
    if not active:
        return 1.0
    return sum(left_value == right_value for left_value, right_value in active) / len(active)


def _quadrant(position: Any) -> str | None:
    try:
        x, y = int(position[0]), int(position[1])
    except (IndexError, TypeError, ValueError):
        return None
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def _day_plans(
    steps: list[Any], player: int, tracker: LandStateTracker
) -> list[dict[str, Any]]:
    """Annotate each expert day without changing its executable action trace."""
    plans = []
    for day in range(30):
        start = day * 24
        if start >= len(steps):
            break
        first_observation = steps[start][player].get("observation") or {}
        tracker.update(first_observation)
        start_fingerprint = observation_fingerprint(
            first_observation, unlock_days=tracker.unlock_days
        )
        hires = 0
        worker_hours = {name: 0 for name in QUADRANTS}
        operations: dict[str, dict[str, int]] = {name: {} for name in QUADRANTS}
        for step_index in range(start, min(start + 24, len(steps) - 1)):
            observation = steps[step_index][player].get("observation") or {}
            tracker.update(observation)
            farms = list(observation.get("farms", []) or [])
            farm = farms[player] if player < len(farms) else {}
            action = canonical_action(steps[step_index + 1][player].get("action"))
            positions = [farm.get("farmer"), *(farm.get("hands") or [])]
            orders = [action["farmer"], *action["hands"]]
            while len(orders) < len(positions):
                orders.append(["PASS"])
            for position, order in zip(positions, orders):
                name = _quadrant(position)
                if name is None:
                    continue
                worker_hours[name] += 1
                op = str((order or ["PASS"])[0])
                if op not in {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}:
                    operations[name][op] = operations[name].get(op, 0) + 1
            hires += sum(
                str((order or ["NONE"])[0]) == "HIRE" for order in action["market"]
            )
        plans.append(
            {
                "day": day,
                "unlock_days": dict(tracker.unlock_days),
                "lands": start_fingerprint["lands"],
                "planned_hires": hires,
                "worker_hours": worker_hours,
                "operations": operations,
            }
        )
    return plans


def _checkpoint_fingerprints(
    steps: list[Any], player: int, checkpoints: tuple[int, ...]
) -> dict[str, dict[str, Any]]:
    tracker = LandStateTracker()
    wanted = set(checkpoints)
    fingerprints: dict[str, dict[str, Any]] = {}
    for step_index in range(min(len(steps), max(wanted, default=0) + 1)):
        observation = steps[step_index][player].get("observation") or {}
        tracker.update(observation)
        if step_index in wanted:
            fingerprints[str(step_index)] = observation_fingerprint(
                observation, unlock_days=tracker.unlock_days
            )
    return fingerprints


def _build_record(task: tuple[str, dict[str, Any], tuple[int, ...]]) -> dict[str, Any]:
    replay_path_text, target, checkpoints = task
    replay = json.loads(Path(replay_path_text).read_text(encoding="utf-8"))
    steps = replay.get("steps", [])
    player = int(target["player_index"])
    actions = [canonical_action(steps[index][player].get("action")) for index in range(1, len(steps))]
    info = replay.get("info", {}) or {}
    names = info.get("TeamNames", []) or []
    tracker = LandStateTracker()
    return {
        "route_id": f"{int(info.get('EpisodeId') or 0)}:{player}",
        "episode_id": int(info.get("EpisodeId") or 0),
        "team": target.get("team_name"),
        "opponent": names[1 - player] if len(names) >= 2 else None,
        "player_index": player,
        "leaderboard_score": float(target.get("leaderboard_score", 0.0) or 0.0),
        "final_reward": float(target.get("final_reward", 0.0) or 0.0),
        "actions": actions,
        "prefix_hashes": {
            str(checkpoint): action_hash(actions[:checkpoint]) for checkpoint in checkpoints
        },
        "semantic_prefix_hashes": {
            str(checkpoint): action_hash(actions[:checkpoint], semantic=True) for checkpoint in checkpoints
        },
        "fingerprints": _checkpoint_fingerprints(steps, player, checkpoints),
        "day_plans": _day_plans(steps, player, tracker),
    }


def load_library(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.name.endswith(".pkl.gz"):
        with gzip.open(source, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid runtime route library: {source}")
        return payload
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(source.read_text(encoding="utf-8"))


def save_library(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".pkl.gz"):
        with gzip.open(destination, "wb", compresslevel=9) as handle:
            pickle.dump(payload, handle, protocol=5)
    elif destination.suffix == ".gz":
        with gzip.open(destination, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    else:
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-target-sides", type=int, default=96)
    parser.add_argument(
        "--team",
        action="append",
        default=[],
        help="Optional exact team-name filter; may be repeated.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=(0, 24, 48, 72, 168, 240, 480, 648)
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = [
        (row, target)
        for row in manifest.get("replays", [])
        if row.get("error") is None
        for target in row.get("targets", [])
        if not args.team or target.get("team_name") in set(args.team)
    ]
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.max_target_sides]
    tasks = [
        (str(args.manifest.parent / row["replay"]), target, tuple(args.checkpoints))
        for row, target in selected
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(_build_record, tasks))
    payload = {
        "schema_version": 2,
        "manifest": str(args.manifest),
        "seed": args.seed,
        "checkpoints": list(args.checkpoints),
        "records": records,
    }
    save_library(payload, args.output)
    print(
        json.dumps(
            {
                "records": len(records),
                "teams": len({record["team"] for record in records}),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
