#!/usr/bin/env python3
"""Recover failure-independent macro intent from Kaggriculture replay files.

The Kaggle replay stores the action for turn ``t`` in replay step ``t + 1``.
For spatial intent we therefore take the actor position from the preceding
observation, while success is checked in the resulting observation.

This script treats replay JSON as data only. It never imports or executes agent
code embedded in, adjacent to, or referenced by a replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANCHORS = (168, 288, 432, 576, 719)
STAGE_WIDTH = 72
STAGE_COUNT = 10
INTENT_CATEGORIES = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "GOOSE",
    "COW",
    "SHEEP",
    "COOP",
    "PASTURE",
)
MACRO_ACTIONS = (*INTENT_CATEGORIES, "BUY_LAND", "HIRE")
INTENT_CODE = {name: index + 1 for index, name in enumerate(INTENT_CATEGORIES)}
MACRO_INDEX = {name: index for index, name in enumerate(MACRO_ACTIONS)}
SPATIAL_OPS = {"PLANT", "PLACE", "BUILD_COOP", "BUILD_PASTURE"}


@dataclass(frozen=True)
class Target:
    rank: int
    team_id: str
    team: str
    submission_id: str
    public_score: float
    episode_type: str
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-root",
        action="append",
        type=Path,
        default=[],
        help="Search this root for an alternate copy only when the manifest copy cannot be parsed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many unique episodes; 0 means all.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def normalize_name(value: Any) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        # The two supplied manifests use `team` and `team_name` respectively.
        if not row.get("team") and row.get("team_name"):
            row["team"] = str(row["team_name"])
    required = {
        "rank",
        "team_id",
        "team",
        "submission_id",
        "public_score",
        "episode_id",
        "episode_type",
        "replay_path",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def unique_targets(rows: Sequence[Mapping[str, str]]) -> list[Target]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Target] = []
    for row in rows:
        key = (str(row["team_id"]), str(row["submission_id"]), str(row["team"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            Target(
                rank=integer(row["rank"]),
                team_id=str(row["team_id"]),
                team=str(row["team"]),
                submission_id=str(row["submission_id"]),
                public_score=number(row["public_score"]),
                episode_type=str(row["episode_type"]),
                source_path=str(row["replay_path"]),
            )
        )
    return result


def group_manifest(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[Target]], dict[str, list[Path]]]:
    targets: dict[str, list[Target]] = defaultdict(list)
    paths: dict[str, list[Path]] = defaultdict(list)
    grouped_rows: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["episode_id"])].append(row)
        candidate = Path(str(row["replay_path"]))
        if candidate not in paths[str(row["episode_id"])]:
            paths[str(row["episode_id"])].append(candidate)
    for episode_id, episode_rows in grouped_rows.items():
        targets[episode_id] = unique_targets(episode_rows)
    return dict(targets), dict(paths)


def choose_replay_path(candidates: Sequence[Path]) -> Path | None:
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    # Prefer the largest copy when duplicated downloads differ or one is partial.
    return max(existing, key=lambda path: path.stat().st_size)


def load_replay_with_fallback(
    episode_id: str,
    primary_candidates: Sequence[Path],
    reuse_roots: Sequence[Path],
) -> tuple[Mapping[str, Any] | None, Path | None, list[str]]:
    errors: list[str] = []
    attempted: set[str] = set()

    def try_paths(paths: Iterable[Path]) -> tuple[Mapping[str, Any] | None, Path | None]:
        existing = [path for path in paths if path.is_file()]
        for path in sorted(existing, key=lambda value: value.stat().st_size, reverse=True):
            key = str(path.resolve()).casefold()
            if key in attempted:
                continue
            attempted.add(key)
            try:
                with path.open("r", encoding="utf-8") as stream:
                    return json.load(stream), path
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        return None, None

    payload, source_path = try_paths(primary_candidates)
    if payload is not None:
        return payload, source_path, errors

    filename = f"episode-{episode_id}-replay.json"
    alternatives: list[Path] = []
    for root in reuse_roots:
        if root.is_dir():
            alternatives.extend(root.rglob(filename))
    payload, source_path = try_paths(alternatives)
    return payload, source_path, errors


def replay_episode_id(payload: Mapping[str, Any]) -> str:
    info = payload.get("info") or {}
    return str(info.get("EpisodeId") or payload.get("id") or "")


def team_names(payload: Mapping[str, Any]) -> list[str]:
    info = payload.get("info") or {}
    names = list(info.get("TeamNames") or [])
    if not names:
        names = [str(agent.get("Name") or "") for agent in info.get("Agents") or []]
    return [str(value) for value in names]


def target_seats(
    target: Target,
    names: Sequence[str],
    inferred_aliases: Sequence[str] = (),
) -> tuple[list[int], str]:
    exact = [index for index, value in enumerate(names) if value == target.team]
    if exact:
        return exact, "exact"
    normalized_target = normalize_name(target.team)
    normalized = [
        index for index, value in enumerate(names) if normalize_name(value) == normalized_target
    ]
    if normalized:
        return normalized, "normalized"
    for inferred_alias in inferred_aliases:
        alias_exact = [index for index, value in enumerate(names) if value == inferred_alias]
        if alias_exact:
            return alias_exact, "inferred_alias"
        normalized_alias = normalize_name(inferred_alias)
        alias_normalized = [
            index for index, value in enumerate(names) if normalize_name(value) == normalized_alias
        ]
        if alias_normalized:
            return alias_normalized, "inferred_alias_normalized"
    return [], "unmatched"


def infer_team_aliases(
    targets_by_episode: Mapping[str, Sequence[Target]],
    paths_by_episode: Mapping[str, Sequence[Path]],
    reuse_roots: Sequence[Path],
    sample_per_submission: int = 12,
) -> dict[tuple[str, str], list[str]]:
    episodes_by_submission: dict[tuple[str, str], list[str]] = defaultdict(list)
    target_by_submission: dict[tuple[str, str], Target] = {}
    for episode_id in sorted(targets_by_episode, key=lambda value: integer(value, 10**18)):
        for target in targets_by_episode[episode_id]:
            key = (target.team_id, target.submission_id)
            target_by_submission[key] = target
            if len(episodes_by_submission[key]) < sample_per_submission:
                episodes_by_submission[key].append(episode_id)

    aliases: dict[tuple[str, str], list[str]] = {}
    for key, episode_ids in episodes_by_submission.items():
        counts: Counter[str] = Counter()
        parsed = 0
        for episode_id in episode_ids:
            payload, _, _ = load_replay_with_fallback(
                episode_id, paths_by_episode[episode_id], reuse_roots
            )
            if payload is None:
                continue
            parsed += 1
            counts.update(team_names(payload))
        if not counts or parsed == 0:
            continue
        alias, frequency = counts.most_common(1)[0]
        # A target appears in every sampled replay (twice in self-play), while
        # ordinary opponents vary. Requiring 75% protects against a repeated
        # opponent being mistaken for an old display name.
        if frequency >= max(2, int(np.ceil(parsed * 0.75))):
            aliases[key] = [alias]
    return aliases


def add_directory_aliases(
    aliases: dict[tuple[str, str], list[str]],
    targets_by_episode: Mapping[str, Sequence[Target]],
    manifest_root: Path,
) -> None:
    targets: dict[tuple[str, str], Target] = {}
    for episode_targets in targets_by_episode.values():
        for target in episode_targets:
            targets[(target.team_id, target.submission_id)] = target
    for key, target in targets.items():
        values = aliases.setdefault(key, [])
        for directory in manifest_root.glob(f"{target.team_id}_*"):
            if not directory.is_dir():
                continue
            alias = directory.name[len(target.team_id) + 1 :].replace("_", " ")
            if not alias:
                continue
            if all(normalize_name(alias) != normalize_name(value) for value in values):
                values.append(alias)


def farm_from_observation(observation: Mapping[str, Any], seat: int) -> Mapping[str, Any]:
    farms = list(observation.get("farms") or [])
    value = farms[seat] if 0 <= seat < len(farms) else {}
    return value if isinstance(value, Mapping) else {}


def actor_pairs(
    action: Mapping[str, Any], previous_farm: Mapping[str, Any]
) -> Iterable[tuple[str, int, Sequence[Any], tuple[int, int] | None]]:
    farmer_position = previous_farm.get("farmer")
    farmer_coordinate = coordinate(farmer_position)
    farmer_action = action.get("farmer")
    if isinstance(farmer_action, Sequence) and not isinstance(farmer_action, (str, bytes)):
        yield "farmer", -1, farmer_action, farmer_coordinate

    positions = list(previous_farm.get("hands") or [])
    for actor_index, order in enumerate(action.get("hands") or []):
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            continue
        actor_coordinate = coordinate(positions[actor_index]) if actor_index < len(positions) else None
        yield "hand", actor_index, order, actor_coordinate


def coordinate(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    row, column = integer(value[0], -1), integer(value[1], -1)
    if 0 <= row < 10 and 0 <= column < 10:
        return row, column
    return None


def tile_at(farm: Mapping[str, Any], location: tuple[int, int] | None) -> Any:
    if location is None:
        return None
    tiles = list(farm.get("tiles") or [])
    row, column = location
    if row >= len(tiles) or not isinstance(tiles[row], Sequence) or column >= len(tiles[row]):
        return None
    return tiles[row][column]


def spatial_intent(order: Sequence[Any]) -> tuple[str, str] | None:
    if not order:
        return None
    operation = str(order[0])
    if operation == "PLANT" and len(order) >= 2:
        item = str(order[1])
    elif operation == "PLACE" and len(order) >= 2:
        item = str(order[1])
    elif operation == "BUILD_COOP":
        item = "COOP"
    elif operation == "BUILD_PASTURE":
        item = "PASTURE"
    else:
        return None
    if item not in INTENT_CODE:
        return None
    return operation, item


def observed_success(operation: str, item: str, resulting_tile: Any) -> bool:
    if not isinstance(resulting_tile, Mapping):
        return False
    if operation == "PLANT":
        return str(resulting_tile.get("crop") or "") == item
    if operation == "PLACE":
        return str(resulting_tile.get("animal") or "") == item
    if operation == "BUILD_COOP":
        return str(resulting_tile.get("kind") or "") == "COOP"
    if operation == "BUILD_PASTURE":
        return str(resulting_tile.get("kind") or "") == "PASTURE"
    return False


def market_macro_actions(action: Mapping[str, Any]) -> Iterable[tuple[str, Sequence[Any]]]:
    for raw in action.get("market") or []:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            continue
        operation = str(raw[0])
        if operation in {"BUY_LAND", "HIRE"}:
            yield operation, raw


def snapshot_layout(
    events: Sequence[tuple[int, int, int]], anchors: Sequence[int] = ANCHORS
) -> np.ndarray:
    result = np.zeros((len(anchors), 100), dtype=np.uint8)
    layout = np.zeros(100, dtype=np.uint8)
    event_index = 0
    for anchor_index, anchor in enumerate(anchors):
        while event_index < len(events) and events[event_index][0] < anchor:
            _, flat_coordinate, code = events[event_index]
            layout[flat_coordinate] = code
            event_index += 1
        result[anchor_index] = layout
    return result


def seat_metadata(
    payload: Mapping[str, Any],
    target: Target,
    episode_id: str,
    seat: int,
    names: Sequence[str],
    mapping_method: str,
    source_path: Path,
    steps_length: int,
) -> dict[str, Any]:
    rewards = list(payload.get("rewards") or [])
    reward = number(rewards[seat] if seat < len(rewards) else None)
    opponent_seat = 1 - seat if len(names) == 2 else -1
    opponent_reward = number(
        rewards[opponent_seat] if 0 <= opponent_seat < len(rewards) else None
    )
    result = "TIE" if reward == opponent_reward else ("WIN" if reward > opponent_reward else "LOSS")
    return {
        "seat_index": -1,
        "episode_id": episode_id,
        "player_index": seat,
        "rank": target.rank,
        "team_id": target.team_id,
        "team": target.team,
        "submission_id": target.submission_id,
        "public_score": target.public_score,
        "episode_type": target.episode_type,
        "replay_team_name": names[seat] if seat < len(names) else "",
        "opponent_team": names[opponent_seat] if 0 <= opponent_seat < len(names) else "",
        "reward": reward,
        "opponent_reward": opponent_reward,
        "result": result,
        "mapping_method": mapping_method,
        "replay_steps": steps_length,
        "source_bytes": source_path.stat().st_size,
        "source_path": str(source_path),
    }


def extract_seat(
    payload: Mapping[str, Any],
    target: Target,
    episode_id: str,
    seat: int,
    names: Sequence[str],
    mapping_method: str,
    source_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    steps = list(payload.get("steps") or [])
    schedule = np.zeros((STAGE_COUNT, len(MACRO_ACTIONS)), dtype=np.uint16)
    spatial_events: list[tuple[int, int, int]] = []
    event_rows: list[dict[str, Any]] = []

    for replay_step in range(1, len(steps)):
        step_slots = list(steps[replay_step] or [])
        previous_slots = list(steps[replay_step - 1] or [])
        if seat >= len(step_slots) or seat >= len(previous_slots):
            continue
        slot = step_slots[seat] or {}
        previous_slot = previous_slots[seat] or {}
        action = slot.get("action") or {}
        if not isinstance(action, Mapping):
            continue
        turn = replay_step - 1
        stage = min(STAGE_COUNT - 1, max(0, turn // STAGE_WIDTH))
        previous_farm = farm_from_observation(previous_slot.get("observation") or {}, seat)
        resulting_farm = farm_from_observation(slot.get("observation") or {}, seat)

        for actor_kind, actor_index, order, location in actor_pairs(action, previous_farm):
            recovered = spatial_intent(order)
            if recovered is None:
                continue
            operation, item = recovered
            schedule[stage, MACRO_INDEX[item]] += 1
            flat_coordinate = -1
            success: bool | None = None
            if location is not None:
                row, column = location
                flat_coordinate = row * 10 + column
                spatial_events.append((turn, flat_coordinate, INTENT_CODE[item]))
                success = observed_success(operation, item, tile_at(resulting_farm, location))
            event_rows.append(
                {
                    "episode_id": episode_id,
                    "player_index": seat,
                    "team_id": target.team_id,
                    "submission_id": target.submission_id,
                    "turn": turn,
                    "stage": stage,
                    "actor_kind": actor_kind,
                    "actor_index": actor_index,
                    "operation": operation,
                    "intent_item": item,
                    "row": location[0] if location is not None else "",
                    "column": location[1] if location is not None else "",
                    "flat_coordinate": flat_coordinate if flat_coordinate >= 0 else "",
                    "observed_success": "" if success is None else int(success),
                }
            )

        for operation, _ in market_macro_actions(action):
            schedule[stage, MACRO_INDEX[operation]] += 1
            event_rows.append(
                {
                    "episode_id": episode_id,
                    "player_index": seat,
                    "team_id": target.team_id,
                    "submission_id": target.submission_id,
                    "turn": turn,
                    "stage": stage,
                    "actor_kind": "market",
                    "actor_index": -1,
                    "operation": operation,
                    "intent_item": operation,
                    "row": "",
                    "column": "",
                    "flat_coordinate": "",
                    "observed_success": "",
                }
            )

    spatial_events.sort(key=lambda value: value[0])
    layout = snapshot_layout(spatial_events)
    cumulative = np.cumsum(schedule, axis=0, dtype=np.uint32)
    metadata = seat_metadata(
        payload,
        target,
        episode_id,
        seat,
        names,
        mapping_method,
        source_path,
        len(steps),
    )
    return metadata, layout, schedule, cumulative, event_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(manifest_path)
    targets_by_episode, paths_by_episode = group_manifest(manifest_rows)
    reuse_roots = [path.resolve() for path in args.reuse_root]
    inferred_aliases = infer_team_aliases(
        targets_by_episode, paths_by_episode, reuse_roots
    )
    add_directory_aliases(inferred_aliases, targets_by_episode, manifest_path.parent)
    episode_ids = sorted(targets_by_episode, key=lambda value: integer(value, 10**18))
    if args.limit > 0:
        episode_ids = episode_ids[: args.limit]

    metadata_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    layouts: list[np.ndarray] = []
    schedules: list[np.ndarray] = []
    cumulatives: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    episode_type_counts = Counter(row["episode_type"] for row in manifest_rows)
    step_length_counts: Counter[int] = Counter()
    mapping_counts: Counter[str] = Counter()
    parsed_episodes = 0
    fallback_replays = 0

    for position, episode_id in enumerate(episode_ids, start=1):
        manifest_source_path = choose_replay_path(paths_by_episode[episode_id])
        if manifest_source_path is None:
            failures.append(
                {
                    "episode_id": episode_id,
                    "failure_kind": "missing_replay",
                    "detail": "no manifest replay_path exists",
                }
            )
            continue
        payload, source_path, parse_errors = load_replay_with_fallback(
            episode_id,
            paths_by_episode[episode_id],
            reuse_roots,
        )
        if payload is None or source_path is None:
            failures.append(
                {
                    "episode_id": episode_id,
                    "failure_kind": "json_parse_error",
                    "detail": " | ".join(parse_errors) or "no parseable replay copy",
                }
            )
            continue
        if source_path.resolve() != manifest_source_path.resolve():
            fallback_replays += 1

        actual_episode_id = replay_episode_id(payload)
        if actual_episode_id and actual_episode_id != episode_id:
            failures.append(
                {
                    "episode_id": episode_id,
                    "failure_kind": "episode_id_mismatch",
                    "detail": f"replay reports {actual_episode_id}",
                }
            )
            continue

        names = team_names(payload)
        if len(names) != 2:
            failures.append(
                {
                    "episode_id": episode_id,
                    "failure_kind": "unexpected_team_count",
                    "detail": json.dumps(names, ensure_ascii=False),
                }
            )
            continue

        steps_length = len(payload.get("steps") or [])
        step_length_counts[steps_length] += 1
        parsed_episodes += 1
        emitted_seat_keys: set[tuple[str, int]] = set()
        for target in targets_by_episode[episode_id]:
            seats, mapping_method = target_seats(
                target,
                names,
                inferred_aliases.get((target.team_id, target.submission_id)),
            )
            mapping_counts[mapping_method] += 1
            if not seats:
                failures.append(
                    {
                        "episode_id": episode_id,
                        "failure_kind": "target_team_unmatched",
                        "detail": json.dumps(
                            {"target": target.team, "replay_names": names}, ensure_ascii=False
                        ),
                    }
                )
                continue
            for seat in seats:
                seat_key = (target.submission_id, seat)
                if seat_key in emitted_seat_keys:
                    continue
                emitted_seat_keys.add(seat_key)
                try:
                    metadata, layout, schedule, cumulative, events = extract_seat(
                        payload,
                        target,
                        episode_id,
                        seat,
                        names,
                        mapping_method,
                        source_path,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "episode_id": episode_id,
                            "failure_kind": "seat_extraction_error",
                            "detail": f"seat={seat} {type(error).__name__}: {error}",
                        }
                    )
                    continue
                metadata["seat_index"] = len(metadata_rows)
                metadata_rows.append(metadata)
                layouts.append(layout)
                schedules.append(schedule)
                cumulatives.append(cumulative)
                event_rows.extend(events)

        del payload
        if args.progress_every > 0 and (
            position % args.progress_every == 0 or position == len(episode_ids)
        ):
            print(
                f"processed {position}/{len(episode_ids)} episodes; "
                f"seats={len(metadata_rows)} failures={len(failures)}",
                flush=True,
            )

    layout_array = (
        np.stack(layouts).astype(np.uint8, copy=False)
        if layouts
        else np.zeros((0, len(ANCHORS), 100), dtype=np.uint8)
    )
    schedule_array = (
        np.stack(schedules).astype(np.uint16, copy=False)
        if schedules
        else np.zeros((0, STAGE_COUNT, len(MACRO_ACTIONS)), dtype=np.uint16)
    )
    cumulative_array = (
        np.stack(cumulatives).astype(np.uint32, copy=False)
        if cumulatives
        else np.zeros((0, STAGE_COUNT, len(MACRO_ACTIONS)), dtype=np.uint32)
    )
    seat_ids = np.asarray(
        [
            f"{row['episode_id']}:{row['player_index']}:{row['submission_id']}"
            for row in metadata_rows
        ],
        dtype=str,
    )

    npz_path = output_dir / "intent_features_v1.npz"
    temporary_npz = output_dir / "intent_features_v1.partial.npz"
    np.savez_compressed(
        temporary_npz,
        seat_id=seat_ids,
        layout=layout_array,
        stage_actions=schedule_array,
        cumulative_actions=cumulative_array,
        anchors=np.asarray(ANCHORS, dtype=np.int16),
        intent_categories=np.asarray(INTENT_CATEGORIES, dtype=str),
        macro_actions=np.asarray(MACRO_ACTIONS, dtype=str),
        stage_width=np.asarray(STAGE_WIDTH, dtype=np.int16),
    )
    os.replace(temporary_npz, npz_path)

    metadata_fields = (
        "seat_index",
        "episode_id",
        "player_index",
        "rank",
        "team_id",
        "team",
        "submission_id",
        "public_score",
        "episode_type",
        "replay_team_name",
        "opponent_team",
        "reward",
        "opponent_reward",
        "result",
        "mapping_method",
        "replay_steps",
        "source_bytes",
        "source_path",
    )
    event_fields = (
        "episode_id",
        "player_index",
        "team_id",
        "submission_id",
        "turn",
        "stage",
        "actor_kind",
        "actor_index",
        "operation",
        "intent_item",
        "row",
        "column",
        "flat_coordinate",
        "observed_success",
    )
    failure_fields = ("episode_id", "failure_kind", "detail")
    write_csv(output_dir / "intent_seats_v1.csv", metadata_rows, metadata_fields)
    write_csv(output_dir / "macro_events_v1.csv", event_rows, event_fields)
    write_csv(output_dir / "failures_v1.csv", failures, failure_fields)

    spatial_rows = [row for row in event_rows if row["operation"] in SPATIAL_OPS]
    success_values = [
        integer(row["observed_success"])
        for row in spatial_rows
        if row["observed_success"] != ""
    ]
    per_submission: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        key = str(row["submission_id"])
        value = per_submission.setdefault(
            key,
            {
                "rank": row["rank"],
                "team_id": row["team_id"],
                "team": row["team"],
                "seat_count": 0,
                "WIN": 0,
                "LOSS": 0,
                "TIE": 0,
            },
        )
        value["seat_count"] += 1
        value[str(row["result"])] += 1

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest_path),
        "output_grain": "one row per (episode_id, player_index, submission_id)",
        "manifest_rows": len(manifest_rows),
        "manifest_exact_duplicate_rows": len(manifest_rows)
        - len({tuple(sorted(row.items())) for row in manifest_rows}),
        "manifest_unique_episodes": len(targets_by_episode),
        "selected_unique_episodes": len(episode_ids),
        "duplicate_episode_references": len(manifest_rows) - len(targets_by_episode),
        "parsed_episodes": parsed_episodes,
        "fallback_replays": fallback_replays,
        "output_seats": len(metadata_rows),
        "failures": len(failures),
        "failure_kinds": dict(Counter(row["failure_kind"] for row in failures)),
        "mapping_methods": dict(mapping_counts),
        "inferred_team_aliases": {
            f"{team_id}:{submission_id}": aliases
            for (team_id, submission_id), aliases in sorted(inferred_aliases.items())
        },
        "manifest_episode_types": dict(episode_type_counts),
        "replay_step_lengths": {str(k): v for k, v in sorted(step_length_counts.items())},
        "anchors": list(ANCHORS),
        "stage_width": STAGE_WIDTH,
        "stage_count": STAGE_COUNT,
        "intent_categories": list(INTENT_CATEGORIES),
        "macro_actions": list(MACRO_ACTIONS),
        "feature_shapes": {
            "layout": list(layout_array.shape),
            "stage_actions": list(schedule_array.shape),
            "cumulative_actions": list(cumulative_array.shape),
        },
        "macro_event_rows": len(event_rows),
        "macro_event_operations": dict(Counter(row["operation"] for row in event_rows)),
        "spatial_intent_rows": len(spatial_rows),
        "spatial_missing_coordinate": sum(row["row"] == "" for row in spatial_rows),
        "spatial_observed_success_rate": (
            sum(success_values) / len(success_values) if success_values else None
        ),
        "submissions": per_submission,
        "method_notes": [
            "Replay action for turn t is read from replay step t+1.",
            "Spatial intent uses the actor position in the preceding observation.",
            "Layout identity follows issued PLANT/PLACE/BUILD instructions even when execution fails.",
            "The latest issued production intent at a cell is retained at each anchor.",
            "Movement, care, watering, weeding, harvesting, and ordinary trades are excluded.",
            "BUY_LAND and HIRE are included only in stage action counts.",
        ],
    }
    summary_path = output_dir / "summary_v1.json"
    temporary_summary = output_dir / "summary_v1.json.partial"
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_summary, summary_path)

    # Keep progress output compatible with Windows consoles whose active code
    # page cannot represent every team name.
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
