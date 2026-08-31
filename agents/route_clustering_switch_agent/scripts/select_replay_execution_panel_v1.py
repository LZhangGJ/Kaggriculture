#!/usr/bin/env python3
"""Select a frozen, action-aware replay execution panel for block extraction."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.route_plan import normalized_action  # noqa: E402


SCHEMA = "replay-execution-panel-v1"
HORIZON = 719
ANCHORS = tuple(range(216, 697, 24))
PUBLIC = "EPISODE_TYPE_PUBLIC"
VALIDATION = "EPISODE_TYPE_VALIDATION"
ALLOWED_DATASETS = frozenset({"top40", "our_latest"})
UNIT_BUCKETS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "PRODUCTION", "MAINTENANCE", "OTHER",
)
MARKET_BUCKETS = ("BUY", "SELL", "HIRE", "LAND", "OTHER")
ACTOR_TIME_HASH_DIM = 512
ACTOR_TIME_HASH_ALGORITHM = "sha256-signed-feature-hash-v1"
PRODUCTION = frozenset({"PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE"})
MAINTENANCE = frozenset({
    "HARVEST", "WATER", "FERTILIZE", "PICK", "PICKUP", "DROP", "CLEAN",
})


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frozen_records(
    path: Path, start: int, stop: int, expected_sha256: str,
) -> list[tuple[str, dict[str, Any]]]:
    size = path.stat().st_size
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("pool segment SHA-256 must be 64 lowercase/uppercase hex digits")
    if not (0 <= start < stop <= size):
        raise ValueError(f"invalid pool byte range {start}:{stop} for {size} bytes")
    with path.open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                raise ValueError("pool start byte is not a JSONL boundary")
        handle.seek(stop - 1)
        if handle.read(1) != b"\n":
            raise ValueError("pool end byte is not a committed JSONL boundary")
        handle.seek(start)
        digest = hashlib.sha256()
        result = []
        position = start
        while position < stop:
            line = handle.readline(stop - position)
            if not line.endswith(b"\n"):
                raise ValueError("frozen pool segment contains an incomplete JSONL row")
            position += len(line)
            digest.update(line)
            if line.strip():
                payload = json.loads(line)
                result.append((_sha256_bytes(_canonical_bytes(payload)), payload))
    if digest.hexdigest() != expected:
        raise ValueError(
            f"pool segment SHA-256 mismatch: expected {expected}, got {digest.hexdigest()}"
        )
    if not result:
        raise ValueError("frozen pool segment has no records")
    return result


def _split_rows(
    path: Path, expected_sha256: str,
) -> tuple[dict[int, dict[str, Any]], str, dict[str, Any]]:
    raw = path.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("split manifest SHA-256 must be 64 hex digits")
    if actual_sha256 != expected:
        raise ValueError(
            f"split manifest SHA-256 mismatch: expected {expected}, got {actual_sha256}"
        )
    payload = json.loads(raw)
    rows = {}
    for row in payload.get("episodes", ()):
        episode = int(row["episode_id"])
        if episode in rows:
            raise ValueError(f"duplicate episode in split manifest: {episode}")
        rows[episode] = dict(row)
    if not rows:
        raise ValueError("split manifest has no episodes")
    return rows, actual_sha256, payload


def _gate_source(source: Mapping[str, Any], split: Mapping[str, Any]) -> None:
    episode = int(source["episode_id"])
    if str(source.get("ingestion_split", "")) != "train":
        raise ValueError(f"non-train record reached panel input: {episode}")
    if str(split.get("split", "")) != "train":
        raise ValueError(f"split manifest does not classify episode as train: {episode}")
    datasets = {str(value) for value in source.get("datasets", ()) or ()}
    split_datasets = {str(value) for value in split.get("datasets", ()) or ()}
    if not datasets or not datasets <= ALLOWED_DATASETS:
        raise ValueError(f"record has out-of-scope datasets at episode {episode}: {datasets}")
    if not datasets <= split_datasets:
        raise ValueError(f"record/split dataset provenance mismatch at episode {episode}")
    episode_types = {str(value) for value in split.get("episode_types", ()) or ()}
    if VALIDATION in episode_types:
        raise ValueError(f"validation episode reached panel input: {episode}")
    if "top40" in datasets and PUBLIC not in episode_types:
        raise ValueError(f"top40 episode is not public in frozen split manifest: {episode}")


def _reference(source: Mapping[str, Any]) -> tuple[Path | None, list[dict[str, Any]]]:
    references = [dict(value) for value in source.get("replay_sources", ()) or ()]
    for reference in references:
        path = Path(str(reference.get("absolute_path") or ""))
        if path.is_file():
            return path.resolve(), references
    return None, references


def _loads_replay(raw: bytes) -> Mapping[str, Any]:
    try:
        import orjson
    except ImportError:
        return json.loads(raw)
    return orjson.loads(raw)


def _tape(replay: Mapping[str, Any], player: int) -> list[dict[str, Any]]:
    if player not in (0, 1):
        raise ValueError(f"invalid player index: {player}")
    steps = list(replay.get("steps", ()) or ())
    if len(steps) < HORIZON + 1:
        raise ValueError(f"incomplete replay with {len(steps)} steps")
    try:
        result = [
            normalized_action(steps[step + 1][player].get("action") or {})
            for step in range(HORIZON)
        ]
    except (IndexError, TypeError, AttributeError) as exc:
        raise ValueError("replay steps do not contain both player actions") from exc
    if len(result) != HORIZON:
        raise AssertionError("replay tape horizon changed unexpectedly")
    return result


def _unit_tape(tape: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", ()) or ()],
        "market": [],
    } for action in tape]


def _block_hashes(
    tape: Sequence[Mapping[str, Any]], unit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{
        "anchor": anchor,
        "duration": min(anchor + 24, HORIZON) - anchor,
        "unit_sha256": _sha256_bytes(_canonical_bytes(unit[anchor:anchor + 24])),
        "full_sha256": _sha256_bytes(_canonical_bytes(tape[anchor:anchor + 24])),
    } for anchor in ANCHORS]


def _unit_bucket(order: Sequence[Any]) -> int:
    operation = str(order[0]) if order else "PASS"
    if operation in UNIT_BUCKETS[:5]:
        return UNIT_BUCKETS.index(operation)
    if operation in PRODUCTION:
        return 5
    if operation in MAINTENANCE:
        return 6
    return 7


def _market_bucket(order: Sequence[Any]) -> int:
    operation = str(order[0]) if order else ""
    if operation == "SELL":
        return 1
    if operation == "HIRE":
        return 2
    if operation == "BUY_LAND":
        return 3
    if operation.startswith("BUY_"):
        return 0
    return 4


def _quantity(order: Sequence[Any]) -> int:
    if len(order) < 3:
        return 1
    try:
        return max(0, int(order[2] or 0))
    except (TypeError, ValueError):
        return 0


def pooled_action_feature(tape: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Pool macro action counts without reading replay-observed prices."""

    width = len(UNIT_BUCKETS) * 2 + 1 + len(MARKET_BUCKETS) * 2
    values = np.zeros((len(ANCHORS), width), dtype=np.float32)
    for anchor_index, anchor in enumerate(ANCHORS):
        row = values[anchor_index]
        for action in tape[anchor:anchor + 24]:
            farmer = list(action.get("farmer") or ["PASS"])
            row[_unit_bucket(farmer)] += 1
            hands = [list(value or ["PASS"]) for value in action.get("hands", ()) or ()]
            row[2 * len(UNIT_BUCKETS)] += len(hands)
            for order in hands:
                row[len(UNIT_BUCKETS) + _unit_bucket(order)] += 1
            for order in action.get("market", ()) or ():
                raw = list(order or ())
                bucket = _market_bucket(raw)
                offset = 2 * len(UNIT_BUCKETS) + 1
                row[offset + bucket] += 1
                row[offset + len(MARKET_BUCKETS) + bucket] += _quantity(raw)
    return np.log1p(values).reshape(-1)


def actor_time_hash_feature(tape: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Hash normalized unit orders with stable farmer/hand actor-time identities."""

    values = np.zeros(ACTOR_TIME_HASH_DIM, dtype=np.float32)
    for anchor in ANCHORS:
        for offset, action in enumerate(tape[anchor:anchor + 24]):
            orders = [("farmer", list(action.get("farmer") or ["PASS"]))]
            orders.extend(
                (f"hand:{slot}", list(order or ["PASS"]))
                for slot, order in enumerate(action.get("hands", ()) or ())
            )
            for actor_slot, order in orders:
                token = _canonical_bytes({
                    "anchor": anchor,
                    "offset": offset,
                    "actor_slot": actor_slot,
                    "normalized_unit_order": order,
                })
                digest = hashlib.sha256(token).digest()
                bucket = int.from_bytes(digest[:8], "big") % ACTOR_TIME_HASH_DIM
                values[bucket] += 1.0 if digest[8] & 1 else -1.0
    return np.sign(values) * np.log1p(np.abs(values))


def action_feature(tape: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Combine pooled macro counts and deterministic actor-time unit tokens."""

    return np.concatenate((
        pooled_action_feature(tape), actor_time_hash_feature(tape),
    )).astype(np.float32, copy=False)


def _result_score(value: str) -> float:
    return 1.0 if value == "win" else .5 if value == "tie" else 0.0


def _source_rank(record: Mapping[str, Any], focus_dataset: str) -> tuple[Any, ...]:
    return (
        _result_score(str(record["result"])),
        focus_dataset in set(record["datasets"]),
        float(record["cash_minimum"]) >= 0.0,
        float(record["historical_margin"]),
        float(record["historical_reward"]),
        -int(record["record_index"]),
    )


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    return inverse.astype(np.float32) / float(len(unique) - 1)


def _scaled_features(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    scale = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.std(values, axis=0))
    active = scale > 1e-6
    if not bool(np.any(active)):
        return np.zeros((len(values), 1), dtype=np.float32)
    result = ((values[:, active] - median[active]) / scale[active]).astype(np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.where(norms > 1e-9, norms, 1.0)


def _select(
    groups: Sequence[dict[str, Any]], *, count: int, macro_cap: int, team_cap: int,
    focus_dataset: str, focus_count: int, team_seed_count: int,
    quality_weight: float,
) -> tuple[list[int], dict[int, str], dict[int, float | None], dict[str, Any]]:
    rewards = np.asarray([row["historical_reward"] for row in groups], np.float64)
    margins = np.asarray([row["historical_margin"] for row in groups], np.float64)
    outcomes = np.asarray([_result_score(row["result"]) for row in groups], np.float64)
    supports = np.asarray([row["support"] for row in groups], np.float64)
    quality = (
        .35 * outcomes
        + .25 * _percentile_ranks(rewards)
        + .25 * _percentile_ranks(margins)
        + .15 * _percentile_ranks(np.log1p(supports))
    ).astype(np.float32)
    features = _scaled_features(np.stack([row["feature"] for row in groups]))
    teams = [str(row["team"] or "unknown") for row in groups]
    selected: list[int] = []
    selected_set: set[int] = set()
    team_counts: Counter[str] = Counter()
    macro_counts: Counter[str] = Counter()
    reasons: dict[int, str] = {}
    distances: dict[int, float | None] = {}

    def add(index: int, reason: str, distance: float | None = None) -> bool:
        if (
            index in selected_set or len(selected) >= count
            or team_counts[teams[index]] >= team_cap
            or macro_counts[str(groups[index]["genome_id"])] >= macro_cap
        ):
            return False
        selected.append(index)
        selected_set.add(index)
        team_counts[teams[index]] += 1
        macro_counts[str(groups[index]["genome_id"])] += 1
        reasons[index] = reason
        distances[index] = distance
        return True

    def ordered(indices: Iterable[int]) -> list[int]:
        return sorted(indices, key=lambda index: (
            -float(quality[index]), -float(margins[index]),
            -float(rewards[index]), str(groups[index]["execution_id"]),
        ))

    best_per_team = []
    for team in sorted(set(teams)):
        best_per_team.append(ordered(
            index for index, value in enumerate(teams) if value == team
        )[0])
    for index in ordered(best_per_team)[:team_seed_count]:
        add(index, "best_per_team")
    team_seed_added = sum(value == "best_per_team" for value in reasons.values())

    focus_selected = sum(
        focus_dataset in set(groups[index]["datasets"]) for index in selected
    )
    focus_candidates = ordered(
        index for index, row in enumerate(groups)
        if focus_dataset in set(row["datasets"])
    )
    for index in focus_candidates:
        if focus_selected >= focus_count:
            break
        if add(index, "focus_dataset"):
            focus_selected += 1

    if selected:
        similarities = features @ features[np.asarray(selected)].T
        minimum_distance = np.min(2.0 - 2.0 * similarities, axis=1)
    else:
        minimum_distance = np.full(len(groups), 2.0, np.float32)
    while len(selected) < count:
        eligible = np.asarray([
            index not in selected_set
            and team_counts[teams[index]] < team_cap
            and macro_counts[str(groups[index]["genome_id"])] < macro_cap
            for index in range(len(groups))
        ])
        if not bool(np.any(eligible)):
            raise ValueError(
                f"team/macro caps permit only {len(selected)} of requested {count} executions"
            )
        objective = minimum_distance + quality_weight * quality
        objective[~eligible] = -np.inf
        index = int(np.argmax(objective))
        add(index, "action_quality_diversity", float(minimum_distance[index]))
        minimum_distance = np.minimum(
            minimum_distance, 2.0 - 2.0 * (features @ features[index]),
        )

    for index in selected:
        groups[index]["quality_score"] = float(quality[index])
    audit = {
        "feature_dimensions": int(features.shape[1]),
        "observed_market_prices_used": False,
        "team_seed_target": team_seed_count,
        "team_seed_added": team_seed_added,
        "focus_target": focus_count,
        "focus_selected": focus_selected,
        "focus_shortfall": max(0, focus_count - focus_selected),
        "team_counts": dict(team_counts.most_common()),
        "macro_genome_counts": dict(macro_counts.most_common()),
        "reason_counts": dict(Counter(reasons.values())),
    }
    return selected, reasons, distances, audit


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _artifact(payload: bytes, name: str) -> dict[str, Any]:
    return {"file": name, "sha256": _sha256_bytes(payload), "bytes": len(payload)}


def build_panel(
    *, pool: Path, split_manifest: Path, output_root: Path,
    pool_start_byte: int, pool_end_byte: int, pool_segment_sha256: str,
    split_manifest_sha256: str,
    count: int = 256, macro_cap: int = 2, team_cap: int = 4,
    focus_dataset: str = "our_latest", focus_count: int = 24,
    team_seed_count: int = 64, quality_weight: float = .15,
    progress_every: int = 128,
) -> dict[str, Any]:
    started = time.perf_counter()
    pool, split_manifest, output_root = (
        pool.resolve(), split_manifest.resolve(), output_root.resolve()
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    if count <= 0 or macro_cap <= 0 or team_cap <= 0:
        raise ValueError("count/macro-cap/team-cap must be positive")
    if focus_count < 0 or team_seed_count < 0 or quality_weight < 0:
        raise ValueError("quotas/quality-weight must be non-negative")
    if progress_every <= 0:
        raise ValueError("progress-every must be positive")

    split_rows, split_sha256, split_payload = _split_rows(
        split_manifest, split_manifest_sha256,
    )
    frozen = _frozen_records(
        pool, pool_start_byte, pool_end_byte, pool_segment_sha256,
    )
    source_ids: set[str] = set()
    catalog: list[dict[str, Any]] = []
    by_path: dict[Path, list[int]] = defaultdict(list)
    by_episode: dict[int, list[int]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    genome_ids: set[str] = set()
    for record_index, (record_sha, payload) in enumerate(frozen):
        source = dict(payload.get("source") or {})
        episode = int(source.get("episode_id", -1))
        if episode not in split_rows:
            raise ValueError(f"episode missing from split manifest: {episode}")
        _gate_source(source, split_rows[episode])
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in source_ids:
            raise ValueError(f"missing or duplicate source_id: {source_id!r}")
        source_ids.add(source_id)
        datasets = sorted({str(value) for value in source.get("datasets", ()) or ()})
        dataset_counts.update(datasets)
        result = str(source.get("result", "unknown"))
        result_counts[result] += 1
        genome_id = str(payload.get("genome_id") or "")
        if not genome_id:
            raise ValueError(f"record has no genome_id: {source_id}")
        genome_ids.add(genome_id)
        reward = float(source.get("final_reward", 0.0) or 0.0)
        opponent_reward = float(source.get("opponent_reward", 0.0) or 0.0)
        if not math.isfinite(reward) or not math.isfinite(opponent_reward):
            raise ValueError(f"non-finite reward in source {source_id}")
        path, references = _reference(source)
        if path is None:
            raise FileNotFoundError(
                f"episode {episode} player {source.get('player_index')} has no readable replay"
            )
        row = {
            "record_index": record_index,
            "source_record_sha256": record_sha,
            "source_id": source_id,
            "episode_id": episode,
            "player_index": int(source.get("player_index", -1)),
            "genome_id": genome_id,
            "team_name": str(source.get("team_name", "")),
            "opponent_team_name": str(source.get("opponent_team_name", "")),
            "datasets": datasets,
            "ingestion_split": str(source.get("ingestion_split", "")),
            "result": result,
            "historical_reward": reward,
            "opponent_reward": opponent_reward,
            "historical_margin": reward - opponent_reward,
            "cash_minimum": float(
                (payload.get("cash_profile") or {}).get("minimum", 0.0) or 0.0
            ),
            "replay_sources": references,
            "selected_replay_path": str(path),
            "status": "pending",
            "rejection_reason": None,
            "replay_sha256": None,
            "execution_id": None,
            "unit_tape_sha256": None,
            "full_tape_sha256": None,
            "block_hashes": None,
        }
        catalog.append(row)
        by_episode[episode].append(record_index)
        by_path[path].append(record_index)

    for episode, indices in sorted(by_episode.items()):
        players = sorted(int(catalog[index]["player_index"]) for index in indices)
        if len(indices) != 2 or players != [0, 1]:
            raise ValueError(
                f"episode {episode} must have exactly player0/player1 records; got {players}"
            )
        paths = {str(catalog[index]["selected_replay_path"]) for index in indices}
        if len(paths) != 1:
            raise ValueError(
                f"episode {episode} player records must reference the same readable replay"
            )

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    replay_file_count = 0
    replay_bytes = 0
    replay_paths = sorted(by_path, key=str)
    # ponytail: keep this bounded-memory one-pass scan until measured bootstrap latency
    # justifies sharding the ~100 GiB input.
    for replay_path in replay_paths:
        try:
            raw = replay_path.read_bytes()
            replay_sha = _sha256_bytes(raw)
            replay = _loads_replay(raw)
        except Exception as exc:
            raise ValueError(f"failed to load replay: {replay_path}") from exc
        replay_file_count += 1
        replay_bytes += len(raw)
        tape_cache: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for index in by_path[replay_path]:
            row = catalog[index]
            player = int(row["player_index"])
            if player not in tape_cache:
                try:
                    tape = _tape(replay, player)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid replay tape for episode {row['episode_id']} player {player}"
                    ) from exc
                tape_cache[player] = (tape, _unit_tape(tape))
            tape, unit = tape_cache[player]
            full_sha = _sha256_bytes(_canonical_bytes(tape))
            unit_sha = _sha256_bytes(_canonical_bytes(unit))
            blocks = _block_hashes(tape, unit)
            feature = action_feature(tape)
            execution_key = (str(row["genome_id"]), unit_sha)
            execution_id = f"REX1_{_sha256_bytes(_canonical_bytes({
                'genome_id': execution_key[0], 'unit_tape_sha256': unit_sha,
            }))[:24]}"
            row.update({
                "status": "eligible",
                "replay_sha256": replay_sha,
                "execution_id": execution_id,
                "unit_tape_sha256": unit_sha,
                "full_tape_sha256": full_sha,
                "block_hashes": blocks,
            })
            candidate = {
                "record_index": index,
                "result": row["result"],
                "datasets": row["datasets"],
                "cash_minimum": row["cash_minimum"],
                "historical_reward": row["historical_reward"],
                "historical_margin": row["historical_margin"],
            }
            group = groups.get(execution_key)
            if group is None:
                group = {
                    "execution_id": execution_id,
                    "genome_id": execution_key[0],
                    "unit_tape_sha256": unit_sha,
                    "unit_block_sha256s": [value["unit_sha256"] for value in blocks],
                    "record_indices": [],
                    "representative": index,
                    "representative_rank": _source_rank(candidate, focus_dataset),
                    "feature": feature,
                    "variants": {},
                }
                groups[execution_key] = group
            elif group["unit_block_sha256s"] != [
                value["unit_sha256"] for value in blocks
            ]:
                raise AssertionError("unit tape SHA collision changed block fingerprints")
            group["record_indices"].append(index)
            rank = _source_rank(candidate, focus_dataset)
            if rank > group["representative_rank"]:
                group["representative"] = index
                group["representative_rank"] = rank
                group["feature"] = feature
            variant = group["variants"].setdefault(full_sha, {
                "record_indices": [], "representative": index,
                "representative_rank": rank,
            })
            variant["record_indices"].append(index)
            if rank > variant["representative_rank"]:
                variant["representative"] = index
                variant["representative_rank"] = rank

        if (
            replay_file_count % progress_every == 0
            or replay_file_count == len(replay_paths)
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(json.dumps({
                "status": "replay_execution_scan",
                "files": replay_file_count,
                "total_files": len(replay_paths),
                "gib": replay_bytes / (1024 ** 3),
                "elapsed_seconds": elapsed,
                "mib_per_second": replay_bytes / (1024 ** 2) / elapsed,
            }, sort_keys=True), flush=True)

    candidates = []
    for key in sorted(groups):
        group = groups[key]
        representative = catalog[int(group["representative"])]
        group.update({
            "support": len(group["record_indices"]),
            "team": str(representative["team_name"]),
            "datasets": list(representative["datasets"]),
            "result": str(representative["result"]),
            "historical_reward": float(representative["historical_reward"]),
            "historical_margin": float(representative["historical_margin"]),
        })
        candidates.append(group)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible execution identities for cap {count}")

    selected, reasons, distances, selection_audit = _select(
        candidates, count=count, macro_cap=macro_cap, team_cap=team_cap,
        focus_dataset=focus_dataset, focus_count=focus_count,
        team_seed_count=team_seed_count, quality_weight=quality_weight,
    )

    selected_groups = [candidates[index] for index in selected]
    requests: dict[Path, list[tuple[str, str, int, str]]] = defaultdict(list)
    for group in selected_groups:
        representative = catalog[int(group["representative"])]
        requests[Path(str(representative["selected_replay_path"]))].append((
            "path", str(group["execution_id"]), int(group["representative"]),
            str(group["unit_tape_sha256"]),
        ))
        for full_sha, variant in sorted(group["variants"].items()):
            record_index = int(variant["representative"])
            path = Path(str(catalog[record_index]["selected_replay_path"]))
            requests[path].append(("overlay", full_sha, record_index, full_sha))

    path_base: dict[str, list[dict[str, Any]]] = {}
    market_overlays: dict[str, list[list[list[Any]]]] = {}
    for replay_path in sorted(requests, key=str):
        raw = replay_path.read_bytes()
        replay = _loads_replay(raw)
        replay_sha = _sha256_bytes(raw)
        player_tapes: dict[int, list[dict[str, Any]]] = {}
        for kind, key, record_index, expected_sha in requests[replay_path]:
            row = catalog[record_index]
            if replay_sha != row["replay_sha256"]:
                raise ValueError(f"selected replay changed during run: {replay_path}")
            player = int(row["player_index"])
            if player not in player_tapes:
                player_tapes[player] = _tape(replay, player)
            tape = player_tapes[player]
            if kind == "path":
                unit = _unit_tape(tape)
                if _sha256_bytes(_canonical_bytes(unit)) != expected_sha:
                    raise AssertionError("selected path tape hash changed")
                path_base[key] = unit
            else:
                if _sha256_bytes(_canonical_bytes(tape)) != expected_sha:
                    raise AssertionError("selected full tape hash changed")
                market_overlays[key] = [
                    [list(order) for order in action.get("market", ()) or ()]
                    for action in tape
                ]

    route_rows = []
    selected_position = {index: position for position, index in enumerate(selected)}
    for candidate_index in selected:
        group = candidates[candidate_index]
        representative = catalog[int(group["representative"])]
        variants = []
        for full_sha, variant in sorted(group["variants"].items()):
            rows = [catalog[int(value)] for value in variant["record_indices"]]
            variant_representative = catalog[int(variant["representative"])]
            variants.append({
                "full_tape_sha256": full_sha,
                "market_overlay_ref": {
                    "file": "market_overlays.json.zlib", "key": full_sha,
                },
                "representative_source_id": variant_representative["source_id"],
                "source_ids": sorted(row["source_id"] for row in rows),
                "source_record_sha256s": sorted(
                    row["source_record_sha256"] for row in rows
                ),
                "replay_sha256s": sorted({row["replay_sha256"] for row in rows}),
                "full_block_sha256s": [
                    value["full_sha256"]
                    for value in variant_representative["block_hashes"]
                ],
            })
        source_rows = [catalog[int(value)] for value in group["record_indices"]]
        route_rows.append({
            "rank": selected_position[candidate_index] + 1,
            "execution_id": group["execution_id"],
            "execution_identity": {
                "genome_id": group["genome_id"],
                "unit_tape_sha256": group["unit_tape_sha256"],
            },
            "path_base_ref": {
                "file": "path_base_actions.json.zlib", "key": group["execution_id"],
            },
            "unit_block_sha256s": group["unit_block_sha256s"],
            "full_action_variants": variants,
            "support": group["support"],
            "representative_source_id": representative["source_id"],
            "representative_team_name": representative["team_name"],
            "datasets": group["datasets"],
            "source_ids": sorted(row["source_id"] for row in source_rows),
            "source_record_sha256s": sorted(
                row["source_record_sha256"] for row in source_rows
            ),
            "selection": {
                "reason": reasons[candidate_index],
                "quality_score": group["quality_score"],
                "nearest_selected_distance_when_added": distances[candidate_index],
            },
        })

    catalog_payload = _jsonl_bytes(catalog)
    selected_payload = _jsonl_bytes(route_rows)
    path_payload = zlib.compress(_canonical_bytes(path_base), level=9)
    overlay_payload = zlib.compress(_canonical_bytes(market_overlays), level=9)
    feature_rows = sorted(candidates, key=lambda row: str(row["execution_id"]))
    feature_buffer = io.BytesIO()
    np.savez_compressed(
        feature_buffer,
        execution_ids=np.asarray(
            [row["execution_id"] for row in feature_rows], dtype=np.str_,
        ),
        action_features=np.stack(
            [row["feature"] for row in feature_rows],
        ).astype(np.float32, copy=False),
        anchors=np.asarray(ANCHORS, dtype=np.int16),
    )
    feature_payload = feature_buffer.getvalue()
    artifacts = {
        "catalog": _artifact(catalog_payload, "record_catalog.jsonl"),
        "selected_executions": _artifact(
            selected_payload, "selected_executions.jsonl",
        ),
        "path_base": _artifact(path_payload, "path_base_actions.json.zlib"),
        "market_overlays": _artifact(
            overlay_payload, "market_overlays.json.zlib",
        ),
        "execution_action_features": _artifact(
            feature_payload, "execution_action_features.npz",
        ),
    }
    implementation_sha = _sha256_bytes(Path(__file__).read_bytes())
    config = {
        "count": count,
        "macro_cap": macro_cap,
        "team_cap": team_cap,
        "focus_dataset": focus_dataset,
        "focus_count": focus_count,
        "team_seed_count": team_seed_count,
        "quality_weight": quality_weight,
        "anchors": list(ANCHORS),
    }
    content_core = {
        "schema": SCHEMA,
        "implementation_sha256": implementation_sha,
        "pool_segment_sha256": pool_segment_sha256.lower(),
        "split_manifest_sha256": split_sha256,
        "config": config,
        "artifact_sha256s": {
            key: value["sha256"] for key, value in sorted(artifacts.items())
        },
    }
    content_sha = _sha256_bytes(_canonical_bytes(content_core))
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "library_id": f"REP1_{content_sha[:24]}",
        "content_sha256": content_sha,
        "content_hash_contract": (
            "SHA-256 of schema, implementation, frozen inputs, selection config, "
            "and every output artifact digest; runtime manifest paths/timing excluded, "
            "while catalog provenance paths are covered by its artifact digest"
        ),
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3]
            ).as_posix(),
            "sha256": implementation_sha,
        },
        "input": {
            "pool": str(pool),
            "pool_start_byte": pool_start_byte,
            "pool_end_byte": pool_end_byte,
            "pool_segment_sha256": pool_segment_sha256.lower(),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": split_sha256,
            "top40_manifest_sha256": split_payload.get("top40_manifest_sha256"),
            "train_only": True,
            "top40_public_only": True,
        },
        "catalog": {
            "records": len(catalog),
            "episodes": len(by_episode),
            "episode_pair_contract": (
                "exactly two records with player indices {0,1} referencing one readable replay"
            ),
            "unique_source_ids": len(source_ids),
            "unique_macro_genomes": len(genome_ids),
            "eligible_records": sum(row["status"] == "eligible" for row in catalog),
            "eligible_execution_identities": len(candidates),
            "unique_replay_files_read": replay_file_count,
            "replay_bytes_read_first_pass": replay_bytes,
            "dataset_counts": dict(dataset_counts.most_common()),
            "result_counts": dict(result_counts.most_common()),
            "status_counts": dict(Counter(row["status"] for row in catalog)),
            "rejection_counts": dict(rejection_counts.most_common()),
        },
        "selection": {
            **config,
            **selection_audit,
            "execution_identity": "(genome_id, unit_tape_sha256)",
            "action_feature": (
                "21 anchor-local pooled macro counts plus deterministic signed "
                "actor-time unit-order hashes; no observed market prices"
            ),
            "raw_feature_dimensions": int(candidates[0]["feature"].shape[0]),
            "pooled_feature_dimensions": int(
                pooled_action_feature([]).shape[0]
            ),
            "actor_time_hash": {
                "algorithm": ACTOR_TIME_HASH_ALGORITHM,
                "dimensions": ACTOR_TIME_HASH_DIM,
                "token": (
                    "(anchor, offset, actor_slot, normalized_unit_order); "
                    "farmer is a dedicated slot and hands use stable list indices"
                ),
                "order_parameters_included": True,
            },
        },
        "representation": {
            "path_base": "farmer+hands actions with market=[]",
            "full_action": (
                "path_base_ref plus one market_overlay_ref; full_tape_sha256 verifies reconstruction"
            ),
            "selected_execution_count": len(route_rows),
            "selected_full_action_variant_count": sum(
                len(row["full_action_variants"]) for row in route_rows
            ),
            "eligible_feature_sidecar": {
                "file": "execution_action_features.npz",
                "rows": len(feature_rows),
                "execution_id_order": "ascending",
                "contains_replay_or_tape": False,
                "arrays": {
                    "execution_ids": "unicode",
                    "action_features": "float32",
                    "anchors": "int16",
                },
            },
        },
        "artifacts": artifacts,
        "routes": route_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.", dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary)
        payloads = {
            "record_catalog.jsonl": catalog_payload,
            "selected_executions.jsonl": selected_payload,
            "path_base_actions.json.zlib": path_payload,
            "market_overlays.json.zlib": overlay_payload,
            "execution_action_features.npz": feature_payload,
        }
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        manifest_payload = json.dumps(
            manifest, ensure_ascii=False, indent=2,
        ).encode("utf-8") + b"\n"
        manifest_sha = _sha256_bytes(manifest_payload)
        (staging / "panel_manifest.json").write_bytes(manifest_payload)
        (staging / "panel_manifest.json.sha256").write_text(
            f"{manifest_sha}  panel_manifest.json\n", encoding="ascii",
        )
        if output_root.exists():
            raise FileExistsError(f"output directory appeared during run: {output_root}")
        staging.replace(output_root)
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pool-start-byte", type=int, default=0)
    parser.add_argument("--pool-end-byte", type=int, required=True)
    parser.add_argument("--pool-segment-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--macro-cap", type=int, default=2)
    parser.add_argument("--team-cap", type=int, default=4)
    parser.add_argument("--focus-dataset", default="our_latest")
    parser.add_argument("--focus-count", type=int, default=24)
    parser.add_argument("--team-seed-count", type=int, default=64)
    parser.add_argument("--quality-weight", type=float, default=.15)
    parser.add_argument("--progress-every", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_panel(
        pool=args.pool, split_manifest=args.split_manifest,
        output_root=args.output_root, pool_start_byte=args.pool_start_byte,
        pool_end_byte=args.pool_end_byte,
        pool_segment_sha256=args.pool_segment_sha256,
        split_manifest_sha256=args.split_manifest_sha256, count=args.count,
        macro_cap=args.macro_cap, team_cap=args.team_cap,
        focus_dataset=args.focus_dataset, focus_count=args.focus_count,
        team_seed_count=args.team_seed_count, quality_weight=args.quality_weight,
        progress_every=args.progress_every,
    )
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "library_id": manifest["library_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "records": manifest["catalog"]["records"],
        "execution_identities": manifest["catalog"]["eligible_execution_identities"],
        "selected": manifest["representation"]["selected_execution_count"],
        "focus_shortfall": manifest["selection"]["focus_shortfall"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



