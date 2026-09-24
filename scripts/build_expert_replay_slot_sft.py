#!/usr/bin/env python3
"""Build a causally safe post-step288 SFT pilot from native-aligned expert replays.

This is intentionally a thin consumer of the frozen full-daily replay pipeline.
Raw Kaggle actions are never reinterpreted here: successful placements and their
before/after state come from the native ordered receipts in ``days.jsonl.gz``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("/root/kaggriculture-research-20260907/full-daily-corpus-v1/frozen")
EPISODE_FILE = re.compile(r"episode-(\d+)-replay\.json$")
TOKENIZER_ROOT = Path("/root/kaggriculture_transformer_ppo_starter")
CLASS_KINDS = (-1, 0, 1, 2, 3, 4, 9, 10, 11)
ITEM_KIND = {
    "WHEAT": 0, "CARROT": 1, "TOMATO": 2, "STRAWBERRY": 3,
    "MELON": 4, "GOOSE": 9, "COW": 10, "SHEEP": 11,
}
FIRST_HARVEST_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8,
                     "STRAWBERRY": 10, "MELON": 10}
DEPOTS = (44, 45, 54, 55)
TOKEN_FIELDS = ("token_type", "category_a", "category_b", "category_c",
                "x", "y", "owner")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_episode_key(episode_id: int) -> str:
    return hashlib.sha256(str(episode_id).encode()).hexdigest()


def _validate_finished_file(path_text: str) -> dict[str, Any]:
    """Use the repository's downloader validator; the frozen aligner is final gate."""
    sys.path.insert(0, str(ROOT))
    from scripts.collect_latest_top_routes import replay_summary

    path = Path(path_text)
    match = EPISODE_FILE.fullmatch(path.name)
    summary = replay_summary(path) if match else None
    return {
        "path": str(path.resolve()),
        "episode_id": int(match.group(1)) if match else None,
        "complete_json": summary is not None,
        "bytes": int(summary["bytes"]) if summary else int(path.stat().st_size),
        "team_names": summary["team_names"] if summary else None,
        "rewards": summary["rewards"] if summary else None,
    }


def _select_pilot(source: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Top-player, reward-blind round-robin using the mature selection ordering."""
    by_episode = {
        int(row["episode_id"]): row for row in source.get("replays", [])
        if row.get("error") is None and row.get("targets")
    }
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    targets = sorted(source["leaderboard_targets"],
                     key=lambda row: (-float(row["leaderboard_score"]), int(row["team_id"])))
    queues: list[list[dict[str, Any]]] = []
    for target in targets:
        rows = [row for row in by_episode.values()
                if any(int(value["team_id"]) == int(target["team_id"])
                       for value in row.get("requested_targets", []))]
        rows.sort(key=lambda row: _stable_episode_key(int(row["episode_id"])))
        queues.append(rows)
    offset = 0
    while len(chosen) < count and any(offset < len(queue) for queue in queues):
        for queue in queues:
            if offset >= len(queue):
                continue
            row = queue[offset]
            episode = int(row["episode_id"])
            if episode not in seen:
                chosen.append(row)
                seen.add(episode)
                if len(chosen) == count:
                    break
        offset += 1
    if len(chosen) != count:
        raise ValueError(f"only {len(chosen)} unique expert episodes available, wanted {count}")
    return chosen


def inventory(args: argparse.Namespace) -> None:
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if len(source.get("replays", [])) < args.pilot_replays:
        raise ValueError("source manifest is smaller than requested pilot")
    roots = [path.resolve() for path in args.replay_root]
    paths = [path for root in roots for path in root.glob("episode-*-replay.json")]
    # Exact filename matching excludes .tmp/.part and all other downloader debris.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        indexed = list(pool.map(_validate_finished_file, map(str, paths), chunksize=1))
    valid = [row for row in indexed if row["complete_json"]]
    copies: dict[int, list[str]] = defaultdict(list)
    for row in valid:
        copies[int(row["episode_id"])].append(str(row["path"]))
    for values in copies.values():
        values.sort(key=lambda value: (roots.index(Path(value).parent)
                    if Path(value).parent in roots else len(roots), value))

    selected = _select_pilot(source, args.pilot_replays)
    missing = [int(row["episode_id"]) for row in selected
               if int(row["episode_id"]) not in copies]
    if missing:
        raise FileNotFoundError(f"pilot has {len(missing)} missing replay files: {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output / "pilot-inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("episode_id", "paths"))
        writer.writeheader()
        for row in selected:
            episode = int(row["episode_id"])
            writer.writerow({"episode_id": episode,
                             "paths": " | ".join(copies[episode])})

    # Keep a standard collector manifest so the mature macro extractor/clustering
    # can consume the pilot unchanged.
    pilot_rows = []
    for row in selected:
        copied = dict(row)
        copied["replay"] = os.path.relpath(copies[int(row["episode_id"])][0], args.output)
        pilot_rows.append(copied)
    pilot_manifest = {
        "competition": source.get("competition"),
        "collected_at_utc": source.get("collected_at_utc"),
        "selection": {
            "pilot_replays": args.pilot_replays,
            "ordering": "leaderboard team round-robin; SHA256(episode_id) within team",
            "reward_filter": None,
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": _sha256(args.source_manifest),
        },
        "leaderboard_targets": source["leaderboard_targets"],
        "replays": pilot_rows,
    }
    pilot_manifest_path = args.output / "pilot-source-manifest.json"
    pilot_manifest_path.write_text(json.dumps(pilot_manifest, indent=2,
                                              ensure_ascii=False) + "\n")
    target_sides = [target for row in selected for target in row.get("targets", [])]
    report = {
        "schema": "expert-replay-pilot-inventory-v1",
        "status": "accepted_for_native_alignment",
        "input_roots": [str(path) for path in roots],
        "scanned_exact_name_files": len(paths),
        "valid_json_files": len(valid),
        "invalid_or_incomplete_files": len(indexed) - len(valid),
        "unique_episode_ids": len(copies),
        "duplicate_file_copies": len(valid) - len(copies),
        "pilot_replays": len(selected),
        "pilot_target_sides": len(target_sides),
        "pilot_teams": len({int(row["team_id"]) for row in target_sides}),
        "leaderboard_score": {
            "min": min(float(row["leaderboard_score"]) for row in target_sides),
            "max": max(float(row["leaderboard_score"]) for row in target_sides),
        },
        "reward_filter": None,
        "files": {
            "inventory_csv": inventory_path.name,
            "inventory_sha256": _sha256(inventory_path),
            "pilot_source_manifest": pilot_manifest_path.name,
            "pilot_source_manifest_sha256": _sha256(pilot_manifest_path),
        },
    }
    (args.output / "inventory-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def _slot_key(cell: int) -> tuple[int, int]:
    x, y = cell % 10, cell // 10
    return min(abs(x - depot % 10) + abs(y - depot // 10) for depot in DEPOTS), cell


def _tile_kind(tile: Any) -> Any:
    return tile.get("kind") if isinstance(tile, dict) else tile


def _structural_mask(tile: Any, day: int) -> int:
    """Physical alternatives visible at the boundary; resources are unknown."""
    mask = 1  # SKIP is always structurally possible.
    if tile is None:
        return mask | sum(1 << index for index in range(1, 6))
    if not isinstance(tile, dict):
        return mask
    kind = tile.get("kind")
    if kind == "COOP" and not tile.get("animal"):
        return mask | (1 << CLASS_KINDS.index(9))
    if kind == "PASTURE" and not tile.get("animal"):
        return mask | (1 << CLASS_KINDS.index(10)) | (1 << CLASS_KINDS.index(11))
    crop = tile.get("crop")
    finite = kind == "PLANT" and int(tile.get("max_lifespan_step", -1)) >= 0
    mature = crop in FIRST_HARVEST_DAY and day - int(tile.get("planted_day", day)) >= FIRST_HARVEST_DAY[crop]
    if finite and mature and int(tile.get("yield_units", 0)) > 0:
        return mask | sum(1 << index for index in range(1, 6))
    return mask


def _canonical_observation(observation: dict[str, Any]) -> dict[str, Any]:
    # Import the already audited seat-canonicalization used by the R1 student.
    from experiments.train_midgame_student_v1 import _canonical_observation as canonical
    return canonical(observation)


def _extract_native_episode(job: dict[str, Any]) -> dict[str, Any]:
    import orjson

    sys.path.insert(0, str(TOKENIZER_ROOT))
    from kaggrl.tokenizer import ObservationTokenizer

    replay_path = Path(job["source"])
    replay = orjson.loads(replay_path.read_bytes())
    episode = int((replay.get("info") or {}).get("EpisodeId"))
    if episode != int(job["episode_id"]):
        raise ValueError("replay/native episode identity mismatch")
    targets: dict[int, dict[str, Any]] = {}
    for target in job["targets"]:
        player = int(target["player_index"])
        previous = targets.get(player)
        if previous is None or float(target["leaderboard_score"]) > float(previous["leaderboard_score"]):
            targets[player] = target
    tokenizer = ObservationTokenizer()
    states = []
    stats: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    attribution: Counter[str] = Counter()
    with gzip.open(job["daily"], "rb") as source:
        for packed in source:
            day_row = orjson.loads(packed)
            player = int(day_row["player"])
            if player not in targets:
                stats["discard.non_target_player_day"] += 1
                continue
            stats["target_player_days"] += 1
            start = int(day_row["start_step"])
            if start < 288:
                stats["discard.before_step288_day"] += 1
                continue
            quality = day_row.get("quality") or {}
            evidence = quality.get("evidence") or {}
            if (day_row.get("quality_status") != "computed" or
                    evidence.get("native_aligned") is not True or
                    evidence.get("unit_receipts_complete") is not True):
                stats["discard.quality_not_native_complete_day"] += 1
                continue
            stats["eligible_native_target_days"] += 1
            observation = replay["steps"][start][player]["observation"]
            if (int(observation["player"]) != player or
                    int(observation["day"]) * 24 + int(observation["hour"]) != start or
                    int(day_row["source"]["observation_frame"]) != start):
                raise ValueError("daily evidence does not match boundary observation")
            farm = observation["farms"][player]
            tiles = farm["tiles"]
            day = int(observation["day"])
            masks = {
                y * 10 + x: _structural_mask(tile, day)
                for y, row in enumerate(tiles) for x, tile in enumerate(row)
            }
            stats["structural_candidate_cells"] += sum(mask != 1 for mask in masks.values())
            retired: set[int] = set()
            creates: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
            for payload in day_row["payloads"]:
                receipt = payload.get("receipt")
                if receipt is None:
                    # The official loop emits no unit event for an omitted hand
                    # or a submitted slot beyond the current workforce.  The
                    # frozen quality audit explicitly treats these as known
                    # no-events, not as successful labels.
                    stats["payload_without_native_unit_event"] += 1
                    action = payload.get("action") or []
                    if action and action[0] in ("PLANT", "PLACE"):
                        stats["discard.establish_without_native_receipt"] += 1
                    continue
                if receipt.get("effect") is not True:
                    continue
                action = payload.get("action") or []
                op = action[0] if action else None
                before, after = receipt["before"], receipt["after"]
                x, y = map(int, before["position"])
                if [x, y] != list(after["position"]) or x != int(receipt["x"]) or y != int(receipt["y"]):
                    # Effective movement receipts change position and are irrelevant.
                    if op in ("PLANT", "PLACE", "HARVEST"):
                        raise ValueError("asset receipt position mismatch")
                    continue
                cell = y * 10 + x
                if (op == "HARVEST" and _tile_kind(before["tile"]) == "PLANT" and
                        _tile_kind(after["tile"]) != "PLANT"):
                    retired.add(cell)
                if op not in ("PLANT", "PLACE"):
                    continue
                item = str(action[1]) if len(action) >= 2 else ""
                label_kind = ITEM_KIND.get(item)
                created = ((op == "PLANT" and _tile_kind(before["tile"]) != "PLANT" and
                            _tile_kind(after["tile"]) == "PLANT" and after["tile"].get("crop") == item) or
                           (op == "PLACE" and not (isinstance(before["tile"], dict) and before["tile"].get("animal")) and
                            isinstance(after["tile"], dict) and after["tile"].get("animal") == item))
                if label_kind is None or not created:
                    stats["discard.effective_establish_not_verified_create"] += 1
                    continue
                stats["native_success_create_events"] += 1
                label_class = CLASS_KINDS.index(label_kind)
                boundary_tile = tiles[y][x]
                if not masks[cell] & (1 << label_class):
                    category = (_tile_kind(boundary_tile) if boundary_tile is not None else "EMPTY")
                    stats[f"discard.boundary_structural_ineligible.{category}"] += 1
                    continue
                if isinstance(boundary_tile, dict) and boundary_tile.get("kind") == "PLANT":
                    if cell not in retired:
                        stats["discard.successor_without_native_retirement"] += 1
                        continue
                    source_kind = "harvest_successor"
                else:
                    source_kind = "direct_new_placement"
                creates[cell].append((label_kind, masks[cell], source_kind))
            accepted = []
            for cell, values in creates.items():
                if len(values) != 1:
                    stats["discard.multiple_successful_creations_same_cell_day"] += len(values)
                    continue
                label_kind, mask, source_kind = values[0]
                accepted.append((cell, label_kind, mask, source_kind))
            if not accepted:
                stats["discard.no_causally_safe_positive_label_day"] += 1
                continue
            accepted.sort(key=lambda row: _slot_key(row[0]))
            encoded = tokenizer.encode(_canonical_observation(observation))
            prefix = np.zeros(len(CLASS_KINDS), dtype=np.float32)
            slots = []
            for cell, label_kind, mask, source_kind in accepted:
                label_class = CLASS_KINDS.index(label_kind)
                slots.append({"cell": cell, "label_kind": label_kind,
                              "label_class": label_class, "legal_mask": mask,
                              "prefix_counts": prefix.copy(),
                              "attribution": 1 if source_kind == "harvest_successor" else 0})
                prefix[label_class] += 1
                class_counts[str(label_kind)] += 1
                attribution[source_kind] += 1
                stats["accepted_native_success_labels"] += 1
            stats["accepted_states"] += 1
            states.append({
                "episode_id": episode, "player": player, "step": start,
                "split": day_row["split"], "target": targets[player],
                "token_continuous": encoded.continuous.numpy(),
                **{name: getattr(encoded, name).numpy() for name in TOKEN_FIELDS},
                "slots": slots,
            })
    return {"episode_id": episode, "states": states, "stats": dict(stats),
            "class_counts": dict(class_counts), "attribution": dict(attribution)}


def _save_array(directory: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {"path": path.name, "dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": _sha256(path)}


def shard(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty shard output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    native = json.loads((args.native / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_rows = {int(row["episode_id"]): row for row in source["replays"]}
    jobs = []
    for row in native["episodes"]:
        if row["status"] != "complete":
            continue
        episode = int(row["episode_id"])
        daily = args.native / row["output_directory"] / row["artifacts"]["daily_records"]["path"]
        source_row = source_rows[episode]
        jobs.append({"episode_id": episode, "source": row["source"],
                     "daily": str(daily), "targets": source_row["targets"]})
    if not jobs:
        raise ValueError("native manifest contains no complete expert episodes")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_extract_native_episode, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    extract_seconds = time.perf_counter() - started
    results.sort(key=lambda row: row["episode_id"])
    states = sorted((state for row in results for state in row["states"]),
                    key=lambda row: (row["episode_id"], row["player"], row["step"]))
    if not states:
        raise ValueError("no causally safe post-step288 expert labels")
    max_tokens = 320
    if max(len(state["token_type"]) for state in states) > max_tokens:
        raise ValueError("mature observation tokenizer exceeded 320 tokens")
    state_count = len(states)
    slot_count = sum(len(state["slots"]) for state in states)
    arrays: dict[str, np.ndarray] = {
        "token_continuous": np.zeros((state_count, max_tokens, 24), dtype="<f4"),
        "token_type": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_a": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_b": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_c": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_x": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_y": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_owner": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_count": np.empty(state_count, dtype="<u2"),
        "state_slot_offsets": np.zeros(state_count + 1, dtype="<u4"),
        "slot_cell": np.empty(slot_count, dtype="u1"),
        "slot_label_kind": np.empty(slot_count, dtype="i1"),
        "slot_label_class": np.empty(slot_count, dtype="u1"),
        "slot_legal_mask": np.empty(slot_count, dtype="<u2"),
        "slot_prefix_counts": np.empty((slot_count, len(CLASS_KINDS)), dtype="<f4"),
        "slot_attribution": np.empty(slot_count, dtype="u1"),
        "split": np.empty(state_count, dtype="u1"),
        "group_hash128": np.empty((state_count, 16), dtype="u1"),
    }
    source_names = dict(zip(
        ("token_type", "token_category_a", "token_category_b", "token_category_c",
         "token_x", "token_y", "token_owner"), TOKEN_FIELDS))
    groups = []
    cursor = 0
    for index, state in enumerate(states):
        count = len(state["token_type"])
        arrays["token_count"][index] = count
        arrays["token_continuous"][index, :count] = state["token_continuous"]
        for target_name, source_name in source_names.items():
            arrays[target_name][index, :count] = state[source_name]
        arrays["state_slot_offsets"][index] = cursor
        for slot in state["slots"]:
            arrays["slot_cell"][cursor] = slot["cell"]
            arrays["slot_label_kind"][cursor] = slot["label_kind"]
            arrays["slot_label_class"][cursor] = slot["label_class"]
            arrays["slot_legal_mask"][cursor] = slot["legal_mask"]
            arrays["slot_prefix_counts"][cursor] = slot["prefix_counts"]
            arrays["slot_attribution"][cursor] = slot["attribution"]
            cursor += 1
        arrays["split"][index] = int(state["split"] == "holdout")
        group = hashlib.sha256(str(state["episode_id"]).encode()).digest()[:16]
        arrays["group_hash128"][index] = np.frombuffer(group, dtype=np.uint8)
        groups.append({
            "group_hash128": group.hex(), "episode_id": state["episode_id"],
            "player": state["player"], "step": state["step"],
            "leaderboard_score": float(state["target"]["leaderboard_score"]),
            "team_name": state["target"]["team_name"],
        })
    arrays["state_slot_offsets"][-1] = cursor
    if cursor != slot_count or not np.any(arrays["split"] == 0) or not np.any(arrays["split"] == 1):
        raise RuntimeError("invalid slot offsets or empty episode-group split")
    legal = ((arrays["slot_legal_mask"] >> arrays["slot_label_class"]) & 1).astype(bool)
    if not np.all(legal):
        raise RuntimeError("native-success label violates structural mask")
    conflicts: defaultdict[bytes, set[int]] = defaultdict(set)
    for state_index in range(state_count):
        start, stop = arrays["state_slot_offsets"][state_index:state_index + 2]
        token_hash = hashlib.sha256(arrays["token_continuous"][state_index].tobytes() +
                                    arrays["token_type"][state_index].tobytes()).digest()
        for slot in range(int(start), int(stop)):
            key = hashlib.sha256(token_hash + arrays["slot_cell"][slot:slot + 1].tobytes() +
                                 arrays["slot_prefix_counts"][slot].tobytes()).digest()
            conflicts[key].add(int(arrays["slot_label_class"][slot]))
    conflict_count = sum(len(values) > 1 for values in conflicts.values())
    totals: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    attribution: Counter[str] = Counter()
    for result in results:
        totals.update(result["stats"])
        class_counts.update(result["class_counts"])
        attribution.update(result["attribution"])
    specs = {name: _save_array(args.output, name, value) for name, value in arrays.items()}
    schema = {
        "schema_name": "expert-replay-outcome-slot-sft-v1",
        "source": "expert_replay_outcome",
        "label_semantics": "native-successful PLANT/PLACE during the next day window",
        "class_kinds": list(CLASS_KINDS),
        "slot_order": "distance-to-central-depot then original-board cell index",
        "mask_precision": "structural_only/resource_unknown",
        "resource_prefix": "unknown and absent",
        "loss_scope": "native-success and structurally legal labels only",
        "skip_semantics": "legal alternative only; no inferred negative SKIP target",
        "model_input_arrays": [
            "token_continuous", "token_type", "token_category_a", "token_category_b",
            "token_category_c", "token_x", "token_y", "token_owner", "token_count",
            "state_slot_offsets", "slot_cell", "slot_legal_mask", "slot_prefix_counts",
        ],
        "target_only_arrays": ["slot_label_kind", "slot_label_class", "slot_attribution"],
        "audit_only_arrays": ["split", "group_hash128"],
        "identity_policy": "seed/team/opponent/player absent from forward; team metadata audit-only",
        "causal_input_boundary": "steps[start_step][acting_player].observation only",
        "forbidden_input": "receipt/action-time before.inventory and all future payloads",
    }
    schema_sha256 = hashlib.sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "format": "kaggriculture-midgame-mmap", "container_version": 1,
        "validation_status": "accepted_expert_pilot",
        "schema": schema, "schema_sha256": schema_sha256, "byte_order": "little",
        "counts": {
            "inventory_replays": int(native["inventory_episodes"]),
            "native_complete_replays": len(jobs),
            "native_quarantined_replays": int(native["inventory_episodes"]) - len(jobs),
            "states": state_count, "slots": slot_count,
            "train_states": int(np.sum(arrays["split"] == 0)),
            "heldout_states": int(np.sum(arrays["split"] == 1)),
        },
        "arrays": specs,
        "class_distribution": dict(sorted(class_counts.items(), key=lambda row: int(row[0]))),
        "attribution": dict(attribution), "audit_counts": dict(totals),
        "validation": {
            "native_success_labels": int(totals["native_success_create_events"]),
            "accepted_labels": slot_count,
            "accepted_over_native_success": slot_count / max(1, totals["native_success_create_events"]),
            "structural_label_legal_fraction": float(np.mean(legal)),
            "identical_input_label_conflicts": conflict_count,
            "action_time_inventory_in_forward": False,
            "future_payload_in_forward": False,
            "all_steps_at_least_288": all(row["step"] >= 288 for row in groups),
        },
        "quality_filter": {
            "player": "only player_index matched to current top100 manifest target",
            "leaderboard_score_min": min(row["leaderboard_score"] for row in groups),
            "leaderboard_score_max": max(row["leaderboard_score"] for row in groups),
            "reward_filter": None,
            "day": "quality_status=computed and native unit receipts complete",
        },
        "group_split": "frozen SHA256(seed) modulus-10 split; all episode seats/days remain together",
        "audit_groups": groups,
        "provenance": {
            "native_manifest": str((args.native / "manifest.json").resolve()),
            "native_manifest_sha256": _sha256(args.native / "manifest.json"),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": _sha256(args.source_manifest),
            "tokenizer": str((TOKENIZER_ROOT / "kaggrl/tokenizer.py").resolve()),
            "tokenizer_sha256": _sha256(TOKENIZER_ROOT / "kaggrl/tokenizer.py"),
        },
        "timing": {"extract_seconds": extract_seconds,
                   "states_per_second": state_count / extract_seconds,
                   "slots_per_second": slot_count / extract_seconds},
        "limitations": [
            "This conditional kind SFT does not supervise whether an untouched cell should be SKIP.",
            "Structural masks do not claim day-start cash/inventory or prefix-resource feasibility.",
            "The tokenizer has current full observation but no reconstructed R1 commitment/ledger belief.",
            "Quarantined new action encodings remain excluded rather than guessed.",
        ],
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      **manifest["counts"], **manifest["validation"],
                      **manifest["timing"]}, ensure_ascii=False))


def _load_shard(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("validation_status") != "accepted_expert_pilot" or
            manifest.get("schema", {}).get("source") != "expert_replay_outcome" or
            manifest["schema"].get("mask_precision") != "structural_only/resource_unknown"):
        raise ValueError("not an accepted isolated expert-outcome shard")
    arrays = {}
    for name, spec in manifest["arrays"].items():
        file = path / spec["path"]
        if _sha256(file) != spec["sha256"]:
            raise ValueError(f"{name}: digest mismatch")
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
            raise ValueError(f"{name}: dtype/shape mismatch")
        arrays[name] = value
    labels = arrays["slot_label_class"]
    if not np.all((arrays["slot_legal_mask"] >> labels) & 1):
        raise ValueError("stored expert label is structurally illegal")
    return manifest, arrays


def _build_model():
    import torch

    class ExpertOutcomeStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(size, 8) for size in (7, 32, 32, 32, 64, 64, 4))
            self.token = torch.nn.Sequential(torch.nn.Linear(24 + 7 * 8, 64), torch.nn.ReLU(),
                                             torch.nn.Linear(64, 128), torch.nn.Tanh())
            self.cell = torch.nn.Embedding(100, 16)
            self.previous = torch.nn.Embedding(10, 16)
            self.prefix = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU())
            self.gru = torch.nn.GRUCell(64, 128)
            self.head = torch.nn.Linear(128, 9)

        def initial_hidden(self, continuous, categories, count):
            positions = torch.arange(continuous.shape[1], device=continuous.device)[None]
            valid = positions < count[:, None]
            denominator = count.clamp_min(1).float()[:, None]
            pooled = [(continuous * valid[:, :, None]).sum(1) / denominator]
            pooled.extend((table(value) * valid[:, :, None]).sum(1) / denominator
                          for table, value in zip(self.embeddings, categories))
            return self.token(torch.cat(pooled, dim=1))

        def step(self, hidden, cell, previous, prefix, legal):
            inputs = torch.cat((self.cell(cell), self.previous(previous), self.prefix(prefix)), dim=1)
            hidden = self.gru(inputs, hidden)
            return self.head(hidden).masked_fill(~legal, -1e9), hidden

    return ExpertOutcomeStudent()


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional

    manifest, arrays = _load_shard(args.shard)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tensor = lambda value, dtype=None: torch.as_tensor(np.asarray(value).copy(), dtype=dtype)
    values = {
        "continuous": tensor(arrays["token_continuous"], torch.float32),
        "categories": [tensor(arrays[name], torch.long) for name in
                       ("token_type", "token_category_a", "token_category_b",
                        "token_category_c", "token_x", "token_y", "token_owner")],
        "count": tensor(arrays["token_count"], torch.long),
        "cell": tensor(arrays["slot_cell"], torch.long),
        "label": tensor(arrays["slot_label_class"], torch.long),
        "mask": tensor(arrays["slot_legal_mask"], torch.long),
        "prefix": tensor(arrays["slot_prefix_counts"], torch.float32),
    }
    offsets = np.asarray(arrays["state_slot_offsets"], dtype=np.int64)
    split = np.asarray(arrays["split"])
    train_states = np.flatnonzero(split == 0)
    heldout_states = np.flatnonzero(split == 1)

    def forward(model, states: np.ndarray):
        state_tensor = tensor(states, torch.long)
        hidden = model.initial_hidden(values["continuous"][state_tensor],
                                      [value[state_tensor] for value in values["categories"]],
                                      values["count"][state_tensor])
        previous = torch.full((len(states),), 9, dtype=torch.long)
        losses, predictions, labels_out, legal_out = [], [], [], []
        max_slots = max(int(offsets[state + 1] - offsets[state]) for state in states)
        for local_slot in range(max_slots):
            active_rows = [row for row, state in enumerate(states)
                           if local_slot < offsets[state + 1] - offsets[state]]
            active = tensor(active_rows, torch.long)
            global_rows = tensor([int(offsets[state]) + local_slot for state in states
                                  if local_slot < offsets[state + 1] - offsets[state]], torch.long)
            bits = values["mask"][global_rows]
            legal = ((bits[:, None] >> torch.arange(9)[None]) & 1).bool()
            logits, next_hidden = model.step(hidden[active], values["cell"][global_rows],
                                             previous[active], values["prefix"][global_rows], legal)
            labels = values["label"][global_rows]
            losses.append(functional.cross_entropy(logits, labels, reduction="none"))
            predictions.append(logits.argmax(1)); labels_out.append(labels); legal_out.append(legal)
            hidden = hidden.clone(); hidden[active] = next_hidden
            previous = previous.clone(); previous[active] = labels
        return torch.cat(losses), torch.cat(predictions), torch.cat(labels_out), torch.cat(legal_out)

    def metrics(model, states: np.ndarray) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            losses, predicted, labels, legal = forward(model, states)
        recalls = {}
        for index, kind in enumerate(CLASS_KINDS):
            selected = labels == index
            support = int(selected.sum())
            recalls[str(kind)] = {"support": support,
                                  "recall": float((predicted[selected] == index).float().mean()) if support else None}
        return {"loss": float(losses.mean()),
                "masked_accuracy": float((predicted == labels).float().mean()),
                "class_recall": recalls,
                "illegal_argmax": int((~legal[torch.arange(len(predicted)), predicted]).sum())}

    probe = _build_model()
    parameters = sum(value.numel() for value in probe.parameters())
    overfit_states = train_states[:min(8, len(train_states))]
    overfit = _build_model()
    optimizer = torch.optim.Adam(overfit.parameters(), lr=0.02)
    first_overfit = None
    for _ in range(args.overfit_steps):
        overfit.train(); optimizer.zero_grad(set_to_none=True)
        loss = forward(overfit, overfit_states)[0].mean()
        if first_overfit is None:
            first_overfit = float(loss.detach())
        loss.backward(); optimizer.step()
    overfit_metrics = metrics(overfit, overfit_states)
    if overfit_metrics["loss"] >= first_overfit or overfit_metrics["masked_accuracy"] < .999:
        raise RuntimeError("expert real-batch overfit failed")

    torch.manual_seed(args.seed)
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial_train = metrics(model, train_states)
    initial_heldout = metrics(model, heldout_states)
    generator = np.random.default_rng(args.seed)
    started = time.perf_counter(); trained_slots = 0
    for _ in range(args.epochs):
        order = generator.permutation(train_states)
        for at in range(0, len(order), args.batch_states):
            batch = order[at:at + args.batch_states]
            model.train(); optimizer.zero_grad(set_to_none=True)
            losses = forward(model, batch)[0]
            losses.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); trained_slots += len(losses)
    elapsed = time.perf_counter() - started
    final_train = metrics(model, train_states)
    final_heldout = metrics(model, heldout_states)
    if final_train["loss"] >= initial_train["loss"]:
        raise RuntimeError("expert SFT training loss did not decrease")
    if final_train["illegal_argmax"] or final_heldout["illegal_argmax"]:
        raise RuntimeError("structural mask emitted illegal argmax")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard_hash = _sha256(args.shard / "manifest.json")
    torch.save({"model": model.state_dict(), "class_kinds": CLASS_KINDS,
                "schema_sha256": manifest["schema_sha256"],
                "shard_manifest_sha256": shard_hash}, args.output)
    result = {
        "status": "PASS_EXPERT_PILOT", "training_task": "expert_replay_outcome_conditional_slot_sft",
        "schema_name": manifest["schema"]["schema_name"],
        "schema_sha256": manifest["schema_sha256"], "shard_manifest_sha256": shard_hash,
        "source": "expert_replay_outcome", "mask_precision": "structural_only/resource_unknown",
        "device": "cpu", "parameters": parameters, "states": len(split),
        "slots": len(arrays["slot_cell"]), "train_states": len(train_states),
        "heldout_states": len(heldout_states),
        "native_complete_replays": manifest["counts"]["native_complete_replays"],
        "label_legal_fraction": manifest["validation"]["structural_label_legal_fraction"],
        "action_time_inventory_in_forward": False,
        "one_batch_overfit": {"states": len(overfit_states), "initial_loss": first_overfit,
                              **overfit_metrics},
        "train": {"initial": initial_train, "final": final_train},
        "heldout": {"initial": initial_heldout, "final": final_heldout},
        "timing": {"epochs": args.epochs, "seconds": elapsed,
                   "trained_slot_examples_per_second": trained_slots / elapsed},
        "checkpoint": str(args.output),
        "limitations": manifest["limitations"],
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("inventory")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--replay-root", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--pilot-replays", type=int, default=128)
    prepare.add_argument("--workers", type=int, default=31)
    prepare.set_defaults(function=inventory)
    export = commands.add_parser("shard")
    export.add_argument("--native", type=Path, required=True)
    export.add_argument("--source-manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--workers", type=int, default=31)
    export.set_defaults(function=shard)
    fit = commands.add_parser("train")
    fit.add_argument("--shard", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--overfit-steps", type=int, default=200)
    fit.add_argument("--batch-states", type=int, default=64)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--threads", type=int, default=16)
    fit.add_argument("--seed", type=int, default=20260922)
    fit.set_defaults(function=train)
    args = parser.parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("workers must be positive")
    if hasattr(args, "pilot_replays") and args.pilot_replays < 1:
        parser.error("pilot-replays must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
