#!/usr/bin/env python3
"""Build real 147-D continuation prototypes for a frozen replay panel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402
import select_replay_execution_panel_v1 as panel  # noqa: E402


SCHEMA = "replay-execution-continuation-bank-v1"
OUTPUT_SCHEMA = "block-mvp-continuation-multitail-v1"
HORIZON = panel.HORIZON
ANCHORS = panel.ANCHORS
REQUIRED_ARTIFACTS = {
    "catalog", "selected_executions", "path_base", "market_overlays",
    "execution_action_features",
}

if ANCHORS != continuation.ANCHORS or HORIZON != continuation.HORIZON:
    raise RuntimeError("panel and continuation schemas disagree")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_sha256(value: Any, label: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a 64-digit SHA-256")
    return result


def _load_replay(path: Path) -> tuple[str, Mapping[str, Any], int]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    size = len(raw)
    try:
        replay = panel._loads_replay(raw)
    except Exception as exc:
        raise ValueError(f"failed to load selected replay: {path}") from exc
    del raw
    return digest, replay, size


def _ordered_replay_paths(
    groups: Mapping[Path, Sequence[Any]],
) -> list[Path]:
    """I/O order is deliberately independent from logical output ordinals."""

    return sorted(groups, key=lambda path: str(path))


def _report_replay_progress(files: int, byte_count: int, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "progress": SCHEMA,
        "files": int(files),
        "bytes": int(byte_count),
        "elapsed": round(elapsed, 3),
        "MiB/s": round(byte_count / (1024.0 ** 2) / elapsed, 3),
    }, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)


def _safe_file(root: Path, name: Any) -> Path:
    path = (root / str(name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"panel artifact escapes its root: {name}") from exc
    return path


def _jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} JSONL is empty or non-object-valued")
    return rows


def _load_panel(
    root: Path, expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
    dict[str, list[dict[str, Any]]], dict[str, list[list[list[Any]]]],
]:
    root = root.resolve()
    manifest_path = root / "panel_manifest.json"
    manifest_raw = manifest_path.read_bytes()
    actual_manifest_sha = _sha256_bytes(manifest_raw)
    if actual_manifest_sha != _expected_sha256(
        expected_manifest_sha256, "panel manifest SHA-256",
    ):
        raise ValueError("panel manifest SHA-256 mismatch")
    sidecar = (root / "panel_manifest.json.sha256").read_text(
        encoding="ascii",
    ).split()
    if not sidecar or sidecar[0].lower() != actual_manifest_sha:
        raise ValueError("panel manifest SHA-256 sidecar mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != panel.SCHEMA or manifest.get("status") != "complete":
        raise ValueError("invalid or incomplete replay execution panel")
    source_input = dict(manifest.get("input") or {})
    if source_input.get("train_only") is not True or source_input.get(
        "top40_public_only",
    ) is not True:
        raise ValueError("panel does not assert train/public-only provenance")
    if tuple(map(int, manifest.get("selection", {}).get("anchors", ()))) != ANCHORS:
        raise ValueError("panel anchors disagree with the continuation contract")

    artifacts = dict(manifest.get("artifacts") or {})
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("panel artifact set changed")
    payloads: dict[str, bytes] = {}
    for key in sorted(REQUIRED_ARTIFACTS):
        entry = dict(artifacts[key] or {})
        raw = _safe_file(root, entry.get("file")).read_bytes()
        if len(raw) != int(entry.get("bytes", -1)):
            raise ValueError(f"panel artifact byte count mismatch: {key}")
        if _sha256_bytes(raw) != str(entry.get("sha256", "")).lower():
            raise ValueError(f"panel artifact SHA-256 mismatch: {key}")
        payloads[key] = raw

    config_keys = (
        "count", "macro_cap", "team_cap", "focus_dataset", "focus_count",
        "team_seed_count", "quality_weight", "anchors",
    )
    config = {key: manifest["selection"][key] for key in config_keys}
    content_core = {
        "schema": panel.SCHEMA,
        "implementation_sha256": str(manifest["implementation"]["sha256"]),
        "pool_segment_sha256": str(source_input["pool_segment_sha256"]).lower(),
        "split_manifest_sha256": str(source_input["split_manifest_sha256"]).lower(),
        "config": config,
        "artifact_sha256s": {
            key: str(artifacts[key]["sha256"]).lower() for key in sorted(artifacts)
        },
    }
    content_sha = _sha256_bytes(_canonical_bytes(content_core))
    if content_sha != str(manifest.get("content_sha256", "")):
        raise ValueError("panel content hash contract mismatch")
    if str(manifest.get("library_id", "")) != f"REP1_{content_sha[:24]}":
        raise ValueError("panel library id mismatch")

    catalog = _jsonl(payloads["catalog"], "record catalog")
    selected = _jsonl(payloads["selected_executions"], "selected executions")
    if _canonical_bytes(selected) != _canonical_bytes(manifest.get("routes", ())):
        raise ValueError("selected executions disagree with panel manifest routes")
    try:
        path_base = json.loads(zlib.decompress(payloads["path_base"]))
        overlays = json.loads(zlib.decompress(payloads["market_overlays"]))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid panel action archive") from exc
    if not isinstance(path_base, dict) or not isinstance(overlays, dict):
        raise ValueError("panel action archives must be object-valued")
    return manifest, catalog, selected, path_base, overlays


def _source_records(
    manifest: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    source_input = dict(manifest["input"])
    pool = Path(str(source_input["pool"])).resolve()
    split_path = Path(str(source_input["split_manifest"])).resolve()
    split_rows, _, _ = panel._split_rows(
        split_path, str(source_input["split_manifest_sha256"]),
    )
    frozen = panel._frozen_records(
        pool, int(source_input["pool_start_byte"]),
        int(source_input["pool_end_byte"]),
        str(source_input["pool_segment_sha256"]),
    )
    if len(frozen) != len(catalog):
        raise ValueError("pool segment and catalog record counts disagree")

    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for index, ((record_sha, record), catalog_raw) in enumerate(zip(frozen, catalog)):
        catalog_row = dict(catalog_raw)
        source = dict(record.get("source") or {})
        episode = int(source.get("episode_id", -1))
        if episode not in split_rows:
            raise ValueError(f"episode missing from split manifest: {episode}")
        panel._gate_source(source, split_rows[episode])
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"missing or duplicate pool source id: {source_id!r}")
        path, references = panel._reference(source)
        expected = {
            "record_index": index,
            "source_record_sha256": record_sha,
            "source_id": source_id,
            "episode_id": episode,
            "player_index": int(source.get("player_index", -1)),
            "genome_id": str(record.get("genome_id") or ""),
            "datasets": sorted({str(value) for value in source.get("datasets", ()) or ()}),
            "ingestion_split": str(source.get("ingestion_split", "")),
            "replay_sources": references,
            "selected_replay_path": str(path) if path is not None else "",
        }
        if any(catalog_row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"pool source and catalog disagree: {source_id}")
        if path is None:
            raise FileNotFoundError(f"selected source has no readable replay: {source_id}")
        result[source_id] = (record, catalog_row)
    return result


def _full_tape(
    base: Sequence[Mapping[str, Any]], overlay: Sequence[Sequence[Sequence[Any]]],
) -> list[dict[str, Any]]:
    if len(base) != HORIZON or len(overlay) != HORIZON:
        raise ValueError("path base and market overlay must both have 719 steps")
    result = []
    for action, orders in zip(base, overlay):
        if not isinstance(action, Mapping) or not isinstance(orders, list):
            raise ValueError("invalid path-base or market-overlay step")
        if list(action.get("market", ()) or ()):
            raise ValueError("path base contains market actions")
        if any(not isinstance(order, list) for order in orders):
            raise ValueError("market overlay order must be list-valued")
        result.append({
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(value or ["PASS"]) for value in action.get("hands", ()) or ()],
            "market": [list(order) for order in orders],
        })
    return result


def _variant_rows(
    selected: Sequence[Mapping[str, Any]], path_base: Mapping[str, Any],
    overlays: Mapping[str, Any], sources: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_execution_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    expected_path_keys: set[str] = set()
    expected_overlay_keys: set[str] = set()
    for selected_row_raw in selected:
        selected_row = dict(selected_row_raw)
        execution_id = str(selected_row.get("execution_id") or "")
        if not execution_id or execution_id in seen_execution_ids:
            raise ValueError(f"missing or duplicate selected execution id: {execution_id!r}")
        seen_execution_ids.add(execution_id)
        identity = dict(selected_row.get("execution_identity") or {})
        genome_id = str(identity.get("genome_id") or "")
        unit_sha = _expected_sha256(
            identity.get("unit_tape_sha256"), "unit tape SHA-256",
        )
        derived_id = "REX1_" + _sha256_bytes(_canonical_bytes({
            "genome_id": genome_id, "unit_tape_sha256": unit_sha,
        }))[:24]
        if not genome_id or execution_id != derived_id:
            raise ValueError(f"selected execution identity mismatch: {execution_id}")
        path_ref = dict(selected_row.get("path_base_ref") or {})
        if path_ref != {"file": "path_base_actions.json.zlib", "key": execution_id}:
            raise ValueError(f"invalid path-base reference: {execution_id}")
        if execution_id not in path_base:
            raise ValueError(f"missing selected path base: {execution_id}")
        base = list(path_base[execution_id])
        if _sha256_bytes(_canonical_bytes(panel._unit_tape(base))) != unit_sha:
            raise ValueError(f"selected path-base SHA-256 mismatch: {execution_id}")
        expected_path_keys.add(execution_id)
        selected_sources = list(map(str, selected_row.get("source_ids", ()) or ()))
        if not selected_sources or len(selected_sources) != len(set(selected_sources)):
            raise ValueError(f"selected execution has invalid source ids: {execution_id}")
        catalog_rows = []
        for source_id in selected_sources:
            if source_id not in sources:
                raise ValueError(f"selected source is absent from frozen pool: {source_id}")
            catalog_row = sources[source_id][1]
            if catalog_row.get("status") != "eligible" or str(
                catalog_row.get("execution_id") or "",
            ) != execution_id:
                raise ValueError(f"selected source execution mismatch: {source_id}")
            catalog_rows.append(catalog_row)
        if sorted(row["source_record_sha256"] for row in catalog_rows) != sorted(
            map(str, selected_row.get("source_record_sha256s", ()) or ()),
        ):
            raise ValueError(f"selected source-record hashes mismatch: {execution_id}")

        assigned: set[str] = set()
        unit_blocks: list[str] | None = None
        for variant_raw in selected_row.get("full_action_variants", ()) or ():
            variant = dict(variant_raw)
            full_sha = _expected_sha256(
                variant.get("full_tape_sha256"), "full tape SHA-256",
            )
            overlay_ref = dict(variant.get("market_overlay_ref") or {})
            if overlay_ref != {"file": "market_overlays.json.zlib", "key": full_sha}:
                raise ValueError(f"invalid market-overlay reference: {full_sha}")
            if full_sha not in overlays:
                raise ValueError(f"missing selected market overlay: {full_sha}")
            full = _full_tape(base, overlays[full_sha])
            if _sha256_bytes(_canonical_bytes(full)) != full_sha:
                raise ValueError(f"reconstructed full tape SHA-256 mismatch: {full_sha}")
            if _sha256_bytes(_canonical_bytes(panel._unit_tape(full))) != unit_sha:
                raise ValueError(f"market overlay changed path identity: {full_sha}")
            hashes = panel._block_hashes(full, panel._unit_tape(full))
            current_unit = [str(row["unit_sha256"]) for row in hashes]
            current_full = [str(row["full_sha256"]) for row in hashes]
            if current_unit != list(map(str, selected_row.get("unit_block_sha256s", ()))):
                raise ValueError(f"unit block fingerprints mismatch: {execution_id}")
            if current_full != list(map(str, variant.get("full_block_sha256s", ()))):
                raise ValueError(f"full block fingerprints mismatch: {full_sha}")
            unit_blocks = current_unit
            variant_sources = list(map(str, variant.get("source_ids", ()) or ()))
            if not variant_sources or assigned.intersection(variant_sources):
                raise ValueError(f"overlapping or empty market variant lineage: {full_sha}")
            assigned.update(variant_sources)
            variant_catalog = [sources[source_id][1] for source_id in variant_sources]
            if any(
                str(row.get("execution_id") or "") != execution_id
                or str(row.get("full_tape_sha256") or "") != full_sha
                for row in variant_catalog
            ):
                raise ValueError(f"catalog/full variant mismatch: {full_sha}")
            if sorted(row["source_record_sha256"] for row in variant_catalog) != sorted(
                map(str, variant.get("source_record_sha256s", ()) or ()),
            ):
                raise ValueError(f"variant source-record hashes mismatch: {full_sha}")
            expected_overlay_keys.add(full_sha)
            route_id = "REPV1_" + _sha256_bytes(_canonical_bytes({
                "execution_id": execution_id, "full_tape_sha256": full_sha,
            }))[:24]
            result.append({
                "rank": int(selected_row["rank"]),
                "route_id": route_id,
                "execution_id": execution_id,
                "genome_id": genome_id,
                "unit_tape_sha256": unit_sha,
                "full_tape_sha256": full_sha,
                "unit_block_sha256s": unit_blocks,
                "full_block_sha256s": current_full,
                "path_base_ref": path_ref,
                "market_overlay_ref": overlay_ref,
                "source_ids": sorted(variant_sources),
                "tape": full,
            })
        if assigned != set(selected_sources):
            raise ValueError(f"market variants do not partition selected lineage: {execution_id}")
        if unit_blocks is None:
            raise ValueError(f"selected execution has no full-action variants: {execution_id}")
    if set(path_base) != expected_path_keys or set(overlays) != expected_overlay_keys:
        raise ValueError("panel action archives contain unreferenced or missing selected keys")
    if len({row["route_id"] for row in result}) != len(result):
        raise ValueError("derived route id collision")
    return sorted(result, key=lambda row: (row["rank"], row["full_tape_sha256"]))


def build_bank(
    *, panel_root: Path, panel_manifest_sha256: str, output_root: Path,
) -> dict[str, Any]:
    panel_root, output_root = panel_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    manifest, catalog, selected, path_base, overlays = _load_panel(
        panel_root, panel_manifest_sha256,
    )
    sources = _source_records(manifest, catalog)
    variants = _variant_rows(selected, path_base, overlays, sources)

    route_actions: dict[str, list[dict[str, Any]]] = {}
    route_rows: list[dict[str, Any]] = []
    replay_groups: dict[Path, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
    task_count = 0
    for variant in variants:
        route_id = str(variant["route_id"])
        tape = list(variant["tape"])
        route_actions[route_id] = tape
        route_provenance = list(map(str, variant["source_ids"]))
        for source_id in variant["source_ids"]:
            catalog_row = sources[source_id][1]
            replay_path = Path(str(catalog_row["selected_replay_path"])).resolve()
            replay_groups[replay_path].append((task_count, variant, str(source_id)))
            task_count += 1
        tail_sha = _sha256_bytes(_canonical_bytes(tape[continuation.SWITCH_STEP:]))
        route_rows.append({
            "route_id": route_id,
            "execution_id": variant["execution_id"],
            "genome_id": variant["genome_id"],
            "unit_tape_sha256": variant["unit_tape_sha256"],
            "full_tape_sha256": variant["full_tape_sha256"],
            "tail_sha256_step216": tail_sha,
            "path_base_ref": variant["path_base_ref"],
            "market_overlay_ref": variant["market_overlay_ref"],
            "representative_source_id": route_provenance[0],
            "provenance_ids": route_provenance,
        })

    if task_count < 1:
        raise ValueError("selected panel produced no replay contract tasks")
    contracts = np.empty(
        (task_count, len(ANCHORS), continuation.FEATURE_DIM), np.float32,
    )
    layouts = np.empty(
        (task_count, len(ANCHORS), continuation.LAYOUT_SIZE), np.uint8,
    )
    masks = np.empty(
        (task_count, len(ANCHORS), len(continuation.QUADRANTS)), np.uint8,
    )
    provenance_slots: list[dict[str, Any] | None] = [None] * task_count
    filled = np.zeros(task_count, dtype=np.bool_)
    replay_reads = 0
    replay_bytes = 0
    progress_started = time.perf_counter()
    for replay_path in _ordered_replay_paths(replay_groups):
        replay_sha, replay, replay_size = _load_replay(replay_path)
        replay_reads += 1
        replay_bytes += replay_size
        try:
            for ordinal, variant, source_id in replay_groups[replay_path]:
                catalog_row = sources[source_id][1]
                tape = variant["tape"]
                if filled[ordinal]:
                    raise RuntimeError(f"duplicate replay task ordinal: {ordinal}")
                if replay_sha != str(catalog_row.get("replay_sha256") or ""):
                    raise ValueError(f"selected replay SHA-256 mismatch: {source_id}")
                player = int(catalog_row["player_index"])
                try:
                    observed_tape = panel._tape(replay, player)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid 719-step selected replay/player: {source_id}"
                    ) from exc
                if _sha256_bytes(
                    _canonical_bytes(observed_tape)
                ) != variant["full_tape_sha256"]:
                    raise ValueError(f"selected source/full tape mismatch: {source_id}")
                if _sha256_bytes(
                    _canonical_bytes(panel._unit_tape(observed_tape))
                ) != variant["unit_tape_sha256"]:
                    raise ValueError(f"selected source/path identity mismatch: {source_id}")
                rewards = list(replay.get("rewards", ()) or ())
                if len(rewards) < 2 or not all(
                    math.isfinite(float(value)) for value in rewards[:2]
                ):
                    raise ValueError(f"invalid selected replay rewards: {source_id}")
                vector, layout, mask = continuation.replay_contracts(
                    replay, player, tape, ANCHORS,
                )
                if (
                    vector.shape != (len(ANCHORS), continuation.FEATURE_DIM)
                    or layout.shape != (len(ANCHORS), continuation.LAYOUT_SIZE)
                    or mask.shape != (len(ANCHORS), len(continuation.QUADRANTS))
                    or not np.isfinite(vector).all()
                ):
                    raise ValueError(f"invalid continuation contract: {source_id}")
                provenance_id = source_id
                contracts[ordinal] = vector
                layouts[ordinal] = layout
                masks[ordinal] = mask
                provenance_slots[ordinal] = {
                    "provenance_id": provenance_id,
                    "route_id": str(variant["route_id"]),
                    "execution_id": variant["execution_id"],
                    "genome_id": variant["genome_id"],
                    "unit_tape_sha256": variant["unit_tape_sha256"],
                    "full_tape_sha256": variant["full_tape_sha256"],
                    "episode_id": int(catalog_row["episode_id"]),
                    "player_index": player,
                    "team_name": str(catalog_row.get("team_name", "")),
                    "opponent_team_name": str(
                        catalog_row.get("opponent_team_name", "")
                    ),
                    "datasets": list(catalog_row.get("datasets", ())),
                    "ingestion_split": str(catalog_row.get("ingestion_split", "")),
                    "result": str(catalog_row.get("result", "")),
                    "final_reward": float(
                        catalog_row.get("historical_reward", rewards[player]) or 0
                    ),
                    "opponent_reward": float(
                        catalog_row.get("opponent_reward", rewards[1-player]) or 0
                    ),
                    "source_record_sha256": str(catalog_row["source_record_sha256"]),
                    "replay_path": str(replay_path),
                    "replay_sha256": replay_sha,
                    "replay_bytes": replay_size,
                }
                filled[ordinal] = True
                del observed_tape, rewards, vector, layout, mask
        finally:
            # Do not let the loop variable retain the prior parsed replay while
            # the next group is loaded.
            del replay
        if replay_reads % 32 == 0:
            _report_replay_progress(replay_reads, replay_bytes, progress_started)

    if not bool(np.all(filled)) or any(row is None for row in provenance_slots):
        raise RuntimeError("one or more replay task ordinals were not populated")
    provenance = [row for row in provenance_slots if row is not None]

    actions_payload = zlib.compress(_canonical_bytes(route_actions), level=9)
    contract_buffer = io.BytesIO()
    np.savez_compressed(
        contract_buffer,
        contracts=contracts,
        layouts=layouts,
        unlocked_masks=masks,
        provenance_ids=np.asarray([row["provenance_id"] for row in provenance]),
        route_ids=np.asarray([row["route_id"] for row in provenance]),
        anchors=np.asarray(ANCHORS, dtype=np.int16),
    )
    contracts_payload = contract_buffer.getvalue()
    implementation_sha = _sha256_bytes(Path(__file__).read_bytes())
    panel_manifest_sha = _expected_sha256(
        panel_manifest_sha256, "panel manifest SHA-256",
    )
    output_manifest = {
        "schema": OUTPUT_SCHEMA,
        "producer_schema": SCHEMA,
        "status": "prepared",
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3]
            ).as_posix(),
            "sha256": implementation_sha,
        },
        "panel": {
            "root": str(panel_root),
            "manifest_sha256": panel_manifest_sha,
            "library_id": str(manifest["library_id"]),
            "content_sha256": str(manifest["content_sha256"]),
            "pool": str(manifest["input"]["pool"]),
            "pool_start_byte": int(manifest["input"]["pool_start_byte"]),
            "pool_end_byte": int(manifest["input"]["pool_end_byte"]),
            "pool_segment_sha256": str(manifest["input"]["pool_segment_sha256"]),
            "split_manifest": str(manifest["input"]["split_manifest"]),
            "split_manifest_sha256": str(manifest["input"]["split_manifest_sha256"]),
            "artifact_sha256s": {
                key: str(value["sha256"])
                for key, value in sorted(manifest["artifacts"].items())
            },
        },
        "anchors": list(ANCHORS),
        "logical_anchor_records": len(provenance) * len(ANCHORS),
        "provenance_count": len(provenance),
        "unique_tail_count": len(route_actions),
        "selected_execution_count": len(selected),
        "full_action_variant_count": len(route_rows),
        "replay_files_read": replay_reads,
        "replay_bytes_read": replay_bytes,
        "replay_loading": {
            "grouping_key": "resolved selected_replay_path",
            "one_load_per_unique_path": True,
            "peak_live_parsed_replays": 1,
            "output_order": (
                "preallocated ordinal slots in existing variant(rank,full_tape_sha256) "
                "then sorted-source_id order; replay path I/O order cannot change artifacts"
            ),
            "progress_every_unique_replays": 32,
        },
        "actions_file": "tail_actions.json.zlib",
        "actions_sha256": _sha256_bytes(actions_payload),
        "contracts_file": "contracts.npz",
        "contracts_sha256": _sha256_bytes(contracts_payload),
        "routes": route_rows,
        "provenance": provenance,
        "contract": {
            "feature_dim": continuation.FEATURE_DIM,
            "layout_dim": continuation.LAYOUT_SIZE,
            "unlocked_mask_dim": len(continuation.QUADRANTS),
            "distance_excluded_feature_indices": list(range(
                continuation.MARKET_START, continuation.MARKET_STOP,
            )),
            "opponent_private_included": False,
            "future_capital": "fixed BUY_SEED/BUY_ANIMAL/BUY_LAND/HIRE only",
            "future_capital_is_full_action_variant_specific": True,
        },
        "identity": {
            "path_base": "(genome_id, unit_tape_sha256)",
            "route_variant": "(execution_id, full_tape_sha256)",
            "observed_market_price_in_path_identity": False,
            "observed_market_price_source": (
                "replay state contract only; excluded by downstream path clustering"
            ),
        },
        "data_gate": {
            "pool_records_checked": len(sources),
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_replay_read": False,
            "fail_closed_replay_horizon": HORIZON,
        },
    }
    manifest_payload = json.dumps(
        output_manifest, ensure_ascii=False, indent=2,
    ).encode("utf-8") + b"\n"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.", dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary)
        (staging / "tail_actions.json.zlib").write_bytes(actions_payload)
        (staging / "contracts.npz").write_bytes(contracts_payload)
        (staging / "candidate_manifest.json").write_bytes(manifest_payload)
        manifest_sha = _sha256_bytes(manifest_payload)
        (staging / "candidate_manifest.json.sha256").write_text(
            f"{manifest_sha}  candidate_manifest.json\n", encoding="ascii",
        )
        if output_root.exists():
            raise FileExistsError(f"output directory appeared during run: {output_root}")
        staging.replace(output_root)
    output_manifest["manifest_sha256"] = manifest_sha
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--panel-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_bank(
        panel_root=args.panel_root,
        panel_manifest_sha256=args.panel_manifest_sha256,
        output_root=args.output_root,
    )
    print(json.dumps({
        "schema": manifest["producer_schema"],
        "panel_library_id": manifest["panel"]["library_id"],
        "selected_executions": manifest["selected_execution_count"],
        "full_action_variants": manifest["full_action_variant_count"],
        "provenance": manifest["provenance_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
