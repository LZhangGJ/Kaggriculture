#!/usr/bin/env python3
"""Build cold/hot/active references for one frozen unit-path block library."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_adaptive_tail_block_library_v0 as builder  # noqa: E402
import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402


SCHEMA = "offline-block-portfolio-v1"
VECTOR_ARRAYS = {
    "block_ids", "anchors", "durations", "action_sha256", "action_vectors",
    "event_profiles", "audit_full_action_vectors", "audit_full_event_profiles",
    "entry_contracts", "layouts", "unlocked_masks", "source_provenance_ids",
    "source_route_ids", "cluster_ids", "active",
}
PROTOTYPE_ARRAYS = {
    "block_ids", "anchors", "entry_contracts", "layouts", "unlocked_masks",
    "provenance_ids", "route_ids", "team_names",
}
DISTANCE_WEIGHTS = {
    "path_action": .35,
    "contract_prototype": .25,
    "capital_risk": .15,
    "layout": .05,
    "source_team": .15,
    "historical_outcome": .05,
}
HOT_QUALITY_BONUS = .15
ACTIVE_QUALITY_BONUS = .25


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


def _safe_file(root: Path, name: Any) -> Path:
    path = (root / str(name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"library artifact escapes its root: {name}") from exc
    return path


def _artifact(root: Path, name: Any, expected: Any, label: str) -> bytes:
    raw = _safe_file(root, name).read_bytes()
    if _sha256_bytes(raw) != _expected_sha256(expected, f"{label} SHA-256"):
        raise ValueError(f"{label} artifact SHA-256 mismatch")
    return raw


def _npz(raw: bytes, expected_names: set[str], label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise ValueError(f"{label} NPZ array set changed")
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(label):
            raise
        raise ValueError(f"invalid {label} NPZ") from exc


def _load_prepared_quality(
    manifest: Mapping[str, Any], used_provenance: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    roots = []
    for reference_raw in manifest.get("prepared_roots", ()) or ():
        reference = dict(reference_raw)
        root = Path(str(reference.get("path") or "")).resolve()
        candidate_path = root / "candidate_manifest.json"
        raw = candidate_path.read_bytes()
        digest = _sha256_bytes(raw)
        if digest != _expected_sha256(
            reference.get("candidate_manifest_sha256"),
            "prepared candidate manifest SHA-256",
        ):
            raise ValueError(f"prepared candidate manifest SHA-256 mismatch: {root}")
        candidate = json.loads(raw)
        if (
            candidate.get("schema") != builder.INPUT_SCHEMA
            or candidate.get("status") != "prepared"
            or str(candidate.get("actions_sha256", "")).lower()
            != str(reference.get("actions_sha256", "")).lower()
            or str(candidate.get("contracts_sha256", "")).lower()
            != str(reference.get("contracts_sha256", "")).lower()
        ):
            raise ValueError(f"prepared candidate reference contract mismatch: {root}")
        gate = dict(candidate.get("data_gate") or {})
        if (
            gate.get("train_only") is not True
            or gate.get("top40_public_only") is not True
            or gate.get("validation_or_sealed_replay_read") is not False
        ):
            raise ValueError(f"prepared candidate is not train/public-only: {root}")
        rows = list(candidate.get("provenance", ()) or ())
        if len(rows) != int(candidate.get("provenance_count", -1)):
            raise ValueError(f"prepared provenance count mismatch: {root}")
        for row_raw in rows:
            row = dict(row_raw)
            provenance_id = str(row.get("provenance_id") or "")
            if not provenance_id or provenance_id in provenance:
                raise ValueError(f"missing or duplicate prepared provenance: {provenance_id!r}")
            if str(row.get("ingestion_split", "")) != "train":
                raise ValueError(f"non-train prepared provenance: {provenance_id}")
            provenance[provenance_id] = row
        roots.append({
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": digest,
        })
    if not roots:
        raise ValueError("block library has no prepared-root provenance")
    missing = sorted(used_provenance - set(provenance))
    if missing:
        raise ValueError(f"block lineage is absent from prepared manifests: {missing[:3]}")

    incomplete = []
    for provenance_id in sorted(used_provenance):
        row = provenance[provenance_id]
        result = str(row.get("result", ""))
        try:
            reward = float(row["final_reward"])
            opponent = float(row["opponent_reward"])
        except (KeyError, TypeError, ValueError):
            incomplete.append(provenance_id)
            continue
        expected_result = "win" if reward > opponent else (
            "tie" if reward == opponent else "loss"
        )
        if (
            result not in {"win", "tie", "loss"}
            or not math.isfinite(reward)
            or not math.isfinite(opponent)
            or result != expected_result
        ):
            incomplete.append(provenance_id)
    quality = {
        "mode": (
            "historical_provenance_tiebreaker"
            if not incomplete else "diversity_only_missing_historical_outcomes"
        ),
        "available": not incomplete,
        "missing_or_invalid_count": len(incomplete),
        "prepared_manifest_refs": roots,
        "semantics": (
            "source replay result/reward; weak non-causal tie-breaker, not a block evaluation"
            if not incomplete else
            "no quality score used because some source result/reward metadata is incomplete"
        ),
    }
    return provenance, quality


def _load_library(
    root: Path, expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray],
    dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any], str,
]:
    root = root.resolve()
    manifest_raw = (root / "block_library_manifest.json").read_bytes()
    manifest_sha = _sha256_bytes(manifest_raw)
    if manifest_sha != _expected_sha256(
        expected_manifest_sha256, "block-library manifest SHA-256",
    ):
        raise ValueError("block-library manifest SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("schema") != builder.SCHEMA
        or manifest.get("deduplication_mode") != "unit_only"
        or tuple(map(int, manifest.get("anchors", ()))) != builder.ANCHORS
    ):
        raise ValueError("portfolio requires a fixed-anchor unit-only block library")
    blocks = [dict(row) for row in manifest.get("blocks", ()) or ()]
    count = int(manifest.get("block_count", -1))
    if count <= 0 or len(blocks) != count:
        raise ValueError("block-library manifest has an invalid block count")
    block_ids = [str(row.get("block_id") or "") for row in blocks]
    if any(not value for value in block_ids) or len(set(block_ids)) != count:
        raise ValueError("block-library manifest has missing or duplicate block ids")

    action_raw = _artifact(
        root, manifest.get("actions_file"), manifest.get("actions_sha256"),
        "block actions",
    )
    vector_raw = _artifact(
        root, manifest.get("vectors_file"), manifest.get("vectors_sha256"),
        "block vectors",
    )
    prototype_raw = _artifact(
        root, manifest.get("contract_prototypes_file"),
        manifest.get("contract_prototypes_sha256"), "contract prototypes",
    )
    try:
        actions = json.loads(zlib.decompress(action_raw))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid block action archive") from exc
    if not isinstance(actions, dict) or set(actions) != set(block_ids):
        raise ValueError("block action archive and manifest ids disagree")
    for row in blocks:
        block_id = str(row["block_id"])
        duration = int(row["duration"])
        tape = actions[block_id]
        if (
            not isinstance(tape, list) or len(tape) != duration
            or any(not isinstance(action, dict) for action in tape)
            or any(list(action.get("market", ()) or ()) for action in tape)
            or _sha256_bytes(_canonical_bytes(tape)) != str(row["action_sha256"])
        ):
            raise ValueError(f"invalid unit-only block action: {block_id}")

    vectors = _npz(vector_raw, VECTOR_ARRAYS, "block vectors")
    prototypes = _npz(prototype_raw, PROTOTYPE_ARRAYS, "contract prototypes")
    expected_vector_shapes = {
        "block_ids": (count,), "anchors": (count,), "durations": (count,),
        "action_sha256": (count,), "action_vectors": (count, builder.ACTION_DIM),
        "event_profiles": (count, builder.EVENT_DIM),
        "audit_full_action_vectors": (count, builder.ACTION_DIM),
        "audit_full_event_profiles": (count, builder.EVENT_DIM),
        "entry_contracts": (count, builder.FEATURE_DIM),
        "layouts": (count, builder.LAYOUT_DIM),
        "unlocked_masks": (count, builder.MASK_DIM),
        "source_provenance_ids": (count,), "source_route_ids": (count,),
        "cluster_ids": (count,), "active": (count,),
    }
    if any(vectors[name].shape != shape for name, shape in expected_vector_shapes.items()):
        raise ValueError("block vector arrays have invalid fixed shapes")
    if (
        list(map(str, vectors["block_ids"])) != block_ids
        or list(map(int, vectors["anchors"])) != [int(row["anchor"]) for row in blocks]
        or list(map(str, vectors["action_sha256"]))
        != [str(row["action_sha256"]) for row in blocks]
        or not all(np.isfinite(vectors[name]).all() for name in (
            "action_vectors", "event_profiles", "audit_full_action_vectors",
            "audit_full_event_profiles", "entry_contracts",
        ))
    ):
        raise ValueError("block vector arrays and manifest disagree")

    prototype_count = int(manifest.get("contract_prototype_count", -1))
    expected_prototype_shapes = {
        "block_ids": (prototype_count,), "anchors": (prototype_count,),
        "entry_contracts": (prototype_count, builder.FEATURE_DIM),
        "layouts": (prototype_count, builder.LAYOUT_DIM),
        "unlocked_masks": (prototype_count, builder.MASK_DIM),
        "provenance_ids": (prototype_count,), "route_ids": (prototype_count,),
        "team_names": (prototype_count,),
    }
    if prototype_count <= 0 or any(
        prototypes[name].shape != shape for name, shape in expected_prototype_shapes.items()
    ) or not np.isfinite(prototypes["entry_contracts"]).all():
        raise ValueError("contract prototype arrays have invalid fixed shapes")

    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    used_provenance: set[str] = set()
    lineage_by_block: dict[str, dict[str, str]] = {}
    for row in blocks:
        block_id = str(row["block_id"])
        lineage = {
            str(value["provenance_id"]): str(value["team_name"])
            for value in row.get("source_lineage", ()) or ()
        }
        if not lineage or len(lineage) != len(row.get("source_lineage", ()) or ()):
            raise ValueError(f"invalid source lineage: {block_id}")
        if set(lineage) != set(map(str, row.get("source_provenance_ids", ()) or ())):
            raise ValueError(f"source lineage/provenance ids disagree: {block_id}")
        lineage_by_block[block_id] = lineage
        used_provenance.update(lineage)
        for value in row.get("source_full_action_sha256s", ()) or ():
            _expected_sha256(value, f"full-action variant of {block_id}")

    prototype_rows_by_block: dict[str, list[int]] = {value: [] for value in block_ids}
    for index, raw_block_id in enumerate(prototypes["block_ids"]):
        block_id = str(raw_block_id)
        if block_id not in block_index:
            raise ValueError(f"prototype references unknown block: {block_id}")
        row = blocks[block_index[block_id]]
        provenance_id = str(prototypes["provenance_ids"][index])
        route_id = str(prototypes["route_ids"][index])
        team = str(prototypes["team_names"][index])
        if (
            int(prototypes["anchors"][index]) != int(row["anchor"])
            or lineage_by_block[block_id].get(provenance_id) != team
            or route_id not in set(map(str, row.get("source_route_ids", ()) or ()))
        ):
            raise ValueError(f"prototype lineage mismatch: {block_id}")
        prototype_rows_by_block[block_id].append(index)
    if any(not rows for rows in prototype_rows_by_block.values()):
        raise ValueError("every block must have at least one contract prototype")

    declared_active = {
        str(row["block_id"])
        for rows in (manifest.get("active_representatives") or {}).values()
        for row in rows
    }
    array_active = {
        block_ids[index] for index in np.flatnonzero(vectors["active"])
    }
    if declared_active != array_active:
        raise ValueError("builder active references and vector flags disagree")
    provenance, quality = _load_prepared_quality(manifest, used_provenance)
    for block_id, lineage in lineage_by_block.items():
        if any(
            not team or str(provenance[provenance_id].get("team_name", "")) != team
            for provenance_id, team in lineage.items()
        ):
            raise ValueError(f"prepared/source team lineage mismatch: {block_id}")
    quality["used_provenance_count"] = len(used_provenance)
    quality["source_scope"] = "train/public only; validation/sealed forbidden"
    manifest["_prototype_rows_by_block"] = prototype_rows_by_block
    manifest["_lineage_by_block"] = lineage_by_block
    manifest["_declared_active"] = declared_active
    return manifest, blocks, vectors, prototypes, provenance, quality, manifest_sha


def _scaled(values: np.ndarray) -> np.ndarray:
    result = builder._scaled(np.asarray(values, np.float64))
    return np.clip(result, -8.0, 8.0).astype(np.float32)


def _features(
    anchor_indices: Sequence[int], blocks: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, np.ndarray], prototypes: Mapping[str, np.ndarray],
    prototype_rows_by_block: Mapping[str, Sequence[int]],
    provenance: Mapping[str, Mapping[str, Any]], quality_available: bool,
) -> dict[str, Any]:
    action_raw = np.concatenate((
        vectors["action_vectors"][anchor_indices],
        vectors["event_profiles"][anchor_indices],
    ), axis=1)[:, builder.PATH_CLUSTER_ACTION_FEATURES]
    contract_rows = []
    risk_rows = []
    layout_rows = []
    teams = []
    outcome_rows = []
    margins = []
    score_rates = []
    supports = []
    state_indices = np.asarray([
        *range(continuation.MARKET_START),
        *range(continuation.MARKET_STOP, continuation.PLAN_START),
    ], np.int64)
    for block_index in anchor_indices:
        block = blocks[block_index]
        block_id = str(block["block_id"])
        rows = np.asarray(prototype_rows_by_block[block_id], np.int64)
        contracts = prototypes["entry_contracts"][rows].astype(np.float64)
        state = contracts[:, state_indices]
        contract_rows.append(np.concatenate((
            np.median(state, axis=0),
            np.percentile(state, 75, axis=0) - np.percentile(state, 25, axis=0),
        )))
        capital = contracts[:, continuation.PLAN_START:]
        risk_rows.append(np.concatenate((
            np.percentile(capital, 10, axis=0),
            np.median(capital, axis=0),
            np.percentile(capital, 90, axis=0),
        )))
        layouts = prototypes["layouts"][rows].astype(np.int64)
        histogram = np.bincount(layouts.reshape(-1), minlength=14)[:14]
        histogram = histogram / max(1, layouts.size)
        layout_rows.append(np.concatenate((
            histogram, np.mean(prototypes["unlocked_masks"][rows], axis=0),
        )))
        source_ids = list(map(str, block.get("source_provenance_ids", ()) or ()))
        teams.append(frozenset(
            str(provenance[source_id].get("team_name", "")) for source_id in source_ids
        ))
        supports.append(int(block.get("occurrence_support", len(source_ids))))
        if quality_available:
            results = [str(provenance[source_id]["result"]) for source_id in source_ids]
            counts = np.asarray([
                results.count("loss"), results.count("tie"), results.count("win"),
            ], np.float64)
            outcome_rows.append(counts / max(1, counts.sum()))
            score_rates.append(float((counts[2] + .5 * counts[1]) / counts.sum()))
            margins.append(float(np.median([
                float(provenance[source_id]["final_reward"])
                - float(provenance[source_id]["opponent_reward"])
                for source_id in source_ids
            ])))
        else:
            outcome_rows.append(np.zeros(3, np.float64))
            score_rates.append(0.0)
            margins.append(0.0)

    quality_scores: list[float] | None = None
    if quality_available:
        unique_margins = sorted(set(margins))
        percentile = {
            value: (index / max(1, len(unique_margins) - 1))
            for index, value in enumerate(unique_margins)
        }
        quality_scores = [
            .75 * score + .25 * percentile[margin]
            for score, margin in zip(score_rates, margins)
        ]
    return {
        "path_action": _scaled(action_raw),
        "contract_prototype": _scaled(np.stack(contract_rows)),
        "capital_risk": _scaled(np.stack(risk_rows)),
        "layout": _scaled(np.stack(layout_rows)),
        "source_team": teams,
        "historical_outcome": np.stack(outcome_rows).astype(np.float32),
        "quality": quality_scores,
        "support": supports,
    }


def _distance(features: Mapping[str, Any], left: int, right: int) -> tuple[float, dict[str, float]]:
    parts: dict[str, float] = {}
    for name in ("path_action", "contract_prototype", "capital_risk", "layout"):
        raw = float(np.mean(
            (features[name][left].astype(np.float64)
             - features[name][right].astype(np.float64)) ** 2
        ))
        parts[name] = raw / (1.0 + raw)
    left_teams = features["source_team"][left]
    right_teams = features["source_team"][right]
    union = left_teams | right_teams
    parts["source_team"] = 1.0 - len(left_teams & right_teams) / max(1, len(union))
    parts["historical_outcome"] = float(
        .5 * np.sum(np.abs(
            features["historical_outcome"][left]
            - features["historical_outcome"][right]
        ))
    )
    weights = dict(DISTANCE_WEIGHTS)
    if features["quality"] is None:
        weights["historical_outcome"] = 0.0
    total_weight = sum(weights.values())
    total = sum(weights[name] * parts[name] for name in weights) / total_weight
    return float(total), parts


def _select(
    candidates: Sequence[int], limit: int, features: Mapping[str, Any],
    quality_bonus: float, blocks: Sequence[Mapping[str, Any]],
    declared_active: set[str],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    if limit <= 0:
        raise ValueError("portfolio tier limits must be positive")
    candidates = list(candidates)
    if len(candidates) <= limit:
        ordered = sorted(candidates, key=lambda value: str(blocks[value]["block_id"]))
        return ordered, {value: {
            "kind": "capacity_not_binding",
            "candidate_count": len(candidates),
            "limit": limit,
        } for value in ordered}
    local = {block_index: index for index, block_index in enumerate(candidates)}
    quality = features["quality"]
    first = min(candidates, key=lambda value: (
        -(quality[local[value]] if quality is not None else 0.0),
        -int(features["support"][local[value]]),
        str(blocks[value]["block_id"]),
    ))
    selected = [first]
    reasons = {first: {
        "kind": "historical_quality_seed" if quality is not None else "support_seed",
        "historical_quality": quality[local[first]] if quality is not None else None,
        "occurrence_support": int(features["support"][local[first]]),
        "builder_active": str(blocks[first]["block_id"]) in declared_active,
    }}
    while len(selected) < limit:
        covered_teams = set().union(*(
            features["source_team"][local[value]] for value in selected
        ))
        best = None
        for block_index in candidates:
            if block_index in reasons:
                continue
            local_index = local[block_index]
            nearest = min(
                (_distance(features, local_index, local[value])[0], value)
                for value in selected
            )
            diversity, nearest_block = nearest
            quality_value = quality[local_index] if quality is not None else 0.0
            utility = (1.0 - quality_bonus) * diversity + quality_bonus * quality_value
            key = (
                utility,
                diversity,
                quality_value,
                int(features["support"][local_index]),
                str(blocks[block_index]["block_id"]),
            )
            if best is None or key[:-1] > best[0][:-1] or (
                key[:-1] == best[0][:-1] and key[-1] < best[0][-1]
            ):
                best = (key, block_index, nearest_block)
        assert best is not None
        key, block_index, nearest_block = best
        local_index = local[block_index]
        nearest_local = local[nearest_block]
        total, parts = _distance(features, local_index, nearest_local)
        new_teams = features["source_team"][local_index] - covered_teams
        reasons[block_index] = {
            "kind": "maximin_multiview",
            "nearest_selected_block_id": str(blocks[nearest_block]["block_id"]),
            "nearest_selected_distance": total,
            "distance_parts": parts,
            "new_source_team_count": len(new_teams),
            "historical_quality": quality[local_index] if quality is not None else None,
            "quality_bonus_weight": quality_bonus if quality is not None else 0.0,
            "selection_utility": float(key[0]),
            "occurrence_support": int(features["support"][local_index]),
            "builder_active": str(blocks[block_index]["block_id"]) in declared_active,
        }
        selected.append(block_index)
    return selected, reasons


def _selection_ref(
    index: int, block: Mapping[str, Any], prototype_rows: Sequence[int],
    manifest: Mapping[str, Any], reason: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "block_id": str(block["block_id"]),
        "block_ref": {
            "file": "block_library_manifest.json", "block_index": index,
        },
        "action_ref": {
            "file": str(manifest["actions_file"]), "key": str(block["block_id"]),
        },
        "vector_ref": {
            "file": str(manifest["vectors_file"]), "row": index,
        },
        "contract_prototype_ref": {
            "file": str(manifest["contract_prototypes_file"]),
            "rows": list(map(int, prototype_rows)),
        },
        "market_variant_ref": {
            "file": "block_library_manifest.json", "block_index": index,
            "field": "source_full_action_sha256s",
            "variant_count": int(block["full_action_variant_count"]),
            "slot_semantics": "variants_of_one_unit_path_not_portfolio_candidates",
        },
        "reason": dict(reason),
    }


def build_portfolio(
    *, library_root: Path, library_manifest_sha256: str, output_root: Path,
    hot_limit: int = 32, active_limit: int = 8,
) -> dict[str, Any]:
    library_root, output_root = library_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    if not (0 < active_limit <= hot_limit):
        raise ValueError("portfolio limits require 0 < active <= hot")
    (
        manifest, blocks, vectors, prototypes, provenance, quality, manifest_sha,
    ) = _load_library(library_root, library_manifest_sha256)
    prototype_rows = manifest.pop("_prototype_rows_by_block")
    manifest.pop("_lineage_by_block")
    declared_active = manifest.pop("_declared_active")
    by_anchor = {
        anchor: [
            index for index, block in enumerate(blocks)
            if int(block["anchor"]) == anchor
        ]
        for anchor in builder.ANCHORS
    }
    if any(not rows for rows in by_anchor.values()):
        raise ValueError("every fixed anchor requires at least one cold block")

    per_anchor = {}
    for anchor, cold in by_anchor.items():
        cold_features = _features(
            cold, blocks, vectors, prototypes, prototype_rows, provenance,
            bool(quality["available"]),
        )
        hot, hot_reasons = _select(
            cold, hot_limit, cold_features, HOT_QUALITY_BONUS, blocks,
            declared_active,
        )
        hot_features = _features(
            hot, blocks, vectors, prototypes, prototype_rows, provenance,
            bool(quality["available"]),
        )
        active, active_reasons = _select(
            hot, active_limit, hot_features, ACTIVE_QUALITY_BONUS, blocks,
            declared_active,
        )
        per_anchor[str(anchor)] = {
            "cold": {
                "count": len(cold),
                "block_indices": cold,
                "reason": "all unique unit-path blocks at this anchor",
            },
            "hot": {
                "limit": hot_limit,
                "count": len(hot),
                "blocks": [
                    _selection_ref(
                        index, blocks[index], prototype_rows[str(blocks[index]["block_id"])],
                        manifest, hot_reasons[index],
                    ) for index in hot
                ],
            },
            "active": {
                "limit": active_limit,
                "count": len(active),
                "blocks": [
                    _selection_ref(
                        index, blocks[index], prototype_rows[str(blocks[index]["block_id"])],
                        manifest, active_reasons[index],
                    ) for index in active
                ],
            },
        }

    implementation_sha = _sha256_bytes(Path(__file__).read_bytes())
    config = {"hot_limit_per_anchor": hot_limit, "active_limit_per_anchor": active_limit}
    core = {
        "schema": SCHEMA,
        "implementation_sha256": implementation_sha,
        "library_manifest_sha256": manifest_sha,
        "artifact_sha256s": {
            "actions": str(manifest["actions_sha256"]),
            "vectors": str(manifest["vectors_sha256"]),
            "contract_prototypes": str(manifest["contract_prototypes_sha256"]),
        },
        "config": config,
        "quality_mode": quality["mode"],
        "per_anchor": per_anchor,
    }
    content_sha = _sha256_bytes(_canonical_bytes(core))
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "portfolio_id": f"OBP1_{content_sha[:24]}",
        "content_sha256": content_sha,
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3]
            ).as_posix(),
            "sha256": implementation_sha,
        },
        "input": {
            "library_root": str(library_root),
            "library_manifest": "block_library_manifest.json",
            "library_manifest_sha256": manifest_sha,
            "artifact_sha256s": core["artifact_sha256s"],
            "deduplication_mode": "unit_only",
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_used": False,
        },
        "config": config,
        "tiers": {
            "cold": "all unique unit-path blocks; references only",
            "hot": "deterministic maximin multi-view subset, at most 32 by default",
            "active": (
                "static default maximin subset of hot, at most 8 by default; "
                "runtime state-conditioned retrieval may replace it"
            ),
            "market_overlay": "nested variant reference; never consumes another path slot",
        },
        "selection": {
            "runtime_state_conditioned": False,
            "distance_weights": DISTANCE_WEIGHTS,
            "hot_historical_quality_bonus": (
                HOT_QUALITY_BONUS if quality["available"] else 0.0
            ),
            "active_historical_quality_bonus": (
                ACTIVE_QUALITY_BONUS if quality["available"] else 0.0
            ),
            "contract_summary": (
                "prototype median+IQR excluding observed market environment features 86:104"
            ),
            "capital_risk_summary": "prototype p10/median/p90 of fixed-capital features 129:147",
            "layout_summary": "prototype tile-code histogram plus unlocked-mask mean",
            "historical_quality": quality,
            "builder_active_used_as": "audited tie metadata, not a forced slot",
        },
        "storage": {
            "actions_copied": False,
            "contracts_copied": False,
            "output_contains": "block/artifact row refs, selection reasons, and hashes only",
        },
        "per_anchor": per_anchor,
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.", dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary)
        (staging / "portfolio_manifest.json").write_bytes(payload)
        digest = _sha256_bytes(payload)
        (staging / "portfolio_manifest.json.sha256").write_text(
            f"{digest}  portfolio_manifest.json\n", encoding="ascii",
        )
        if output_root.exists():
            raise FileExistsError(f"output directory appeared during run: {output_root}")
        staging.replace(output_root)
    output["manifest_sha256"] = digest
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--library-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hot-limit", type=int, default=32)
    parser.add_argument("--active-limit", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_portfolio(
        library_root=args.library_root,
        library_manifest_sha256=args.library_manifest_sha256,
        output_root=args.output_root,
        hot_limit=args.hot_limit,
        active_limit=args.active_limit,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "portfolio_id": manifest["portfolio_id"],
        "quality_mode": manifest["selection"]["historical_quality"]["mode"],
        "cold_blocks": sum(value["cold"]["count"] for value in manifest["per_anchor"].values()),
        "hot_blocks": sum(value["hot"]["count"] for value in manifest["per_anchor"].values()),
        "active_blocks": sum(value["active"]["count"] for value in manifest["per_anchor"].values()),
        "manifest_sha256": manifest["manifest_sha256"],
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
