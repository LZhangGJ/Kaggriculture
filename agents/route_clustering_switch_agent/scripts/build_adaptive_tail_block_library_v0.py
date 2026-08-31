#!/usr/bin/env python3
"""Build a deduplicated, continuation-compatible replay block library."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_block_mvp_96_216 import (  # noqa: E402
    MACRO_KEYS,
    _action_parts,
    _scaled,
    _unit_role,
)


SCHEMA = "adaptive-tail-block-library-v0"
INPUT_SCHEMA = "block-mvp-continuation-multitail-v1"
HORIZON = 719
ANCHORS = tuple(range(216, 697, 24))
FEATURE_DIM = 147
LAYOUT_DIM = 100
MASK_DIM = 4
MARKET_START, MARKET_STOP = 86, 104
ACTION_DIM = 54
EVENT_DIM = 14
MARKET_ACTION_START, MARKET_ACTION_STOP = 35, 53
ALL_CLUSTER_ACTION_FEATURES = np.arange(ACTION_DIM + EVENT_DIM, dtype=np.int64)
PATH_CLUSTER_ACTION_FEATURES = np.asarray([
    *range(MARKET_ACTION_START),
    *range(MARKET_ACTION_STOP, ACTION_DIM + EVENT_DIM),
], dtype=np.int64)
GROUP_WEIGHTS = {
    "action_intent_and_event": 0.60,
    "entry_contract_without_market_environment": 0.25,
    "layout_and_unlocked_mask": 0.15,
}
NON_MARKET_FEATURES = np.asarray(
    [*range(MARKET_START), *range(MARKET_STOP, FEATURE_DIM)], np.int64,
)
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
DEDUPLICATION_MODES = ("full_action", "unit_only")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_artifact(root: Path, name: Any) -> Path:
    path = (root / str(name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"prepared artifact escapes its root: {name}") from exc
    return path


def _canonical_block_actions(
    tape: Sequence[Mapping[str, Any]], anchor: int, mode: str,
) -> list[dict[str, Any]]:
    if mode not in DEDUPLICATION_MODES:
        raise ValueError(f"invalid deduplication mode: {mode}")
    actions = list(tape[anchor:min(anchor + 24, HORIZON)])
    if mode == "full_action":
        return [dict(action) for action in actions]
    return [{
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [
            list(raw or ["PASS"]) for raw in (action.get("hands", ()) or ())
        ],
        "market": [],
    } for action in actions]


@dataclass(frozen=True)
class PreparedBank:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    actions: dict[str, list[dict[str, Any]]]
    contracts: np.ndarray
    layouts: np.ndarray
    masks: np.ndarray
    provenance_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    provenance: tuple[dict[str, Any], ...]


def load_prepared(root: Path) -> PreparedBank:
    """Strictly load one immutable continuation prepared bank."""

    root = root.resolve()
    manifest_path = root / "candidate_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("invalid continuation candidate manifest schema")
    if tuple(map(int, manifest.get("anchors", ()))) != ANCHORS:
        raise ValueError("continuation manifest anchors are not the fixed 21 anchors")

    action_path = _safe_artifact(root, manifest.get("actions_file"))
    contract_path = _safe_artifact(root, manifest.get("contracts_file"))
    action_bytes = action_path.read_bytes()
    contract_bytes = contract_path.read_bytes()
    if _sha256_bytes(action_bytes) != str(manifest.get("actions_sha256", "")):
        raise ValueError("continuation action archive digest mismatch")
    if _sha256_bytes(contract_bytes) != str(manifest.get("contracts_sha256", "")):
        raise ValueError("continuation contract archive digest mismatch")

    try:
        actions = json.loads(zlib.decompress(action_bytes).decode("utf-8"))
    except (OSError, zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid continuation action archive") from exc
    if not isinstance(actions, dict) or not actions:
        raise ValueError("continuation action archive is empty")
    if any(
        not isinstance(route_id, str)
        or not isinstance(tape, list)
        or len(tape) != HORIZON
        or any(not isinstance(action, dict) for action in tape)
        for route_id, tape in actions.items()
    ):
        raise ValueError("continuation bank requires dict-valued 719-step tapes")

    with np.load(io.BytesIO(contract_bytes), allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    count = int(manifest.get("provenance_count", -1))
    expected = {
        "contracts": (count, len(ANCHORS), FEATURE_DIM),
        "layouts": (count, len(ANCHORS), LAYOUT_DIM),
        "unlocked_masks": (count, len(ANCHORS), MASK_DIM),
        "provenance_ids": (count,),
        "route_ids": (count,),
        "anchors": (len(ANCHORS),),
    }
    if set(arrays) != set(expected) or any(
        arrays[name].shape != shape for name, shape in expected.items()
    ):
        raise ValueError("continuation contract arrays have an invalid fixed shape")
    if tuple(map(int, arrays["anchors"])) != ANCHORS:
        raise ValueError("continuation contract anchors disagree with the manifest")
    if not all(
        np.issubdtype(arrays[name].dtype, np.number)
        and np.isfinite(arrays[name]).all()
        for name in ("contracts", "layouts", "unlocked_masks")
    ):
        raise ValueError("continuation contract arrays contain non-finite values")
    if (
        np.any(arrays["layouts"] < 0)
        or np.any(arrays["layouts"] > 13)
        or np.any(arrays["unlocked_masks"] < 0)
        or np.any(arrays["unlocked_masks"] > 1)
    ):
        raise ValueError("continuation layout or unlocked-mask codes are invalid")

    provenance = tuple(dict(row) for row in manifest.get("provenance", ()) or ())
    routes = tuple(dict(row) for row in manifest.get("routes", ()) or ())
    provenance_ids = tuple(map(str, arrays["provenance_ids"]))
    route_ids = tuple(map(str, arrays["route_ids"]))
    manifest_provenance_ids = tuple(str(row.get("provenance_id")) for row in provenance)
    manifest_route_ids = tuple(str(row.get("route_id")) for row in provenance)
    bank_routes = set(actions)
    listed_routes = {str(row.get("route_id")) for row in routes}
    if (
        count <= 0
        or len(provenance) != count
        or len(set(provenance_ids)) != count
        or provenance_ids != manifest_provenance_ids
        or route_ids != manifest_route_ids
        or set(route_ids) != bank_routes
        or listed_routes != bank_routes
        or int(manifest.get("unique_tail_count", -1)) != len(actions)
    ):
        raise ValueError("continuation manifest, route bank, and arrays disagree")
    for row in routes:
        route_id = str(row["route_id"])
        expected_tail = _sha256_bytes(_canonical_bytes(actions[route_id][216:]))
        if str(row.get("tail_sha256_step216", "")) != expected_tail:
            raise ValueError(f"continuation route tail digest mismatch: {route_id}")

    return PreparedBank(
        root=root,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        actions=actions,
        contracts=np.asarray(arrays["contracts"], np.float32),
        layouts=np.asarray(arrays["layouts"], np.uint8),
        masks=np.asarray(arrays["unlocked_masks"], np.uint8),
        provenance_ids=provenance_ids,
        route_ids=route_ids,
        provenance=provenance,
    )


def action_intent(
    tape: Sequence[Mapping[str, Any]], anchor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 54-D plan-only intent and 14-D event profile for one block."""

    stop = min(anchor + 24, HORIZON)
    duration = stop - anchor
    window = list(tape[anchor:stop])
    padded = [*window, *[PASS_ACTION] * (24 - duration)]
    macro, _, market, events = _action_parts(padded, 0, 24)
    events[:7] = np.minimum(events[:7], duration)

    roles = np.zeros(5, np.float32)
    for action in window:
        for raw in [action.get("farmer"), *(action.get("hands", ()) or ())]:
            roles[_unit_role(list(raw or ["PASS"]))] += 1
    roles /= max(1.0, float(roles.sum()))

    totals = macro.reshape(-1, len(MACRO_KEYS)).sum(axis=0)
    planned = np.asarray((
        *totals[:5], *totals[7:10], totals[5], totals[6], totals[10], totals[11],
    ), np.float32)
    trades = market.reshape(-1, 2)
    timing = {name: [] for name in ("buy_step", "buy_weight", "sell_step", "sell_weight")}
    for offset, action in enumerate(window):
        for raw in action.get("market", ()) or ():
            order = list(raw or ())
            if not order:
                continue
            op = str(order[0])
            try:
                quantity = max(0, int(order[2])) if len(order) >= 3 else 1
            except (TypeError, ValueError):
                quantity = 0
            if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND", "HIRE"}:
                timing["buy_step"].append(offset)
                timing["buy_weight"].append(max(1, quantity))
            elif op == "SELL":
                timing["sell_step"].append(offset)
                timing["sell_weight"].append(max(1, quantity))

    def time_summary(kind: str) -> tuple[float, float]:
        steps = np.asarray(timing[f"{kind}_step"], np.float32)
        weights = np.asarray(timing[f"{kind}_weight"], np.float32)
        if not len(steps):
            return float(duration), float(duration)
        return float(np.average(steps, weights=weights)), float(np.min(steps))

    buy_mean, buy_first = time_summary("buy")
    sell_mean, sell_first = time_summary("sell")
    trade_timing = np.asarray((
        trades[:, 0].sum(), trades[:, 1].sum(),
        buy_mean, sell_mean, buy_first, sell_first,
    ), np.float32)
    vector = np.concatenate((
        macro, planned, trade_timing, roles, market,
        np.asarray([duration], np.float32),
    )).astype(np.float32)
    if vector.shape != (ACTION_DIM,) or events.shape != (EVENT_DIM,):
        raise AssertionError("adaptive block feature dimensions changed")
    if not np.isfinite(vector).all() or not np.isfinite(events).all():
        raise ValueError("adaptive block features contain non-finite values")
    return vector, events.astype(np.float32)


@dataclass
class Block:
    block_id: str
    anchor: int
    duration: int
    action_sha256: str
    actions: list[dict[str, Any]]
    action_vector: np.ndarray
    event_profile: np.ndarray
    contract: np.ndarray
    layout: np.ndarray
    mask: np.ndarray
    source_provenance_id: str
    source_route_id: str
    sources: list[dict[str, Any]]
    quality: float
    cluster: int = -1
    audit_action_vector: np.ndarray | None = None
    audit_event_profile: np.ndarray | None = None


@dataclass(frozen=True)
class FeatureSpace:
    action: np.ndarray
    contract: np.ndarray
    layout: np.ndarray


def _feature_space(
    blocks: Sequence[Block],
    action_feature_indices: np.ndarray = ALL_CLUSTER_ACTION_FEATURES,
) -> FeatureSpace:
    action = np.concatenate((
        np.stack([block.action_vector for block in blocks]),
        np.stack([block.event_profile for block in blocks]),
    ), axis=1).astype(np.float64)
    # Counts/quantities are heavy-tailed; timing, shares and duration stay linear.
    count_indices = np.asarray([
        *range(24), *range(35, 53), *range(ACTION_DIM + 7, ACTION_DIM + 14),
    ], np.int64)
    action[:, count_indices] = np.sign(action[:, count_indices]) * np.log1p(
        np.abs(action[:, count_indices])
    )
    contract = np.stack([block.contract for block in blocks])[:, NON_MARKET_FEATURES]
    layout = np.concatenate((
        np.stack([block.layout for block in blocks]),
        np.stack([block.mask for block in blocks]),
    ), axis=1)
    indices = np.asarray(action_feature_indices, np.int64)
    if (
        indices.ndim != 1 or not len(indices)
        or np.any(indices < 0) or np.any(indices >= action.shape[1])
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("invalid cluster action feature indices")
    return FeatureSpace(
        action=_scaled(action[:, indices]).astype(np.float32),
        contract=_scaled(contract.astype(np.float64)).astype(np.float32),
        layout=layout.astype(np.uint8),
    )


def _distance_to(space: FeatureSpace, center: int) -> np.ndarray:
    action = np.mean((space.action - space.action[center]) ** 2, axis=1)
    contract = np.mean((space.contract - space.contract[center]) ** 2, axis=1)
    layout = np.mean(space.layout != space.layout[center], axis=1)
    return (
        GROUP_WEIGHTS["action_intent_and_event"] * action
        + GROUP_WEIGHTS["entry_contract_without_market_environment"] * contract
        + GROUP_WEIGHTS["layout_and_unlocked_mask"] * layout
    )


def _medoid(space: FeatureSpace, members: np.ndarray, blocks: Sequence[Block]) -> int:
    count = len(members)
    costs = np.zeros(count, np.float64)
    for values, weight in (
        (space.action[members].astype(np.float64), GROUP_WEIGHTS["action_intent_and_event"]),
        (space.contract[members].astype(np.float64), GROUP_WEIGHTS["entry_contract_without_market_environment"]),
    ):
        norms = np.sum(values * values, axis=1)
        costs += weight * (
            count * norms - 2 * values.dot(np.sum(values, axis=0)) + np.sum(norms)
        ) / max(1, values.shape[1])
    categorical = space.layout[members]
    mismatch = np.zeros(count, np.float64)
    for column in range(categorical.shape[1]):
        _, inverse, frequencies = np.unique(
            categorical[:, column], return_inverse=True, return_counts=True,
        )
        mismatch += count - frequencies[inverse]
    costs += GROUP_WEIGHTS["layout_and_unlocked_mask"] * mismatch / categorical.shape[1]
    return min(
        range(count),
        key=lambda local: (
            float(costs[local]), -blocks[int(members[local])].quality,
            blocks[int(members[local])].block_id,
        ),
    )


def _cluster(
    blocks: Sequence[Block], k: int,
    action_feature_indices: np.ndarray = ALL_CLUSTER_ACTION_FEATURES,
    occurrence_weighted_support: bool = False,
) -> list[dict[str, Any]]:
    if not blocks:
        raise ValueError("cannot cluster an empty anchor")
    space = _feature_space(blocks, action_feature_indices)
    k = min(k, len(blocks))
    first = min(
        range(len(blocks)), key=lambda index: (-blocks[index].quality, blocks[index].block_id),
    )
    centers = [first]
    nearest = _distance_to(space, first)
    while len(centers) < k:
        index = min(
            (value for value in range(len(blocks)) if value not in centers),
            key=lambda value: (-float(nearest[value]), -blocks[value].quality, blocks[value].block_id),
        )
        centers.append(index)
        nearest = np.minimum(nearest, _distance_to(space, index))
    distances = np.stack([_distance_to(space, center) for center in centers], axis=1)
    assignment = np.argmin(distances, axis=1)
    # Equal semantic features can otherwise send every exact-action center to C00.
    assignment[np.asarray(centers, np.int64)] = np.arange(k, dtype=np.int64)

    result = []
    for cluster in range(k):
        members = np.flatnonzero(assignment == cluster)
        representative = int(members[_medoid(space, members, blocks)])
        for index in members:
            blocks[int(index)].cluster = cluster
        block = blocks[representative]
        occurrence_support = int(sum(
            len(blocks[int(index)].sources) for index in members
        ))
        unique_member_count = int(len(members))
        result.append({
            "block_id": block.block_id,
            "source_route_id": block.source_route_id,
            "source_provenance_id": block.source_provenance_id,
            "cluster_id": f"C{cluster:02d}",
            "support": (
                occurrence_support if occurrence_weighted_support
                else unique_member_count
            ),
            "unique_member_count": unique_member_count,
            "unique_block_support": unique_member_count,
            "occurrence_support": occurrence_support,
        })
    return result


def _quality(source: Mapping[str, Any]) -> float:
    reward = float(source.get("final_reward", 0) or 0)
    opponent = float(source.get("opponent_reward", 0) or 0)
    if not math.isfinite(reward) or not math.isfinite(opponent):
        raise ValueError("non-finite source reward")
    return (1_000_000.0 if reward > opponent else 0.0) + reward - opponent + reward * 1e-6


def _atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = _sha256_bytes(payload)
    if _sha256_bytes(path.read_bytes()) != digest:
        raise OSError(f"post-write hash verification failed: {path}")
    return digest


def build_library(
    prepared_roots: Sequence[Path], output_root: Path, clusters_per_anchor: int = 8,
    heldout_team_names: Sequence[str] = (),
    deduplication_mode: str | None = None,
) -> dict[str, Any]:
    if clusters_per_anchor <= 0:
        raise ValueError("clusters_per_anchor must be positive")
    roots = sorted({Path(value).resolve() for value in prepared_roots}, key=str)
    if not roots:
        raise ValueError("at least one --prepared-root is required")
    banks = [load_prepared(root) for root in roots]
    heldout_teams = tuple(sorted({str(value) for value in heldout_team_names}))
    if any(not value for value in heldout_teams):
        raise ValueError("heldout team names must be non-empty")
    heldout_set = set(heldout_teams)
    deduplication_mode = deduplication_mode or (
        "unit_only" if heldout_teams else "full_action"
    )
    if deduplication_mode not in DEDUPLICATION_MODES:
        raise ValueError(f"invalid deduplication mode: {deduplication_mode}")
    cluster_action_features = (
        PATH_CLUSTER_ACTION_FEATURES if deduplication_mode == "unit_only"
        else ALL_CLUSTER_ACTION_FEATURES
    )

    route_actions: dict[str, list[dict[str, Any]]] = {}
    seen_provenance: set[str] = set()
    raw: dict[tuple[int, str], dict[str, Any]] = {}
    occurrence_count = {anchor: 0 for anchor in ANCHORS}
    input_occurrence_count = {anchor: 0 for anchor in ANCHORS}
    full_action_fingerprints = {anchor: set() for anchor in ANCHORS}
    bank_filter_counts: dict[Path, tuple[int, int]] = {}
    for bank in banks:
        retained = 0
        for row_index, source in enumerate(bank.provenance):
            for anchor in ANCHORS:
                input_occurrence_count[anchor] += 1
            team_name = source.get("team_name")
            if heldout_teams and (not isinstance(team_name, str) or not team_name):
                raise ValueError("strict lineage filtering requires source team_name")
            if team_name in heldout_set:
                continue
            retained += 1
            provenance_id = bank.provenance_ids[row_index]
            route_id = bank.route_ids[row_index]
            if provenance_id in seen_provenance:
                raise ValueError(f"duplicate provenance across prepared roots: {provenance_id}")
            seen_provenance.add(provenance_id)
            tape = bank.actions[route_id]
            if route_id in route_actions and _canonical_bytes(route_actions[route_id][216:]) != _canonical_bytes(tape[216:]):
                raise ValueError(f"route id has conflicting tails across prepared roots: {route_id}")
            route_actions.setdefault(route_id, tape)
            for anchor_index, anchor in enumerate(ANCHORS):
                actions = _canonical_block_actions(
                    tape, anchor, deduplication_mode,
                )
                full_actions = _canonical_block_actions(tape, anchor, "full_action")
                full_action_sha = _sha256_bytes(_canonical_bytes(full_actions))
                full_action_fingerprints[anchor].add(full_action_sha)
                action_sha = _sha256_bytes(_canonical_bytes(actions))
                key = (anchor, action_sha)
                occurrence_count[anchor] += 1
                occurrence = {
                    "source": source,
                    "provenance_id": provenance_id,
                    "route_id": route_id,
                    "contract": bank.contracts[row_index, anchor_index],
                    "layout": bank.layouts[row_index, anchor_index],
                    "mask": bank.masks[row_index, anchor_index],
                    "quality": _quality(source),
                    "tape": tape,
                    "full_action_sha256": full_action_sha,
                }
                if key not in raw:
                    raw[key] = {
                        "actions": actions, "sources": [occurrence],
                    }
                else:
                    if _canonical_bytes(raw[key]["actions"]) != _canonical_bytes(actions):
                        raise AssertionError("SHA-256 collision in action slices")
                    raw[key]["sources"].append(occurrence)
        bank_filter_counts[bank.root] = (retained, len(bank.provenance) - retained)

    if not raw:
        raise ValueError("lineage filter removed every provenance row")

    blocks: list[Block] = []
    for (anchor, action_sha), value in sorted(raw.items()):
        representative = min(
            value["sources"],
            key=lambda item: (
                -float(item["quality"]), str(item["provenance_id"]), str(item["route_id"]),
            ),
        )
        audit_vector, audit_events = action_intent(
            representative["tape"], anchor,
        )
        if deduplication_mode == "unit_only":
            unit_tape = [PASS_ACTION] * HORIZON
            unit_tape[anchor:anchor + len(value["actions"])] = value["actions"]
            vector, events = action_intent(unit_tape, anchor)
        else:
            vector, events = audit_vector, audit_events
        blocks.append(Block(
            block_id=f"ATB{anchor}_{action_sha}",
            anchor=anchor,
            duration=min(anchor + 24, HORIZON) - anchor,
            action_sha256=action_sha,
            actions=value["actions"],
            action_vector=vector,
            event_profile=events,
            contract=np.asarray(representative["contract"], np.float32),
            layout=np.asarray(representative["layout"], np.uint8),
            mask=np.asarray(representative["mask"], np.uint8),
            source_provenance_id=str(representative["provenance_id"]),
            source_route_id=str(representative["route_id"]),
            sources=value["sources"],
            quality=float(representative["quality"]),
            audit_action_vector=audit_vector,
            audit_event_profile=audit_events,
        ))

    active: dict[str, list[dict[str, Any]]] = {}
    for anchor in ANCHORS:
        anchor_blocks = [block for block in blocks if block.anchor == anchor]
        active[str(anchor)] = _cluster(
            anchor_blocks, clusters_per_anchor, cluster_action_features,
            occurrence_weighted_support=deduplication_mode == "unit_only",
        )
    active_ids = {
        row["block_id"] for rows in active.values() for row in rows
    }

    actions_payload = zlib.compress(_canonical_bytes({
        block.block_id: block.actions for block in blocks
    }), level=9)
    arrays = {
        "block_ids": np.asarray([block.block_id for block in blocks]),
        "anchors": np.asarray([block.anchor for block in blocks], np.int16),
        "durations": np.asarray([block.duration for block in blocks], np.int8),
        "action_sha256": np.asarray([block.action_sha256 for block in blocks]),
        "action_vectors": np.stack([block.action_vector for block in blocks]).astype(np.float32),
        "event_profiles": np.stack([block.event_profile for block in blocks]).astype(np.float32),
        "audit_full_action_vectors": np.stack([
            block.audit_action_vector for block in blocks
        ]).astype(np.float32),
        "audit_full_event_profiles": np.stack([
            block.audit_event_profile for block in blocks
        ]).astype(np.float32),
        "entry_contracts": np.stack([block.contract for block in blocks]).astype(np.float32),
        "layouts": np.stack([block.layout for block in blocks]).astype(np.uint8),
        "unlocked_masks": np.stack([block.mask for block in blocks]).astype(np.uint8),
        "source_provenance_ids": np.asarray([block.source_provenance_id for block in blocks]),
        "source_route_ids": np.asarray([block.source_route_id for block in blocks]),
        "cluster_ids": np.asarray([block.cluster for block in blocks], np.int16),
        "active": np.asarray([block.block_id in active_ids for block in blocks], np.uint8),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    vectors_payload = buffer.getvalue()

    prototype_rows = [
        (block, source) for block in blocks for source in block.sources
    ]
    prototype_arrays = {
        "block_ids": np.asarray([
            block.block_id for block, _ in prototype_rows
        ]),
        "anchors": np.asarray([
            block.anchor for block, _ in prototype_rows
        ], np.int16),
        "entry_contracts": np.stack([
            source["contract"] for _, source in prototype_rows
        ]).astype(np.float32),
        "layouts": np.stack([
            source["layout"] for _, source in prototype_rows
        ]).astype(np.uint8),
        "unlocked_masks": np.stack([
            source["mask"] for _, source in prototype_rows
        ]).astype(np.uint8),
        "provenance_ids": np.asarray([
            str(source["provenance_id"]) for _, source in prototype_rows
        ]),
        "route_ids": np.asarray([
            str(source["route_id"]) for _, source in prototype_rows
        ]),
        "team_names": np.asarray([
            str(source["source"].get("team_name", ""))
            for _, source in prototype_rows
        ]),
    }
    prototype_buffer = io.BytesIO()
    np.savez_compressed(prototype_buffer, **prototype_arrays)
    prototypes_payload = prototype_buffer.getvalue()

    output_root = output_root.resolve()
    action_path = output_root / "block_actions.json.zlib"
    vectors_path = output_root / "block_vectors.npz"
    prototypes_path = output_root / "contract_prototypes.npz"
    actions_digest = _atomic_write(action_path, actions_payload)
    vectors_digest = _atomic_write(vectors_path, vectors_payload)
    prototypes_digest = _atomic_write(prototypes_path, prototypes_payload)
    source_lineage = {
        block.block_id: sorted(
            ({
                "provenance_id": str(row["provenance_id"]),
                "team_name": str(row["source"].get("team_name", "")),
            } for row in block.sources),
            key=lambda row: (row["provenance_id"], row["team_name"]),
        )
        for block in blocks
    }
    leaked_lineage = [
        row for rows in source_lineage.values() for row in rows
        if row["team_name"] in heldout_set
    ]
    if leaked_lineage:
        raise AssertionError("heldout source lineage reached the block library")
    manifest = {
        "schema": SCHEMA,
        "implementation": {
            "builder_path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3]
            ).as_posix(),
            "builder_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        },
        "prepared_roots": [{
            "path": str(bank.root),
            "candidate_manifest_sha256": bank.manifest_sha256,
            "actions_sha256": str(bank.manifest["actions_sha256"]),
            "contracts_sha256": str(bank.manifest["contracts_sha256"]),
            "provenance_count": len(bank.provenance),
            "retained_provenance_count": bank_filter_counts[bank.root][0],
            "excluded_provenance_count": bank_filter_counts[bank.root][1],
        } for bank in banks],
        "lineage_filter": {
            "mode": (
                "exclude_source_team_before_dedup_scale_cluster_medoid_support"
                if heldout_teams else "disabled"
            ),
            "heldout_team_names": list(heldout_teams),
            "input_provenance_count": sum(len(bank.provenance) for bank in banks),
            "retained_provenance_count": len(seen_provenance),
            "excluded_provenance_count": sum(
                count[1] for count in bank_filter_counts.values()
            ),
            "member_provenance_audit": {
                "checked_count": sum(map(len, source_lineage.values())),
                "leaked_count": len(leaked_lineage),
                "passed": not leaked_lineage,
            },
        },
        "anchors": list(ANCHORS),
        "deduplication_mode": deduplication_mode,
        "block_count": len(blocks),
        "active_representatives": active,
        "actions_file": action_path.name,
        "actions_sha256": actions_digest,
        "vectors_file": vectors_path.name,
        "vectors_sha256": vectors_digest,
        "contract_prototypes_file": prototypes_path.name,
        "contract_prototypes_sha256": prototypes_digest,
        "contract_prototype_count": len(prototype_rows),
        "contract_prototypes": {
            "mapping": "many state prototypes map to one canonical action block_id",
            "retrieval": "top-M prototype distance, then deduplicate by block_id",
            "retrieval_entry_contract_feature_indices": list(map(
                int, NON_MARKET_FEATURES,
            )),
            "audit_only_market_environment_feature_indices": list(range(
                MARKET_START, MARKET_STOP,
            )),
            "arrays": {
                "block_ids": ["N"],
                "anchors": ["N"],
                "entry_contracts": ["N", FEATURE_DIM],
                "layouts": ["N", LAYOUT_DIM],
                "unlocked_masks": ["N", MASK_DIM],
                "provenance_ids": ["N"],
                "route_ids": ["N"],
                "team_names": ["N"],
            },
        },
        "feature_contract": {
            "action_intent_dim": ACTION_DIM,
            "action_intent_parts": {
                "macro": 12, "planned": 12, "trade_timing": 6,
                "worker_roles": 5, "market_action_quantities": 18, "duration": 1,
            },
            "event_profile_dim": EVENT_DIM,
            "entry_contract_dim": FEATURE_DIM,
            "layout_dim": LAYOUT_DIM,
            "unlocked_mask_dim": MASK_DIM,
            "market_environment_excluded_feature_indices": list(range(MARKET_START, MARKET_STOP)),
            "market_action_quantities_included": True,
            "market_action_quantities_retained_for_audit": True,
            "market_action_quantities_included_in_cluster_distance": (
                deduplication_mode != "unit_only"
            ),
            "cluster_vector_source": (
                "unit-only canonical farmer+hands actions with market removed"
                if deduplication_mode == "unit_only"
                else "full action"
            ),
            "audit_full_action_vector_arrays": [
                "audit_full_action_vectors", "audit_full_event_profiles",
            ],
            "market_action_quantity_feature_indices": list(range(
                MARKET_ACTION_START, MARKET_ACTION_STOP,
            )),
            "cluster_action_and_event_feature_indices": list(map(
                int, cluster_action_features,
            )),
            "cluster_action_and_event_excluded_feature_indices": sorted(
                set(map(int, ALL_CLUSTER_ACTION_FEATURES))
                - set(map(int, cluster_action_features))
            ),
            "cluster_group_weights": GROUP_WEIGHTS,
            "scaling": {
                "action_intent_and_event": "median/IQR, std fallback, per dimension",
                "entry_contract_without_market_environment": "median/IQR, std fallback, per dimension",
                "layout_and_unlocked_mask": "categorical Hamming, normalized by 104 dimensions",
            },
            "distance": "weighted mean squared action/contract plus categorical Hamming",
            "selection": "deterministic farthest-first k-center, then exact within-cluster medoid",
            "active_support_semantics": (
                "support=occurrence_support; unique_member_count retained"
                if deduplication_mode == "unit_only"
                else "support=unique_member_count; occurrence_support retained"
            ),
            "deduplication": (
                "per-anchor canonical farmer+hands action-slice SHA-256; market=[]"
                if deduplication_mode == "unit_only"
                else "per-anchor canonical full action-slice SHA-256"
            ),
        },
        "per_anchor": {
            str(anchor): {
                "duration": min(anchor + 24, HORIZON) - anchor,
                "occurrences": occurrence_count[anchor],
                "input_occurrences": input_occurrence_count[anchor],
                "excluded_occurrences": input_occurrence_count[anchor] - occurrence_count[anchor],
                "unique_blocks": sum(block.anchor == anchor for block in blocks),
                "full_action_unique_blocks": len(full_action_fingerprints[anchor]),
                "unit_only_merged_full_action_blocks": (
                    len(full_action_fingerprints[anchor])
                    - sum(block.anchor == anchor for block in blocks)
                ),
                "deduplicated_occurrences": occurrence_count[anchor] - sum(block.anchor == anchor for block in blocks),
                "active_count": len(active[str(anchor)]),
            }
            for anchor in ANCHORS
        },
        "blocks": [{
            "block_id": block.block_id,
            "anchor": block.anchor,
            "duration": block.duration,
            "action_sha256": block.action_sha256,
            "source_provenance_id": block.source_provenance_id,
            "source_route_id": block.source_route_id,
            "source_provenance_ids": sorted({str(row["provenance_id"]) for row in block.sources}),
            "source_route_ids": sorted({str(row["route_id"]) for row in block.sources}),
            "source_lineage": source_lineage[block.block_id],
            "occurrence_support": len(block.sources),
            "source_full_action_sha256s": sorted({
                str(row["full_action_sha256"]) for row in block.sources
            }),
            "full_action_variant_count": len({
                str(row["full_action_sha256"]) for row in block.sources
            }),
            "cluster_id": f"C{block.cluster:02d}",
            "active": block.block_id in active_ids,
        } for block in blocks],
    }
    manifest_payload = _canonical_bytes(manifest) + b"\n"
    _atomic_write(output_root / "block_library_manifest.json", manifest_payload)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--clusters-per-anchor", type=int, default=8)
    parser.add_argument(
        "--heldout-team", action="append", default=[],
        help=(
            "exact source team name to remove before deduplication and path-only "
            "clustering; repeat for multiple heldout teams"
        ),
    )
    parser.add_argument(
        "--deduplication-mode", choices=DEDUPLICATION_MODES,
        help=(
            "fingerprint full actions, or farmer+hands with market removed; "
            "defaults to unit_only when heldout teams are configured"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_library(
        args.prepared_root, args.output_root, args.clusters_per_anchor,
        args.heldout_team,
        args.deduplication_mode,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "prepared_roots": len(manifest["prepared_roots"]),
        "block_count": manifest["block_count"],
        "deduplication_mode": manifest["deduplication_mode"],
        "contract_prototype_count": manifest["contract_prototype_count"],
        "active_count": sum(len(rows) for rows in manifest["active_representatives"].values()),
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
