#!/usr/bin/env python3
"""Compare active, contract-retrieved, and full replay-block path oracles.

The NR020 trajectory is immutable. Retrieval happens at each 24-step anchor;
the selected unit-only blocks are evaluated at anchor+1 for four steps and are
always discarded. R0 and R1 each retrieve eight donors and use the same
action-diversity selector to retain three. R2 exposes every block at that
anchor and is the oracle upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from meta_agent.src.route_compiler import _positions
from meta_agent.src.route_switch_features import route_switch_vector
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual
import scan_multi_farmer_path_oracle_v1 as scanner


SCHEMA = "contract-retrieval-oracle-ablation-v1"
ARMS = ("R0_active8", "R1_contract8", "R2_all")
STATE_INDICES = np.asarray([
    *range(continuation.MARKET_START),
    *range(continuation.MARKET_STOP, continuation.PLAN_START),
], np.int64)
PLAN_INDICES = np.arange(
    continuation.PLAN_START, continuation.FEATURE_DIM, dtype=np.int64,
)
DEFAULT_LIBRARY = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\adaptive_tail_oracle_v0"
    r"\block_library_strict_unit_path_v2_prototypes_20260829"
)
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_retrieval_oracle_ablation_v1_20260829"
)


@dataclass(frozen=True)
class StrictLibrary:
    root: Path
    manifest: Mapping[str, Any]
    blocks: Mapping[str, path_v1.DonorBlock]
    by_anchor: Mapping[int, tuple[path_v1.DonorBlock, ...]]
    active_by_anchor: Mapping[int, tuple[path_v1.DonorBlock, ...]]
    prototypes: Mapping[str, np.ndarray]
    prototype_indices: Mapping[int, np.ndarray]
    scales: Mapping[int, np.ndarray]
    inputs: Mapping[str, Any]


def retrieval_and_decision_steps(anchor: int) -> tuple[int, int]:
    """Keep the contract boundary distinct from the post-HIRE unit decision."""

    if anchor not in path_v1.ANCHORS:
        raise ValueError("retrieval anchor is outside the 21 frozen boundaries")
    return int(anchor), int(anchor + 1)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    scale = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.std(values, axis=0))
    return np.where(scale > 1e-6, scale, 1.0)


def contract_distances(
    query: np.ndarray,
    query_layout: np.ndarray,
    query_mask: np.ndarray,
    contracts: np.ndarray,
    layouts: np.ndarray,
    masks: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Continuation-v1 L1 distance with market environment 86:104 removed."""

    query = np.asarray(query, np.float32)
    contracts = np.asarray(contracts, np.float32)
    scale = np.asarray(scale, np.float64)
    if query.shape != (continuation.FEATURE_DIM,):
        raise ValueError("query contract must have 147 features")
    if contracts.ndim != 2 or contracts.shape[1] != continuation.FEATURE_DIM:
        raise ValueError("prototype contracts must have shape [N,147]")
    state = np.mean(
        np.abs(contracts[:, STATE_INDICES] - query[STATE_INDICES])
        / scale[STATE_INDICES], axis=1,
    )
    plan = np.mean(
        np.abs(contracts[:, PLAN_INDICES] - query[PLAN_INDICES])
        / scale[PLAN_INDICES], axis=1,
    )
    spatial = (
        np.sum(np.asarray(layouts) != np.asarray(query_layout), axis=1)
        + np.sum(np.asarray(masks) != np.asarray(query_mask), axis=1)
    ) / float(continuation.LAYOUT_SIZE + len(continuation.QUADRANTS))
    result = (state + plan + spatial) / 3.0
    if result.shape != (len(contracts),) or not np.isfinite(result).all():
        raise ValueError("invalid prototype contract distances")
    return result


def nearest_distinct_blocks(
    block_ids: Sequence[str], distances: Sequence[float], k: int,
) -> tuple[list[str], int, dict[str, float]]:
    """Scan nearest prototypes until k distinct block IDs have been observed."""

    if k < 1 or len(block_ids) != len(distances):
        raise ValueError("invalid nearest-distinct request")
    order = sorted(
        range(len(block_ids)),
        key=lambda index: (float(distances[index]), str(block_ids[index]), index),
    )
    selected: list[str] = []
    best: dict[str, float] = {}
    inspected = 0
    target = min(k, len(set(map(str, block_ids))))
    for index in order:
        inspected += 1
        block_id = str(block_ids[index])
        best.setdefault(block_id, float(distances[index]))
        if block_id not in selected:
            selected.append(block_id)
            if len(selected) == target:
                break
    return selected, inspected, best


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value = {
        "path": str(path.resolve()), "sha256": residual._sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {name: raw[name].copy() for name in raw.files}


def load_strict_library(root: Path) -> StrictLibrary:
    root = root.resolve()
    manifest_path = root / "block_library_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "adaptive-tail-block-library-v0"
        or manifest.get("deduplication_mode") != "unit_only"
        or tuple(map(int, manifest.get("anchors", ()))) != path_v1.ANCHORS
    ):
        raise ValueError("not a 21-anchor strict unit-only block library")
    lineage = dict(manifest.get("lineage_filter") or {})
    audit = dict(lineage.get("member_provenance_audit") or {})
    if lineage.get("mode") == "disabled" or not audit.get("passed"):
        raise ValueError("strict lineage filtering was not audited")

    actions_path = root / str(manifest["actions_file"])
    vectors_path = root / str(manifest["vectors_file"])
    prototypes_path = root / str(manifest["contract_prototypes_file"])
    for path, expected in (
        (actions_path, manifest["actions_sha256"]),
        (vectors_path, manifest["vectors_sha256"]),
        (prototypes_path, manifest["contract_prototypes_sha256"]),
    ):
        if residual._sha256_file(path) != str(expected):
            raise ValueError(f"strict library digest mismatch: {path.name}")

    actions = continuation.load_action_tapes(actions_path)
    vectors = _load_npz(vectors_path)
    prototypes = _load_npz(prototypes_path)
    block_rows = {str(row["block_id"]): row for row in manifest["blocks"]}
    block_ids = list(map(str, vectors["block_ids"]))
    if (
        len(block_ids) != int(manifest["block_count"])
        or set(block_ids) != set(block_rows) or set(block_ids) != set(actions)
        or len(prototypes["block_ids"]) != int(manifest["contract_prototype_count"])
    ):
        raise ValueError("strict block library arrays disagree with manifest")

    blocks: dict[str, path_v1.DonorBlock] = {}
    by_anchor: dict[int, list[path_v1.DonorBlock]] = defaultdict(list)
    active: dict[int, list[path_v1.DonorBlock]] = defaultdict(list)
    for index, block_id in enumerate(block_ids):
        row = block_rows[block_id]
        anchor = int(vectors["anchors"][index])
        donor = path_v1.DonorBlock(
            block_id=block_id,
            anchor=anchor,
            cluster_id=f"C{int(vectors['cluster_ids'][index]):02d}",
            source_provenance_id=str(vectors["source_provenance_ids"][index]),
            source_route_id=str(vectors["source_route_ids"][index]),
            support=int(row.get("occurrence_support", 0)),
            actions=tuple(actions[block_id]),
            team_name="",
        )
        blocks[block_id] = donor
        by_anchor[anchor].append(donor)
        if int(vectors["active"][index]):
            active[anchor].append(donor)

    prototype_indices: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    for anchor in path_v1.ANCHORS:
        by_anchor[anchor].sort(key=lambda donor: donor.block_id)
        active[anchor].sort(key=lambda donor: donor.block_id)
        indices = np.flatnonzero(prototypes["anchors"] == anchor)
        if len(active[anchor]) != path_v1.MAX_ACTIVE_DONORS or len(indices) < 8:
            raise ValueError(f"strict library has invalid coverage at {anchor}")
        prototype_indices[anchor] = indices
        scales[anchor] = _robust_scale(prototypes["entry_contracts"][indices])
    inputs = {
        "manifest": _artifact(manifest_path),
        "actions": _artifact(actions_path),
        "vectors": _artifact(vectors_path),
        "contract_prototypes": _artifact(prototypes_path),
        "block_count": len(block_ids),
        "prototype_count": len(prototypes["block_ids"]),
        "lineage_filter": lineage,
    }
    return StrictLibrary(
        root, manifest, blocks,
        {anchor: tuple(values) for anchor, values in by_anchor.items()},
        {anchor: tuple(values) for anchor, values in active.items()},
        prototypes, prototype_indices, scales, inputs,
    )


def snapshot_contract(
    observation: Mapping[str, Any], history: Any,
    plan_tape: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contract = route_switch_vector(observation, history, plan_tape).astype(
        np.float32, copy=True,
    )
    contract[continuation.PLAN_START:] = continuation.fixed_commitments(
        observation, plan_tape,
    )
    layout, mask = continuation.own_layout(observation)
    if contract.shape != (continuation.FEATURE_DIM,) or not np.isfinite(contract).all():
        raise ValueError("invalid live retrieval contract")
    return contract, layout, mask


def _canonical_candidates(
    base_tape: Sequence[Mapping[str, Any]],
    donors: Sequence[path_v1.DonorBlock],
    anchor: int,
    actor_count: int,
    decision_step: int,
) -> list[path_v1.PathCandidate]:
    base = path_v1._unit_trace(base_tape, decision_step, actor_count)
    start_offset = decision_step - anchor
    raw: list[tuple[Any, ...]] = [
        ("KEEP", "KEEP", tuple(), None, None, 0, -1, 0, base),
    ]
    for rank, donor in enumerate(donors):
        raw.extend(path_v1._donor_variants(
            base, donor, actor_count, path_v1.PLAN_HORIZON,
            start_offset, rank, True,
        ))
    canonical: dict[bytes, list[Any]] = {}
    order: list[bytes] = []
    for row in raw:
        payload = path_v1._canonical_bytes(row[-1])
        if payload not in canonical:
            canonical[payload] = [*row, [str(row[0])]]
            order.append(payload)
        else:
            canonical[payload][-1].append(str(row[0]))
    result = []
    for payload in order:
        (code, kind, assignments, donor_id, cluster, support, rank,
         novelty, units, aliases) = canonical[payload]
        result.append(path_v1.PathCandidate(
            code=str(code), kind=str(kind), units=units,
            assignments=tuple(assignments), donor_id=donor_id,
            donor_cluster=cluster, donor_support=int(support),
            donor_rank=int(rank), trace_novelty=int(novelty),
            aliases=tuple(aliases), raw_sha256=hashlib.sha256(payload).hexdigest(),
        ))
    if not result or not result[0].keep:
        raise AssertionError("canonical candidate list lost KEEP")
    return result


def _online_candidates(
    base_tape: Sequence[Mapping[str, Any]],
    donor_pool: Sequence[path_v1.DonorBlock],
    anchor: int,
    actor_count: int,
    decision_step: int,
    budget: int,
) -> tuple[list[path_v1.PathCandidate], dict[str, Any]]:
    selected, audit = _select_diverse_fast(
        base_tape, donor_pool, anchor, actor_count, budget, decision_step,
    )
    return (
        _canonical_candidates(base_tape, selected, anchor, actor_count, decision_step),
        audit,
    )


def _select_diverse_fast(
    base_tape: Sequence[Mapping[str, Any]],
    donor_pool: Sequence[path_v1.DonorBlock],
    anchor: int,
    actor_count: int,
    budget: int,
    decision_step: int,
) -> tuple[tuple[path_v1.DonorBlock, ...], dict[str, Any]]:
    """Exact v1 diversity choice without rebuilding variants per combination."""

    ordered = tuple(sorted(donor_pool, key=lambda donor: donor.block_id))
    count = min(budget, len(ordered))
    if count < 1:
        raise ValueError("empty online donor pool")
    base = path_v1._unit_trace(base_tape, decision_step, actor_count)
    start_offset = decision_step - anchor
    payloads: dict[str, frozenset[bytes]] = {}
    for donor in ordered:
        payloads[donor.block_id] = frozenset(
            path_v1._canonical_bytes(row[-1])
            for row in path_v1._donor_variants(
                base, donor, actor_count, path_v1.PLAN_HORIZON,
                start_offset, 0, True,
            )
        )

    support_top = tuple(sorted(
        ordered, key=lambda donor: (-donor.support, donor.block_id),
    )[:count])
    best: tuple[path_v1.DonorBlock, ...] | None = None
    best_score = (-1, -1)
    for values in combinations(ordered, count):
        unique = set().union(*(payloads[donor.block_id] for donor in values))
        score = (len(unique), sum(donor.support for donor in values))
        if score > best_score:
            best, best_score = values, score
    if best is None:
        raise AssertionError("diversity selection produced no combination")
    selected = tuple(sorted(best, key=lambda donor: donor.block_id))
    support_unique = set().union(*(
        payloads[donor.block_id] for donor in support_top
    ))
    return selected, {
        "anchor": int(anchor), "allowed": len(ordered), "budget": count,
        "support_top_ids": [donor.block_id for donor in support_top],
        "support_top_unique_residuals": len(support_unique),
        "diversity_ids": [donor.block_id for donor in selected],
        "diversity_unique_residuals": best_score[0],
        "diversity_support": best_score[1],
    }


def _slice_from_r2(
    candidates: Sequence[path_v1.PathCandidate],
    r2_candidates: Sequence[path_v1.PathCandidate],
    rewards: np.ndarray,
    market_diff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {candidate.raw_sha256: index for index, candidate in enumerate(r2_candidates)}
    try:
        indices = np.asarray([lookup[candidate.raw_sha256] for candidate in candidates])
    except KeyError as exc:
        raise AssertionError("online arm is not a subset of R2") from exc
    return rewards[indices], market_diff[indices]


def _arm_result(
    candidates: Sequence[path_v1.PathCandidate], rewards: np.ndarray,
    market_diff: np.ndarray, seat: int, donor_pool: int,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    best = scanner.select_path_pure_index(rewards, market_diff, seat)
    keep_rank = adaptive.reward_rank(rewards[0], seat)
    best_rank = adaptive.reward_rank(rewards[best], seat)
    selected = candidates[best]
    strict = best_rank > keep_rank
    delta = float(best_rank[1] - keep_rank[1]) if strict else 0.0
    return {
        "donor_pool_count": int(donor_pool),
        "selected_donor_count": len(selected_ids),
        "selected_donor_ids": list(selected_ids),
        "candidate_count": len(candidates),
        "path_pure_candidate_count": int(np.count_nonzero(market_diff == 0)),
        "market_diff_candidate_count": int(np.count_nonzero(market_diff)),
        "market_diff_step_count": int(np.sum(market_diff)),
        "best_code": selected.code,
        "best_kind": selected.kind,
        "best_donor_id": selected.donor_id,
        "best_rewards": list(map(float, rewards[best])),
        "best_outcome": int(best_rank[0]),
        "best_margin": float(best_rank[1]),
        "strict_headroom": bool(strict),
        "oracle_gain": delta,
        "loss_to_win": bool(keep_rank[0] == 0 and best_rank[0] == 2),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r2_positive = [row for row in rows if row["arms"]["R2_all"]["strict_headroom"]]
    r2_gain = sum(float(row["arms"]["R2_all"]["oracle_gain"]) for row in rows)
    result: dict[str, Any] = {
        "states": len(rows),
        "r2_positive_states": len(r2_positive),
        "r2_positive_rate": len(r2_positive) / len(rows) if rows else 0.0,
        "arms": {},
    }
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        nonkeep = np.asarray([
            max(0, int(value["candidate_count"]) - 1) for value in values
        ], np.int32)
        recalled = sum(bool(row["arms"][arm]["strict_headroom"]) for row in r2_positive)
        gain = sum(float(value["oracle_gain"]) for value in values)
        result["arms"][arm] = {
            "positive_states": sum(bool(value["strict_headroom"]) for value in values),
            "positive_state_recall": recalled / len(r2_positive) if r2_positive else 1.0,
            "oracle_gain_sum": gain,
            "oracle_gain_retention": gain / r2_gain if r2_gain > 0 else 1.0,
            "loss_to_win_states": sum(bool(value["loss_to_win"]) for value in values),
            "candidate_count_sum": sum(int(value["candidate_count"]) for value in values),
            "candidate_count_one_states": int(np.count_nonzero(nonkeep == 0)),
            "effective_nonkeep_mean": float(np.mean(nonkeep)) if len(nonkeep) else 0.0,
            "effective_nonkeep_median": float(np.median(nonkeep)) if len(nonkeep) else 0.0,
            "effective_nonkeep_p90": float(np.quantile(nonkeep, .9)) if len(nonkeep) else 0.0,
            "market_diff_candidates": sum(
                int(value["market_diff_candidate_count"]) for value in values
            ),
            "market_diff_steps": sum(int(value["market_diff_step_count"]) for value in values),
        }
    return result


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    opponents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    splits: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[str(row["anchor"])].append(row)
        opponents[str(row["opponent"])].append(row)
        splits[str(row["split"])].append(row)
    return {
        "overall": _group_summary(rows),
        "by_anchor": {key: _group_summary(value) for key, value in sorted(anchors.items())},
        "by_opponent": {key: _group_summary(value) for key, value in sorted(opponents.items())},
        "by_split": {key: _group_summary(value) for key, value in sorted(splits.items())},
    }


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    strict: StrictLibrary,
    opponent: str,
    seed: int,
    seat: int,
    split: str,
    anchors: Sequence[int],
    online_pool: int,
    donor_budget: int,
    parity_check: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(tapes[baseline_id], tapes, genome)
    anchor_set = set(map(int, anchors))
    decision_to_anchor = {retrieval_and_decision_steps(anchor)[1]: anchor for anchor in anchors}
    pending: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    physical_rollouts = reference_rollouts = 0
    native_seconds = 0.0
    parity_rows: list[dict[str, Any]] = []
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        route_id = str(genome[segment])
        if step in anchor_set:
            contract, layout, mask = snapshot_contract(observation, history, plan_tape)
            indices = strict.prototype_indices[step]
            distances = contract_distances(
                contract, layout, mask,
                strict.prototypes["entry_contracts"][indices],
                strict.prototypes["layouts"][indices],
                strict.prototypes["unlocked_masks"][indices],
                strict.scales[step],
            )
            ids, inspected, best = nearest_distinct_blocks(
                strict.prototypes["block_ids"][indices], distances, online_pool,
            )
            pending[step] = {
                "retrieval_step": step,
                "decision_step": step + 1,
                "query_sha256": path_v1._sha256({
                    "contract": contract.tolist(), "layout": layout.tolist(),
                    "mask": mask.tolist(),
                }),
                "prototype_count": len(indices),
                "raw_prototypes_inspected": inspected,
                "unique_block_ids": ids,
                "unique_block_distances": {block_id: best[block_id] for block_id in ids},
            }
        if step in decision_to_anchor:
            anchor = decision_to_anchor[step]
            retrieval = pending.pop(anchor)
            actor_count = max(1, len(_positions(observation)))
            base_tape = tapes[route_id]
            r0, r0_audit = _online_candidates(
                base_tape, strict.active_by_anchor[anchor], anchor,
                actor_count, step, donor_budget,
            )
            r1_pool = tuple(strict.blocks[block_id] for block_id in retrieval["unique_block_ids"])
            r1, r1_audit = _online_candidates(
                base_tape, r1_pool, anchor, actor_count, step, donor_budget,
            )
            if parity_check and len(parity_rows) < 6:
                for name, pool, fast_audit in (
                    ("R0_active8", strict.active_by_anchor[anchor], r0_audit),
                    ("R1_contract8", r1_pool, r1_audit),
                ):
                    fast_started = time.perf_counter()
                    fast_selected, verified_fast_audit = _select_diverse_fast(
                        base_tape, pool, anchor, actor_count,
                        donor_budget, step,
                    )
                    fast_seconds = time.perf_counter() - fast_started
                    old_started = time.perf_counter()
                    old_selected, old_audit = path_v1.select_diverse_donors(
                        base_tape, pool, anchor, actor_count,
                        top_k=donor_budget, decision_step=step,
                    )
                    old_seconds = time.perf_counter() - old_started
                    keys = (
                        "support_top_ids", "support_top_unique_residuals",
                        "diversity_ids", "diversity_unique_residuals",
                        "diversity_support",
                    )
                    if (
                        [donor.block_id for donor in fast_selected]
                        != [donor.block_id for donor in old_selected]
                        or any(verified_fast_audit[key] != old_audit[key] for key in keys)
                        or any(fast_audit[key] != old_audit[key] for key in keys)
                    ):
                        raise AssertionError("fast diversity selection changed v1 semantics")
                    parity_rows.append({
                        "arm": name, "anchor": int(anchor),
                        "actor_count": int(actor_count),
                        "selected_ids": [donor.block_id for donor in fast_selected],
                        "audit": {key: old_audit[key] for key in keys},
                        "fast_seconds": fast_seconds,
                        "old_seconds": old_seconds,
                    })
            r2 = _canonical_candidates(
                base_tape, strict.by_anchor[anchor], anchor, actor_count, step,
            )
            packed, counts = path_v1.pack_unit_plans(r2)
            started = time.perf_counter()
            raw = bundle.executor.rollout_schedule_unit_override_sequence_batch(
                env, states[0], states[1], schedule[segment:],
                path_v1.STOPS[segment:], seat, packed, counts,
            )
            native_elapsed = time.perf_counter() - started
            native_seconds += native_elapsed
            rewards, market_diff = raw if isinstance(raw, tuple) else (
                raw, np.zeros(len(r2), np.int32),
            )
            rewards = np.asarray(rewards, np.float64)
            market_diff = np.asarray(market_diff, np.int32)
            physical_rollouts += len(r2)
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                path_v1.STOPS[segment:],
            ), np.float64)[0]
            reference_rollouts += 1
            if not np.array_equal(rewards[0], reference):
                raise RuntimeError("R2 KEEP does not reproduce frozen NR020")
            arm_candidates = {
                "R0_active8": r0, "R1_contract8": r1, "R2_all": r2,
            }
            selected = {
                "R0_active8": r0_audit["diversity_ids"],
                "R1_contract8": r1_audit["diversity_ids"],
                "R2_all": [donor.block_id for donor in strict.by_anchor[anchor]],
            }
            pool_counts = {
                "R0_active8": len(strict.active_by_anchor[anchor]),
                "R1_contract8": len(r1_pool),
                "R2_all": len(strict.by_anchor[anchor]),
            }
            arms = {}
            for arm, candidates in arm_candidates.items():
                arm_rewards, arm_market = _slice_from_r2(
                    candidates, r2, rewards, market_diff,
                )
                arms[arm] = _arm_result(
                    candidates, arm_rewards, arm_market, seat,
                    pool_counts[arm], selected[arm],
                )
            if (
                arms["R0_active8"]["donor_pool_count"] != online_pool
                or arms["R1_contract8"]["donor_pool_count"] != online_pool
                or arms["R0_active8"]["selected_donor_count"] != donor_budget
                or arms["R1_contract8"]["selected_donor_count"] != donor_budget
            ):
                raise AssertionError("R0/R1 online donor budgets diverged")
            keep_rank = adaptive.reward_rank(reference, seat)
            rows.append({
                "schema": SCHEMA,
                "split": split, "opponent": opponent,
                "seed": int(seed), "seat": int(seat),
                "genome_id": "NR020", "route_id": route_id,
                "anchor": int(anchor),
                "retrieval_step": int(retrieval["retrieval_step"]),
                "decision_step": int(step),
                "actor_count": int(actor_count),
                "query_sha256": retrieval["query_sha256"],
                "prototype_count": int(retrieval["prototype_count"]),
                "r1_raw_prototypes_inspected": int(retrieval["raw_prototypes_inspected"]),
                "r1_unique_block_ids": list(retrieval["unique_block_ids"]),
                "r1_unique_block_distances": retrieval["unique_block_distances"],
                "keep_rewards": list(map(float, reference)),
                "keep_outcome": int(keep_rank[0]),
                "keep_margin": float(keep_rank[1]),
                "arms": arms,
                "candidate_rollout_seconds": float(native_elapsed),
                "committed": "KEEP",
            })
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            )
            for player in (0, 1)
        ])
    if not env.done or pending:
        raise RuntimeError("frozen scenario did not complete cleanly")
    return rows, {
        "split": split, "opponent": opponent, "seed": int(seed), "seat": int(seat),
        "decisions": len(rows), "physical_candidate_rollouts": physical_rollouts,
        "reference_rollouts": reference_rollouts,
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": physical_rollouts / native_seconds,
        "diversity_parity_checks": parity_rows,
        "completed": True,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    overall = report["summary"]["overall"]
    lines = []
    for arm in ARMS:
        row = overall["arms"][arm]
        lines.append(
            f"| {arm} | {row['positive_states']}/{overall['states']} | "
            f"{100 * row['positive_state_recall']:.2f}% | "
            f"{100 * row['oracle_gain_retention']:.2f}% | "
            f"{row['loss_to_win_states']} | {row['market_diff_candidates']} |"
        )
    throughput = report["throughput"]
    return (
        "# CONTRACT-RETRIEVAL-ORACLE-ABLATION-v1\n\n"
        f"Status: `{report['status']}`\n\n"
        "Contract retrieval is computed at each anchor boundary. Unit-path oracle "
        "evaluation happens at anchor+1 after the frozen anchor action; candidates "
        "are never committed. R0 and R1 both use an 8-donor pool and the same "
        "action-diversity budget of 3.\n\n"
        "| arm | positive states | R2-positive recall | R2 gain retained | loss->win states | H4 market-diff excluded |\n"
        "|---|---:|---:|---:|---:|---:|\n" + "\n".join(lines) + "\n\n"
        f"Physical R2 rollouts: {throughput['physical_candidate_rollouts']} at "
        f"{throughput['candidate_rollouts_per_second']:.1f}/s; logical candidate "
        f"evaluations across the three nested arms: {throughput['logical_candidate_evaluations']}.\n\n"
        "## Decision\n\n" + report["decision"] + "\n\n"
        "## Evidence boundary\n\n"
        "The strict replay library excludes the three named replay team lineages "
        "before deduplication and clustering. The proxy-opponent split is still not "
        "a strict baseline OOD test: NR020 and the fixed-route proxy pool were not "
        "created under a sealed opponent-generation protocol. These are native "
        "fixed-route proxies, not submitted agents. Market equality is diagnosed "
        "only over the local H4 unit override; locally market-different candidates "
        "are excluded, and no candidate is committed.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    opponents = tuple(args.opponent or (*path_v1.TRAIN_OPPONENTS, *path_v1.HELDOUT_OPPONENTS))
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    if len(set(opponents)) != len(opponents) or set(seeds) & residual.SEALED_SEEDS:
        raise ValueError("opponents must be unique and seeds must remain unsealed")
    if not args.smoke and (
        len(opponents) != 12 or len(seeds) != 2 or args.max_windows != len(path_v1.ANCHORS)
    ):
        raise ValueError("formal ablation requires 12 opponents x 2 seeds x 21 anchors")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    load_args = argparse.Namespace(
        frozen_run=args.frozen_run, prepared_root=args.prepared_root,
        v1_root=args.v1_root, block_library=args.block_library,
        heldout_team=None, base_genome_id="NR020", composition_genome_id=None,
        train_opponent=list(opponents), validation_opponent=[],
    )
    (bundle, baseline_id, _, tapes, genomes, _, active_donors,
     experiment_inputs, source) = path_v1.load_experiment(load_args)
    strict = load_strict_library(args.block_library)
    genome = genomes["NR020"]
    for anchor in path_v1.ANCHORS:
        if {donor.block_id for donor in active_donors[anchor]} != {
            donor.block_id for donor in strict.active_by_anchor[anchor]
        }:
            raise ValueError(f"R0 active8 mismatch at {anchor}")

    split_by_opponent = {
        opponent: (
            "train_proxy" if opponent in path_v1.TRAIN_OPPONENTS else "heldout_proxy"
        ) for opponent in opponents
    }
    anchors = path_v1.ANCHORS[:args.max_windows]
    scenarios = [
        (opponent, seed, seat) for opponent in opponents for seed in seeds
        for seat in (0, 1)
    ]
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, (opponent, seed, seat) in enumerate(scenarios, 1):
        decisions, trajectory = _scenario(
            bundle, baseline_id, tapes, genome, strict, opponent, seed, seat,
            split_by_opponent[opponent], anchors,
            args.online_pool, args.donor_budget,
            parity_check=index == 1,
        )
        rows.extend(decisions)
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "contract_ablation_progress", "index": index,
            "total": len(scenarios), "opponent": opponent,
            "seed": seed, "seat": seat,
            "states": len(rows),
        }, sort_keys=True), flush=True)

    summary = summarize(rows)
    overall = summary["overall"]
    r0 = overall["arms"]["R0_active8"]
    r1 = overall["arms"]["R1_contract8"]
    if overall["r2_positive_states"] == 0:
        status = "r2_no_path_headroom"
        decision = "The full strict block vocabulary has no positive oracle state; do not train a selector on this H4 formulation."
    elif (
        r1["positive_state_recall"] > r0["positive_state_recall"]
        or r1["oracle_gain_retention"] > r0["oracle_gain_retention"] + 1e-12
    ):
        status = "contract_retrieval_improves_oracle_coverage"
        decision = "Keep two-stage contract retrieval: it recovers more of the full-library oracle signal at the same 8->3 donor budget."
    else:
        status = "contract_retrieval_not_better_than_active8"
        decision = "Do not add contract retrieval to the selector yet; at equal donor budget it does not improve positive-state recall or oracle gain retention."

    physical = sum(int(row["physical_candidate_rollouts"]) for row in trajectories)
    native_seconds = sum(float(row["native_candidate_seconds"]) for row in trajectories)
    logical = sum(
        int(row["arms"][arm]["candidate_count"]) for row in rows for arm in ARMS
    )
    throughput = {
        "physical_candidate_rollouts": physical,
        "reference_rollouts": sum(int(row["reference_rollouts"]) for row in trajectories),
        "logical_candidate_evaluations": logical,
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": physical / native_seconds,
        "nested_arm_reuse": "R0/R1 canonical traces sliced from the single R2 batch",
    }
    inspection = [int(row["r1_raw_prototypes_inspected"]) for row in rows]
    parity = [
        check for trajectory in trajectories
        for check in trajectory["diversity_parity_checks"]
    ]
    retrieval = {
        "retrieval_step": "anchor",
        "decision_step": "anchor+1",
        "distance": "mean of robust-scaled L1 state-without-86:104, robust-scaled L1 fixed capital plan, and layout+mask Hamming",
        "prototype_universe": int(strict.inputs["prototype_count"]),
        "nearest_distinct_unique_blocks": args.online_pool,
        "raw_prototypes_inspected_mean": float(np.mean(inspection)),
        "raw_prototypes_inspected_median": float(np.median(inspection)),
        "raw_prototypes_inspected_p90": float(np.quantile(inspection, .9)),
        "raw_prototypes_inspected_min": min(inspection),
        "raw_prototypes_inspected_max": max(inspection),
        "online_donor_budget": args.donor_budget,
        "r0_r1_budget_equal": all(
            row["arms"]["R0_active8"]["donor_pool_count"] == args.online_pool
            and row["arms"]["R1_contract8"]["donor_pool_count"] == args.online_pool
            and row["arms"]["R0_active8"]["selected_donor_count"] == args.donor_budget
            and row["arms"]["R1_contract8"]["selected_donor_count"] == args.donor_budget
            for row in rows
        ),
    }
    optimization = {
        "method": "cache each donor's canonical variant SHA set, then score combinations by set union",
        "real_state_parity_checks": len(parity),
        "selected_ids_and_audit_exact": len(parity) >= 6,
        "old_selection_seconds": sum(float(row["old_seconds"]) for row in parity),
        "fast_selection_seconds": sum(float(row["fast_seconds"]) for row in parity),
    }
    optimization["measured_speedup"] = (
        optimization["old_selection_seconds"]
        / optimization["fast_selection_seconds"]
    )

    decisions_path = output / "oracle_decisions.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    summary_path = output / "summary.json"
    residual._atomic_text(decisions_path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))
    residual._atomic_text(trajectories_path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in trajectories
    ))
    residual._atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": SCHEMA, "status": status, "decision": decision,
        "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020", "route_ids": list(genome),
            "sha256": path_v1._sha256(list(genome)),
        },
        "scenario_contract": {
            "opponents": list(opponents), "seeds": list(seeds), "seats": [0, 1],
            "anchors": list(map(int, anchors)),
            "states_expected": len(opponents) * len(seeds) * 2 * len(anchors),
            "states_observed": len(rows), "candidate_committed": False,
        },
        "retrieval": retrieval, "summary": summary, "throughput": throughput,
        "selection_optimization": optimization,
        "implementation": {
            "runner": _artifact(Path(__file__).resolve()),
            "path_candidate_runtime": _artifact(Path(path_v1.__file__).resolve()),
            "native_extension": _artifact(
                Path(path_v1.native_extension.__file__).resolve()
            ),
        },
        "strict_library": strict.inputs, "experiment_inputs": experiment_inputs,
        "source": source,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": {
            "proxy_opponents_not_submitted_agents": True,
            "proxy_split_not_strict_baseline_ood": True,
            "strict_replay_lineage_filter_passed": True,
            "market_diagnostic_scope": "local H4 only",
            "sealed_test_executed": False,
        },
        "artifacts": {
            decisions_path.name: _artifact(decisions_path, len(rows)),
            trajectories_path.name: _artifact(trajectories_path, len(trajectories)),
            summary_path.name: _artifact(summary_path),
        },
    }
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "contract_ablation_complete", "status": status,
        "output": str(output), "overall": overall,
        "throughput": throughput,
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--block-library", type=Path, default=DEFAULT_LIBRARY)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--opponent", action="append", default=None)
    result.add_argument("--seed-start", type=int, default=2026086340)
    result.add_argument("--seed-count", type=int, default=2)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--online-pool", type=int, default=8)
    result.add_argument("--donor-budget", type=int, default=3)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        args.opponent = [(args.opponent or list(path_v1.TRAIN_OPPONENTS))[0]]
        args.seed_count = 1
        args.max_windows = 1
    if (
        args.seed_count < 1 or not 1 <= args.max_windows <= len(path_v1.ANCHORS)
        or args.online_pool != path_v1.MAX_ACTIVE_DONORS
        or args.donor_budget != path_v1.MAX_PAIRS
    ):
        raise ValueError("v1 fixes online pool=8, donor budget=3, and valid positive budgets")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
