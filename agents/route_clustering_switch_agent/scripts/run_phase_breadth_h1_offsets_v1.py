#!/usr/bin/env python3
"""Evaluate frozen phase breadth candidate-set oracles at H1 offsets 1..4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from meta_agent.src.route_compiler import _positions
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_contract_breadth_sweep_v2 as breadth
import run_contract_retrieval_oracle_ablation_v1 as retrieval_v1
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual


SCHEMA = "phase-breadth-h1-offsets-v1"
ARMS = ("R0_active8", "phase_frozen", "R2_all")
VIEWS = ("all_candidates_oracle", "path_pure_oracle")
OFFSETS = (1, 2, 3, 4)
EXPECTED_LOPO_SHA256 = "7cf8ba402c2d4f6df395309f254eb388a4df3719a857dc49284b4b9dd2b92829"
EXPECTED_BREADTH_REPORT_SHA256 = "3499ee38c0899fc5763a38a235452eed3bf922774970a264d927a4dbf7d8d054"
EXPECTED_BREADTH_ROWS_SHA256 = "170e98855694a3d6efa782596a3d04b1235fad7d76bdb035c196bbd0e419633b"
DEFAULT_BREADTH_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_breadth_sweep_v2_formal_20260829a"
)
DEFAULT_LOPO = DEFAULT_BREADTH_ROOT / "PHASE_BREADTH_LOPO_STABILITY.json"
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\phase_breadth_h1_offsets_v1_20260829"
)


def decision_points(
    anchors: Sequence[int],
) -> dict[int, tuple[int, int]]:
    result = {
        int(anchor + offset): (int(anchor), int(offset))
        for anchor in anchors for offset in OFFSETS
        if anchor + offset < path_v1.HORIZON
    }
    if len(result) != len(anchors) * len(OFFSETS):
        raise ValueError("anchor offsets overlap or exceed the frozen horizon")
    return result


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(path_v1._canonical_bytes(value)).hexdigest()


def h1_payload_masks(
    cache: breadth.VariantCache,
) -> dict[str, int]:
    payloads = {
        block_id: {
            path_v1._canonical_bytes(row[-1][:1])
            for row in cache.variants[block_id]
        }
        for block_id in cache.donors
    }
    universe = {
        payload: index
        for index, payload in enumerate(sorted({
            payload for values in payloads.values() for payload in values
        }))
    }
    return {
        block_id: sum(1 << universe[payload] for payload in values)
        for block_id, values in payloads.items()
    }


def build_h1_diverse_arm(
    cache: breadth.VariantCache,
    pool_ids: Iterable[str],
    payload_masks: Mapping[str, int],
) -> tuple[list[path_v1.PathCandidate], dict[str, Any]]:
    pool = tuple(sorted(set(pool_ids)))
    selected, audit = breadth.select_diverse_ids(
        pool, payload_masks, cache.support,
    )
    h4_candidates = breadth.candidates_from_cache(cache, selected)
    h1_candidates = path_v1.effective_candidates(h4_candidates, 1)
    pool_union = 0
    for block_id in pool:
        pool_union |= int(payload_masks[block_id])
    return h1_candidates, {
        **audit,
        "generator_kind": "horizon_aligned_h1_action_diversity",
        "pool_unique_h1_actions": pool_union.bit_count(),
        "selected_unique_h1_actions": audit["diversity_unique_residuals"],
        "pre_h1_candidate_count": len(h4_candidates),
        "h1_candidate_count": len(h1_candidates),
        "h1_collapse_count": len(h4_candidates) - len(h1_candidates),
        "candidate_sha256": [
            candidate.raw_sha256 for candidate in h1_candidates
        ],
    }


def legacy_h4_then_h1_diagnostic(
    cache: breadth.VariantCache,
    pool_ids: Iterable[str],
) -> dict[str, Any]:
    selected, audit = breadth.select_diverse_ids(
        pool_ids, cache.payload_masks, cache.support,
    )
    h4_candidates = breadth.candidates_from_cache(cache, selected)
    h1_candidates = path_v1.effective_candidates(h4_candidates, 1)
    return {
        "generator_kind": "legacy_h4_diversity_then_h1_truncation_diagnostic_only",
        "selected_donor_ids": list(selected),
        "h4_unique_variant_coverage": audit["diversity_unique_residuals"],
        "pre_h1_candidate_count": len(h4_candidates),
        "h1_candidate_count": len(h1_candidates),
        "h1_collapse_count": len(h4_candidates) - len(h1_candidates),
        "candidate_sha256": [
            candidate.raw_sha256 for candidate in h1_candidates
        ],
    }


def build_h1_r2(
    cache: breadth.VariantCache,
    payload_masks: Mapping[str, int],
) -> tuple[list[path_v1.PathCandidate], dict[str, Any]]:
    block_ids = tuple(sorted(cache.donors))
    h4_candidates = breadth.candidates_from_cache(cache, block_ids)
    h1_candidates = path_v1.effective_candidates(h4_candidates, 1)
    return h1_candidates, {
        "generator_kind": "full_strict_h1_action_oracle",
        "allowed": len(block_ids),
        "budget": len(block_ids),
        "diversity_ids": list(block_ids),
        "pre_h1_candidate_count": len(h4_candidates),
        "h1_candidate_count": len(h1_candidates),
        "h1_collapse_count": len(h4_candidates) - len(h1_candidates),
        "pool_unique_h1_actions": _mask_union_count(
            block_ids, payload_masks,
        ),
    }


def _mask_union_count(
    block_ids: Iterable[str],
    payload_masks: Mapping[str, int],
) -> int:
    union = 0
    for block_id in block_ids:
        union |= int(payload_masks[block_id])
    return union.bit_count()


def slice_indices(
    candidates: Sequence[path_v1.PathCandidate],
    r2_candidates: Sequence[path_v1.PathCandidate],
) -> np.ndarray:
    lookup = {
        candidate.raw_sha256: index
        for index, candidate in enumerate(r2_candidates)
    }
    if len(lookup) != len(r2_candidates):
        raise AssertionError("R2 H1 canonical action SHA is not unique")
    try:
        return np.asarray([
            lookup[candidate.raw_sha256] for candidate in candidates
        ], np.int32)
    except KeyError as exc:
        raise AssertionError("online H1 arm is not a canonical action subset of R2") from exc


def _best_index(
    rewards: np.ndarray,
    seat: int,
    indices: Iterable[int],
) -> int:
    values = tuple(map(int, indices))
    if not values:
        raise ValueError("oracle view has no eligible candidates")
    best = values[0]
    best_rank = adaptive.reward_rank(rewards[best], seat)
    for index in values[1:]:
        rank = adaptive.reward_rank(rewards[index], seat)
        if rank > best_rank:
            best, best_rank = index, rank
    return best


def oracle_view(
    candidates: Sequence[path_v1.PathCandidate],
    rewards: np.ndarray,
    market_diff: np.ndarray,
    seat: int,
    *,
    path_pure: bool,
) -> dict[str, Any]:
    if (
        rewards.shape != (len(candidates), 2)
        or market_diff.shape != (len(candidates),)
        or not candidates
        or not candidates[0].keep
        or int(market_diff[0]) != 0
    ):
        raise ValueError("invalid aligned H1 candidate batch")
    eligible = (
        np.flatnonzero(market_diff == 0)
        if path_pure else np.arange(len(candidates), dtype=np.int32)
    )
    best = _best_index(rewards, seat, eligible)
    keep_rank = adaptive.reward_rank(rewards[0], seat)
    best_rank = adaptive.reward_rank(rewards[best], seat)
    strict = best_rank > keep_rank
    return {
        "view": "path_pure_oracle" if path_pure else "all_candidates_oracle",
        "eligible_candidate_count": len(eligible),
        "best_index": int(best),
        "best_sha256": candidates[best].raw_sha256,
        "best_code": candidates[best].code,
        "best_kind": candidates[best].kind,
        "best_donor_id": candidates[best].donor_id,
        "best_market_diff": int(market_diff[best]),
        "best_rewards": list(map(float, rewards[best])),
        "best_outcome": int(best_rank[0]),
        "best_margin": float(best_rank[1]),
        "strict_headroom": bool(strict),
        "oracle_gain": (
            float(best_rank[1] - keep_rank[1]) if strict else 0.0
        ),
        "loss_to_win": bool(
            keep_rank[0] == 0 and best_rank[0] == 2
        ),
    }


def arm_result(
    candidates: Sequence[path_v1.PathCandidate],
    rewards: np.ndarray,
    market_diff: np.ndarray,
    seat: int,
    audit: Mapping[str, Any],
    r2_indices: Sequence[int],
) -> dict[str, Any]:
    path_pure_mask = (market_diff == 0)
    return {
        "evaluation_kind": "candidate_set_oracle",
        "selector_executed": False,
        "donor_pool_count": int(audit["allowed"]),
        "selected_donor_count": len(audit["diversity_ids"]),
        "selected_donor_ids": list(audit["diversity_ids"]),
        "pre_h1_candidate_count": int(audit["pre_h1_candidate_count"]),
        "candidate_count": len(candidates),
        "h1_collapse_count": int(audit["h1_collapse_count"]),
        "candidate_sha256": [candidate.raw_sha256 for candidate in candidates],
        "r2_indices": list(map(int, r2_indices)),
        "market_diff": list(map(int, market_diff)),
        "path_pure_mask": list(map(bool, path_pure_mask)),
        "path_pure_candidate_count": int(np.count_nonzero(path_pure_mask)),
        "market_diff_candidate_count": int(np.count_nonzero(~path_pure_mask)),
        "market_diff_step_count": int(np.sum(market_diff)),
        "all_candidates_oracle": oracle_view(
            candidates, rewards, market_diff, seat, path_pure=False,
        ),
        "path_pure_oracle": oracle_view(
            candidates, rewards, market_diff, seat, path_pure=True,
        ),
    }


def _phase_pool_ids(
    phase_arm: str,
    anchor: int,
    retrieval: Mapping[str, Any],
    strict: retrieval_v1.StrictLibrary,
) -> tuple[str, ...]:
    if phase_arm == "R0_active8":
        return tuple(
            donor.block_id for donor in strict.active_by_anchor[anchor]
        )
    if not phase_arm.startswith("K") or not phase_arm.endswith("_contract"):
        raise ValueError(f"invalid frozen phase arm: {phase_arm}")
    k = int(phase_arm[1:].split("_", 1)[0])
    if k not in breadth.KS:
        raise ValueError(f"unsupported frozen breadth: {phase_arm}")
    return tuple(retrieval["prefixes"][k])


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    strict: retrieval_v1.StrictLibrary,
    phase_mapping: Mapping[int, str],
    opponent: str,
    seed: int,
    seat: int,
    split: str,
    anchors: Sequence[int],
    parity_check: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(tapes[baseline_id], tapes, genome)
    anchor_set = set(map(int, anchors))
    points = decision_points(anchors)
    pending: dict[int, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    native_seconds = 0.0
    physical_rollouts = reference_rollouts = physical_batches = 0
    parity_rows: list[dict[str, Any]] = []
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        route_id = str(genome[segment])
        if step in anchor_set:
            contract, layout, mask = retrieval_v1.snapshot_contract(
                observation, history, plan_tape,
            )
            indices = strict.prototype_indices[step]
            prototype_ids = list(map(
                str, strict.prototypes["block_ids"][indices],
            ))
            distances = retrieval_v1.contract_distances(
                contract, layout, mask,
                strict.prototypes["entry_contracts"][indices],
                strict.prototypes["layouts"][indices],
                strict.prototypes["unlocked_masks"][indices],
                strict.scales[step],
            )
            prefixes, inspected = breadth.nearest_distinct_prefixes(
                prototype_ids, distances, breadth.KS,
            )
            pending[step] = {
                "query_sha256": _sha256_payload({
                    "contract": contract.tolist(),
                    "layout": layout.tolist(),
                    "mask": mask.tolist(),
                }),
                "prototype_count": len(indices),
                "prefixes": prefixes,
                "raw_prototypes_inspected": inspected,
            }
        if step in points:
            anchor, offset = points[step]
            retrieval = pending[anchor]
            actor_count = max(1, len(_positions(observation)))
            base_tape = tapes[route_id]
            cache = breadth.build_variant_cache(
                base_tape, strict.by_anchor[anchor],
                anchor, actor_count, step,
            )
            masks = h1_payload_masks(cache)
            r0_pool = tuple(
                donor.block_id for donor in strict.active_by_anchor[anchor]
            )
            phase_arm = phase_mapping[anchor]
            phase_pool = _phase_pool_ids(
                phase_arm, anchor, retrieval, strict,
            )
            r0_candidates, r0_audit = build_h1_diverse_arm(
                cache, r0_pool, masks,
            )
            if phase_arm == "R0_active8":
                phase_candidates = list(r0_candidates)
                phase_audit = dict(r0_audit)
            else:
                phase_candidates, phase_audit = build_h1_diverse_arm(
                    cache, phase_pool, masks,
                )
            r2_candidates, r2_audit = build_h1_r2(cache, masks)
            legacy = {
                "R0_active8": legacy_h4_then_h1_diagnostic(
                    cache, r0_pool,
                ),
                "phase_frozen": legacy_h4_then_h1_diagnostic(
                    cache, phase_pool,
                ),
            }
            if parity_check and not parity_rows:
                old_r0 = path_v1.effective_candidates(
                    path_v1.donor_candidates(
                        base_tape, strict.active_by_anchor[anchor],
                        anchor, actor_count, decision_step=step,
                    ),
                    1,
                )
                old_sha = [
                    candidate.raw_sha256 for candidate in old_r0
                ]
                if old_sha != legacy["R0_active8"]["candidate_sha256"]:
                    raise AssertionError(
                        "legacy H4-to-H1 diagnostic changed existing semantics"
                    )
                parity_rows.append({
                    "anchor": int(anchor),
                    "offset": int(offset),
                    "candidate_sha256": old_sha,
                    "passed": True,
                })
            packed, counts = path_v1.pack_unit_plans(r2_candidates)
            native_started = time.perf_counter()
            raw = bundle.executor.rollout_schedule_unit_override_sequence_batch(
                env, states[0], states[1], schedule[segment:],
                path_v1.STOPS[segment:], seat, packed, counts,
            )
            native_elapsed = time.perf_counter() - native_started
            native_seconds += native_elapsed
            physical_batches += 1
            rewards, market_diff = raw if isinstance(raw, tuple) else (
                raw, np.zeros(len(r2_candidates), np.int32),
            )
            rewards = np.asarray(rewards, np.float64)
            market_diff = np.asarray(market_diff, np.int32)
            if (
                rewards.shape != (len(r2_candidates), 2)
                or market_diff.shape != (len(r2_candidates),)
                or not np.isfinite(rewards).all()
                or np.any(market_diff < 0)
            ):
                raise RuntimeError("native H1 batch returned invalid aligned results")
            physical_rollouts += len(r2_candidates)
            reference = np.asarray(
                bundle.executor.rollout_schedule_batch(
                    env, states[0], states[1],
                    schedule[segment:][None, :, :],
                    path_v1.STOPS[segment:],
                ),
                np.float64,
            )[0]
            reference_rollouts += 1
            if (
                not np.array_equal(rewards[0], reference)
                or int(market_diff[0]) != 0
            ):
                raise RuntimeError("H1 R2 KEEP does not reproduce frozen NR020")

            arm_candidates = {
                "R0_active8": r0_candidates,
                "phase_frozen": phase_candidates,
                "R2_all": r2_candidates,
            }
            arm_audits = {
                "R0_active8": r0_audit,
                "phase_frozen": phase_audit,
                "R2_all": r2_audit,
            }
            arms: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                candidates = arm_candidates[arm]
                indices_in_r2 = (
                    np.arange(len(r2_candidates), dtype=np.int32)
                    if arm == "R2_all"
                    else slice_indices(candidates, r2_candidates)
                )
                arms[arm] = arm_result(
                    candidates,
                    rewards[indices_in_r2],
                    market_diff[indices_in_r2],
                    seat,
                    arm_audits[arm],
                    indices_in_r2,
                )

            state_id = (
                f"{opponent}:{seed}:{seat}:{anchor}:{offset}"
            )
            keep_rank = adaptive.reward_rank(reference, seat)
            for index, candidate in enumerate(r2_candidates):
                rank = adaptive.reward_rank(rewards[index], seat)
                candidate_rows.append({
                    "schema": SCHEMA,
                    "state_id": state_id,
                    "split": split,
                    "opponent": opponent,
                    "seed": int(seed),
                    "seat": int(seat),
                    "anchor": int(anchor),
                    "offset": int(offset),
                    "step": int(step),
                    "candidate_index": int(index),
                    "candidate_sha256": candidate.raw_sha256,
                    "code": candidate.code,
                    "kind": candidate.kind,
                    "donor_id": candidate.donor_id,
                    "aliases": list(candidate.aliases),
                    "assignments": [list(value) for value in candidate.assignments],
                    "unit_actions": [
                        list(action) for action in candidate.units[0]
                    ],
                    "terminal_rewards": list(map(float, rewards[index])),
                    "outcome": int(rank[0]),
                    "margin": float(rank[1]),
                    "strict_vs_keep": bool(rank > keep_rank),
                    "market_diff": int(market_diff[index]),
                    "path_pure": bool(market_diff[index] == 0),
                })
            decisions.append({
                "schema": SCHEMA,
                "state_id": state_id,
                "split": split,
                "opponent": opponent,
                "seed": int(seed),
                "seat": int(seat),
                "genome_id": "NR020",
                "route_id": route_id,
                "anchor": int(anchor),
                "offset": int(offset),
                "retrieval_step": int(anchor),
                "decision_step": int(step),
                "actor_count": int(actor_count),
                "query_sha256": retrieval["query_sha256"],
                "prototype_count": int(retrieval["prototype_count"]),
                "phase_pool_arm": phase_arm,
                "phase_pool_block_ids": list(phase_pool),
                "phase_raw_prototypes_inspected": (
                    None if phase_arm == "R0_active8"
                    else int(retrieval["raw_prototypes_inspected"][
                        int(phase_arm[1:].split("_", 1)[0])
                    ])
                ),
                "generator_contract": (
                    "phase freezes pool breadth only; each offset selects three "
                    "donors by exact canonical H1 action coverage"
                ),
                "evaluation_kind": "candidate_set_oracle",
                "selector_executed": False,
                "keep_rewards": list(map(float, reference)),
                "keep_outcome": int(keep_rank[0]),
                "keep_margin": float(keep_rank[1]),
                "arms": arms,
                "legacy_h4_then_h1_diagnostic": legacy,
                "physical_r2_batch_count": 1,
                "physical_r2_candidate_count": len(r2_candidates),
                "candidate_rollout_seconds": float(native_elapsed),
                "committed": "KEEP",
            })
            if offset == OFFSETS[-1]:
                pending.pop(anchor)
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            )
            for player in (0, 1)
        ])
    if not env.done or pending:
        raise RuntimeError("frozen H1 offset trajectory did not complete cleanly")
    return decisions, candidate_rows, {
        "split": split,
        "opponent": opponent,
        "seed": int(seed),
        "seat": int(seat),
        "decisions": len(decisions),
        "candidate_rows": len(candidate_rows),
        "physical_r2_batches": physical_batches,
        "physical_candidate_rollouts": physical_rollouts,
        "reference_rollouts": reference_rollouts,
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": (
            physical_rollouts / native_seconds if native_seconds else 0.0
        ),
        "legacy_h1_api_parity": parity_rows,
        "completed": True,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    if not len(array):
        return {
            "mean": 0.0, "median": 0.0, "p90": 0.0,
            "min": 0.0, "max": 0.0,
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _view_metrics(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    view: str,
) -> dict[str, Any]:
    r2_positive = [
        row for row in rows
        if row["arms"]["R2_all"][view]["strict_headroom"]
    ]
    values = [row["arms"][arm][view] for row in rows]
    gain = sum(float(value["oracle_gain"]) for value in values)
    r2_gain = sum(
        float(row["arms"]["R2_all"][view]["oracle_gain"])
        for row in rows
    )
    recoveries = sum(
        not row["arms"]["R0_active8"][view]["strict_headroom"]
        and row["arms"][arm][view]["strict_headroom"]
        for row in r2_positive
    )
    regressions = sum(
        row["arms"]["R0_active8"][view]["strict_headroom"]
        and not row["arms"][arm][view]["strict_headroom"]
        for row in r2_positive
    )
    return {
        "evaluation_kind": "candidate_set_oracle",
        "selector_executed": False,
        "positive_states": sum(
            bool(value["strict_headroom"]) for value in values
        ),
        "positive_state_recall": (
            sum(
                bool(row["arms"][arm][view]["strict_headroom"])
                for row in r2_positive
            ) / len(r2_positive)
            if r2_positive else 1.0
        ),
        "oracle_gain_sum": gain,
        "oracle_gain_retention": (
            gain / r2_gain if r2_gain > 0 else 1.0
        ),
        "coverage_recoveries_vs_R0": recoveries,
        "coverage_regressions_vs_R0": regressions,
        "loss_to_win_states": sum(
            bool(value["loss_to_win"]) for value in values
        ),
        "best_market_diff_nonzero_states": sum(
            int(value["best_market_diff"]) != 0 for value in values
        ),
    }


def _candidate_audit(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    values = [row["arms"][arm] for row in rows]
    result = {
        "candidate_count": _distribution([
            int(value["candidate_count"]) for value in values
        ]),
        "pre_h1_candidate_count": _distribution([
            int(value["pre_h1_candidate_count"]) for value in values
        ]),
        "h1_collapse_count_sum": sum(
            int(value["h1_collapse_count"]) for value in values
        ),
        "candidate_count_one_states": sum(
            int(value["candidate_count"]) == 1 for value in values
        ),
        "path_pure_candidate_count": _distribution([
            int(value["path_pure_candidate_count"]) for value in values
        ]),
        "path_pure_count_one_states": sum(
            int(value["path_pure_candidate_count"]) == 1
            for value in values
        ),
        "market_diff_candidate_count": sum(
            int(value["market_diff_candidate_count"]) for value in values
        ),
        "market_diff_step_count": sum(
            int(value["market_diff_step_count"]) for value in values
        ),
    }
    if arm in ("R0_active8", "phase_frozen"):
        legacy = [
            row["legacy_h4_then_h1_diagnostic"][arm]
            for row in rows
        ]
        result["legacy_h4_then_h1"] = {
            "diagnostic_only": True,
            "candidate_count": _distribution([
                int(value["h1_candidate_count"]) for value in legacy
            ]),
            "h1_collapse_count_sum": sum(
                int(value["h1_collapse_count"]) for value in legacy
            ),
        }
    return result


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "states": len(rows),
        "views": {
            view: {
                "r2_positive_states": sum(
                    bool(row["arms"]["R2_all"][view]["strict_headroom"])
                    for row in rows
                ),
                "arms": {
                    arm: _view_metrics(rows, arm, view)
                    for arm in ARMS
                },
            }
            for view in VIEWS
        },
        "candidate_audit": {
            arm: _candidate_audit(rows, arm) for arm in ARMS
        },
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "by_split": defaultdict(list),
        "by_offset": defaultdict(list),
        "by_anchor": defaultdict(list),
        "by_phase_arm": defaultdict(list),
    }
    for row in rows:
        groups["by_split"][str(row["split"])].append(row)
        groups["by_offset"][str(row["offset"])].append(row)
        groups["by_anchor"][str(row["anchor"])].append(row)
        groups["by_phase_arm"][str(row["phase_pool_arm"])].append(row)
    return {
        "overall": _group_summary(rows),
        **{
            name: {
                key: _group_summary(values)
                for key, values in sorted(group.items())
            }
            for name, group in groups.items()
        },
    }


def _atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# PHASE-BREADTH-H1-OFFSETS-v1",
        "",
        f"- Status: {report['status']}",
        f"- Decisions: {report['scenario_contract']['decisions_observed']}",
        "- Phase mapping freezes pool breadth only; donor IDs are selected anew at each offset.",
        "- Donors use exact canonical H1 action diversity with budget three.",
        "- Every metric below is a candidate-set oracle; no learned selector was executed.",
        "- All-candidate and path-pure oracle views are reported separately.",
        "",
    ]
    for view in VIEWS:
        lines.extend([
            f"## {view}",
            "",
            "| offset | arm | recall | gain retention | loss-to-win | market-best states |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for offset, group in report["summary"]["by_offset"].items():
            for arm in ARMS:
                value = group["views"][view]["arms"][arm]
                lines.append(
                    f"| {offset} | {arm} | "
                    f"{100 * value['positive_state_recall']:.2f}% | "
                    f"{100 * value['oracle_gain_retention']:.2f}% | "
                    f"{value['loss_to_win_states']} | "
                    f"{value['best_market_diff_nonzero_states']} |"
                )
        lines.append("")
    lines.extend([
        "## Evidence boundary",
        "",
        "- Frozen NR020 trajectory; every candidate was discarded.",
        "- One physical R2 H1 batch per decision; R0 and phase arms are SHA slices.",
        "- The legacy H4-diversity then H1-truncation path is diagnostic only.",
        "- Proxy-heldout is not strict baseline OOD.",
        "",
    ])
    return "\n".join(lines)


def load_frozen_phase_mapping(
    breadth_root: Path,
    lopo_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, str], dict[str, Any]]:
    breadth_root = breadth_root.resolve()
    lopo_path = lopo_path.resolve()
    report_path = breadth_root / "FINAL_REPORT.json"
    rows_path = breadth_root / "oracle_decisions.jsonl"
    if (
        residual._sha256_file(report_path) != EXPECTED_BREADTH_REPORT_SHA256
        or residual._sha256_file(rows_path) != EXPECTED_BREADTH_ROWS_SHA256
        or residual._sha256_file(lopo_path) != EXPECTED_LOPO_SHA256
    ):
        raise ValueError("frozen breadth or LOPO artifact SHA changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lopo = json.loads(lopo_path.read_text(encoding="utf-8"))
    pinned_rows = report.get("artifacts", {}).get(rows_path.name, {})
    if (
        report.get("schema") != "contract-breadth-sweep-v2"
        or str(pinned_rows.get("sha256", "")).lower()
        != EXPECTED_BREADTH_ROWS_SHA256
        or int(pinned_rows.get("rows", -1)) != 1008
        or lopo.get("schema") != "phase-breadth-lopo-stability-v1"
        or lopo.get("status") != "stable_hard_phase_router_survives"
        or lopo.get("evidence_boundary", {}).get(
            "heldout_used_for_selection_or_threshold"
        ) is not False
    ):
        raise ValueError("frozen breadth/LOPO semantic contract changed")
    mapping = {
        int(anchor): str(arm)
        for anchor, arm in lopo["frozen_stable_mapping"].items()
    }
    if (
        set(mapping) != set(path_v1.ANCHORS)
        or any(
            arm != "R0_active8"
            and arm not in {f"K{k}_contract" for k in breadth.KS}
            for arm in mapping.values()
        )
    ):
        raise ValueError("frozen phase mapping does not cover 21 valid anchors")
    inputs = {
        "breadth_formal_report": retrieval_v1._artifact(report_path),
        "breadth_oracle_decisions": retrieval_v1._artifact(rows_path, 1008),
        "phase_breadth_lopo_stability": retrieval_v1._artifact(lopo_path),
        "phase_mapping_sha256": _sha256_payload({
            str(anchor): arm for anchor, arm in sorted(mapping.items())
        }),
    }
    return report, lopo, mapping, inputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    opponents = tuple(
        args.opponent
        or (*path_v1.TRAIN_OPPONENTS, *path_v1.HELDOUT_OPPONENTS)
    )
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    if len(set(opponents)) != len(opponents) or set(seeds) & residual.SEALED_SEEDS:
        raise ValueError("opponents must be unique and seeds must remain unsealed")
    if not args.smoke and (
        len(opponents) != 12
        or len(seeds) != 2
        or args.max_windows != len(path_v1.ANCHORS)
    ):
        raise ValueError(
            "formal H1 offsets requires 12 opponents x 2 seeds x 21 anchors"
        )
    breadth_report, lopo, phase_mapping, frozen_inputs = (
        load_frozen_phase_mapping(args.breadth_root, args.lopo_report)
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    load_args = argparse.Namespace(
        frozen_run=args.frozen_run,
        prepared_root=args.prepared_root,
        v1_root=args.v1_root,
        block_library=args.block_library,
        heldout_team=None,
        base_genome_id="NR020",
        composition_genome_id=None,
        train_opponent=list(opponents),
        validation_opponent=[],
    )
    (bundle, baseline_id, _, tapes, genomes, _, active_donors,
     experiment_inputs, source) = path_v1.load_experiment(load_args)
    strict = retrieval_v1.load_strict_library(args.block_library)
    genome = genomes["NR020"]
    for anchor in path_v1.ANCHORS:
        if {donor.block_id for donor in active_donors[anchor]} != {
            donor.block_id for donor in strict.active_by_anchor[anchor]
        }:
            raise ValueError(f"R0 active8 mismatch at {anchor}")

    split_by_opponent = {
        opponent: (
            "train_proxy"
            if opponent in path_v1.TRAIN_OPPONENTS
            else "heldout_proxy"
        )
        for opponent in opponents
    }
    anchors = path_v1.ANCHORS[:args.max_windows]
    scenarios = [
        (opponent, seed, seat)
        for opponent in opponents for seed in seeds for seat in (0, 1)
    ]
    decisions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, (opponent, seed, seat) in enumerate(scenarios, 1):
        scenario_decisions, scenario_candidates, trajectory = _scenario(
            bundle, baseline_id, tapes, genome, strict,
            phase_mapping, opponent, seed, seat,
            split_by_opponent[opponent], anchors,
            parity_check=index == 1,
        )
        decisions.extend(scenario_decisions)
        candidate_rows.extend(scenario_candidates)
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "phase_breadth_h1_progress",
            "index": index,
            "total": len(scenarios),
            "opponent": opponent,
            "seed": seed,
            "seat": seat,
            "decisions": len(decisions),
            "candidate_rows": len(candidate_rows),
        }, sort_keys=True), flush=True)

    expected = len(opponents) * len(seeds) * 2 * len(anchors) * len(OFFSETS)
    physical_batches = sum(
        int(row["physical_r2_batches"]) for row in trajectories
    )
    physical_rollouts = sum(
        int(row["physical_candidate_rollouts"]) for row in trajectories
    )
    if (
        len(decisions) != expected
        or physical_batches != len(decisions)
        or len(candidate_rows) != physical_rollouts
        or any(
            int(row["physical_r2_batch_count"]) != 1
            for row in decisions
        )
    ):
        raise AssertionError("H1 single-R2 physical batch contract failed")
    summary = summarize(decisions)
    pure = summary["overall"]["views"]["path_pure_oracle"]["arms"]
    phase_beats_r0 = bool(
        pure["phase_frozen"]["positive_state_recall"]
        > pure["R0_active8"]["positive_state_recall"]
        and pure["phase_frozen"]["oracle_gain_retention"]
        > pure["R0_active8"]["oracle_gain_retention"]
    )
    status = (
        "smoke_passed" if args.smoke
        else (
            "phase_h1_candidate_oracle_beats_r0"
            if phase_beats_r0
            else "phase_h1_candidate_oracle_not_better_on_both"
        )
    )
    decision = (
        "This is only a horizon-aligned candidate-generator oracle signal; "
        "train and evaluate a selector before any agent integration."
    )
    decisions_path = output / "h1_decisions.jsonl"
    candidates_path = output / "h1_candidate_evaluations.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    summary_path = output / "summary.json"
    _atomic_jsonl(decisions_path, decisions)
    _atomic_jsonl(candidates_path, candidate_rows)
    _atomic_jsonl(trajectories_path, trajectories)
    residual._atomic_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    native_seconds = sum(
        float(row["native_candidate_seconds"]) for row in trajectories
    )
    throughput = {
        "physical_r2_batches": physical_batches,
        "physical_candidate_rollouts": physical_rollouts,
        "reference_rollouts": sum(
            int(row["reference_rollouts"]) for row in trajectories
        ),
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": (
            physical_rollouts / native_seconds
        ),
        "one_r2_physical_batch_per_decision": True,
        "arm_reuse": "R0 and phase arms slice R2 by canonical H1 action SHA",
    }
    script_path = Path(__file__).resolve()
    test_path = (
        script_path.parents[1]
        / "tests"
        / "test_run_phase_breadth_h1_offsets_v1.py"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020",
            "route_ids": list(genome),
            "sha256": path_v1._sha256(list(genome)),
        },
        "frozen_phase_mapping": {
            str(anchor): arm
            for anchor, arm in sorted(phase_mapping.items())
        },
        "scenario_contract": {
            "opponents": list(opponents),
            "seeds": list(seeds),
            "seats": [0, 1],
            "anchors": list(map(int, anchors)),
            "offsets": list(OFFSETS),
            "decisions_expected": expected,
            "decisions_observed": len(decisions),
            "candidate_evaluations": len(candidate_rows),
            "candidate_committed": False,
        },
        "candidate_generator_contract": {
            "frozen_object": "phase to donor-pool breadth only",
            "donor_ids_frozen": False,
            "online_per_offset_selection": (
                "exact canonical H1 action-coverage diversity, budget 3"
            ),
            "horizon_aligned_generator": True,
            "not_the_original_h4_selector": True,
            "legacy_h4_then_h1_truncation": "diagnostic only",
        },
        "evaluation_contract": {
            "all_candidates_reported": True,
            "path_pure_subset_reported": True,
            "market_diff_alignment": "one integer per physical candidate",
            "batch_discard_on_any_market_diff": False,
            "candidate_set_oracle": True,
            "selector_executed": False,
            "oracle_selector_distinction": (
                "R0/phase/R2 scores select the best terminal candidate in hindsight; "
                "they are candidate-set oracle ceilings, not deployable selector scores"
            ),
        },
        "summary": summary,
        "throughput": throughput,
        "legacy_api_parity": {
            "checks": [
                value
                for trajectory in trajectories
                for value in trajectory["legacy_h1_api_parity"]
            ],
            "passed": all(
                value["passed"]
                for trajectory in trajectories
                for value in trajectory["legacy_h1_api_parity"]
            ),
        },
        "implementation": {
            "runner": retrieval_v1._artifact(script_path),
            "tests": retrieval_v1._artifact(test_path),
            "breadth_runtime": retrieval_v1._artifact(
                Path(breadth.__file__).resolve()
            ),
            "path_candidate_runtime": retrieval_v1._artifact(
                Path(path_v1.__file__).resolve()
            ),
            "native_extension": retrieval_v1._artifact(
                Path(path_v1.native_extension.__file__).resolve()
            ),
        },
        "frozen_inputs": {
            **frozen_inputs,
            "breadth_status": breadth_report["status"],
            "lopo_status": lopo["status"],
        },
        "strict_library": strict.inputs,
        "experiment_inputs": experiment_inputs,
        "source": source,
        "evidence_boundary": {
            "proxy_split_not_strict_baseline_ood": True,
            "sealed_test_executed": False,
            "selector_evaluated": False,
            "candidate_committed": False,
            "market_diagnostics_are_candidate_level": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            decisions_path.name: retrieval_v1._artifact(
                decisions_path, len(decisions)
            ),
            candidates_path.name: retrieval_v1._artifact(
                candidates_path, len(candidate_rows)
            ),
            trajectories_path.name: retrieval_v1._artifact(
                trajectories_path, len(trajectories)
            ),
            summary_path.name: retrieval_v1._artifact(summary_path),
        },
    }
    report_path = output / (
        "SMOKE_REPORT.json" if args.smoke else "FINAL_REPORT.json"
    )
    markdown_path = output / (
        "SMOKE_REPORT.md" if args.smoke else "FINAL_REPORT.md"
    )
    residual._atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "phase_breadth_h1_complete",
        "status": status,
        "output": str(output),
        "decisions": len(decisions),
        "candidate_rows": len(candidate_rows),
        "candidate_rollouts_per_second": (
            throughput["candidate_rollouts_per_second"]
        ),
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--block-library", type=Path, default=retrieval_v1.DEFAULT_LIBRARY)
    result.add_argument("--breadth-root", type=Path, default=DEFAULT_BREADTH_ROOT)
    result.add_argument("--lopo-report", type=Path, default=DEFAULT_LOPO)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--opponent", action="append", default=None)
    result.add_argument("--seed-start", type=int, default=2026086340)
    result.add_argument("--seed-count", type=int, default=2)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        args.opponent = [
            (args.opponent or list(path_v1.TRAIN_OPPONENTS))[0]
        ]
        args.seed_count = 1
        args.max_windows = 1
    if (
        args.seed_count < 1
        or not 1 <= args.max_windows <= len(path_v1.ANCHORS)
    ):
        raise ValueError("invalid positive H1 scenario budgets")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
