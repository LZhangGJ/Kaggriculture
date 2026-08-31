"""CONTRACT-BREADTH-SWEEP-v2 experiment runner.

One native R2 batch is evaluated per state; all fixed arms are exact SHA slices.
Retrieval happens at anchor and the unit action is decided at anchor + 1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
import run_contract_retrieval_oracle_ablation_v1 as v1
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual


KS = (8, 16, 24, 32)
SCHEMA = "contract-breadth-sweep-v2"
ARM_KEYS = ("R0_active8",) + tuple(f"K{k}_contract" for k in KS) + ("R2_all",)
TRAIN_SPLITS = frozenset(("train", "train_proxy"))
HELDOUT_SPLITS = frozenset(("heldout", "heldout_proxy"))
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_breadth_sweep_v2_20260829"
)
DEFAULT_K8_REFERENCE = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_retrieval_ablation_v1_formal_20260829c"
)
EXPECTED_K8_REPORT_SHA256 = "294a43240b1c4468058603c263befae4bca6eb876c85eb5cad790bb24fc4b9f0"
EXPECTED_K8_DECISIONS_SHA256 = "7cdc36d2b2e655ec7c3dbaa2ffcb4d944f5bf92dc8c5bd90dbf3de5ee8011d64"
K8_PARITY_FIELDS = (
    "selected_donor_ids",
    "candidate_count",
    "path_pure_candidate_count",
    "market_diff_candidate_count",
    "market_diff_step_count",
    "best_code",
    "best_kind",
    "best_donor_id",
    "best_rewards",
    "best_outcome",
    "best_margin",
    "strict_headroom",
    "oracle_gain",
    "loss_to_win",
)


def nearest_distinct_prefixes(
    block_ids: Sequence[str],
    distances: Sequence[float],
    ks: Iterable[int] = KS,
) -> tuple[dict[int, list[str]], dict[int, int]]:
    """Return nearest-distinct prefixes and raw prototype scan counts.

    Ties match v1: distance, block id, then original prototype position. A K
    arm scans until it has K unique blocks (or exhausts the anchor).
    """
    if len(block_ids) != len(distances):
        raise ValueError("block_ids and distances must have the same length")
    requested = tuple(sorted(set(int(k) for k in ks)))
    if not requested or requested[0] <= 0:
        raise ValueError("ks must contain positive integers")

    order = sorted(range(len(block_ids)), key=lambda i: (float(distances[i]), block_ids[i], i))
    distinct: list[str] = []
    seen: set[str] = set()
    inspected_at: dict[int, int] = {}
    for raw_position, prototype_index in enumerate(order, start=1):
        block_id = block_ids[prototype_index]
        if block_id in seen:
            continue
        seen.add(block_id)
        distinct.append(block_id)
        if len(distinct) in requested:
            inspected_at[len(distinct)] = raw_position

    exhausted = len(order)
    prefixes: dict[int, list[str]] = {}
    inspected: dict[int, int] = {}
    for k in requested:
        capped = min(k, len(distinct))
        prefixes[k] = distinct[:capped]
        inspected[k] = inspected_at.get(capped, exhausted) if capped else 0
    return prefixes, inspected


@dataclass(frozen=True)
class DiversityAudit:
    selected_ids: tuple[str, ...]
    unique_payloads: int
    total_support: int
    combinations_checked: int


def exact_action_diverse_three(
    pool_ids: Iterable[str],
    payload_masks: Mapping[str, int],
    support_by_id: Mapping[str, int],
) -> DiversityAudit:
    """Exact v1 diversity objective using cached canonical-payload bitsets."""
    pool = tuple(sorted(set(pool_ids)))
    if not pool:
        return DiversityAudit((), 0, 0, 0)
    missing = [block_id for block_id in pool if block_id not in payload_masks]
    if missing:
        raise KeyError(f"missing payload mask(s): {missing[:3]}")
    count = min(3, len(pool))
    best_ids: tuple[str, ...] = ()
    best_score = (-1, -1)
    checked = 0
    for combo in itertools.combinations(pool, count):
        checked += 1
        union = 0
        support = 0
        for block_id in combo:
            union |= int(payload_masks[block_id])
            support += int(support_by_id.get(block_id, 0))
        score = (union.bit_count(), support)
        if score > best_score:
            best_ids = combo
            best_score = score
    return DiversityAudit(best_ids, best_score[0], best_score[1], checked)


def select_diverse_ids(
    pool_ids: Iterable[str],
    payload_masks: Mapping[str, int],
    support_by_id: Mapping[str, int],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return exact selected ids plus the legacy v1 audit fields."""
    pool = tuple(sorted(set(pool_ids)))
    audit = exact_action_diverse_three(pool, payload_masks, support_by_id)
    count = min(3, len(pool))
    support_top = tuple(sorted(pool, key=lambda block_id: (-support_by_id[block_id], block_id))[:count])
    support_union = 0
    for block_id in support_top:
        support_union |= int(payload_masks[block_id])
    return audit.selected_ids, {
        "allowed": len(pool),
        "budget": count,
        "support_top_ids": list(support_top),
        "support_top_unique_residuals": support_union.bit_count(),
        "diversity_ids": list(audit.selected_ids),
        "diversity_unique_residuals": audit.unique_payloads,
        "diversity_support": audit.total_support,
        "combinations_checked": audit.combinations_checked,
    }


@dataclass(frozen=True)
class VariantCache:
    base: tuple[Any, ...]
    donors: Mapping[str, path_v1.DonorBlock]
    variants: Mapping[str, tuple[tuple[Any, ...], ...]]
    payloads: Mapping[str, tuple[bytes, ...]]
    payload_masks: Mapping[str, int]
    support: Mapping[str, int]
    build_seconds: float


def build_variant_cache(
    base_tape: Sequence[Mapping[str, Any]],
    donors: Sequence[path_v1.DonorBlock],
    anchor: int,
    actor_count: int,
    decision_step: int,
) -> VariantCache:
    """Build every strict donor variant/payload once for this live state."""
    started = time.perf_counter()
    ordered = tuple(sorted(donors, key=lambda donor: donor.block_id))
    base = path_v1._unit_trace(base_tape, decision_step, actor_count)
    start_offset = decision_step - anchor
    if start_offset < 1:
        raise ValueError("unit donor residual must start after its retrieval anchor")
    donor_map = {donor.block_id: donor for donor in ordered}
    variants: dict[str, tuple[tuple[Any, ...], ...]] = {}
    payloads: dict[str, tuple[bytes, ...]] = {}
    for donor in ordered:
        rows = tuple(path_v1._donor_variants(
            base, donor, actor_count, path_v1.PLAN_HORIZON,
            start_offset, 0, True,
        ))
        variants[donor.block_id] = rows
        payloads[donor.block_id] = tuple(
            path_v1._canonical_bytes(row[-1]) for row in rows
        )
    universe = {
        payload: index
        for index, payload in enumerate(sorted({
            payload for values in payloads.values() for payload in values
        }))
    }
    masks = {
        block_id: sum(1 << universe[payload] for payload in set(values))
        for block_id, values in payloads.items()
    }
    return VariantCache(
        base=base,
        donors=donor_map,
        variants=variants,
        payloads=payloads,
        payload_masks=masks,
        support={donor.block_id: int(donor.support) for donor in ordered},
        build_seconds=time.perf_counter() - started,
    )


def candidates_from_cache(
    cache: VariantCache,
    block_ids: Iterable[str],
) -> list[path_v1.PathCandidate]:
    """Materialize one arm without regenerating donor traces or payload SHAs."""
    selected = tuple(sorted(set(block_ids)))
    missing = [block_id for block_id in selected if block_id not in cache.donors]
    if missing:
        raise KeyError(f"unknown cached donor(s): {missing[:3]}")
    keep = ("KEEP", "KEEP", tuple(), None, None, 0, -1, 0, cache.base)
    entries: list[tuple[tuple[Any, ...], bytes]] = [
        (keep, path_v1._canonical_bytes(cache.base)),
    ]
    for donor_rank, block_id in enumerate(selected):
        for row, payload in zip(cache.variants[block_id], cache.payloads[block_id]):
            adjusted = (*row[:6], donor_rank, *row[7:])
            entries.append((adjusted, payload))

    canonical: dict[bytes, list[Any]] = {}
    order: list[bytes] = []
    for row, payload in entries:
        if payload not in canonical:
            canonical[payload] = [*row, [str(row[0])]]
            order.append(payload)
        else:
            canonical[payload][-1].append(str(row[0]))
    result: list[path_v1.PathCandidate] = []
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
        raise AssertionError("cached candidate list lost KEEP")
    return result


def build_diverse_arm(
    cache: VariantCache,
    pool_ids: Iterable[str],
) -> tuple[list[path_v1.PathCandidate], dict[str, Any]]:
    selection_started = time.perf_counter()
    selected, audit = select_diverse_ids(pool_ids, cache.payload_masks, cache.support)
    selection_seconds = time.perf_counter() - selection_started
    materialize_started = time.perf_counter()
    candidates = candidates_from_cache(cache, selected)
    materialization_seconds = time.perf_counter() - materialize_started
    return candidates, {
        **audit,
        "selection_seconds": selection_seconds,
        "materialization_seconds": materialization_seconds,
        "construction_seconds": selection_seconds + materialization_seconds,
    }


def decision_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["opponent"]), int(row["seed"]), int(row["seat"]), int(row["anchor"]))


def k8_parity_mismatches(
    current_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Compare K=8 against frozen v1 R1 rows without another rollout."""
    reference = {decision_key(row): row for row in reference_rows}
    mismatches: list[str] = []
    for row in current_rows:
        key = decision_key(row)
        old = reference.get(key)
        if old is None:
            mismatches.append(f"{key}: missing reference row")
            continue
        new_arm = row["arms"]["K8_contract"]
        old_arm = old["arms"]["R1_contract8"]
        for field in K8_PARITY_FIELDS:
            if new_arm.get(field) != old_arm.get(field):
                mismatches.append(
                    f"{key}: {field}: {new_arm.get(field)!r} != {old_arm.get(field)!r}"
                )
        if row.get("k8_unique_block_ids") != old.get("r1_unique_block_ids"):
            mismatches.append(f"{key}: nearest-distinct K8 block ids differ")
        if row.get("keep_rewards") != old.get("keep_rewards"):
            mismatches.append(f"{key}: KEEP rewards differ")
    return mismatches


def _arm_gain(row: Mapping[str, Any], arm: str) -> float:
    return float(row["arms"][arm]["oracle_gain"])


def choose_global_k_train_only(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one global contract breadth from train only; smaller K wins ties."""
    train = [row for row in rows if row["split"] in TRAIN_SPLITS]
    if not train:
        raise ValueError("global K selection requires train rows")
    gains = {k: sum(_arm_gain(row, f"K{k}_contract") for row in train) for k in KS}
    chosen = max(KS, key=lambda k: (gains[k], -k))
    return {
        "chosen_k": chosen,
        "chosen_arm": f"K{chosen}_contract",
        "train_gain_by_k": {str(k): gains[k] for k in KS},
        "heldout_used_for_selection": False,
        "tie_break": "max train oracle_gain sum, then smaller K",
    }


def choose_per_anchor_router_train_only(rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """High-variance diagnostic router; never used as the main conclusion."""
    options = ("R0_active8",) + tuple(f"K{k}_contract" for k in KS)
    priority = {arm: -i for i, arm in enumerate(options)}
    by_anchor: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in TRAIN_SPLITS:
            by_anchor[int(row["anchor"])].append(row)
    return {
        anchor: max(
            options,
            key=lambda arm: (sum(_arm_gain(row, arm) for row in anchor_rows), priority[arm]),
        )
        for anchor, anchor_rows in sorted(by_anchor.items())
    }


def _distance_minima(block_ids: Sequence[str], distances: Sequence[float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for block_id, distance in zip(block_ids, distances):
        key = str(block_id)
        result[key] = min(result.get(key, math.inf), float(distance))
    return result


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    strict: v1.StrictLibrary,
    opponent: str,
    seed: int,
    seat: int,
    split: str,
    anchors: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(tapes[baseline_id], tapes, genome)
    anchor_set = set(map(int, anchors))
    decision_to_anchor = {v1.retrieval_and_decision_steps(anchor)[1]: anchor for anchor in anchors}
    pending: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    physical_rollouts = reference_rollouts = 0
    native_seconds = cache_seconds = parity_seconds = 0.0
    parity_checks: list[dict[str, Any]] = []
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        route_id = str(genome[segment])
        if step in anchor_set:
            retrieval_started = time.perf_counter()
            contract, layout, mask = v1.snapshot_contract(observation, history, plan_tape)
            indices = strict.prototype_indices[step]
            prototype_ids = list(map(str, strict.prototypes["block_ids"][indices]))
            distances = v1.contract_distances(
                contract, layout, mask,
                strict.prototypes["entry_contracts"][indices],
                strict.prototypes["layouts"][indices],
                strict.prototypes["unlocked_masks"][indices],
                strict.scales[step],
            )
            prefixes, inspected = nearest_distinct_prefixes(prototype_ids, distances, KS)
            best = _distance_minima(prototype_ids, distances)
            pending[step] = {
                "retrieval_step": step,
                "decision_step": step + 1,
                "query_sha256": path_v1._sha256({
                    "contract": contract.tolist(),
                    "layout": layout.tolist(),
                    "mask": mask.tolist(),
                }),
                "prototype_count": len(indices),
                "prefixes": prefixes,
                "inspected": inspected,
                "distances": {
                    k: {block_id: best[block_id] for block_id in prefixes[k]}
                    for k in KS
                },
                "construction_seconds": time.perf_counter() - retrieval_started,
            }
        if step in decision_to_anchor:
            anchor = decision_to_anchor[step]
            retrieval = pending.pop(anchor)
            actor_count = max(1, len(_positions(observation)))
            base_tape = tapes[route_id]
            cache = build_variant_cache(
                base_tape, strict.by_anchor[anchor], anchor, actor_count, step,
            )
            cache_seconds += cache.build_seconds
            candidates_by_arm: dict[str, list[path_v1.PathCandidate]] = {}
            audits: dict[str, dict[str, Any]] = {}

            active_ids = [donor.block_id for donor in strict.active_by_anchor[anchor]]
            candidates_by_arm["R0_active8"], audits["R0_active8"] = build_diverse_arm(
                cache, active_ids,
            )
            for k in KS:
                arm = f"K{k}_contract"
                candidates_by_arm[arm], audits[arm] = build_diverse_arm(
                    cache, retrieval["prefixes"][k],
                )

            parity_started = time.perf_counter()
            k8_pool = tuple(
                strict.blocks[block_id] for block_id in retrieval["prefixes"][8]
            )
            legacy_selected, legacy_audit = v1._select_diverse_fast(
                base_tape, k8_pool, anchor, actor_count, path_v1.MAX_PAIRS, step,
            )
            parity_seconds += time.perf_counter() - parity_started
            parity_keys = (
                "support_top_ids", "support_top_unique_residuals",
                "diversity_ids", "diversity_unique_residuals", "diversity_support",
            )
            current_audit = audits["K8_contract"]
            if (
                [donor.block_id for donor in legacy_selected]
                != current_audit["diversity_ids"]
                or any(legacy_audit[key] != current_audit[key] for key in parity_keys)
            ):
                raise AssertionError("cached K8 selection/audit changed v1 semantics")
            parity_checks.append({
                "anchor": int(anchor),
                "actor_count": int(actor_count),
                "selected_ids": list(current_audit["diversity_ids"]),
                "audit": {key: current_audit[key] for key in parity_keys},
            })

            r2_started = time.perf_counter()
            r2_ids = [donor.block_id for donor in strict.by_anchor[anchor]]
            candidates_by_arm["R2_all"] = candidates_from_cache(cache, r2_ids)
            audits["R2_all"] = {
                "allowed": len(r2_ids), "budget": len(r2_ids),
                "diversity_ids": r2_ids,
                "selection_seconds": 0.0,
                "materialization_seconds": time.perf_counter() - r2_started,
            }
            r2 = candidates_by_arm["R2_all"]
            packed, counts = path_v1.pack_unit_plans(r2)
            native_started = time.perf_counter()
            raw = bundle.executor.rollout_schedule_unit_override_sequence_batch(
                env, states[0], states[1], schedule[segment:],
                path_v1.STOPS[segment:], seat, packed, counts,
            )
            native_elapsed = time.perf_counter() - native_started
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

            arms: dict[str, dict[str, Any]] = {}
            for arm, candidates in candidates_by_arm.items():
                arm_rewards, arm_market = v1._slice_from_r2(
                    candidates, r2, rewards, market_diff,
                )
                selected_ids = (
                    audits[arm]["diversity_ids"] if arm != "R2_all" else r2_ids
                )
                value = v1._arm_result(
                    candidates, arm_rewards, arm_market, seat,
                    audits[arm]["allowed"], selected_ids,
                )
                value["selection_audit"] = {
                    key: audits[arm][key] for key in (
                        "allowed", "budget", "diversity_ids",
                    ) if key in audits[arm]
                }
                value["selection_seconds"] = float(audits[arm]["selection_seconds"])
                value["materialization_seconds"] = float(
                    audits[arm]["materialization_seconds"]
                )
                value["construction_seconds"] = (
                    value["selection_seconds"] + value["materialization_seconds"]
                )
                arms[arm] = value
            if any(
                arms[arm]["selected_donor_count"] != path_v1.MAX_PAIRS
                for arm in ARM_KEYS[:-1]
            ):
                raise AssertionError("an online arm violated the exact donor budget of 3")

            keep_rank = adaptive.reward_rank(reference, seat)
            rows.append({
                "schema": SCHEMA,
                "split": split,
                "opponent": opponent,
                "seed": int(seed),
                "seat": int(seat),
                "genome_id": "NR020",
                "route_id": route_id,
                "anchor": int(anchor),
                "retrieval_step": int(retrieval["retrieval_step"]),
                "decision_step": int(step),
                "actor_count": int(actor_count),
                "query_sha256": retrieval["query_sha256"],
                "prototype_count": int(retrieval["prototype_count"]),
                "raw_prototypes_inspected": {
                    str(k): int(retrieval["inspected"][k]) for k in KS
                },
                "unique_block_ids": {
                    str(k): list(retrieval["prefixes"][k]) for k in KS
                },
                "unique_block_distances": {
                    str(k): retrieval["distances"][k] for k in KS
                },
                "k8_unique_block_ids": list(retrieval["prefixes"][8]),
                "keep_rewards": list(map(float, reference)),
                "keep_outcome": int(keep_rank[0]),
                "keep_margin": float(keep_rank[1]),
                "arms": arms,
                "shared_cache_build_seconds": float(cache.build_seconds),
                "retrieval_construction_seconds": float(
                    retrieval["construction_seconds"]
                ),
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
        "split": split,
        "opponent": opponent,
        "seed": int(seed),
        "seat": int(seat),
        "decisions": len(rows),
        "physical_candidate_rollouts": physical_rollouts,
        "reference_rollouts": reference_rollouts,
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": (
            physical_rollouts / native_seconds if native_seconds else 0.0
        ),
        "shared_cache_build_seconds": cache_seconds,
        "k8_parity_seconds": parity_seconds,
        "k8_selected_ids_audit_parity_checks": parity_checks,
        "completed": True,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    if not len(array):
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _arm_metrics(
    rows: Sequence[Mapping[str, Any]],
    arm_for_row: Any,
) -> dict[str, Any]:
    selected = [(row, row["arms"][arm_for_row(row)]) for row in rows]
    r2_positive = [
        row for row in rows if row["arms"]["R2_all"]["strict_headroom"]
    ]
    r2_gain = sum(float(row["arms"]["R2_all"]["oracle_gain"]) for row in rows)
    recalled = sum(
        bool(row["arms"][arm_for_row(row)]["strict_headroom"])
        for row in r2_positive
    )
    gains = [float(value["oracle_gain"]) for _, value in selected]
    nonkeep = [max(0, int(value["candidate_count"]) - 1) for _, value in selected]
    pools = [int(value["donor_pool_count"]) for _, value in selected]
    construction = [float(value["construction_seconds"]) for _, value in selected]
    selection = [float(value["selection_seconds"]) for _, value in selected]
    materialization = [float(value["materialization_seconds"]) for _, value in selected]
    return {
        "positive_states": sum(bool(value["strict_headroom"]) for _, value in selected),
        "positive_state_recall": recalled / len(r2_positive) if r2_positive else 1.0,
        "oracle_gain_sum": sum(gains),
        "oracle_gain_retention": sum(gains) / r2_gain if r2_gain > 0 else 1.0,
        "loss_to_win_states": sum(bool(value["loss_to_win"]) for _, value in selected),
        "candidate_count_sum": sum(int(value["candidate_count"]) for _, value in selected),
        "candidate_count_one_states": sum(value == 0 for value in nonkeep),
        "effective_nonkeep": _distribution(nonkeep),
        "donor_pool": _distribution(pools),
        "market_diff_candidates": sum(
            int(value["market_diff_candidate_count"]) for _, value in selected
        ),
        "market_diff_steps": sum(
            int(value["market_diff_step_count"]) for _, value in selected
        ),
        "selection_seconds_sum": sum(selection),
        "materialization_seconds_sum": sum(materialization),
        "construction_seconds_sum": sum(construction),
        "construction_seconds": _distribution(construction),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r2_positive = sum(
        bool(row["arms"]["R2_all"]["strict_headroom"]) for row in rows
    )
    return {
        "states": len(rows),
        "r2_positive_states": r2_positive,
        "r2_positive_rate": r2_positive / len(rows) if rows else 0.0,
        "arms": {
            arm: _arm_metrics(rows, lambda _row, arm=arm: arm)
            for arm in ARM_KEYS
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "by_split": defaultdict(list),
        "by_anchor": defaultdict(list),
        "by_opponent": defaultdict(list),
    }
    for row in rows:
        groups["by_split"][str(row["split"])].append(row)
        groups["by_anchor"][str(row["anchor"])].append(row)
        groups["by_opponent"][str(row["opponent"])].append(row)
    return {
        "overall": _group_summary(rows),
        **{
            name: {
                key: _group_summary(group)
                for key, group in sorted(values.items())
            }
            for name, values in groups.items()
        },
    }


def _router_evaluation(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, str],
) -> dict[str, Any]:
    def one(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _arm_metrics(group, lambda row: mapping[int(row["anchor"])])

    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_anchor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
        by_anchor[str(row["anchor"])].append(row)
    return {
        "overall": one(rows),
        "by_split": {key: one(value) for key, value in sorted(by_split.items())},
        "by_anchor": {key: one(value) for key, value in sorted(by_anchor.items())},
    }


def _retrieval_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "retrieval_step": "anchor",
        "decision_step": "anchor+1",
        "distance": (
            "mean of robust-scaled L1 state excluding 86:104, robust-scaled "
            "fixed-capital plan, and layout+mask Hamming"
        ),
        "nearest_distinct": {},
        "shared_cache_build_seconds_sum": sum(
            float(row["shared_cache_build_seconds"]) for row in rows
        ),
        "shared_cache_build_seconds": _distribution([
            float(row["shared_cache_build_seconds"]) for row in rows
        ]),
        "retrieval_construction_seconds_sum": sum(
            float(row["retrieval_construction_seconds"]) for row in rows
        ),
        "retrieval_construction_seconds": _distribution([
            float(row["retrieval_construction_seconds"]) for row in rows
        ]),
    }
    for k in KS:
        arm = f"K{k}_contract"
        pools = [int(row["arms"][arm]["donor_pool_count"]) for row in rows]
        result["nearest_distinct"][str(k)] = {
            "requested_k": k,
            "cap_if_anchor_has_fewer": True,
            "unique_pool": _distribution(pools),
            "raw_prototypes_inspected": _distribution([
                int(row["raw_prototypes_inspected"][str(k)]) for row in rows
            ]),
            "exact_action_diverse_budget": path_v1.MAX_PAIRS,
        }
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    chosen = report["global_k_selection"]
    heldout = chosen["evaluation"]["by_split"].get("heldout_proxy")
    lines = [
        "# CONTRACT-BREADTH-SWEEP-v2",
        "",
        f"- Status: {report['status']}",
        f"- States: {report['scenario_contract']['states_observed']}",
        f"- Frozen train-only global K: **{chosen['chosen_k']}**",
        "- Selection rule: max train oracle-gain sum, then smaller K.",
        "- Heldout was not used for K selection.",
        "- Per-anchor router is diagnostic only and high variance (24 train states per anchor).",
        "- No candidate was committed; NR020 stayed frozen.",
        "",
        "## Fixed-arm overall",
        "",
        "| Arm | Recall | Gain retention | Loss->win | Collapse |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, value in report["summary"]["overall"]["arms"].items():
        lines.append(
            f"| {arm} | {value['positive_state_recall']:.4f} | "
            f"{value['oracle_gain_retention']:.4f} | "
            f"{value['loss_to_win_states']} | {value['candidate_count_one_states']} |"
        )
    if heldout:
        lines.extend([
            "",
            "## Frozen heldout readout",
            "",
            f"- Recall: {heldout['positive_state_recall']:.4f}",
            f"- Gain retention: {heldout['oracle_gain_retention']:.4f}",
            f"- Loss->win states: {heldout['loss_to_win_states']}",
        ])
    lines.extend([
        "",
        "## Evidence boundary",
        "",
        "- The proxy heldout split is not strict baseline OOD.",
        "- Market 86:104 is excluded from retrieval distance; market diagnostics are local H4 only.",
        "- This is an oracle retrieval ablation, not a submitted selector.",
        "",
    ])
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    opponents = tuple(args.opponent or (*path_v1.TRAIN_OPPONENTS, *path_v1.HELDOUT_OPPONENTS))
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    if len(set(opponents)) != len(opponents) or set(seeds) & residual.SEALED_SEEDS:
        raise ValueError("opponents must be unique and seeds must remain unsealed")
    if not args.smoke and (
        len(opponents) != 12
        or len(seeds) != 2
        or args.max_windows != len(path_v1.ANCHORS)
    ):
        raise ValueError("formal sweep requires 12 opponents x 2 seeds x 21 anchors")

    reference_report = args.k8_reference / "FINAL_REPORT.json"
    reference_decisions = args.k8_reference / "oracle_decisions.jsonl"
    if residual._sha256_file(reference_report) != EXPECTED_K8_REPORT_SHA256:
        raise ValueError("frozen K8 reference report SHA changed")
    if residual._sha256_file(reference_decisions) != EXPECTED_K8_DECISIONS_SHA256:
        raise ValueError("frozen K8 decision SHA changed")
    reference_rows = _read_jsonl(reference_decisions)

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
    strict = v1.load_strict_library(args.block_library)
    genome = genomes["NR020"]
    for anchor in path_v1.ANCHORS:
        if {donor.block_id for donor in active_donors[anchor]} != {
            donor.block_id for donor in strict.active_by_anchor[anchor]
        }:
            raise ValueError(f"R0 active8 mismatch at {anchor}")

    split_by_opponent = {
        opponent: (
            "train_proxy" if opponent in path_v1.TRAIN_OPPONENTS else "heldout_proxy"
        )
        for opponent in opponents
    }
    anchors = path_v1.ANCHORS[:args.max_windows]
    scenarios = [
        (opponent, seed, seat)
        for opponent in opponents for seed in seeds for seat in (0, 1)
    ]
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, (opponent, seed, seat) in enumerate(scenarios, 1):
        decisions, trajectory = _scenario(
            bundle, baseline_id, tapes, genome, strict,
            opponent, seed, seat, split_by_opponent[opponent], anchors,
        )
        rows.extend(decisions)
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "contract_breadth_progress",
            "index": index,
            "total": len(scenarios),
            "opponent": opponent,
            "seed": seed,
            "seat": seat,
            "states": len(rows),
        }, sort_keys=True), flush=True)

    parity = k8_parity_mismatches(rows, reference_rows)
    if parity:
        raise AssertionError("K8 frozen numerical parity failed: " + parity[0])
    summary = summarize(rows)
    global_choice = choose_global_k_train_only(rows)
    chosen_arm = global_choice["chosen_arm"]
    global_choice["evaluation"] = {
        "overall": summary["overall"]["arms"][chosen_arm],
        "by_split": {
            split: value["arms"][chosen_arm]
            for split, value in summary["by_split"].items()
        },
        "by_anchor": {
            anchor: value["arms"][chosen_arm]
            for anchor, value in summary["by_anchor"].items()
        },
        "heldout_used_for_selection": False,
    }
    router_mapping = choose_per_anchor_router_train_only(rows)
    router_diagnostic = {
        "diagnostic_only": True,
        "high_variance": True,
        "train_states_per_anchor_formal": 24,
        "selection_rule": "max per-anchor train oracle-gain sum; fixed arm priority breaks ties",
        "heldout_used_for_selection": False,
        "frozen_mapping": {
            str(anchor): arm for anchor, arm in sorted(router_mapping.items())
        },
        "evaluation": _router_evaluation(rows, router_mapping),
    }
    physical = sum(
        int(row["physical_candidate_rollouts"]) for row in trajectories
    )
    native_seconds = sum(
        float(row["native_candidate_seconds"]) for row in trajectories
    )
    logical = sum(
        int(row["arms"][arm]["candidate_count"])
        for row in rows for arm in ARM_KEYS
    )
    throughput = {
        "physical_candidate_rollouts": physical,
        "reference_rollouts": sum(
            int(row["reference_rollouts"]) for row in trajectories
        ),
        "logical_candidate_evaluations": logical,
        "native_candidate_seconds": native_seconds,
        "candidate_rollouts_per_second": physical / native_seconds,
        "single_r2_physical_batch_per_state": True,
        "nested_arm_reuse": "all fixed arms are canonical-trace-SHA slices of R2",
    }
    reference_report_value = json.loads(reference_report.read_text(encoding="utf-8"))
    old_k8 = reference_report_value["summary"]["overall"]["arms"]["R1_contract8"]
    new_k8 = summary["overall"]["arms"]["K8_contract"]
    common_k8_fields = tuple(sorted(set(old_k8) & set(new_k8)))
    k8_summary_mismatches = [
        field for field in common_k8_fields if old_k8[field] != new_k8[field]
    ]
    if not args.smoke and k8_summary_mismatches:
        raise AssertionError(
            "formal K8 aggregate parity failed: " + k8_summary_mismatches[0]
        )

    heldout = global_choice["evaluation"]["by_split"].get("heldout_proxy")
    heldout_k8 = (
        summary["by_split"]["heldout_proxy"]["arms"]["K8_contract"]
        if "heldout_proxy" in summary["by_split"] else None
    )
    if args.smoke:
        status = "smoke_passed"
        decision = "Two-state native smoke passed; proceed to the frozen formal sweep."
    elif heldout and heldout_k8 and (
        heldout["oracle_gain_retention"] > heldout_k8["oracle_gain_retention"] + 1e-12
        or heldout["positive_state_recall"] > heldout_k8["positive_state_recall"]
    ):
        status = "train_selected_breadth_improves_heldout"
        decision = "Use the train-selected global K as the next fixed retrieval breadth candidate."
    else:
        status = "train_selected_breadth_not_heldout_better_than_k8"
        decision = "Keep K8 for now; broader retrieval did not improve the frozen proxy-heldout readout."

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
    residual._atomic_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
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
        "scenario_contract": {
            "opponents": list(opponents),
            "seeds": list(seeds),
            "seats": [0, 1],
            "anchors": list(map(int, anchors)),
            "states_expected": len(opponents) * len(seeds) * 2 * len(anchors),
            "states_observed": len(rows),
            "candidate_committed": False,
        },
        "k8_parity": {
            "reference_report": v1._artifact(reference_report),
            "reference_decisions": v1._artifact(
                reference_decisions, len(reference_rows)
            ),
            "frozen_reference_rows": len(reference_rows),
            "compared_rows": len(rows),
            "numeric_mismatches": len(parity),
            "selected_ids_audit_checks": sum(
                len(row["k8_selected_ids_audit_parity_checks"])
                for row in trajectories
            ),
            "selected_ids_audit_exact": all(
                len(row["k8_selected_ids_audit_parity_checks"])
                == int(row["decisions"])
                for row in trajectories
            ),
            "aggregate_common_fields": list(common_k8_fields),
            "aggregate_mismatches": k8_summary_mismatches,
            "aggregate_exact_when_formal": (
                not k8_summary_mismatches if not args.smoke else None
            ),
            "passed": True,
        },
        "retrieval": _retrieval_summary(rows),
        "summary": summary,
        "global_k_selection": global_choice,
        "per_anchor_router_diagnostic": router_diagnostic,
        "throughput": throughput,
        "implementation": {
            "runner": v1._artifact(Path(__file__).resolve()),
            "v1_retrieval_helper": v1._artifact(Path(v1.__file__).resolve()),
            "path_candidate_runtime": v1._artifact(Path(path_v1.__file__).resolve()),
            "native_extension": v1._artifact(
                Path(path_v1.native_extension.__file__).resolve()
            ),
        },
        "strict_library": strict.inputs,
        "experiment_inputs": experiment_inputs,
        "source": source,
        "evidence_boundary": {
            "proxy_opponents_not_submitted_agents": True,
            "proxy_split_not_strict_baseline_ood": True,
            "strict_replay_lineage_filter_passed": True,
            "market_86_104_excluded_from_retrieval": True,
            "market_diagnostic_scope": "local H4 only",
            "sealed_test_executed": False,
            "candidate_committed": False,
        },
        "artifacts": {
            decisions_path.name: v1._artifact(decisions_path, len(rows)),
            trajectories_path.name: v1._artifact(
                trajectories_path, len(trajectories)
            ),
            summary_path.name: v1._artifact(summary_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / ("SMOKE_REPORT.json" if args.smoke else "FINAL_REPORT.json")
    markdown_path = output / ("SMOKE_REPORT.md" if args.smoke else "FINAL_REPORT.md")
    residual._atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "contract_breadth_complete",
        "output": str(output),
        "states": len(rows),
        "k8_parity": True,
        "chosen_k": global_choice["chosen_k"],
        "candidate_rollouts_per_second": throughput["candidate_rollouts_per_second"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--block-library", type=Path, default=v1.DEFAULT_LIBRARY)
    result.add_argument("--k8-reference", type=Path, default=DEFAULT_K8_REFERENCE)
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
        args.opponent = [(args.opponent or list(path_v1.TRAIN_OPPONENTS))[0]]
        args.seed_count = 1
        args.max_windows = 1
    if (
        args.seed_count < 1
        or not 1 <= args.max_windows <= len(path_v1.ANCHORS)
    ):
        raise ValueError("invalid positive scenario budgets")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
