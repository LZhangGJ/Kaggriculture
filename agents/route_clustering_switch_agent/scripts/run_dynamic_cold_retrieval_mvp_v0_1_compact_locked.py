#!/usr/bin/env python3
"""Compact, audit-gated performance adapter for dynamic-cold v0.1."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_offline_block_portfolio_v1 as offline  # noqa: E402
import retrieve_runtime_active_portfolio_v1 as runtime  # noqa: E402
import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402
import run_contract_retrieval_oracle_ablation_v1 as retrieval  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0 as draft  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0_1_locked as v01  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0_locked as v0  # noqa: E402


SCHEMA = "dynamic-cold-retrieval-mvp-v0.1-compact"
THRESHOLDS = dict(v01.THRESHOLDS)
build_query_panel = v0.build_query_panel
semantic_view = v01.semantic_view


def _assert_locks() -> None:
    v01._assert_v0_lock()
    if THRESHOLDS != v0.THRESHOLDS:
        raise ValueError("dynamic-cold compact threshold contract changed")


def _cache_compact_overlay_groups(index: dict[str, Any]) -> None:
    """Cache only immutable row topology, refs, and three capital columns."""

    prototypes = index["prototypes"]
    capital = np.asarray(
        prototypes["entry_contracts"][:, runtime.CAPITAL_REQUIREMENT_INDICES],
        np.float64,
    )
    if not np.isfinite(capital).all() or np.any(capital < 0):
        raise ValueError("invalid fixed-capital requirements in cold prototype bank")
    index["required_fixed_capital"] = np.max(capital, axis=1)

    reference_pool: list[dict[str, Any]] = []
    reference_index: dict[tuple[str, ...], int] = {}
    prototype_rows_by_block = index["library"]["_prototype_rows_by_block"]
    for anchor, cached in index["anchors"].items():
        local_by_row = {
            int(row): local for local, row in enumerate(cached["prototype_rows"])
        }
        groups_by_block: dict[str, tuple[tuple[tuple[int, ...], int], ...]] = {}
        for block_id in cached["rows_by_block"]:
            grouped: dict[tuple[str, str], list[int]] = {}
            for raw_row in prototype_rows_by_block[block_id]:
                row = int(raw_row)
                if int(prototypes["anchors"][row]) != int(anchor):
                    raise ValueError(f"cached prototype anchor mismatch: {block_id}")
                provenance_id = str(prototypes["provenance_ids"][row])
                source = index["provenance"].get(provenance_id)
                if source is None:
                    raise ValueError(
                        f"prototype provenance is unavailable: {provenance_id}"
                    )
                execution_id = str(source.get("execution_id") or "")
                full_sha = runtime._expected_sha256(
                    source.get("full_tape_sha256"),
                    f"prototype variant {provenance_id}",
                )
                grouped.setdefault((execution_id, full_sha), []).append(row)

            compact_groups = []
            for (execution_id, full_sha), rows in sorted(grouped.items()):
                rows = sorted(rows)
                provenance_ids = tuple(
                    str(prototypes["provenance_ids"][row]) for row in rows
                )
                pool_key = (
                    execution_id, full_sha, *tuple(sorted(provenance_ids)),
                )
                pool_row = reference_index.get(pool_key)
                if pool_row is None:
                    pool_row = len(reference_pool)
                    reference_index[pool_key] = pool_row
                    reference_pool.append(runtime._variant_ref(
                        provenance_ids, index["route_refs"], index["provenance"],
                    ))
                compact_groups.append((
                    tuple(local_by_row[row] for row in rows), pool_row,
                ))
            if not compact_groups:
                raise ValueError(f"cold block has no market variants: {block_id}")
            groups_by_block[block_id] = tuple(compact_groups)

        cached["anchor"] = int(anchor)
        cached["overlay_groups_by_block"] = groups_by_block
        cached["rows_by_block"] = {
            block_id: {
                "block_id": block_id,
                "block_ref": dict(row["block_ref"]),
                "action_ref": dict(row["action_ref"]),
                "contract_prototype_ref": dict(row["contract_prototype_ref"]),
            }
            for block_id, row in cached["rows_by_block"].items()
        }
    index["market_variant_reference_pool"] = tuple(reference_pool)


def load_cold_index(
    portfolio_root: Path, portfolio_manifest_sha256: str,
    expected_library_sha256: str, expected_artifact_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    _assert_locks()
    started = time.perf_counter()
    index = v0.load_cold_index(
        portfolio_root, portfolio_manifest_sha256,
        expected_library_sha256, expected_artifact_sha256s,
    )
    _cache_compact_overlay_groups(index)
    index["initialization_ms"] = (time.perf_counter() - started) * 1000.0
    return index


def _variant_reference(index: Mapping[str, Any], pool_row: int) -> dict[str, Any]:
    reference = dict(index["market_variant_reference_pool"][pool_row])
    reference["provenance_ids"] = list(reference["provenance_ids"])
    return reference


def _diagnose_cached(
    index: Mapping[str, Any], cached: Mapping[str, Any], hot_ids: Sequence[str],
    distances: np.ndarray, query: np.ndarray, minimum_cash_reserve: float,
) -> dict[str, Any]:
    reserve = float(minimum_cash_reserve)
    self_money = float(query[0])
    required = index["required_fixed_capital"]
    eligible_variant_candidates = []
    block_audit = {}
    hot_prototype_count = 0
    for block_id in hot_ids:
        hot_ref = cached["rows_by_block"][block_id]
        variants = []
        for positions, reference_row in cached["overlay_groups_by_block"][block_id]:
            hot_prototype_count += len(positions)
            best_position = min(
                positions,
                key=lambda position: (
                    float(distances[position]),
                    int(cached["prototype_rows"][position]),
                ),
            )
            prototype_row = int(cached["prototype_rows"][best_position])
            distance = float(distances[best_position])
            headroom = self_money - float(required[prototype_row])
            variant = {
                "market_variant_ref": _variant_reference(index, reference_row),
                "best_prototype_ref": {
                    "file": str(hot_ref["contract_prototype_ref"]["file"]),
                    "row": prototype_row,
                },
                "distance": distance,
                "capital_shortfall_diagnostic": float(
                    max(0.0, reserve - headroom)
                ),
                "reason": "capital_report_only_not_a_feasibility_proof",
            }
            variants.append(variant)
            eligible_variant_candidates.append({
                "block_id": block_id,
                "distance": distance,
                "prototype_row": prototype_row,
            })
        best_variant = min(variants, key=lambda row: (
            float(row["distance"]),
            str(row["market_variant_ref"]["full_tape_sha256"]),
        ))
        block_audit[block_id] = {
            "block_ref": dict(hot_ref["block_ref"]),
            "action_ref": dict(hot_ref["action_ref"]),
            "distance": float(best_variant["distance"]),
            "best_prototype_ref": dict(best_variant["best_prototype_ref"]),
            "market_variants": variants,
            "rejected_market_variants": [],
        }

    distinct_ids, inspected, best_distances = retrieval.nearest_distinct_blocks(
        [str(row["block_id"]) for row in eligible_variant_candidates],
        [float(row["distance"]) for row in eligible_variant_candidates],
        len(hot_ids),
    )
    selected = []
    for rank, block_id in enumerate(distinct_ids, 1):
        row = dict(block_audit[block_id])
        row.update({
            "rank": rank,
            "block_id": block_id,
            "distance": float(best_distances[block_id]),
            "reason": "nearest_distinct_unit_path_capital_report_only",
        })
        selected.append(row)
    selected_set = set(distinct_ids)
    below_limit = [
        {
            "block_id": block_id,
            "block_ref": dict(row["block_ref"]),
            "distance": float(row["distance"]),
            "reason": "eligible_but_ranked_below_active_limit",
        }
        for block_id, row in sorted(
            block_audit.items(), key=lambda item: (float(item[1]["distance"]), item[0]),
        )
        if block_id not in selected_set
    ]
    return {
        "anchor": int(cached["anchor"]),
        "hot_path_count": len(hot_ids),
        "hot_prototype_count": hot_prototype_count,
        "eligible_path_count": len(block_audit),
        "hard_current_cash_rejected_path_count": 0,
        "active_limit": len(hot_ids),
        "active_count": len(selected),
        "minimum_cash_reserve": reserve,
        "eligible_variant_candidates_inspected": int(inspected),
        "selected": selected,
        "rejected": [],
        "below_active_limit": below_limit,
        "distance": (
            "existing continuation-v1 robust-scaled L1 state/plan plus layout+mask Hamming"
        ),
        "mask_semantics": "existing soft Hamming term; not a new hard subset gate",
        "capital_gate_mode": "soft_report_only",
        "capital_feasibility_proven": False,
        "capital_evaluation": "soft report only: capital shortfall never rejects a path",
        "capital_proof_requires": (
            "a future liquidity curve/projection with income and expense timing; "
            "the current 147D fixed-commitment totals are insufficient"
        ),
        "path_action_diversity": (
            "distinct unit-path collapse; action diversity inherited_from_hot32"
        ),
        "market_variant_slot_semantics": (
            "all retrieval-eligible overlay refs remain nested under one selected unit path"
        ),
    }


def select_dynamic_cold(
    index: Mapping[str, Any], *, query: np.ndarray, query_layout: np.ndarray,
    query_mask: np.ndarray, anchor: int, minimum_cash_reserve: float = 0.0,
    top_limit: int = draft.TOP_LIMIT, active_limit: int = draft.ACTIVE_LIMIT,
) -> dict[str, Any]:
    query = np.asarray(query, np.float32)
    query_layout = np.asarray(query_layout, np.uint8)
    query_mask = np.asarray(query_mask, np.uint8)
    if (
        anchor not in index["anchors"]
        or query.shape != (continuation.FEATURE_DIM,)
        or query_layout.shape != (continuation.LAYOUT_SIZE,)
        or query_mask.shape != (len(continuation.QUADRANTS),)
        or not np.isfinite(query).all()
        or np.any(query_layout > 13)
        or np.any(query_mask > 1)
        or not (0 < active_limit <= top_limit)
        or not math.isfinite(float(minimum_cash_reserve))
        or float(minimum_cash_reserve) < 0
    ):
        raise ValueError("invalid dynamic cold query or limits")

    started = time.perf_counter()
    cached = index["anchors"][anchor]
    prototype_rows = cached["prototype_rows"]
    prototypes = index["prototypes"]
    distances = retrieval.contract_distances(
        query, query_layout, query_mask,
        prototypes["entry_contracts"][prototype_rows],
        prototypes["layouts"][prototype_rows],
        prototypes["unlocked_masks"][prototype_rows], cached["scale"],
    )
    hot_ids, inspected, best = retrieval.nearest_distinct_blocks(
        cached["prototype_block_ids"], distances, top_limit,
    )
    cold_scoring_ms = (time.perf_counter() - started) * 1000.0
    diagnosed = _diagnose_cached(
        index, cached, hot_ids, distances, query, minimum_cash_reserve,
    )
    if [str(row["block_id"]) for row in diagnosed["selected"]] != hot_ids or any(
        abs(float(row["distance"]) - float(best[row["block_id"]])) > 1e-6
        for row in diagnosed["selected"]
    ):
        raise AssertionError("cached cold retrieval disagrees with overlay diagnostics")

    nearest_id = hot_ids[0]
    remaining = [index["block_by_id"][value] for value in hot_ids[1:]]
    diverse_count = min(active_limit - 1, len(remaining))
    diverse: list[int] = []
    diverse_reasons: dict[int, dict[str, Any]] = {}
    if diverse_count:
        features = offline._features(
            remaining, index["blocks"], index["vectors"], prototypes,
            index["library"]["_prototype_rows_by_block"], index["provenance"],
            index["quality_available"],
        )
        diverse, diverse_reasons = offline._select(
            remaining, diverse_count, features,
            offline.ACTIVE_QUALITY_BONUS if index["quality_available"] else 0.0,
            index["blocks"], index["declared_active"],
        )
    active_ids = [nearest_id, *[
        str(index["blocks"][value]["block_id"]) for value in diverse
    ]]
    diagnosed_by_id = {
        str(row["block_id"]): dict(row) for row in diagnosed["selected"]
    }
    active = []
    for rank, block_id in enumerate(active_ids, 1):
        row = diagnosed_by_id[block_id]
        row["rank"] = rank
        row["active_reason"] = (
            {"kind": "global_nearest_exact_contract"}
            if rank == 1 else diverse_reasons[index["block_by_id"][block_id]]
        )
        active.append(row)
    return {
        "anchor": int(anchor),
        "runtime_state_conditioned": True,
        "cold_path_count": len(cached["block_indices"]),
        "cold_prototype_count": len(prototype_rows),
        "raw_prototypes_inspected_until_top32": int(inspected),
        "dynamic_hot_count": len(diagnosed["selected"]),
        "active_count": len(active),
        "cold_scoring_ms": cold_scoring_ms,
        "end_to_end_ms": (time.perf_counter() - started) * 1000.0,
        "minimum_cash_reserve": float(minimum_cash_reserve),
        "capital_gate_mode": "soft_report_only",
        "capital_feasibility_proven": False,
        "market_environment_distance_feature_indices": list(range(86, 104)),
        "market_variant_slot_semantics": diagnosed["market_variant_slot_semantics"],
        "dynamic_hot": diagnosed["selected"],
        "active": active,
    }


def assert_v0_equivalence(
    index: Mapping[str, Any], *, query: np.ndarray, query_layout: np.ndarray,
    query_mask: np.ndarray, anchor: int, minimum_cash_reserve: float = 0.0,
) -> None:
    baseline = v0.select_dynamic_cold(
        index, query=query, query_layout=query_layout, query_mask=query_mask,
        anchor=anchor, minimum_cash_reserve=minimum_cash_reserve,
    )
    optimized = select_dynamic_cold(
        index, query=query, query_layout=query_layout, query_mask=query_mask,
        anchor=anchor, minimum_cash_reserve=minimum_cash_reserve,
    )
    if semantic_view(baseline) != semantic_view(optimized):
        raise AssertionError("compact v0.1 semantic output differs from locked v0")


def cache_coverage(index: Mapping[str, Any]) -> dict[int, bool]:
    return {
        int(anchor): sorted(
            position
            for groups in cached["overlay_groups_by_block"].values()
            for positions, _ in groups
            for position in positions
        ) == list(range(len(cached["prototype_rows"])))
        for anchor, cached in index["anchors"].items()
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    raise RuntimeError(
        "compact v0.1 formal is audit-gated until independent 252-query "
        "equivalence and RSS checks pass"
    )


if __name__ == "__main__":
    raise SystemExit(main())
