#!/usr/bin/env python3
"""Falsify direct state-conditioned cold->top32->active8 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_offline_block_portfolio_v1 as offline  # noqa: E402
import retrieve_runtime_active_portfolio_v1 as runtime  # noqa: E402
import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402
import run_contract_retrieval_oracle_ablation_v1 as retrieval  # noqa: E402


SCHEMA = "dynamic-cold-retrieval-mvp-v0"
PANEL_SCHEMA = "dynamic-cold-query-panel-v0"
PANEL_SEED = "DYNAMIC-COLD-RETRIEVAL-MVP-v0"
PANEL_PER_KIND_PER_ANCHOR = 4
TOP_LIMIT = 32
ACTIVE_LIMIT = 8
LOCKED_LIBRARY_SHA256 = (
    "391555c41e1b7abf05837563d07636f70e57c0689fa50e7086d374073ab12242"
)
LOCKED_ARTIFACT_SHA256S = {
    "actions": "66a8982e22bce46c51c8f6876edd55c9b0e4e7d51a27db2d040116ef5cabbc26",
    "vectors": "3829d613adb46f173bfb3a8c149217c551deefbf0ba82ca14a3b268ebd54130a",
    "contract_prototypes": (
        "706e9503f0c9f806dd34788bd88c130085b9e8971cecf7240bdc115e893e8c6b"
    ),
}
LOCKED_PORTFOLIO_SHA256 = (
    "30ae8dcd713910c2642e5014fb9670c765bd253b1a890c8440370b294c762002"
)
LOCKED_PORTFOLIO_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\offline_block_portfolio_v1_base83bdd86e_delta6b6d026b_unitk4_hot32_active8_20260829a"
)
THRESHOLDS = {
    "reference_top32_parity_rate": 1.0,
    "distance_absolute_tolerance": 1e-6,
    "market_invariance_rate": 1.0,
    "near_unseen_nonworse_rate": 1.0,
    "near_unseen_strict_improvement_rate": 0.5,
    "global_nearest_in_active_rate": 1.0,
    "warm_core_p95_ms": 10.0,
    "end_to_end_p95_ms": 100.0,
    "end_to_end_max_ms": 125.0,
    "canonical_21_anchor_total_ms": 2100.0,
    "initialization_ms": 5000.0,
    "incremental_rss_mib": 128.0,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), percentile))


def load_cold_index(
    portfolio_root: Path, portfolio_manifest_sha256: str,
    expected_library_sha256: str, expected_artifact_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Load the existing fail-closed bundle and cache immutable anchor indices."""

    started = time.perf_counter()
    (
        portfolio, library, blocks, vectors, prototypes, provenance, route_refs,
        portfolio_sha,
    ) = runtime._load_portfolio(portfolio_root, portfolio_manifest_sha256)
    if str(portfolio["input"]["library_manifest_sha256"]) != expected_library_sha256:
        raise ValueError("locked combined library manifest SHA-256 mismatch")
    actual_artifacts = dict(portfolio["input"]["artifact_sha256s"])
    if actual_artifacts != dict(expected_artifact_sha256s):
        raise ValueError("locked combined artifact SHA-256 mismatch")
    if (
        portfolio["input"].get("train_only") is not True
        or portfolio["input"].get("top40_public_only") is not True
        or portfolio["input"].get("validation_or_sealed_used") is not False
        or library.get("deduplication_mode") != "unit_only"
        or int(library.get("block_count", -1)) != len(blocks)
    ):
        raise ValueError("dynamic cold input boundary is invalid")

    prototype_anchors = np.asarray(prototypes["anchors"])
    block_by_id = {str(row["block_id"]): index for index, row in enumerate(blocks)}
    prototype_map = dict(library["_prototype_rows_by_block"])
    declared_active = {
        str(row["block_id"])
        for rows in (library.get("active_representatives") or {}).values()
        for row in rows
    }
    anchors: dict[int, dict[str, Any]] = {}
    seen_cold: list[int] = []
    for anchor in continuation.ANCHORS:
        cold = list(map(
            int, portfolio["per_anchor"][str(anchor)]["cold"]["block_indices"],
        ))
        prototype_rows = np.flatnonzero(prototype_anchors == anchor)
        if (
            not cold or not len(prototype_rows)
            or any(int(blocks[index]["anchor"]) != anchor for index in cold)
        ):
            raise ValueError(f"invalid cold anchor index: {anchor}")
        rows = {
            str(blocks[index]["block_id"]): offline._selection_ref(
                index, blocks[index], prototype_map[str(blocks[index]["block_id"])],
                library, {"kind": "dynamic_cold_candidate"},
            )
            for index in cold
        }
        anchors[anchor] = {
            "block_indices": np.asarray(cold, np.int64),
            "prototype_rows": prototype_rows.astype(np.int64),
            "prototype_block_ids": np.asarray(
                prototypes["block_ids"][prototype_rows],
            ),
            "scale": retrieval._robust_scale(
                prototypes["entry_contracts"][prototype_rows],
            ),
            "rows_by_block": rows,
        }
        seen_cold.extend(cold)
    if sorted(seen_cold) != list(range(len(blocks))):
        raise ValueError("cold tiers do not cover every unit path exactly once")
    return {
        "portfolio": portfolio,
        "library": library,
        "blocks": blocks,
        "vectors": vectors,
        "prototypes": prototypes,
        "provenance": provenance,
        "route_refs": route_refs,
        "portfolio_sha256": portfolio_sha,
        "library_sha256": expected_library_sha256,
        "artifact_sha256s": actual_artifacts,
        "block_by_id": block_by_id,
        "declared_active": declared_active,
        "quality_available": bool(
            portfolio["selection"]["historical_quality"]["available"]
        ),
        "anchors": anchors,
        "initialization_ms": (time.perf_counter() - started) * 1000.0,
    }


def select_dynamic_cold(
    index: Mapping[str, Any], *, query: np.ndarray, query_layout: np.ndarray,
    query_mask: np.ndarray, anchor: int, minimum_cash_reserve: float = 0.0,
    top_limit: int = TOP_LIMIT, active_limit: int = ACTIVE_LIMIT,
) -> dict[str, Any]:
    """FP32 cold retrieval with cached scales and existing overlay diagnostics."""

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

    # ponytail: reuse the audited overlay engine; specialize only if the 100 ms
    # end-to-end gate fails, because duplicating its capital/lineage logic is riskier.
    diagnosed = runtime.select_runtime_active(
        query=query, query_layout=query_layout, query_mask=query_mask,
        anchor=anchor,
        hot_rows=[cached["rows_by_block"][block_id] for block_id in hot_ids],
        blocks=index["blocks"], prototypes=prototypes,
        provenance=index["provenance"], route_refs=index["route_refs"],
        active_limit=len(hot_ids),
        minimum_cash_reserve=minimum_cash_reserve,
        capital_gate_mode="soft_report_only",
    )
    diagnosed_ids = [str(row["block_id"]) for row in diagnosed["selected"]]
    if diagnosed_ids != hot_ids or any(
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


def build_query_panel(index: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze outcome-independent exact/near/interpolated train/public queries."""

    queries = []
    prototypes = index["prototypes"]
    for anchor in continuation.ANCHORS:
        cached = index["anchors"][anchor]
        candidates = sorted(
            map(int, cached["prototype_rows"]),
            key=lambda row: _sha256_bytes(_canonical_bytes({
                "seed": PANEL_SEED, "anchor": anchor,
                "provenance_id": str(prototypes["provenance_ids"][row]),
                "block_id": str(prototypes["block_ids"][row]),
            })),
        )
        chosen = []
        used_blocks = set()
        for row in candidates:
            block_id = str(prototypes["block_ids"][row])
            if block_id not in used_blocks:
                chosen.append(row)
                used_blocks.add(block_id)
            if len(chosen) == PANEL_PER_KIND_PER_ANCHOR * 2:
                break
        if len(chosen) != PANEL_PER_KIND_PER_ANCHOR * 2:
            raise ValueError(f"anchor {anchor} lacks distinct panel sources")
        for offset, row in enumerate(chosen[:PANEL_PER_KIND_PER_ANCHOR]):
            partner = chosen[PANEL_PER_KIND_PER_ANCHOR + offset]
            contract = np.asarray(prototypes["entry_contracts"][row], np.float32)
            layout = np.asarray(prototypes["layouts"][row], np.uint8)
            mask = np.asarray(prototypes["unlocked_masks"][row], np.uint8)
            base = {
                "anchor": anchor,
                "source_provenance_ids": [
                    str(prototypes["provenance_ids"][row]),
                ],
                "source_block_ids": [str(prototypes["block_ids"][row])],
            }
            queries.append({
                **base, "query_id": f"exact-{anchor}-{offset}", "kind": "exact",
                "contract": contract.tolist(), "layout": layout.tolist(),
                "mask": mask.tolist(),
            })
            near = contract.copy()
            digest = hashlib.sha256(
                f"{PANEL_SEED}|near|{anchor}|{offset}".encode("ascii"),
            ).digest()
            feature = int(retrieval.STATE_INDICES[int(digest[0]) % len(retrieval.STATE_INDICES)])
            sign = 1.0 if digest[1] & 1 else -1.0
            near[feature] += sign * 0.25 * float(cached["scale"][feature])
            queries.append({
                **base, "query_id": f"near-{anchor}-{offset}", "kind": "near",
                "transform": {"feature": feature, "robust_scale_delta": sign * 0.25},
                "contract": near.tolist(), "layout": layout.tolist(),
                "mask": mask.tolist(),
            })
            other = np.asarray(prototypes["entry_contracts"][partner], np.float32)
            unseen = contract.copy()
            unseen[retrieval.STATE_INDICES] = 0.5 * (
                contract[retrieval.STATE_INDICES] + other[retrieval.STATE_INDICES]
            )
            unseen[retrieval.PLAN_INDICES] = 0.5 * (
                contract[retrieval.PLAN_INDICES] + other[retrieval.PLAN_INDICES]
            )
            queries.append({
                "query_id": f"unseen-{anchor}-{offset}", "kind": "unseen",
                "anchor": anchor,
                "source_provenance_ids": [
                    str(prototypes["provenance_ids"][row]),
                    str(prototypes["provenance_ids"][partner]),
                ],
                "source_block_ids": [
                    str(prototypes["block_ids"][row]),
                    str(prototypes["block_ids"][partner]),
                ],
                "transform": "non_market_contract_midpoint",
                "contract": unseen.tolist(), "layout": layout.tolist(),
                "mask": np.asarray(prototypes["unlocked_masks"][partner], np.uint8).tolist(),
            })
    panel = {
        "schema": PANEL_SCHEMA,
        "seed": PANEL_SEED,
        "selection": "lowest SHA-256 per anchor with distinct source unit paths",
        "source_scope": "frozen train/public combined contract prototypes only",
        "outcome_used_for_panel_selection": False,
        "counts": {
            "total": len(queries),
            "per_kind": {
                kind: sum(row["kind"] == kind for row in queries)
                for kind in ("exact", "near", "unseen")
            },
        },
        "queries": queries,
    }
    panel["content_sha256"] = _sha256_bytes(_canonical_bytes(panel))
    return panel


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["block_id"]) for row in rows]


def evaluate_panel(
    index: Mapping[str, Any], panel: Mapping[str, Any],
    initialization_ms: float, incremental_rss_mib: float,
) -> dict[str, Any]:
    queries = list(panel["queries"])
    first = queries[0]
    select_dynamic_cold(
        index, query=np.asarray(first["contract"], np.float32),
        query_layout=np.asarray(first["layout"], np.uint8),
        query_mask=np.asarray(first["mask"], np.uint8), anchor=int(first["anchor"]),
    )
    rows = []
    parity = []
    market_invariant = []
    canonical_latency = {}
    for raw in queries:
        query = np.asarray(raw["contract"], np.float32)
        layout = np.asarray(raw["layout"], np.uint8)
        mask = np.asarray(raw["mask"], np.uint8)
        anchor = int(raw["anchor"])
        dynamic = select_dynamic_cold(
            index, query=query, query_layout=layout, query_mask=mask, anchor=anchor,
        )
        static = runtime.select_runtime_active(
            query=query, query_layout=layout, query_mask=mask, anchor=anchor,
            hot_rows=runtime._hot_refs(
                index["portfolio"], index["library"], index["blocks"], anchor,
            ),
            blocks=index["blocks"], prototypes=index["prototypes"],
            provenance=index["provenance"], route_refs=index["route_refs"],
            active_limit=ACTIVE_LIMIT,
        )
        dynamic_distance = float(dynamic["active"][0]["distance"])
        static_distance = float(static["selected"][0]["distance"])
        rows.append({
            "query_id": raw["query_id"], "kind": raw["kind"], "anchor": anchor,
            "dynamic_nearest_distance": dynamic_distance,
            "static_nearest_distance": static_distance,
            "dynamic_nonworse": dynamic_distance <= static_distance + 1e-6,
            "dynamic_strictly_better": dynamic_distance < static_distance - 1e-9,
            "global_nearest_in_active": (
                dynamic["active"][0]["block_id"]
                == dynamic["dynamic_hot"][0]["block_id"]
            ),
            "cold_scoring_ms": dynamic["cold_scoring_ms"],
            "end_to_end_ms": dynamic["end_to_end_ms"],
        })
        if raw["kind"] == "exact" and anchor not in canonical_latency:
            canonical_latency[anchor] = float(dynamic["end_to_end_ms"])
            reference = runtime.select_runtime_active(
                query=query, query_layout=layout, query_mask=mask, anchor=anchor,
                hot_rows=list(index["anchors"][anchor]["rows_by_block"].values()),
                blocks=index["blocks"], prototypes=index["prototypes"],
                provenance=index["provenance"], route_refs=index["route_refs"],
                active_limit=TOP_LIMIT,
            )
            parity.append(
                _ids(reference["selected"]) == _ids(dynamic["dynamic_hot"])
                and all(
                    abs(float(left["distance"]) - float(right["distance"])) <= 1e-6
                    for left, right in zip(reference["selected"], dynamic["dynamic_hot"])
                )
            )
            market = query.copy()
            market[continuation.MARKET_START:continuation.MARKET_STOP] += 1_000_000.0
            changed = select_dynamic_cold(
                index, query=market, query_layout=layout, query_mask=mask,
                anchor=anchor,
            )
            market_invariant.append(
                _ids(changed["dynamic_hot"]) == _ids(dynamic["dynamic_hot"])
                and all(
                    float(left["distance"]) == float(right["distance"])
                    for left, right in zip(changed["dynamic_hot"], dynamic["dynamic_hot"])
                )
            )

    near_unseen = [row for row in rows if row["kind"] in {"near", "unseen"}]
    core = [float(row["cold_scoring_ms"]) for row in rows]
    end_to_end = [float(row["end_to_end_ms"]) for row in rows]
    metrics = {
        "query_count": len(rows),
        "reference_top32_parity_rate": sum(parity) / len(parity),
        "market_invariance_rate": sum(market_invariant) / len(market_invariant),
        "near_unseen_nonworse_rate": sum(
            bool(row["dynamic_nonworse"]) for row in near_unseen
        ) / len(near_unseen),
        "near_unseen_strict_improvement_rate": sum(
            bool(row["dynamic_strictly_better"]) for row in near_unseen
        ) / len(near_unseen),
        "global_nearest_in_active_rate": sum(
            bool(row["global_nearest_in_active"]) for row in rows
        ) / len(rows),
        "warm_core_p95_ms": _percentile(core, 95),
        "end_to_end_p95_ms": _percentile(end_to_end, 95),
        "end_to_end_max_ms": max(end_to_end),
        "canonical_21_anchor_total_ms": sum(canonical_latency.values()),
        "initialization_ms": float(initialization_ms),
        "incremental_rss_mib": float(incremental_rss_mib),
    }
    checks = {
        name: (
            value >= threshold
            if name.endswith("_rate") else value <= threshold
        )
        for name, threshold in THRESHOLDS.items()
        for value in [metrics[name]]
    }
    exact_zero = all(
        abs(float(row["dynamic_nearest_distance"])) <= 1e-6
        for row in rows if row["kind"] == "exact"
    )
    checks["exact_query_minimum_distance_zero"] = exact_zero
    return {
        "metrics": metrics,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
    }


def run(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    index = load_cold_index(
        LOCKED_PORTFOLIO_ROOT, LOCKED_PORTFOLIO_SHA256,
        LOCKED_LIBRARY_SHA256, LOCKED_ARTIFACT_SHA256S,
    )
    rss_after = process.memory_info().rss
    panel = build_query_panel(index)
    evaluation = evaluate_panel(
        index, panel, index["initialization_ms"],
        max(0.0, (rss_after - rss_before) / (1024.0 ** 2)),
    )
    panel_payload = zlib.compress(_canonical_bytes(panel), level=9)
    script_sha = _sha256_bytes(Path(__file__).read_bytes())
    result = {
        "schema": SCHEMA,
        "status": "passed" if evaluation["passed"] else "failed",
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3],
            ).as_posix(),
            "sha256": script_sha,
        },
        "input": {
            "portfolio_root": str(LOCKED_PORTFOLIO_ROOT.resolve()),
            "portfolio_manifest_sha256": LOCKED_PORTFOLIO_SHA256,
            "library_manifest_sha256": LOCKED_LIBRARY_SHA256,
            "artifact_sha256s": dict(LOCKED_ARTIFACT_SHA256S),
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_used": False,
        },
        "config": {
            "top_limit": TOP_LIMIT, "active_limit": ACTIVE_LIMIT,
            "capital_gate_mode": "soft_report_only",
            "fp_precision": "float32",
            "scale_cache": "precomputed once per fixed anchor",
            "market_environment_distance_feature_indices": list(range(86, 104)),
        },
        "query_panel": {
            "schema": PANEL_SCHEMA,
            "file": "query_panel.json.zlib",
            "file_sha256": _sha256_bytes(panel_payload),
            "content_sha256": panel["content_sha256"],
            "counts": panel["counts"],
            "source_scope": panel["source_scope"],
            "outcome_used_for_panel_selection": False,
        },
        "evaluation": evaluation,
        "storage": {
            "actions_copied": False,
            "contracts_copied": False,
            "output_contains": "frozen query panel, refs/metrics/reasons, and hashes",
        },
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.", dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        (staging / "query_panel.json.zlib").write_bytes(panel_payload)
        (staging / "result_manifest.json").write_bytes(payload)
        manifest_sha = _sha256_bytes(payload)
        (staging / "result_manifest.json.sha256").write_text(
            f"{manifest_sha}  result_manifest.json\n", encoding="ascii",
        )
        if output_root.exists():
            raise FileExistsError(f"output appeared during run: {output_root}")
        staging.replace(output_root)
    result["manifest_sha256"] = manifest_sha
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv).output_root)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "manifest_sha256": result["manifest_sha256"],
        "query_count": result["evaluation"]["metrics"]["query_count"],
        "near_unseen_strict_improvement_rate": result["evaluation"]["metrics"][
            "near_unseen_strict_improvement_rate"
        ],
        "output_root": str(parse_args(argv).output_root.resolve()),
    }, ensure_ascii=True, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
