#!/usr/bin/env python3
"""Retrieve a state-conditioned active path portfolio with capital diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_offline_block_portfolio_v1 as offline  # noqa: E402
import run_contract_retrieval_oracle_ablation_v1 as retrieval  # noqa: E402
import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402


SCHEMA = "runtime-active-portfolio-v1"
CAPITAL_GATE_MODES = ("soft_report_only", "hard_current_cash")
CAPITAL_REQUIREMENT_INDICES = np.asarray([
    continuation.PLAN_START + 1,
    continuation.PLAN_START + 7,
    continuation.PLAN_START + 13,
], np.int64)


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


def _load_portfolio(
    root: Path, expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]],
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]],
    dict[str, dict[str, Any]], str,
]:
    root = root.resolve()
    raw = (root / "portfolio_manifest.json").read_bytes()
    digest = _sha256_bytes(raw)
    if digest != _expected_sha256(
        expected_manifest_sha256, "offline portfolio manifest SHA-256",
    ):
        raise ValueError("offline portfolio manifest SHA-256 mismatch")
    sidecar = (root / "portfolio_manifest.json.sha256").read_text(
        encoding="ascii",
    ).split()
    if not sidecar or sidecar[0].lower() != digest:
        raise ValueError("offline portfolio manifest SHA-256 sidecar mismatch")
    portfolio = json.loads(raw)
    if portfolio.get("schema") != offline.SCHEMA or portfolio.get("status") != "complete":
        raise ValueError("invalid or incomplete offline block portfolio")
    portfolio_input = dict(portfolio.get("input") or {})
    if (
        portfolio_input.get("train_only") is not True
        or portfolio_input.get("top40_public_only") is not True
        or portfolio_input.get("validation_or_sealed_used") is not False
        or portfolio_input.get("deduplication_mode") != "unit_only"
        or dict(portfolio.get("selection") or {}).get("runtime_state_conditioned") is not False
    ):
        raise ValueError("offline portfolio provenance/runtime contract is invalid")
    core = {
        "schema": offline.SCHEMA,
        "implementation_sha256": str(portfolio["implementation"]["sha256"]),
        "library_manifest_sha256": str(portfolio_input["library_manifest_sha256"]),
        "artifact_sha256s": dict(portfolio_input["artifact_sha256s"]),
        "config": dict(portfolio["config"]),
        "quality_mode": str(portfolio["selection"]["historical_quality"]["mode"]),
        "per_anchor": dict(portfolio["per_anchor"]),
    }
    content_sha = _sha256_bytes(_canonical_bytes(core))
    if (
        content_sha != str(portfolio.get("content_sha256", ""))
        or str(portfolio.get("portfolio_id", "")) != f"OBP1_{content_sha[:24]}"
    ):
        raise ValueError("offline portfolio content hash contract mismatch")

    library_root = Path(str(portfolio_input["library_root"])).resolve()
    (
        library, blocks, vectors, prototypes, provenance, _, library_sha,
    ) = offline._load_library(
        library_root, str(portfolio_input["library_manifest_sha256"]),
    )
    actual_artifacts = {
        "actions": str(library["actions_sha256"]),
        "vectors": str(library["vectors_sha256"]),
        "contract_prototypes": str(library["contract_prototypes_sha256"]),
    }
    if actual_artifacts != dict(portfolio_input["artifact_sha256s"]):
        raise ValueError("offline portfolio/library artifact hashes disagree")

    route_refs: dict[str, dict[str, Any]] = {}
    for prepared_raw in library.get("prepared_roots", ()) or ():
        prepared = dict(prepared_raw)
        candidate_path = Path(str(prepared["path"])).resolve() / "candidate_manifest.json"
        candidate_raw = candidate_path.read_bytes()
        candidate_sha = _sha256_bytes(candidate_raw)
        if candidate_sha != str(prepared["candidate_manifest_sha256"]):
            raise ValueError("prepared route manifest SHA-256 mismatch")
        candidate = json.loads(candidate_raw)
        for route_raw in candidate.get("routes", ()) or ():
            route = dict(route_raw)
            route_id = str(route.get("route_id") or "")
            if not route_id:
                raise ValueError("prepared route has no route id")
            overlay = dict(route.get("market_overlay_ref") or {})
            full_sha = _expected_sha256(
                route.get("full_tape_sha256"), f"route full tape {route_id}",
            )
            if overlay.get("key") != full_sha:
                raise ValueError(f"route market-overlay reference mismatch: {route_id}")
            reference = {
                "candidate_manifest": str(candidate_path),
                "candidate_manifest_sha256": candidate_sha,
                "route_id": route_id,
                "field": "market_overlay_ref",
                "execution_id": str(route.get("execution_id") or ""),
                "full_tape_sha256": full_sha,
            }
            route_provenance = list(map(
                str, route.get("provenance_ids", ()) or (),
            ))
            if not route_provenance:
                raise ValueError(f"prepared route has no provenance refs: {route_id}")
            for provenance_id in route_provenance:
                if provenance_id in route_refs:
                    raise ValueError(
                        f"duplicate prepared route provenance: {provenance_id}"
                    )
                route_refs[provenance_id] = reference
    for provenance_id, source in provenance.items():
        route_id = str(source.get("route_id") or "")
        route = route_refs.get(provenance_id)
        if (
            route is None
            or route["route_id"] != route_id
            or route["execution_id"] != str(source.get("execution_id") or "")
            or route["full_tape_sha256"] != str(source.get("full_tape_sha256") or "")
        ):
            raise ValueError(f"prepared provenance/market variant mismatch: {provenance_id}")
    return (
        portfolio, library, blocks, vectors, prototypes, provenance, route_refs,
        digest,
    )


def _hot_refs(
    portfolio: Mapping[str, Any], library: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]], anchor: int,
) -> list[dict[str, Any]]:
    if anchor not in continuation.ANCHORS:
        raise ValueError("runtime anchor is outside the fixed 21 boundaries")
    tiers = dict((portfolio.get("per_anchor") or {}).get(str(anchor)) or {})
    cold = dict(tiers.get("cold") or {})
    hot = dict(tiers.get("hot") or {})
    cold_indices = list(map(int, cold.get("block_indices", ()) or ()))
    rows = [dict(row) for row in hot.get("blocks", ()) or ()]
    if (
        int(cold.get("count", -1)) != len(cold_indices)
        or int(hot.get("count", -1)) != len(rows)
        or len(rows) > int(hot.get("limit", -1))
        or len({str(row.get("block_id") or "") for row in rows}) != len(rows)
    ):
        raise ValueError("offline portfolio tier counts are invalid")
    cold_set = set(cold_indices)
    prototype_map = dict(library["_prototype_rows_by_block"])
    for row in rows:
        block_id = str(row["block_id"])
        block_ref = dict(row.get("block_ref") or {})
        index = int(block_ref.get("block_index", -1))
        if (
            block_ref.get("file") != "block_library_manifest.json"
            or index not in cold_set
            or index < 0 or index >= len(blocks)
            or str(blocks[index]["block_id"]) != block_id
            or int(blocks[index]["anchor"]) != anchor
        ):
            raise ValueError(f"invalid hot block reference: {block_id}")
        expected_prototypes = list(map(int, prototype_map[block_id]))
        prototype_ref = dict(row.get("contract_prototype_ref") or {})
        if (
            prototype_ref.get("file") != library["contract_prototypes_file"]
            or list(map(int, prototype_ref.get("rows", ()) or ()))
            != expected_prototypes
        ):
            raise ValueError(f"invalid hot prototype reference: {block_id}")
        variant_ref = dict(row.get("market_variant_ref") or {})
        if (
            variant_ref.get("file") != "block_library_manifest.json"
            or int(variant_ref.get("block_index", -1)) != index
            or variant_ref.get("field") != "source_full_action_sha256s"
            or variant_ref.get("slot_semantics")
            != "variants_of_one_unit_path_not_portfolio_candidates"
        ):
            raise ValueError(f"invalid hot market-variant reference: {block_id}")
    return rows


def _variant_ref(
    provenance_ids: Sequence[str], route_refs: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    for value in provenance_ids:
        if value not in provenance:
            raise ValueError(f"market variant provenance is unavailable: {value}")
    references = [route_refs[value] for value in provenance_ids]
    route_ids = sorted({str(row["route_id"]) for row in references})
    if len(route_ids) != 1 or any(
        row["execution_id"] != references[0]["execution_id"]
        or row["full_tape_sha256"] != references[0]["full_tape_sha256"]
        for row in references
    ):
        raise ValueError("one market variant maps to multiple prepared routes")
    reference = dict(references[0])
    reference["provenance_ids"] = sorted(map(str, provenance_ids))
    return reference


def select_runtime_active(
    *, query: np.ndarray, query_layout: np.ndarray, query_mask: np.ndarray,
    anchor: int, hot_rows: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]], prototypes: Mapping[str, np.ndarray],
    provenance: Mapping[str, Mapping[str, Any]],
    route_refs: Mapping[str, Mapping[str, Any]], active_limit: int = 8,
    minimum_cash_reserve: float = 0.0,
    capital_gate_mode: str = "soft_report_only",
) -> dict[str, Any]:
    """Pure query-only retrieval using the frozen continuation-v1 distance."""

    query = np.asarray(query, np.float32)
    query_layout = np.asarray(query_layout, np.uint8)
    query_mask = np.asarray(query_mask, np.uint8)
    if (
        query.shape != (continuation.FEATURE_DIM,)
        or query_layout.shape != (continuation.LAYOUT_SIZE,)
        or query_mask.shape != (len(continuation.QUADRANTS),)
        or not np.isfinite(query).all()
        or np.any(query_layout > 13)
        or np.any(query_mask > 1)
    ):
        raise ValueError("invalid runtime 147D/layout/mask query")
    if active_limit <= 0:
        raise ValueError("active limit must be positive")
    reserve = float(minimum_cash_reserve)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("minimum cash reserve must be a finite non-negative input")
    if capital_gate_mode not in CAPITAL_GATE_MODES:
        raise ValueError("invalid capital gate mode")
    self_money = float(query[0])
    if not math.isfinite(self_money):
        raise ValueError("runtime self money is non-finite")

    hot_ids = [str(row["block_id"]) for row in hot_rows]
    if len(hot_ids) != len(set(hot_ids)):
        raise ValueError("hot portfolio repeats a unit path")
    block_by_id = {str(row["block_id"]): row for row in blocks}
    if any(
        block_id not in block_by_id
        or int(block_by_id[block_id]["anchor"]) != int(anchor)
        for block_id in hot_ids
    ):
        raise ValueError("hot portfolio references a block outside the runtime anchor")
    hot_set = set(hot_ids)
    all_anchor_rows = np.flatnonzero(np.asarray(prototypes["anchors"]) == int(anchor))
    hot_prototype_rows = np.asarray([
        index for index in all_anchor_rows
        if str(prototypes["block_ids"][index]) in hot_set
    ], np.int64)
    if not len(hot_prototype_rows):
        raise ValueError("hot portfolio has no contract prototypes at the anchor")
    scale = retrieval._robust_scale(prototypes["entry_contracts"][all_anchor_rows])
    distances = retrieval.contract_distances(
        query, query_layout, query_mask,
        prototypes["entry_contracts"][hot_prototype_rows],
        prototypes["layouts"][hot_prototype_rows],
        prototypes["unlocked_masks"][hot_prototype_rows], scale,
    )

    grouped: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        block_id: {} for block_id in hot_ids
    }
    for local, prototype_row in enumerate(hot_prototype_rows):
        block_id = str(prototypes["block_ids"][prototype_row])
        provenance_id = str(prototypes["provenance_ids"][prototype_row])
        source = provenance.get(provenance_id)
        if source is None:
            raise ValueError(f"prototype provenance is unavailable: {provenance_id}")
        execution_id = str(source.get("execution_id") or "")
        full_sha = _expected_sha256(
            source.get("full_tape_sha256"), f"prototype variant {provenance_id}",
        )
        requirements = np.asarray(
            prototypes["entry_contracts"][prototype_row, CAPITAL_REQUIREMENT_INDICES],
            np.float64,
        )
        if not np.isfinite(requirements).all() or np.any(requirements < 0):
            raise ValueError(f"invalid fixed-capital requirements: {provenance_id}")
        required = float(np.max(requirements))
        headroom = self_money - required
        row = {
            "prototype_row": int(prototype_row),
            "provenance_id": provenance_id,
            "distance": float(distances[local]),
            "required_fixed_capital": required,
            "capital_headroom": headroom,
            "hard_current_cash_pass": headroom >= reserve,
            "capital_shortfall": max(0.0, reserve - headroom),
        }
        grouped[block_id].setdefault((execution_id, full_sha), []).append(row)

    eligible_variant_candidates: list[dict[str, Any]] = []
    rejected_blocks = []
    block_audit: dict[str, dict[str, Any]] = {}
    hot_by_id = {str(row["block_id"]): dict(row) for row in hot_rows}
    for block_id in hot_ids:
        retrieval_variants = []
        rejected_variants = []
        for _, rows in sorted(grouped[block_id].items()):
            provenance_ids = [str(row["provenance_id"]) for row in rows]
            reference = _variant_ref(provenance_ids, route_refs, provenance)
            eligible_rows = (
                [row for row in rows if bool(row["hard_current_cash_pass"])]
                if capital_gate_mode == "hard_current_cash"
                else rows
            )
            if eligible_rows:
                best = min(
                    eligible_rows,
                    key=lambda row: (float(row["distance"]), int(row["prototype_row"])),
                )
                variant = {
                    "market_variant_ref": reference,
                    "best_prototype_ref": {
                        "file": str(
                            hot_by_id[block_id]["contract_prototype_ref"]["file"]
                        ),
                        "row": int(best["prototype_row"]),
                    },
                    "distance": float(best["distance"]),
                    "capital_shortfall_diagnostic": float(
                        best["capital_shortfall"]
                    ),
                    "reason": (
                        "hard_current_cash_gate_passed"
                        if capital_gate_mode == "hard_current_cash"
                        else "capital_report_only_not_a_feasibility_proof"
                    ),
                }
                retrieval_variants.append(variant)
                eligible_variant_candidates.append({
                    "block_id": block_id,
                    "distance": float(best["distance"]),
                    "prototype_row": int(best["prototype_row"]),
                })
            else:
                closest = min(
                    rows,
                    key=lambda row: (
                        float(row["capital_shortfall"]), float(row["distance"]),
                        int(row["prototype_row"]),
                    ),
                )
                rejected_variants.append({
                    "market_variant_ref": reference,
                    "best_prototype_ref": {
                        "file": str(
                            hot_by_id[block_id]["contract_prototype_ref"]["file"]
                        ),
                        "row": int(closest["prototype_row"]),
                    },
                    "distance": float(closest["distance"]),
                    "reason": "insufficient_current_cash_for_fixed_commitments",
                    "required_fixed_capital": float(closest["required_fixed_capital"]),
                    "minimum_cash_reserve": reserve,
                    "capital_shortfall": float(closest["capital_shortfall"]),
                })
        if retrieval_variants:
            best_variant = min(
                retrieval_variants,
                key=lambda row: (
                    float(row["distance"]),
                    str(row["market_variant_ref"]["full_tape_sha256"]),
                ),
            )
            block_audit[block_id] = {
                "block_ref": dict(hot_by_id[block_id]["block_ref"]),
                "action_ref": dict(hot_by_id[block_id]["action_ref"]),
                "distance": float(best_variant["distance"]),
                "best_prototype_ref": dict(best_variant["best_prototype_ref"]),
                "market_variants": retrieval_variants,
                "rejected_market_variants": rejected_variants,
            }
        else:
            rejected_blocks.append({
                "block_id": block_id,
                "block_ref": dict(hot_by_id[block_id]["block_ref"]),
                "reason": "all_market_variants_fail_hard_current_cash_gate",
                "rejected_market_variants": rejected_variants,
            })

    distinct_ids, inspected, best_distances = retrieval.nearest_distinct_blocks(
        [str(row["block_id"]) for row in eligible_variant_candidates],
        [float(row["distance"]) for row in eligible_variant_candidates],
        active_limit,
    ) if eligible_variant_candidates else ([], 0, {})
    selected = []
    for rank, block_id in enumerate(distinct_ids, 1):
        row = dict(block_audit[block_id])
        row.update({
            "rank": rank,
            "block_id": block_id,
            "distance": float(best_distances[block_id]),
            "reason": (
                "nearest_hard_current_cash_eligible_distinct_unit_path"
                if capital_gate_mode == "hard_current_cash"
                else "nearest_distinct_unit_path_capital_report_only"
            ),
        })
        selected.append(row)
    selected_set = set(distinct_ids)
    below_limit = []
    for block_id, row in sorted(
        block_audit.items(), key=lambda item: (float(item[1]["distance"]), item[0]),
    ):
        if block_id not in selected_set:
            below_limit.append({
                "block_id": block_id,
                "block_ref": dict(row["block_ref"]),
                "distance": float(row["distance"]),
                "reason": "eligible_but_ranked_below_active_limit",
            })
    return {
        "anchor": int(anchor),
        "hot_path_count": len(hot_ids),
        "hot_prototype_count": len(hot_prototype_rows),
        "eligible_path_count": len(block_audit),
        "hard_current_cash_rejected_path_count": len(rejected_blocks),
        "active_limit": int(active_limit),
        "active_count": len(selected),
        "minimum_cash_reserve": reserve,
        "eligible_variant_candidates_inspected": int(inspected),
        "selected": selected,
        "rejected": rejected_blocks,
        "below_active_limit": below_limit,
        "distance": (
            "existing continuation-v1 robust-scaled L1 state/plan plus layout+mask Hamming"
        ),
        "mask_semantics": "existing soft Hamming term; not a new hard subset gate",
        "capital_gate_mode": capital_gate_mode,
        "capital_feasibility_proven": False,
        "capital_evaluation": (
            "hard_current_cash diagnostic: self_money >= max(fixed commitment "
            "requirements at 24/48/72) + explicit minimum_cash_reserve"
            if capital_gate_mode == "hard_current_cash"
            else "soft report only: capital shortfall never rejects a path"
        ),
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


def _load_query(path: Path, expected_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    raw = path.resolve().read_bytes()
    digest = _sha256_bytes(raw)
    if digest != _expected_sha256(expected_sha256, "runtime query SHA-256"):
        raise ValueError("runtime query SHA-256 mismatch")
    try:
        with np.load(path.resolve(), allow_pickle=False) as archive:
            if set(archive.files) != {"contract", "layout", "mask"}:
                raise ValueError("runtime query NPZ array set changed")
            query = np.asarray(archive["contract"], np.float32)
            layout = np.asarray(archive["layout"], np.uint8)
            mask = np.asarray(archive["mask"], np.uint8)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("runtime query"):
            raise
        raise ValueError("invalid runtime query NPZ") from exc
    return query, layout, mask, digest


def retrieve(
    *, portfolio_root: Path, portfolio_manifest_sha256: str,
    query_path: Path, query_sha256: str, output_root: Path, anchor: int,
    active_limit: int = 8, minimum_cash_reserve: float = 0.0,
    capital_gate_mode: str = "soft_report_only",
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    (
        portfolio, library, blocks, _, prototypes, provenance, route_refs,
        portfolio_sha,
    ) = _load_portfolio(portfolio_root, portfolio_manifest_sha256)
    hot = _hot_refs(portfolio, library, blocks, int(anchor))
    query, layout, mask, query_sha = _load_query(query_path, query_sha256)
    selection = select_runtime_active(
        query=query, query_layout=layout, query_mask=mask, anchor=int(anchor),
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
        active_limit=active_limit, minimum_cash_reserve=minimum_cash_reserve,
        capital_gate_mode=capital_gate_mode,
    )
    implementation_sha = _sha256_bytes(Path(__file__).read_bytes())
    core = {
        "schema": SCHEMA,
        "implementation_sha256": implementation_sha,
        "portfolio_manifest_sha256": portfolio_sha,
        "library_manifest_sha256": str(portfolio["input"]["library_manifest_sha256"]),
        "query_sha256": query_sha,
        "anchor": int(anchor),
        "active_limit": int(active_limit),
        "minimum_cash_reserve": float(minimum_cash_reserve),
        "capital_gate_mode": capital_gate_mode,
        "selection": selection,
    }
    content_sha = _sha256_bytes(_canonical_bytes(core))
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "retrieval_id": f"RAP1_{content_sha[:24]}",
        "content_sha256": content_sha,
        "runtime_state_conditioned": True,
        "capital_feasibility_proven": False,
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3]
            ).as_posix(),
            "sha256": implementation_sha,
        },
        "input": {
            "portfolio_root": str(portfolio_root.resolve()),
            "portfolio_manifest_sha256": portfolio_sha,
            "library_manifest_sha256": str(portfolio["input"]["library_manifest_sha256"]),
            "library_artifact_sha256s": dict(portfolio["input"]["artifact_sha256s"]),
            "query": str(query_path.resolve()),
            "query_sha256": query_sha,
            "query_payload_copied": False,
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_used": False,
        },
        "config": {
            "anchor": int(anchor),
            "active_limit": int(active_limit),
            "minimum_cash_reserve": float(minimum_cash_reserve),
            "capital_gate_mode": capital_gate_mode,
        },
        "selection": selection,
        "storage": {
            "actions_copied": False,
            "contracts_copied": False,
            "query_copied": False,
            "output_contains": "refs, distances, rejection reasons, and hashes only",
        },
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.", dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary)
        (staging / "runtime_active_manifest.json").write_bytes(payload)
        digest = _sha256_bytes(payload)
        (staging / "runtime_active_manifest.json.sha256").write_text(
            f"{digest}  runtime_active_manifest.json\n", encoding="ascii",
        )
        if output_root.exists():
            raise FileExistsError(f"output directory appeared during run: {output_root}")
        staging.replace(output_root)
    output["manifest_sha256"] = digest
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-root", type=Path, required=True)
    parser.add_argument("--portfolio-manifest-sha256", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--active-limit", type=int, default=8)
    parser.add_argument("--minimum-cash-reserve", type=float, default=0.0)
    parser.add_argument(
        "--capital-gate-mode", choices=CAPITAL_GATE_MODES,
        default="soft_report_only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = retrieve(
        portfolio_root=args.portfolio_root,
        portfolio_manifest_sha256=args.portfolio_manifest_sha256,
        query_path=args.query,
        query_sha256=args.query_sha256,
        output_root=args.output_root,
        anchor=args.anchor,
        active_limit=args.active_limit,
        minimum_cash_reserve=args.minimum_cash_reserve,
        capital_gate_mode=args.capital_gate_mode,
    )
    selection = manifest["selection"]
    print(json.dumps({
        "schema": manifest["schema"],
        "retrieval_id": manifest["retrieval_id"],
        "anchor": selection["anchor"],
        "hot": selection["hot_path_count"],
        "eligible": selection["eligible_path_count"],
        "active": selection["active_count"],
        "hard_current_cash_rejected": selection[
            "hard_current_cash_rejected_path_count"
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
