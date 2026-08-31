#!/usr/bin/env python3
"""Locked correction and one-shot runner for DYNAMIC-COLD-RETRIEVAL-MVP-v0."""

from __future__ import annotations

import argparse
import json
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import retrieve_runtime_active_portfolio_v1 as runtime  # noqa: E402
import run_block_mvp_continuation_multitail_v1 as continuation  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0 as draft  # noqa: E402


SCHEMA = draft.SCHEMA
DRAFT_SHA256 = "502b76bf2ad9f5c74ea7e30ce08ecdff527e2c368ba65ad6184d67a95bc50527"
THRESHOLDS = {
    "reference_top32_parity_rate": 1.0,
    "reference_max_distance_absolute_error": 1e-6,
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


def _assert_draft_lock() -> None:
    if draft._sha256_bytes(Path(draft.__file__).read_bytes()) != DRAFT_SHA256:
        raise ValueError("dynamic cold draft dependency SHA-256 mismatch")


def load_cold_index(
    portfolio_root: Path, portfolio_manifest_sha256: str,
    expected_library_sha256: str, expected_artifact_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    _assert_draft_lock()
    return draft.load_cold_index(
        portfolio_root, portfolio_manifest_sha256,
        expected_library_sha256, expected_artifact_sha256s,
    )


select_dynamic_cold = draft.select_dynamic_cold
build_query_panel = draft.build_query_panel


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["block_id"]) for row in rows]


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), percentile))


def evaluate_panel(
    index: Mapping[str, Any], panel: Mapping[str, Any],
    initialization_ms: float, incremental_rss_mib: float,
) -> dict[str, Any]:
    """Evaluate the frozen panel without changing any preregistered threshold."""

    queries = list(panel["queries"])
    first = queries[0]
    select_dynamic_cold(
        index, query=np.asarray(first["contract"], np.float32),
        query_layout=np.asarray(first["layout"], np.uint8),
        query_mask=np.asarray(first["mask"], np.uint8), anchor=int(first["anchor"]),
    )
    rows = []
    parity = []
    parity_errors = []
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
            active_limit=draft.ACTIVE_LIMIT,
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
                active_limit=draft.TOP_LIMIT,
            )
            errors = [
                abs(float(left["distance"]) - float(right["distance"]))
                for left, right in zip(reference["selected"], dynamic["dynamic_hot"])
            ]
            parity_errors.extend(errors)
            parity.append(
                _ids(reference["selected"]) == _ids(dynamic["dynamic_hot"])
                and all(value <= 1e-6 for value in errors)
            )
            changed_query = query.copy()
            changed_query[
                continuation.MARKET_START:continuation.MARKET_STOP
            ] += 1_000_000.0
            changed = select_dynamic_cold(
                index, query=changed_query, query_layout=layout,
                query_mask=mask, anchor=anchor,
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
        "reference_max_distance_absolute_error": max(parity_errors),
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
            metrics[name] >= threshold
            if name.endswith("_rate") else metrics[name] <= threshold
        )
        for name, threshold in THRESHOLDS.items()
    }
    checks["exact_query_minimum_distance_zero"] = all(
        abs(float(row["dynamic_nearest_distance"])) <= 1e-6
        for row in rows if row["kind"] == "exact"
    )
    return {
        "metrics": metrics,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
    }


def run(output_root: Path) -> dict[str, Any]:
    """Run exactly one locked formal panel and atomically preserve pass or fail."""

    _assert_draft_lock()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    index = load_cold_index(
        draft.LOCKED_PORTFOLIO_ROOT, draft.LOCKED_PORTFOLIO_SHA256,
        draft.LOCKED_LIBRARY_SHA256, draft.LOCKED_ARTIFACT_SHA256S,
    )
    rss_after = process.memory_info().rss
    panel = build_query_panel(index)
    evaluation = evaluate_panel(
        index, panel, index["initialization_ms"],
        max(0.0, (rss_after - rss_before) / (1024.0 ** 2)),
    )
    panel_payload = zlib.compress(draft._canonical_bytes(panel), level=9)
    script_sha = draft._sha256_bytes(Path(__file__).read_bytes())
    result = {
        "schema": SCHEMA,
        "status": "passed" if evaluation["passed"] else "failed",
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3],
            ).as_posix(),
            "sha256": script_sha,
            "locked_draft_dependency": {
                "path": Path(draft.__file__).resolve().relative_to(
                    Path(draft.__file__).resolve().parents[3],
                ).as_posix(),
                "sha256": DRAFT_SHA256,
            },
        },
        "input": {
            "portfolio_root": str(draft.LOCKED_PORTFOLIO_ROOT.resolve()),
            "portfolio_manifest_sha256": draft.LOCKED_PORTFOLIO_SHA256,
            "library_manifest_sha256": draft.LOCKED_LIBRARY_SHA256,
            "artifact_sha256s": dict(draft.LOCKED_ARTIFACT_SHA256S),
            "train_only": True, "top40_public_only": True,
            "validation_or_sealed_used": False,
        },
        "config": {
            "top_limit": draft.TOP_LIMIT, "active_limit": draft.ACTIVE_LIMIT,
            "capital_gate_mode": "soft_report_only", "fp_precision": "float32",
            "scale_cache": "precomputed once per fixed anchor",
            "market_environment_distance_feature_indices": list(range(86, 104)),
        },
        "query_panel": {
            "schema": draft.PANEL_SCHEMA, "file": "query_panel.json.zlib",
            "file_sha256": draft._sha256_bytes(panel_payload),
            "content_sha256": panel["content_sha256"],
            "counts": panel["counts"], "source_scope": panel["source_scope"],
            "outcome_used_for_panel_selection": False,
        },
        "evaluation": evaluation,
        "storage": {
            "actions_copied": False, "contracts_copied": False,
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
        manifest_sha = draft._sha256_bytes(payload)
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
    args = parse_args(argv)
    result = run(args.output_root)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "manifest_sha256": result["manifest_sha256"],
        "query_count": result["evaluation"]["metrics"]["query_count"],
        "near_unseen_strict_improvement_rate": result["evaluation"]["metrics"][
            "near_unseen_strict_improvement_rate"
        ],
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=True, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
