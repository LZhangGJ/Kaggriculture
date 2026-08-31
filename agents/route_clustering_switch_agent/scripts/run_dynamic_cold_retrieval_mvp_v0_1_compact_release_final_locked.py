#!/usr/bin/env python3
"""Release the audited compact selector into one locked 252-query formal."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest import mock

import psutil


SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_dynamic_cold_retrieval_mvp_v0 as draft  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0_1_compact_locked as compact  # noqa: E402
import run_dynamic_cold_retrieval_mvp_v0_locked as v0  # noqa: E402


SCHEMA = "dynamic-cold-retrieval-mvp-v0.1-compact-release"
EXPECTED_RUNNER_NORMALIZED_SHA256 = "6367b14fa9c9bf71bb0eef470cdeddcb9606898df3f770606a543f1323f00755"
EXPECTED_COMPACT_SHA256 = "fdc2369fbf77d9e04627dc461f2d45f3adab8d1f28dccdc2c2e9b441b8c69926"
EXPECTED_CANONICAL_TEST_SHA256 = "be399692316ceea8bd4e98ae7fa6ee9a5b397ef7172b5cbd9da8d645db75d789"
EXPECTED_V0_RESULT_MANIFEST_SHA256 = "843048dfafdb1631864a5b43cbabbee3f5378b49621bdfff1a96685689f4338d"
EXPECTED_V0_QUERY_PANEL_CONTENT_SHA256 = "cc769d9bf33c1d46a596513bb6dedb757b57476c0716f1d0e5ad6572df1b0fa9"
V0_RESULT_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\dynamic_cold_retrieval_mvp_v0_locked_252q_20260829a"
)
CANONICAL_TEST = (
    SCRIPT_ROOT.parent / "tests"
    / "test_run_dynamic_cold_retrieval_mvp_v0_1_compact_locked_corrected_v3.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = _sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(f"locked artifact SHA mismatch: {resolved}")
    return {"path": str(resolved), "sha256": observed, "bytes": resolved.stat().st_size}


def _runner_normalized_sha256(path: Path) -> str:
    prefix = "EXPECTED_RUNNER_NORMALIZED_SHA256 = "
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1]:
        lines.pop()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("runner normalized-hash marker changed")
    lines[matches[0]] = prefix + '"<normalized-self-hash>"'
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def verify_locked_environment() -> dict[str, Any]:
    observed_self = _runner_normalized_sha256(Path(__file__).resolve())
    if observed_self != EXPECTED_RUNNER_NORMALIZED_SHA256:
        raise ValueError("release runner normalized source hash changed")
    compact._assert_locks()
    sources = {
        "release_runner_normalized_sha256": observed_self,
        "compact_runner": _artifact(Path(compact.__file__), EXPECTED_COMPACT_SHA256),
        "canonical_test": _artifact(CANONICAL_TEST, EXPECTED_CANONICAL_TEST_SHA256),
    }
    manifest_path = V0_RESULT_ROOT / "result_manifest.json"
    sources["v0_result_manifest"] = _artifact(
        manifest_path, EXPECTED_V0_RESULT_MANIFEST_SHA256,
    )
    sidecar = (V0_RESULT_ROOT / "result_manifest.json.sha256").read_text(
        encoding="ascii",
    ).split()
    if not sidecar or sidecar[0] != EXPECTED_V0_RESULT_MANIFEST_SHA256:
        raise ValueError("locked v0 result sidecar mismatch")
    baseline = json.loads(manifest_path.read_bytes())
    baseline_input = dict(baseline.get("input") or {})
    if (
        baseline.get("schema") != v0.SCHEMA
        or baseline.get("status") != "failed"
        or str(baseline_input.get("portfolio_manifest_sha256"))
        != draft.LOCKED_PORTFOLIO_SHA256
        or str(baseline_input.get("library_manifest_sha256"))
        != draft.LOCKED_LIBRARY_SHA256
        or dict(baseline_input.get("artifact_sha256s") or {})
        != draft.LOCKED_ARTIFACT_SHA256S
        or baseline_input.get("validation_or_sealed_used") is not False
        or dict(baseline.get("query_panel") or {}).get("content_sha256")
        != EXPECTED_V0_QUERY_PANEL_CONTENT_SHA256
        or int(dict(baseline.get("evaluation") or {}).get("metrics", {}).get(
            "query_count", -1,
        )) != 252
        or compact.THRESHOLDS != v0.THRESHOLDS
    ):
        raise ValueError("locked v0 formal/input/threshold contract changed")
    return {
        "sources": sources,
        "v0_status": baseline["status"],
        "v0_query_panel_content_sha256": EXPECTED_V0_QUERY_PANEL_CONTENT_SHA256,
        "portfolio_manifest_sha256": draft.LOCKED_PORTFOLIO_SHA256,
        "library_manifest_sha256": draft.LOCKED_LIBRARY_SHA256,
        "artifact_sha256s": dict(draft.LOCKED_ARTIFACT_SHA256S),
        "all_locks_verified_before_index_load": True,
    }


def evaluate_panel(
    index: Mapping[str, Any], panel: Mapping[str, Any],
    initialization_ms: float, incremental_rss_mib: float,
) -> dict[str, Any]:
    with mock.patch.object(v0, "select_dynamic_cold", compact.select_dynamic_cold):
        return v0.evaluate_panel(
            index, panel, initialization_ms, incremental_rss_mib,
        )


def run(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_root}")
    locks = verify_locked_environment()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    index = compact.load_cold_index(
        draft.LOCKED_PORTFOLIO_ROOT, draft.LOCKED_PORTFOLIO_SHA256,
        draft.LOCKED_LIBRARY_SHA256, draft.LOCKED_ARTIFACT_SHA256S,
    )
    rss_after = process.memory_info().rss
    if (
        index["portfolio_sha256"] != draft.LOCKED_PORTFOLIO_SHA256
        or index["library_sha256"] != draft.LOCKED_LIBRARY_SHA256
        or index["artifact_sha256s"] != draft.LOCKED_ARTIFACT_SHA256S
    ):
        raise ValueError("loaded compact index disagrees with locked inputs")
    panel = compact.build_query_panel(index)
    if panel["content_sha256"] != EXPECTED_V0_QUERY_PANEL_CONTENT_SHA256:
        raise ValueError("release query panel differs from locked v0 panel")
    evaluation = evaluate_panel(
        index, panel, index["initialization_ms"],
        max(0.0, (rss_after - rss_before) / (1024.0 ** 2)),
    )
    panel_payload = zlib.compress(draft._canonical_bytes(panel), level=9)
    result = {
        "schema": SCHEMA,
        "status": "passed" if evaluation["passed"] else "failed",
        "implementation": {
            "path": Path(__file__).resolve().relative_to(
                Path(__file__).resolve().parents[3],
            ).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
            "normalized_sha256": EXPECTED_RUNNER_NORMALIZED_SHA256,
            "compact_runner_sha256": EXPECTED_COMPACT_SHA256,
            "canonical_test_sha256": EXPECTED_CANONICAL_TEST_SHA256,
            "locked_v0_result_manifest_sha256": EXPECTED_V0_RESULT_MANIFEST_SHA256,
        },
        "locks": locks,
        "input": {
            "portfolio_root": str(draft.LOCKED_PORTFOLIO_ROOT.resolve()),
            "portfolio_manifest_sha256": draft.LOCKED_PORTFOLIO_SHA256,
            "library_manifest_sha256": draft.LOCKED_LIBRARY_SHA256,
            "artifact_sha256s": dict(draft.LOCKED_ARTIFACT_SHA256S),
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_used": False,
        },
        "config": {
            "top_limit": draft.TOP_LIMIT,
            "active_limit": draft.ACTIVE_LIMIT,
            "capital_gate_mode": "soft_report_only",
            "fp_precision": "float32",
            "selector": "audited compact v0.1; algorithm unchanged",
            "market_environment_distance_feature_indices": list(range(86, 104)),
            "thresholds_changed_from_v0": False,
        },
        "query_panel": {
            "schema": draft.PANEL_SCHEMA,
            "file": "query_panel.json.zlib",
            "file_sha256": draft._sha256_bytes(panel_payload),
            "content_sha256": panel["content_sha256"],
            "counts": panel["counts"],
            "source_scope": panel["source_scope"],
            "outcome_used_for_panel_selection": False,
        },
        "evaluation": evaluation,
        "storage": {
            "actions_copied": False,
            "contracts_copied": False,
            "complete_artifact_on_failed_thresholds": True,
            "partial_output_on_exception": False,
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
        "schema": result["schema"],
        "status": result["status"],
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


