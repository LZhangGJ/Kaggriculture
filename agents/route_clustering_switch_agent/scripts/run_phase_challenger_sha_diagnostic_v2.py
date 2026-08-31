"""Audit train-only action/state aliases with validated input provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_compact_et_ab_v1 as compact_v1
import run_phase_challenger_direct_nested_smoke_v1 as direct_smoke
import run_phase_challenger_sha_diagnostic_v1 as legacy_v1
import run_route_residual_adapter_v0 as residual


SCHEMA = "phase-challenger-state-action-diagnostic-v2"
HEX64 = re.compile(r"[0-9a-f]{64}")
STATE_WIDTH = 337
DEFAULT_LABEL_ROOT = legacy_v1.DEFAULT_LABEL_ROOT
DEFAULT_MATERIALIZED = legacy_v1.DEFAULT_MATERIALIZED
DEFAULT_CHOICES = legacy_v1.DEFAULT_CHOICES
DEFAULT_LEGACY = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_sha_diagnostic_v1_20260829a"
)
DEFAULT_OUTPUT = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_state_action_diagnostic_v2_20260829a"
)


def _hex(value: Any, name: str) -> str:
    result = str(value)
    if HEX64.fullmatch(result) is None:
        raise ValueError(f"{name} is not canonical lowercase SHA256 hex")
    return result


def _digest(*parts: bytes) -> bytes:
    value = hashlib.blake2b(digest_size=16)
    for part in parts:
        value.update(part)
    return value.digest()


def sign_group_summary(keys: Sequence[bytes], delta: np.ndarray) -> dict[str, Any]:
    groups: dict[bytes, list[float]] = defaultdict(list)
    for key, value in zip(keys, np.asarray(delta, np.float64), strict=True):
        groups[bytes(key)].append(float(value))
    positive_negative = []
    positive_nonpositive = []
    for values in groups.values():
        positive = any(value > 0 for value in values)
        negative = any(value < 0 for value in values)
        nonpositive = any(value <= 0 for value in values)
        if positive and negative:
            positive_negative.append(values)
        if positive and nonpositive:
            positive_nonpositive.append(values)
    return {
        "unique_groups": len(groups),
        "positive_groups": sum(
            any(value > 0 for value in values) for values in groups.values()
        ),
        "zero_groups": sum(
            any(value == 0 for value in values) for values in groups.values()
        ),
        "negative_groups": sum(
            any(value < 0 for value in values) for values in groups.values()
        ),
        "nonpositive_groups": sum(
            any(value <= 0 for value in values) for values in groups.values()
        ),
        "mixed_positive_negative_groups": len(positive_negative),
        "rows_in_positive_negative_groups": sum(
            len(values) for values in positive_negative
        ),
        "mixed_positive_nonpositive_groups": len(positive_nonpositive),
        "rows_in_positive_nonpositive_groups": sum(
            len(values) for values in positive_nonpositive
        ),
        "maximum_group_rows": max(map(len, groups.values()), default=0),
    }


def validate_materialized_boundary(
    report: Mapping[str, Any], label_root: Path, materialized_root: Path,
) -> dict[str, Any]:
    expected = {
        "schema": "phase-challenger-direct-u-phase-compact-et-ab-v1",
        "status": "formal_materialization_complete_training_not_requested",
        "training_executed": False,
        "feature_allocation_executed": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"materialized report boundary mismatch: {key}")
    runtime = report.get("runtime_boundary", {})
    if (
        runtime.get("validation_opened") is not False
        or runtime.get("sealed_opened") is not False
        or runtime.get("candidate_committed") is not False
    ):
        raise ValueError("materialized report is outside the train-only boundary")
    if Path(str(report.get("input_label_root", ""))).resolve() != label_root:
        raise ValueError("materialized report points at a different label root")
    panel_rows = int(report.get("panel", {}).get("rows", -1))
    storage = report.get("storage", {})
    expected_paths = {
        "metadata": materialized_root / "metadata.npz",
        "compact1078": materialized_root / "compact_1078.npy",
        "full2844": materialized_root / "full_2844.npy",
    }
    for key, path in expected_paths.items():
        artifact = storage.get(key, {})
        if (
            int(artifact.get("rows", -1)) != panel_rows
            or Path(str(artifact.get("path", ""))).resolve() != path
        ):
            raise ValueError(f"materialized {key} provenance mismatch")
        direct_smoke._artifact_matches(path, artifact)
    return {
        "schema_status_verified": True,
        "training_not_executed_verified": True,
        "train_only_runtime_boundary_verified": True,
        "label_root_verified": True,
        "all_storage_paths_rows_hashes_verified": True,
    }


def load_choices(
    path: Path, metadata: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"focus choices are required: {path}")
    report_path = path.parent / "SMOKE_REPORT.json"
    if not report_path.is_file():
        raise ValueError("choices are missing their adjacent smoke report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    boundary = report.get("evidence_boundary", {})
    if (
        report.get("schema")
        != "phase-challenger-residual-group-quantile-real-feature-smoke-v1"
        or report.get("status") != "real_feature_nested_residual_smoke_complete"
        or boundary.get("train_only") is not True
        or boundary.get("validation_opened") is not False
        or boundary.get("sealed_opened") is not False
        or boundary.get("candidate_committed") is not False
    ):
        raise ValueError("choices report is outside the expected train-only smoke")
    with np.load(path, allow_pickle=False) as choices:
        if not {"decision", "A_materialized_row"} <= set(choices.files):
            raise ValueError("choices file is missing decision provenance")
        raw_decision = choices["decision"]
        raw_selected = choices["A_materialized_row"]
        if raw_decision.dtype.kind not in "iu" or raw_selected.dtype.kind not in "iu":
            raise ValueError("choices indices must have integer dtype")
        decision = np.asarray(raw_decision, np.int64)
        selected = np.asarray(raw_selected, np.int64)
    rows = len(np.asarray(metadata["decision"]))
    if (
        decision.ndim != 1 or selected.ndim != 1
        or len(decision) != len(selected) or not len(selected)
        or len(set(map(int, decision))) != len(decision)
        or np.any(selected < 0) or np.any(selected >= rows)
    ):
        raise ValueError("choices shape, uniqueness or row bounds are invalid")
    if not np.array_equal(
        np.asarray(metadata["decision"], np.int64)[selected], decision,
    ):
        raise ValueError("choices do not align with this materialized panel")
    if np.any(np.asarray(metadata["phase_only"], np.bool_)[selected]):
        raise ValueError("A choices contain phase-only rows")
    expected = int(
        report.get("panels", {}).get("compact1078", {}).get("decisions", -1)
    )
    if expected != len(decision):
        raise ValueError("choices length disagrees with its smoke report")
    return selected, {
        "report_path": str(report_path),
        "report_sha256": residual._sha256_file(report_path),
        "choices_path": str(path),
        "choices_sha256": residual._sha256_file(path),
        "choices": len(selected),
        "row_bounds_decision_membership_verified": True,
        "train_only_smoke_boundary_verified": True,
        "legacy_runner_embedded_source_artifact_sha": False,
    }


def load_legacy_extended(
    root: Path, metadata_path: Path, compact_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report_path = root / "DIAGNOSTIC_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report.get("artifacts", {}).get("extended_metadata", {})
    extended_path = root / "extended_metadata.npz"
    if (
        report.get("schema") != legacy_v1.SCHEMA
        or report.get("status") != "train_only_sha_alias_diagnostic_complete"
        or report.get("artifacts", {}).get("source_metadata_sha256")
        != residual._sha256_file(metadata_path)
        or report.get("artifacts", {}).get("source_compact_sha256")
        != residual._sha256_file(compact_path)
        or str(artifact.get("sha256")) != residual._sha256_file(extended_path)
    ):
        raise ValueError("legacy extended metadata provenance mismatch")
    with np.load(extended_path, allow_pickle=False) as raw:
        extended = {key: raw[key] for key in raw.files}
    return extended, {
        "legacy_report_path": str(report_path),
        "legacy_report_sha256": residual._sha256_file(report_path),
        "extended_path": str(extended_path),
        "extended_sha256": residual._sha256_file(extended_path),
        "source_hashes_reverified": True,
        "formal_rows_reverified_below": True,
    }


def _describe(rows: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    values = delta[rows]
    return {
        "rows": int(len(rows)),
        "positive": int(np.count_nonzero(values > 0)),
        "zero": int(np.count_nonzero(values == 0)),
        "negative": int(np.count_nonzero(values < 0)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def focus_rows(
    selected: np.ndarray, extended: Mapping[str, np.ndarray],
    derived: Mapping[str, np.ndarray], metadata: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    delta = np.asarray(metadata["delta_margin"], np.float64)
    result = []
    for row in selected[delta[selected] < 0]:
        histories = {}
        for name in (
            "candidate_sha256", "anchor_query_offset_action",
            "observable_state337_action", "compact_feature_blake2b128",
            "full_feature_blake2b128",
        ):
            values = extended[name] if name in extended else derived[name]
            histories[f"{name}_history"] = _describe(
                np.flatnonzero(values == values[row]), delta,
            )
        result.append({
            "materialized_row": int(row),
            "decision": int(metadata["decision"][row]),
            "step": int(metadata["step"][row]),
            "opponent": int(metadata["opponent"][row]),
            "seed": int(metadata["seed"][row]),
            "seat": int(metadata["seat"][row]),
            "delta_margin": float(delta[row]),
            "candidate_sha256": bytes(
                extended["candidate_sha256"][row]
            ).decode("ascii"),
            "anchor_query_sha256": bytes(
                extended["query_sha256"][row]
            ).decode("ascii"),
            **histories,
        })
    return result


def fold_coverage(
    folds: np.ndarray, group: np.ndarray, decision: np.ndarray,
    delta: np.ndarray,
) -> list[dict[str, Any]]:
    result = []
    for fold in range(4):
        rows = np.flatnonzero(folds == fold)
        result.append({
            "fold": fold,
            "rows": int(len(rows)),
            "unique_groups": len(set(map(bytes, group[rows]))),
            "decisions": len(set(map(int, decision[rows]))),
            "positive_rows": int(np.count_nonzero(delta[rows] > 0)),
            "zero_rows": int(np.count_nonzero(delta[rows] == 0)),
            "negative_rows": int(np.count_nonzero(delta[rows] < 0)),
        })
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    label_root = args.label_root.resolve()
    materialized_root = args.materialized_root.resolve()
    contract = compact_v1.load_input_contract(label_root, smoke=False)
    materialized_report_path = materialized_root / "FINAL_REPORT.json"
    materialized_report = json.loads(
        materialized_report_path.read_text(encoding="utf-8")
    )
    boundary = validate_materialized_boundary(
        materialized_report, label_root, materialized_root,
    )
    metadata_path = materialized_root / "metadata.npz"
    compact_path = materialized_root / "compact_1078.npy"
    full_path = materialized_root / "full_2844.npy"
    with np.load(metadata_path, allow_pickle=False) as raw:
        metadata = {key: raw[key] for key in raw.files}
    compact = np.load(compact_path, mmap_mode="r")
    full = np.load(full_path, mmap_mode="r")
    rows = len(metadata["decision"])
    if compact.shape != (rows, 1078) or full.shape != (rows, 2844):
        raise ValueError("materialized feature shape changed")
    extended, legacy_audit = load_legacy_extended(
        args.legacy_diagnostic_root.resolve(), metadata_path, compact_path,
    )
    if any(len(value) != rows for value in extended.values()):
        raise ValueError("legacy extended metadata row count changed")
    selected, choices_audit = load_choices(args.choices.resolve(), metadata)

    market_diff_steps = np.empty(rows, np.int16)
    at = 0
    decisions = 0
    for decision, labels in compact_v1.iter_panels(
        contract["paths"]["h1_decisions.jsonl"],
        contract["paths"]["canonical_union_labels.jsonl"],
    ):
        stop = at + len(labels)
        expected_rows = np.arange(at, stop)
        query = _hex(decision["query_sha256"], "anchor query SHA")
        if (
            not np.all(metadata["decision"][expected_rows] == decisions)
            or not np.all(
                metadata["step"][expected_rows] == int(decision["decision_step"])
            )
            or not np.array_equal(
                metadata["union_index"][expected_rows], np.arange(len(labels))
            )
        ):
            raise RuntimeError("materialized metadata row order changed")
        for local, label in enumerate(labels):
            row = at + local
            candidate = _hex(label["candidate_sha256"], "candidate SHA")
            market_diff = int(label["market_diff"])
            if (
                bytes(extended["candidate_sha256"][row]).decode("ascii")
                != candidate
                or bytes(extended["query_sha256"][row]).decode("ascii") != query
                or int(extended["outcome"][row]) != int(label["outcome"])
                or int(extended["anchor"][row]) != int(label["anchor"])
                or int(extended["offset"][row]) != int(label["offset"])
                or bool(extended["market_diff"][row]) != bool(market_diff)
                or int(extended["action_fold"][row]) != legacy_v1._fold(candidate)
                or int(extended["query_fold"][row]) != legacy_v1._fold(query)
            ):
                raise RuntimeError("legacy extended row disagrees with formal labels")
            market_diff_steps[row] = market_diff
        at = stop
        decisions += 1
    if at != rows or decisions != len(set(map(int, metadata["decision"]))):
        raise RuntimeError("formal labels did not cover the materialized panel")

    compact_hash = np.empty(rows, "S16")
    full_hash = np.empty(rows, "S16")
    for row in range(rows):
        compact_hash[row] = _digest(
            np.asarray(compact[row], np.float32).tobytes()
        )
        full_hash[row] = _digest(np.asarray(full[row], np.float32).tobytes())
    if not np.array_equal(
        compact_hash, extended["compact_feature_blake2b128"],
    ):
        raise RuntimeError("legacy compact row fingerprints changed")

    decision_values = np.asarray(metadata["decision"], np.int64)
    starts = np.flatnonzero(np.r_[True, decision_values[1:] != decision_values[:-1]])
    stops = np.r_[starts[1:], rows]
    current_state = np.empty(rows, "S16")
    for start, stop in zip(starts, stops, strict=True):
        state = np.asarray(compact[start, :STATE_WIDTH], np.float32)
        if not np.array_equal(
            np.asarray(compact[start:stop, :STATE_WIDTH], np.float32),
            np.broadcast_to(state, (stop - start, STATE_WIDTH)),
        ):
            raise RuntimeError("state337 is not shared within a decision")
        current_state[start:stop] = _digest(state.tobytes())

    anchor_offset_action = np.empty(rows, "S16")
    state_action = np.empty(rows, "S16")
    for row in range(rows):
        offset = int(extended["offset"][row]).to_bytes(2, "little", signed=True)
        action = bytes(extended["candidate_sha256"][row])
        anchor_offset_action[row] = _digest(
            bytes(extended["query_sha256"][row]), offset, action,
        )
        state_action[row] = _digest(bytes(current_state[row]), action)
    derived = {
        "anchor_query_offset_action": anchor_offset_action,
        "observable_state337_action": state_action,
        "full_feature_blake2b128": full_hash,
    }
    delta = np.asarray(metadata["delta_margin"], np.float64)
    phase = np.asarray(metadata["phase_only"], np.bool_)
    report = {
        "schema": SCHEMA,
        "status": "validated_train_only_state_action_diagnostic_complete",
        "evidence_boundary": {
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "model_trained": False,
            "all_claims_derived_after_preflight": True,
        },
        "input_contract": {
            "materialized_report_path": str(materialized_report_path),
            "materialized_report_sha256": residual._sha256_file(
                materialized_report_path
            ),
            "materialized_boundary": boundary,
            "legacy_extended": legacy_audit,
            "focus_choices": choices_audit,
            "h1_decisions_sha256": residual._sha256_file(
                contract["paths"]["h1_decisions.jsonl"]
            ),
            "canonical_union_labels_sha256": residual._sha256_file(
                contract["paths"]["canonical_union_labels.jsonl"]
            ),
        },
        "panel": {
            "rows": rows,
            "decisions": decisions,
            "compact_width": int(compact.shape[1]),
            "full_width": int(full.shape[1]),
            "observable_state_width": STATE_WIDTH,
            "market_diff_steps_min": int(market_diff_steps.min()),
            "market_diff_steps_max": int(market_diff_steps.max()),
        },
        "semantics": {
            "query_sha256": "anchor_query_sha; not the current offset decision state",
            "candidate_sha256": "executable four-step actor-unit action tensor",
            "observable_state337_action": (
                "current shared state337 bytes plus executable action SHA"
            ),
            "compact_and_full_exact": (
                "include candidate/provenance features; uniqueness alone is not causality"
            ),
            "harm_target_recommendation": "P(delta_margin <= 0)",
        },
        "groups": {
            "candidate_action_sha": sign_group_summary(
                extended["candidate_sha256"], delta,
            ),
            "anchor_query_sha": sign_group_summary(
                extended["query_sha256"], delta,
            ),
            "anchor_query_offset_action": sign_group_summary(
                anchor_offset_action, delta,
            ),
            "observable_state337_action": sign_group_summary(
                state_action, delta,
            ),
            "exact_compact_feature": sign_group_summary(compact_hash, delta),
            "exact_full_feature": sign_group_summary(full_hash, delta),
            "phase_candidate_action_sha": sign_group_summary(
                extended["candidate_sha256"][phase], delta[phase],
            ),
        },
        "focus_harmful_A_choices": focus_rows(
            selected, extended, derived, metadata,
        ),
        "fold_coverage": {
            "action_sha": fold_coverage(
                extended["action_fold"], extended["candidate_sha256"],
                decision_values, delta,
            ),
            "anchor_query_sha": fold_coverage(
                extended["query_fold"], extended["query_sha256"],
                decision_values, delta,
            ),
        },
        "fingerprints": {
            "algorithm": "blake2b-128 over canonical float32 row bytes/composites",
            "compact_source_sha256_reused_after_storage_and_legacy_verification": (
                materialized_report["storage"]["compact1078"]["sha256"]
            ),
            "full_source_sha256_reused_after_storage_verification": (
                materialized_report["storage"]["full2844"]["sha256"]
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    residual._atomic_npz(output / "derived_fingerprints.npz", {
        "current_state337_blake2b128": current_state,
        "anchor_query_offset_action": anchor_offset_action,
        "observable_state337_action": state_action,
        "full_feature_blake2b128": full_hash,
        "market_diff_steps": market_diff_steps,
    })
    report["artifacts"] = {
        "derived_fingerprints": {
            "path": str(output / "derived_fingerprints.npz"),
            "sha256": residual._sha256_file(output / "derived_fingerprints.npz"),
            "bytes": (output / "derived_fingerprints.npz").stat().st_size,
        }
    }
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.md",
        "\n".join((
            "# PHASE-CHALLENGER validated state/action diagnostic v2",
            "", f"Status: `{report['status']}`",
            f"Rows: {rows}; decisions: {decisions}",
            "Anchor query, offset, observable state, compact and full fingerprints are separated.",
            "No model, validation or sealed artifact was opened.", "",
        )),
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "rows": rows, "decisions": decisions,
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    result.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    result.add_argument("--choices", type=Path, default=DEFAULT_CHOICES)
    result.add_argument(
        "--legacy-diagnostic-root", type=Path, default=DEFAULT_LEGACY,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
