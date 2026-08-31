#!/usr/bin/env python3
"""Run the train-only 16-step A/R0 outcome multiclass experiment v1."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_outcome_two_head_A_smoke_v0 as loader_v0
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_outcome_multiclass_A_nested_v1 as outcome_v1
import train_phase_challenger_residual_gate_v1 as base


SCHEMA = "phase-challenger-outcome-multiclass-A-v1"
DEFAULT_STEPS = tuple(loader_v0.DEFAULT_STEPS)
EXPECTED_DECISIONS_PER_STEP = 6 * 8 * 2
EXPECTED_DECISIONS = EXPECTED_DECISIONS_PER_STEP * len(DEFAULT_STEPS)
FORMAL_OPPONENTS = tuple(range(6))
FORMAL_SEATS = (0, 1)
DEFAULT_OUTPUT = loader_v0.smoke.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_outcome_multiclass_A_v1_20260829a"
)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": residual._sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _array_sha256(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(np.asarray(data.shape, np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def implementation_provenance() -> dict[str, dict[str, Any]]:
    runner = Path(__file__).resolve()
    trainer = Path(outcome_v1.__file__).resolve()
    tests = runner.parents[1] / "tests" / (
        "test_train_phase_challenger_outcome_multiclass_A_nested_v1.py"
    )
    return {
        "runner": _artifact(runner),
        "trainer": _artifact(trainer),
        "focused_tests": _artifact(tests),
    }


def choices_provenance(
    choices_path: Path, report_path: Path,
    input_provenance_value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "outcome-multiclass-A-choices-provenance-v1",
        "choices": _artifact(choices_path),
        "final_report": _artifact(report_path),
        "implementation": implementation_provenance(),
        "input_provenance": dict(input_provenance_value),
        "full_decision_coverage": True,
        "train_only": True,
        "validation_opened": False,
        "sealed_opened": False,
    }


def load_full_panel(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    load_args = argparse.Namespace(
        materialized_root=args.materialized_root,
        fingerprint_root=args.fingerprint_root,
        decision_steps=DEFAULT_STEPS,
    )
    arrays, panel, source_indices = loader_v0.load_arrays(load_args)
    metadata_path = args.materialized_root.resolve() / "metadata.npz"
    with np.load(metadata_path, allow_pickle=False) as metadata:
        arrays["step"] = np.asarray(metadata["step"], np.int16)[source_indices]
    return arrays, panel, np.asarray(source_indices, np.int64)


def audit_full_panel(
    arrays: Mapping[str, np.ndarray], panel: Mapping[str, Any],
    source_indices: np.ndarray, *, expected_decisions: int = EXPECTED_DECISIONS,
) -> dict[str, Any]:
    audit = base.validate_arrays(arrays)
    if len(source_indices) != int(audit["rows"]):
        raise ValueError("source_indices do not align with the selected panel")
    if (
        np.any(source_indices < 0)
        or len(np.unique(source_indices)) != len(source_indices)
        or np.any(source_indices[1:] <= source_indices[:-1])
    ):
        raise ValueError("source_indices must be unique increasing materialized rows")
    if tuple(map(int, panel["decision_steps"])) != DEFAULT_STEPS:
        raise ValueError("the 16 pre-registered decision steps changed")
    if tuple(audit["seeds"]) != outcome_v1.FORMAL_SEEDS:
        raise ValueError("full panel lost a pre-registered seed")
    if int(audit["decisions"]) != int(expected_decisions):
        raise ValueError("full decision coverage changed")
    step_counts: dict[str, int] = {}
    cells_by_step: dict[int, set[tuple[int, int, int]]] = {}
    duplicate_cells: dict[int, list[tuple[int, int, int]]] = {}
    if "seat" not in arrays:
        raise ValueError("full panel is missing seat provenance")
    for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        values = set(map(int, np.asarray(arrays["step"])[rows]))
        if len(values) != 1:
            raise ValueError("a decision crosses step boundaries")
        step = str(values.pop())
        step_counts[step] = step_counts.get(step, 0) + 1
        seats = set(map(int, np.asarray(arrays["seat"])[rows]))
        if len(seats) != 1:
            raise ValueError("a decision crosses seat boundaries")
        cell = (
            int(arrays["seed"][rows[0]]),
            int(arrays["opponent"][rows[0]]),
            seats.pop(),
        )
        step_value = int(step)
        cells = cells_by_step.setdefault(step_value, set())
        if cell in cells:
            duplicate_cells.setdefault(step_value, []).append(cell)
        cells.add(cell)
    if expected_decisions == EXPECTED_DECISIONS and (
        set(map(int, step_counts)) != set(DEFAULT_STEPS)
        or any(value != EXPECTED_DECISIONS_PER_STEP for value in step_counts.values())
    ):
        raise ValueError("one of the 16 steps lost opponent/seed/seat decisions")
    if expected_decisions == EXPECTED_DECISIONS:
        expected_cells = set(itertools.product(
            outcome_v1.FORMAL_SEEDS, FORMAL_OPPONENTS, FORMAL_SEATS,
        ))
        failures = {
            step: {
                "missing": sorted(expected_cells - cells_by_step.get(step, set())),
                "extra": sorted(cells_by_step.get(step, set()) - expected_cells),
                "duplicates": duplicate_cells.get(step, []),
            }
            for step in DEFAULT_STEPS
            if cells_by_step.get(step, set()) != expected_cells
            or duplicate_cells.get(step)
        }
        if failures:
            raise ValueError(
                "per-step seed/opponent/seat Cartesian coverage changed: "
                f"{failures}"
            )
    a, _ = outcome_v1.outcome_arrays(arrays)
    a_decisions = len(base.decision_slices(a["decision"]))
    if a_decisions != int(audit["decisions"]):
        raise ValueError("A/R0 filtering lost a complete decision")
    return {
        **audit,
        "A_rows": int(len(a["decision"])),
        "A_decisions": a_decisions,
        "decision_steps": list(DEFAULT_STEPS),
        "decisions_by_step": step_counts,
        "per_step_seed_opponent_seat_Cartesian_complete": True,
        "source_indices_unique_increasing": True,
        "source_indices_sha256": _array_sha256(source_indices),
        "all_decisions_include_KEEP": True,
        "phase_rows_used_by_model": 0,
    }


def map_choice_rows(
    arrays: Mapping[str, np.ndarray], source_indices: np.ndarray,
    a_source_indices: np.ndarray, a_choices: np.ndarray,
) -> dict[str, np.ndarray]:
    a, regenerated_source = outcome_v1.outcome_arrays(arrays)
    a_source = np.asarray(a_source_indices, np.int64)
    choices = np.asarray(a_choices, np.int64)
    if not np.array_equal(a_source, regenerated_source):
        raise ValueError("reported A source mapping changed on regeneration")
    slices = base.decision_slices(a["decision"])
    if len(slices) != len(choices):
        raise ValueError("A choices do not cover every decision")
    for rows, choice in zip(slices, choices, strict=True):
        if not np.any(rows == choice):
            raise ValueError("A choice escaped its source decision")
    input_rows = a_source[choices]
    materialized_rows = np.asarray(source_indices, np.int64)[input_rows]
    decision = np.asarray([a["decision"][rows[0]] for rows in slices], np.int64)
    if not np.array_equal(np.asarray(arrays["decision"])[input_rows], decision):
        raise AssertionError("choice/source decision mapping changed")
    return {
        "decision": decision,
        "A_row_index": choices,
        "input_panel_row_index": input_rows,
        "materialized_row_index": materialized_rows,
        "choice_is_KEEP": (
            np.asarray(a["edit"])[choices] == base.KEEP_EDIT
        ).astype(np.bool_),
    }


def input_provenance(args: argparse.Namespace, panel: Mapping[str, Any]) -> dict[str, Any]:
    materialized_report = args.materialized_root.resolve() / "FINAL_REPORT.json"
    fingerprint_report = args.fingerprint_root.resolve() / "DIAGNOSTIC_REPORT.json"
    return {
        "materialized_report": _artifact(materialized_report),
        "fingerprint_report": _artifact(fingerprint_report),
        "outcome_metadata_sha256": str(panel["outcome_metadata_sha256"]),
        "source_index_semantics": (
            "filtered input row -> canonical materialized row; A source index -> "
            "filtered input row; choice -> A row"
        ),
        "train_only": True,
        "validation_opened": False,
        "sealed_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays, panel, source_indices = load_full_panel(args)
    input_audit = audit_full_panel(arrays, panel, source_indices)
    result = outcome_v1.evaluate(
        arrays, trees=args.trees, random_seed=args.random_seed,
    )
    if result["status"] != "strict_four_seed_fold_outcome_multiclass_A_complete":
        raise RuntimeError("outcome v1 did not complete its four outer folds")
    mapping = map_choice_rows(
        arrays, source_indices, result["A_source_indices"], result["A_choices"],
    )
    if len(mapping["decision"]) != EXPECTED_DECISIONS:
        raise AssertionError("choice artifact lost full decision coverage")
    output.mkdir(parents=True, exist_ok=False)
    choices_path = output / "choices_compact1078.npz"
    residual._atomic_npz(choices_path, mapping)
    reportable = {
        key: value for key, value in result.items()
        if key not in {"A_choices", "A_source_indices"}
    }
    report = {
        "schema": SCHEMA,
        "status": "train_only_outcome_multiclass_A_v1_complete",
        "evidence_boundary": {
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "candidate_committed": False,
            "outer_labels_used_for_threshold_calibration": False,
            "steps_selected_from_outcome_oracle": True,
            "repair_on_dev": True,
            "unbiased_estimate": False,
        },
        "pipeline": (
            "compact1078 A/R0-only shared three-class model; known_pool_loso is "
            "primary, unknown-opponent LOPO is coverage diagnostic/abstain; "
            "KEEP fallback; phase disabled"
        ),
        "parameters": {
            "decision_steps": list(DEFAULT_STEPS),
            "decision_step_origin": "union of train outcome-oracle positive steps",
            "steps_selected_from_outcome_oracle": True,
            "trees_per_group_model": int(args.trees),
            "random_seed": int(args.random_seed),
            "min_samples_leaf": outcome_v1.model_params(
                args.trees, args.random_seed,
            )["min_samples_leaf"],
        },
        "panel": {**panel, **input_audit},
        "result": reportable,
        "input_provenance": input_provenance(args, panel),
        "artifacts": {"choices": _artifact(choices_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "FINAL_REPORT.json"
    residual._atomic_text(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "FINAL_REPORT.md",
        "\n".join((
            "# Outcome-aware A/R0 multiclass v1", "",
            f"Status: `{report['status']}`",
            f"All decisions: {reportable['A_decisions']} across 16 fixed steps.",
            f"Raw win delta: {reportable['all_decision_metrics']['raw_win_delta']}.",
            "Thresholds use outer-train OOF only; every outer decision and KEEP is evaluated.",
            "The 16 steps were selected from train outcome-oracle positives: repair-on-dev, not unbiased.",
            "Phase, validation and sealed data remain disabled/unopened.", "",
        )),
    )
    residual._atomic_text(
        output / "choices_compact1078_PROVENANCE.json",
        json.dumps(choices_provenance(
            choices_path, report_path, report["input_provenance"],
        ), ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "decisions": reportable["A_decisions"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--materialized-root", type=Path,
        default=loader_v0.smoke.DEFAULT_MATERIALIZED,
    )
    result.add_argument(
        "--fingerprint-root", type=Path,
        default=loader_v0.paired_smoke.DEFAULT_FINGERPRINTS,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--trees", type=int, default=48)
    result.add_argument("--random-seed", type=int, default=20260829)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
