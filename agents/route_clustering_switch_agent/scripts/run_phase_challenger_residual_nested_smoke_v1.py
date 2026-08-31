"""Real-feature train-only smoke for the nested R0 A -> phase residual pipeline."""

from __future__ import annotations

import argparse
import json
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

import run_phase_challenger_direct_nested_smoke_v1 as direct_smoke
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as nested


SCHEMA = "phase-challenger-residual-nested-real-feature-smoke-v1"
DEFAULT_MATERIALIZED = direct_smoke.DEFAULT_MATERIALIZED
DEFAULT_OUTPUT = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_residual_nested_real_feature_smoke_v1_20260829a"
)
REPRESENTATIONS = direct_smoke.REPRESENTATIONS


def load_arrays(
    root: Path, representation: str, decision_steps: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    report = json.loads((root / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "formal_materialization_complete_training_not_requested"
        or report.get("training_executed") is not False
        or report.get("runtime_boundary", {}).get("validation_opened") is not False
        or report.get("runtime_boundary", {}).get("sealed_opened") is not False
    ):
        raise ValueError("materialized panel crossed the train-only boundary")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    steps = tuple(sorted(set(map(int, decision_steps))))
    if not steps:
        raise ValueError("decision_steps must not be empty")
    feature_path = root / REPRESENTATIONS[representation]
    metadata_path = root / "metadata.npz"
    direct_smoke._artifact_matches(feature_path, report["storage"][representation])
    direct_smoke._artifact_matches(metadata_path, report["storage"]["metadata"])
    features = np.load(feature_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as raw:
        metadata = {key: raw[key] for key in raw.files}
    indices = np.flatnonzero(np.isin(np.asarray(metadata["step"], np.int64), steps))
    if not len(indices):
        raise ValueError("selected decision steps are absent")
    arrays = {
        "features": np.asarray(features[indices], np.float32),
        "delta_margin": np.asarray(metadata["delta_margin"], np.float64)[indices],
        "membership": np.asarray(metadata["phase_only"], np.int8)[indices],
        "decision": np.asarray(metadata["decision"], np.int64)[indices],
        "opponent": np.asarray(metadata["opponent"], np.int16)[indices],
        "seed": np.asarray(metadata["seed"], np.int64)[indices],
        "edit": np.asarray(metadata["edit"], np.int8)[indices],
    }
    audit = base.validate_arrays(arrays)
    expected_decisions = 96 * len(steps)
    if audit["decisions"] != expected_decisions:
        raise ValueError(
            f"expected {expected_decisions} complete decisions, got {audit['decisions']}"
        )
    if len(audit["opponents"]) != 6 or len(audit["seeds"]) != 8:
        raise ValueError("smoke lost opponent or seed coverage")
    return arrays, {
        "representation": representation,
        "decision_steps": list(steps),
        "rows": int(len(indices)),
        "decisions": int(audit["decisions"]),
        "R0_rows": int(audit["R0_rows"]),
        "phase_only_rows": int(audit["phase_only_rows"]),
        "feature_width": int(audit["feature_width"]),
        "opponents": audit["opponents"],
        "seeds": audit["seeds"],
    }, indices


def incremental_diagnostics(
    arrays: Mapping[str, np.ndarray], a_choices: np.ndarray,
    final_choices: np.ndarray,
) -> dict[str, Any]:
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    if len(slices) != len(a_choices) or len(slices) != len(final_choices):
        raise ValueError("choice arrays do not align with decisions")
    groups: dict[str, dict[str, dict[str, float]]] = {
        "opponent": defaultdict(lambda: defaultdict(float)),
        "seed": defaultdict(lambda: defaultdict(float)),
    }
    overall: dict[str, float] = defaultdict(float)
    delta = np.asarray(arrays["delta_margin"], np.float64)
    membership = np.asarray(arrays["membership"], np.int8)
    for rows, a_choice, final_choice in zip(
        slices, a_choices, final_choices, strict=True,
    ):
        if not np.any(rows == a_choice) or not np.any(rows == final_choice):
            raise ValueError("choice escaped its decision")
        fired = int(a_choice != final_choice)
        if fired and membership[final_choice] != base.PHASE_ONLY:
            raise AssertionError("residual override did not select a phase-only row")
        value = float(delta[final_choice] - delta[a_choice])
        opponent = str(int(arrays["opponent"][rows[0]]))
        seed = str(int(arrays["seed"][rows[0]]))
        for target in (overall, groups["opponent"][opponent], groups["seed"][seed]):
            target["decisions"] += 1
            target["fires"] += fired
            target["fallback_exact_A"] += int(not fired)
            target["harmful"] += int(value < 0)
            target["beneficial"] += int(value > 0)
            target["realized_uplift_sum"] += value
    return {
        "overall": dict(overall),
        **{
            name: {key: dict(value) for key, value in sorted(cells.items())}
            for name, cells in groups.items()
        },
    }


def reportable(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"A_choices", "final_choices"}
    return {key: value for key, value in result.items() if key not in excluded}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    representations: dict[str, Any] = {}
    panels: dict[str, Any] = {}
    for offset, representation in enumerate(args.representations):
        arrays, panel, source_indices = load_arrays(
            args.materialized_root.resolve(), representation, args.decision_steps,
        )
        fit_started = time.perf_counter()
        result = nested.evaluate(
            arrays, trees=args.trees,
            random_seed=args.random_seed + offset * 1_000_003,
        )
        panels[representation] = panel
        entry = reportable(result)
        if result.get("status") == "nested_seed_fold_evaluation_complete":
            a_choices = np.asarray(result["A_choices"], np.int64)
            final_choices = np.asarray(result["final_choices"], np.int64)
            decision_order = np.asarray([
                arrays["decision"][rows[0]] for rows in base.decision_slices(arrays["decision"])
            ], np.int64)
            entry["phase_incremental_vs_A_diagnostics"] = incremental_diagnostics(
                arrays, a_choices, final_choices,
            )
            residual._atomic_npz(output / f"choices_{representation}.npz", {
                "decision": decision_order,
                "A_materialized_row": source_indices[a_choices],
                "final_materialized_row": source_indices[final_choices],
            })
        entry["elapsed_seconds"] = time.perf_counter() - fit_started
        representations[representation] = entry
    statuses = {value["status"] for value in representations.values()}
    report = {
        "schema": SCHEMA,
        "status": (
            "real_feature_nested_residual_smoke_complete"
            if statuses == {"nested_seed_fold_evaluation_complete"}
            else "real_feature_nested_residual_smoke_stopped_on_signal"
        ),
        "evidence_boundary": {
            "smoke_only": True,
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "candidate_committed": False,
            "outer_seed_labels_used_for_calibration": False,
        },
        "pipeline": (
            "learned R0 selector A; phase model scores [phase, phase-A]; "
            "phase rejection returns exact A"
        ),
        "parameters": {
            "decision_steps": list(map(int, args.decision_steps)),
            "trees_per_model": int(args.trees),
            "random_seed": int(args.random_seed),
        },
        "panels": panels,
        "representations": representations,
        "elapsed_seconds": time.perf_counter() - started,
    }
    residual._atomic_text(
        output / "SMOKE_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "SMOKE_REPORT.md",
        "\n".join((
            "# PHASE-CHALLENGER residual nested real-feature smoke",
            "", f"Status: `{report['status']}`",
            f"Steps: `{report['parameters']['decision_steps']}`",
            "Outer seed folds are evaluated once; all A and phase calibration stays inside outer train.",
            "Phase incremental diagnostics are measured against the deployable outer-fold A choice.",
            "This is train-only plumbing evidence, not validation or final agent win-rate evidence.",
            "",
        )),
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "representations": list(representations),
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--decision-steps", type=int, nargs="+", default=(217, 218, 219, 220))
    result.add_argument(
        "--representations", nargs="+", choices=tuple(REPRESENTATIONS),
        default=tuple(REPRESENTATIONS),
    )
    result.add_argument("--trees", type=int, default=2)
    result.add_argument("--random-seed", type=int, default=20260829)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
