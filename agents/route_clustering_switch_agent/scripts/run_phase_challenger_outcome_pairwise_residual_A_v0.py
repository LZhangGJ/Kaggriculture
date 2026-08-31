#!/usr/bin/env python3
"""Run the train-only A candidate-vs-KEEP representation sanity screen."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_outcome_multiclass_A_v1 as v1_runner
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_outcome_pairwise_residual_A_v0 as pair_v0


SCHEMA = "phase-challenger-outcome-pairwise-residual-A-run-v0"
DEFAULT_STEPS = v1_runner.DEFAULT_STEPS
EXPECTED_DECISIONS = v1_runner.EXPECTED_DECISIONS
EXPECTED_PAIRS = 27_552
EXPECTED_PAIRS_BY_STEP = {
    str(step): 1_632 if step == 219 else 1_728 for step in DEFAULT_STEPS
}
DEFAULT_OUTPUT = v1_runner.DEFAULT_OUTPUT.parent / (
    "phase_challenger_outcome_pairwise_residual_A_v0_20260829a"
)

# These aliases deliberately reuse the independently tested v1 trust boundary.
load_full_panel = v1_runner.load_full_panel
audit_full_panel = v1_runner.audit_full_panel
map_choice_rows = v1_runner.map_choice_rows


def _artifact(path: Path) -> dict[str, Any]:
    return v1_runner._artifact(path)


def implementation_provenance() -> dict[str, dict[str, Any]]:
    runner = Path(__file__).resolve()
    trainer = Path(pair_v0.__file__).resolve()
    tests = runner.parents[1] / "tests" / (
        "test_train_phase_challenger_outcome_pairwise_residual_A_v0.py"
    )
    return {
        "runner": _artifact(runner), "trainer": _artifact(trainer),
        "focused_tests": _artifact(tests),
    }


def choices_provenance(
    choices_path: Path, report_path: Path,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "outcome-pairwise-residual-A-choices-provenance-v0",
        "choices": _artifact(choices_path), "final_report": _artifact(report_path),
        "implementation": implementation_provenance(),
        "input_provenance": dict(input_provenance),
        "full_decision_coverage": True, "train_only": True,
        "validation_opened": False, "sealed_opened": False,
    }


def audit_pair_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if int(result["A_decisions"]) != EXPECTED_DECISIONS:
        raise ValueError("pairwise result lost full decision coverage")
    if int(result["A_candidate_KEEP_pairs"]) != EXPECTED_PAIRS:
        raise ValueError("formal candidate-vs-KEEP pair count changed")
    observed = {
        str(step): int(count)
        for step, count in result["A_candidate_KEEP_pairs_by_step"].items()
    }
    if observed != EXPECTED_PAIRS_BY_STEP:
        raise ValueError("formal candidate-vs-KEEP pair count by step changed")
    audits = [fold["outer_pair_weight_audit"] for fold in result["folds"]]
    if not all(
        audit["each_two_seat_cell_total_weight"] == 1.0
        and audit["all_cells_contain_both_seats"] is True
        and audit["equal_candidate_count_by_seat"] is True
        for audit in audits
    ):
        raise ValueError("paired-seat cell weighting contract changed")
    return {
        "candidate_KEEP_pairs": EXPECTED_PAIRS,
        "candidate_KEEP_pairs_by_step": EXPECTED_PAIRS_BY_STEP,
        "paired_seed_opponent_step_cell_weight": 1.0,
        "step219_candidates_per_seat": 17,
        "other_steps_candidates_per_seat": 18,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays, panel, source_indices = load_full_panel(args)
    panel_audit = audit_full_panel(arrays, panel, source_indices)
    result = pair_v0.evaluate(
        arrays, trees=args.trees, random_seed=args.random_seed,
    )
    pair_audit = audit_pair_result(result)
    mapping = map_choice_rows(
        arrays, source_indices, result["A_source_indices"], result["A_choices"],
    )
    output.mkdir(parents=True, exist_ok=False)
    choices_path = output / "choices_compact1078_pair1062.npz"
    residual._atomic_npz(choices_path, mapping)
    reportable = {
        key: value for key, value in result.items()
        if key not in {"A_choices", "A_source_indices"}
    }
    input_provenance = v1_runner.input_provenance(args, panel)
    report = {
        "schema": SCHEMA,
        "status": "pairwise_representation_sanity_screen_run_complete",
        "screen_result": result["status"],
        "evidence_boundary": {
            "train_only": True, "validation_opened": False,
            "sealed_opened": False, "candidate_committed": False,
            "repair_on_dev": True, "steps_selected_from_outcome_oracle": True,
            "unbiased_estimate": False, "not_final_model": True,
        },
        "pipeline": (
            "A/R0 candidate-vs-KEEP pair1062; shallow nonlinear outcome head; "
            "neutral-only margin tie-break; KEEP fallback; phase disabled"
        ),
        "parameters": {
            "decision_steps": list(DEFAULT_STEPS),
            "trees_per_group_model_per_head": int(args.trees),
            "random_seed": int(args.random_seed),
        },
        "panel": {**panel, **panel_audit, **pair_audit},
        "result": reportable, "input_provenance": input_provenance,
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
            "# A candidate-vs-KEEP pairwise representation screen v0", "",
            f"Status: `{report['status']}`",
            f"Screen result: `{report['screen_result']}`",
            f"Pairs: {EXPECTED_PAIRS}; decisions: {EXPECTED_DECISIONS}.",
            "This is a train-only repair-on-dev representation screen, not a final model.",
            "Validation and sealed artifacts remain unopened.", "",
        )),
    )
    residual._atomic_text(
        output / "choices_compact1078_pair1062_PROVENANCE.json",
        json.dumps(choices_provenance(
            choices_path, report_path, input_provenance,
        ), ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "event": report["status"], "screen_result": report["screen_result"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--materialized-root", type=Path,
        default=v1_runner.loader_v0.smoke.DEFAULT_MATERIALIZED,
    )
    result.add_argument(
        "--fingerprint-root", type=Path,
        default=v1_runner.loader_v0.paired_smoke.DEFAULT_FINGERPRINTS,
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
