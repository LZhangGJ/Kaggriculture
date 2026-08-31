#!/usr/bin/env python3
"""Run the frozen train-only action-prior -> state-residual A screen."""

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

import run_phase_challenger_outcome_multiclass_A_v1 as panel_runner
import run_phase_challenger_outcome_pairwise_residual_A_v0 as pair_runner
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_action_prior_residual_A_v1 as action_v1
import train_phase_challenger_outcome_pairwise_residual_A_v0 as pair_v0


SCHEMA = "phase-challenger-action-prior-residual-A-run-v1"
EXPECTED_TRAINER_SCHEMA = "phase-challenger-action-prior-residual-A-v1"
FORMAL_TREES = 48
FORMAL_RANDOM_SEED = 20260829
EXPECTED_INPUT = {
    "materialized_report_sha256": "afb1c6d93c456322cdfba5382a92d851edde957c4c00a0846d8aebd3957ce604",
    "compact1078_sha256": "1da4b0af5536e92d861322057ccd19a97fdbc13671d949e1d9d985e904af4cd7",
    "metadata_sha256": "f6dcb85e8a6ddddda4051e7fd50df88e5e9121e74d64343ad99335cc828ee269",
    "candidate_order_sha256": "69e45374b4c9ebd9f0a822b55d227cd5987adbab905abe6b45c309a25d507c88",
    "diagnostic_report_sha256": "1686748726dca86b03bbc324965081d367af19dc200303fde3befef893b57114",
    "derived_fingerprints_sha256": "7425a1e61a597057ebbe09b24a7e0f6f56423c1dfc5af9b4ae13b1bf0b7ba356",
    "outcome_metadata_sha256": "8a772a870ee548f7184602eb0dbe0fac8b0a7e719df301193b2f8b4644848dd5",
}
EXPECTED_DECISIONS = pair_runner.EXPECTED_DECISIONS
EXPECTED_PAIRS = pair_runner.EXPECTED_PAIRS
EXPECTED_PAIRS_BY_STEP = pair_runner.EXPECTED_PAIRS_BY_STEP
DEFAULT_OUTPUT = pair_runner.DEFAULT_OUTPUT.parent / (
    "phase_challenger_action_prior_residual_A_v1_20260829a"
)

load_full_panel = panel_runner.load_full_panel
audit_full_panel = panel_runner.audit_full_panel
map_choice_rows = panel_runner.map_choice_rows


def _artifact(path: Path) -> dict[str, Any]:
    return panel_runner._artifact(path)


def implementation_provenance() -> dict[str, dict[str, Any]]:
    runner = Path(__file__).resolve()
    tests = runner.parents[1] / "tests" / (
        "test_train_phase_challenger_action_prior_residual_A_v1.py"
    )
    return {
        "runner": _artifact(runner),
        "trainer": _artifact(Path(action_v1.__file__).resolve()),
        "focused_tests": _artifact(tests),
        "pair1062_dependency": _artifact(Path(pair_v0.__file__).resolve()),
    }


def frozen_identity(
    *, materialized_report: Mapping[str, Any],
    diagnostic_report: Mapping[str, Any], panel: Mapping[str, Any],
    materialized_report_sha256: str, diagnostic_report_sha256: str,
    trees: int, random_seed: int,
) -> dict[str, Any]:
    observed = {
        "materialized_report_sha256": str(materialized_report_sha256),
        "compact1078_sha256": str(
            materialized_report["storage"]["compact1078"]["sha256"]
        ),
        "metadata_sha256": str(
            materialized_report["storage"]["metadata"]["sha256"]
        ),
        "candidate_order_sha256": str(
            materialized_report["panel"]["candidate_order_sha256"]
        ),
        "diagnostic_report_sha256": str(diagnostic_report_sha256),
        "derived_fingerprints_sha256": str(
            diagnostic_report["artifacts"]["derived_fingerprints"]["sha256"]
        ),
        "outcome_metadata_sha256": str(panel["outcome_metadata_sha256"]),
    }
    failures = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED_INPUT.items()
        if observed.get(key) != expected
    }
    if failures:
        raise ValueError(f"formal frozen input identity changed: {failures}")
    if int(trees) != FORMAL_TREES or int(random_seed) != FORMAL_RANDOM_SEED:
        raise ValueError("formal trees/random seed changed")
    if action_v1.SCHEMA != EXPECTED_TRAINER_SCHEMA:
        raise ValueError("formal trainer schema changed")
    if (
        action_v1.PAIR_WIDTH != 1062
        or pair_v0.FEATURE_SCHEMA["shared_width"] != 908
        or pair_v0.FEATURE_SCHEMA["relative_action_width"] != 154
    ):
        raise ValueError("formal pair1062 feature schema changed")
    return {
        **observed, "trainer_schema": action_v1.SCHEMA,
        "pair_schema": pair_v0.SCHEMA, "pair_width": 1062,
        "trees": FORMAL_TREES, "random_seed": FORMAL_RANDOM_SEED,
        "all_formal_inputs_and_parameters_locked": True,
    }


def audit_frozen_inputs(
    args: argparse.Namespace, panel: Mapping[str, Any],
) -> dict[str, Any]:
    materialized_path = args.materialized_root.resolve() / "FINAL_REPORT.json"
    diagnostic_path = args.fingerprint_root.resolve() / "DIAGNOSTIC_REPORT.json"
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    return frozen_identity(
        materialized_report=materialized, diagnostic_report=diagnostic,
        panel=panel,
        materialized_report_sha256=residual._sha256_file(materialized_path),
        diagnostic_report_sha256=residual._sha256_file(diagnostic_path),
        trees=args.trees, random_seed=args.random_seed,
    )


def choices_provenance(
    choices_path: Path, report_path: Path,
    input_provenance: Mapping[str, Any], frozen: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "action-prior-residual-A-choices-provenance-v1",
        "choices": _artifact(choices_path), "final_report": _artifact(report_path),
        "implementation": implementation_provenance(),
        "input_provenance": dict(input_provenance),
        "frozen_formal_identity": dict(frozen),
        "full_decision_coverage": True, "train_only": True,
        "validation_opened": False, "sealed_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays, panel, source_indices = load_full_panel(args)
    panel_audit = audit_full_panel(arrays, panel, source_indices)
    frozen = audit_frozen_inputs(args, panel)
    result = action_v1.evaluate(
        arrays, trees=args.trees, random_seed=args.random_seed,
    )
    pair_audit = pair_runner.audit_pair_result(result)
    mapping = map_choice_rows(
        arrays, source_indices, result["A_source_indices"], result["A_choices"],
    )
    output.mkdir(parents=True, exist_ok=False)
    choices_path = output / "choices_compact1078_pair1062_action_prior_v1.npz"
    residual._atomic_npz(choices_path, mapping)
    reportable = {
        key: value for key, value in result.items()
        if key not in {
            "A_choices", "support_only_A_choices", "state_shuffle_A_choices",
            "A_source_indices",
        }
    }
    input_provenance = panel_runner.input_provenance(args, panel)
    report = {
        "schema": SCHEMA,
        "status": "action_prior_state_residual_screen_run_complete",
        "screen_result": result["status"],
        "evidence_boundary": {
            "train_only": True, "validation_opened": False,
            "sealed_opened": False, "repair_on_dev": True,
            "steps_selected_from_outcome_oracle": True,
            "unbiased_estimate": False, "not_final_model": True,
        },
        "pipeline": (
            "exact action-support empirical prior -> class-balanced state-action "
            "gain/risk residual -> KEEP; support-only and state-shuffle controls"
        ),
        "parameters": {
            "decision_steps": list(pair_runner.DEFAULT_STEPS),
            "trees": int(args.trees), "random_seed": int(args.random_seed),
        },
        "frozen_formal_identity": frozen,
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
            "# Action-prior -> state-residual A screen v1", "",
            f"Status: `{report['status']}`",
            f"Screen result: `{report['screen_result']}`",
            f"Pairs: {EXPECTED_PAIRS}; decisions: {EXPECTED_DECISIONS}.",
            "Frozen train-only repair-on-dev screen; not a final model.",
            "Validation and sealed artifacts remain unopened.", "",
        )),
    )
    residual._atomic_text(
        output / "choices_compact1078_pair1062_action_prior_v1_PROVENANCE.json",
        json.dumps(choices_provenance(
            choices_path, report_path, input_provenance, frozen,
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
        default=panel_runner.loader_v0.smoke.DEFAULT_MATERIALIZED,
    )
    result.add_argument(
        "--fingerprint-root", type=Path,
        default=panel_runner.loader_v0.paired_smoke.DEFAULT_FINGERPRINTS,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--trees", type=int, default=FORMAL_TREES)
    result.add_argument("--random-seed", type=int, default=FORMAL_RANDOM_SEED)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
