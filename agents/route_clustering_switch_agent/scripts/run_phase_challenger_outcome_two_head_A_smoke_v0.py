"""Run the train-only 16-step A-stage outcome two-head experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import run_phase_challenger_residual_nested_smoke_v1 as smoke
import run_phase_challenger_residual_paired_seat_two_head_smoke_v0 as paired_smoke
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_outcome_two_head_A_nested_v0 as outcome_v0
import train_phase_challenger_residual_gate_v1 as base


SCHEMA = "phase-challenger-outcome-two-head-A-smoke-v0"
DEFAULT_STEPS = (
    217, 218, 219, 266, 267, 268, 435, 436,
    457, 482, 483, 484, 529, 530, 531, 553,
)
DEFAULT_OUTPUT = smoke.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_outcome_two_head_A_smoke_v0_20260829a"
)


def load_arrays(args: argparse.Namespace):
    paired_smoke._FINGERPRINT_ROOT = args.fingerprint_root.resolve()
    arrays, panel, source_indices = paired_smoke.load_arrays_with_groups(
        args.materialized_root.resolve(), "compact1078", args.decision_steps,
    )
    diagnostic_report = json.loads(
        (args.fingerprint_root.resolve() / "DIAGNOSTIC_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = diagnostic_report["input_contract"]["legacy_extended"]
    extended_path = Path(legacy["extended_path"])
    if residual._sha256_file(extended_path) != legacy["extended_sha256"]:
        raise ValueError("validated outcome metadata hash changed")
    with np.load(extended_path, allow_pickle=False) as extended:
        outcome = np.asarray(extended["outcome"], np.int8)[source_indices]
    arrays["outcome"] = outcome
    panel.update({
        "outcome_encoding": "0=loss, 1=tie, 2=win",
        "outcome_metadata_sha256": legacy["extended_sha256"],
        "KEEP_wins": int(sum(
            int(outcome[rows[0]]) == 2
            for rows in base.decision_slices(arrays["decision"])
        )),
    })
    return arrays, panel, source_indices


def run(args: argparse.Namespace) -> dict:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays, panel, source_indices = load_arrays(args)
    result = outcome_v0.evaluate(
        arrays, trees=args.trees, random_seed=args.random_seed,
    )
    if result.get("status") != "nested_seed_fold_evaluation_complete":
        report = {
            "schema": SCHEMA,
            "status": "outcome_two_head_smoke_stopped",
            "evidence_boundary": {
                "train_only": True, "validation_opened": False,
                "sealed_opened": False, "repair_on_dev": True,
            },
            "panel": panel,
            "result": result,
        }
        output.mkdir(parents=True, exist_ok=False)
    else:
        choices = np.asarray(result["A_choices"], np.int64)
        decision_order = np.asarray([
            arrays["decision"][rows[0]]
            for rows in base.decision_slices(arrays["decision"])
        ], np.int64)
        output.mkdir(parents=True, exist_ok=False)
        choice_path = output / "choices_compact1078.npz"
        residual._atomic_npz(choice_path, {
            "decision": decision_order,
            "A_materialized_row": source_indices[choices],
            "final_materialized_row": source_indices[choices],
        })
        report = {
            "schema": SCHEMA,
            "status": "outcome_two_head_A_nested_smoke_complete",
            "evidence_boundary": {
                "train_only": True,
                "validation_opened": False,
                "sealed_opened": False,
                "repair_on_dev": True,
                "candidate_committed": False,
            },
            "pipeline": (
                "A-only outcome-upgrade/regression heads; KEEP fallback; phase disabled"
            ),
            "parameters": {
                "decision_steps": list(map(int, args.decision_steps)),
                "trees_per_group_model_per_head": int(args.trees),
                "random_seed": int(args.random_seed),
            },
            "panel": panel,
            "result": {
                key: value for key, value in result.items()
                if key not in {"A_choices", "final_choices"}
            },
            "artifacts": {
                "choices": {
                    "path": str(choice_path),
                    "sha256": residual._sha256_file(choice_path),
                    "bytes": choice_path.stat().st_size,
                }
            },
        }
    residual._atomic_text(
        output / "SMOKE_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "SMOKE_REPORT.md",
        "\n".join((
            "# A-stage outcome two-head smoke v0", "",
            f"Status: `{report['status']}`",
            f"Steps: `{list(map(int, args.decision_steps))}`",
            "Primary metric is raw win delta; phase is disabled.",
            "No validation or sealed artifact was opened.", "",
        )),
    )
    if "artifacts" in report:
        report_sha = residual._sha256_file(output / "SMOKE_REPORT.json")
        residual._atomic_text(
            output / "choices_compact1078_PROVENANCE.json",
            json.dumps({
                "schema": "outcome-two-head-A-choices-provenance-v0",
                "choices_sha256": report["artifacts"]["choices"]["sha256"],
                "smoke_report_sha256": report_sha,
                "materialized_report_sha256": residual._sha256_file(
                    args.materialized_root.resolve() / "FINAL_REPORT.json"
                ),
                "fingerprint_report_sha256": residual._sha256_file(
                    args.fingerprint_root.resolve() / "DIAGNOSTIC_REPORT.json"
                ),
                "train_only": True,
                "validation_opened": False,
                "sealed_opened": False,
            }, ensure_ascii=False, indent=2) + "\n",
        )
    print(json.dumps({
        "event": report["status"], "output": str(output),
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--materialized-root", type=Path, default=smoke.DEFAULT_MATERIALIZED,
    )
    result.add_argument(
        "--fingerprint-root", type=Path,
        default=paired_smoke.DEFAULT_FINGERPRINTS,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--decision-steps", type=int, nargs="+", default=DEFAULT_STEPS,
    )
    result.add_argument("--trees", type=int, default=24)
    result.add_argument("--random-seed", type=int, default=20260829)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
