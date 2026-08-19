from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-generation",
        type=Path,
        default=PROJECT / "receipts" / "m37b_emergency_feed_trace_generation_v7.json",
    )
    parser.add_argument(
        "--baseline-diagnostic",
        type=Path,
        default=PROJECT / "receipts" / "m37b_animal_pipeline_diagnostic_v1.json",
    )
    parser.add_argument(
        "--candidate-generation",
        type=Path,
        default=PROJECT / "receipts" / "m37b_day_feasible_animal_trace_generation_v9.json",
    )
    parser.add_argument(
        "--candidate-diagnostic",
        type=Path,
        default=PROJECT / "receipts" / "m37b_animal_pipeline_diagnostic_v9.json",
    )
    parser.add_argument(
        "--candidate-lifecycle",
        type=Path,
        default=PROJECT / "receipts" / "m37a_detailed_lifecycle_v9.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m37b_animal_pipeline_acceptance_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_generation = load(args.baseline_generation)
    baseline = load(args.baseline_diagnostic)
    candidate_generation = load(args.candidate_generation)
    candidate = load(args.candidate_diagnostic)
    lifecycle = load(args.candidate_lifecycle)

    commitments = candidate_generation["calendar_source"]["filled_animal_additions"]
    commitment_count = int(sum(commitments))
    lifecycle_rows = [
        row
        for species_rows in lifecycle["animal_commitment_lifecycle"].values()
        for row in species_rows
    ]
    admitted_to_active = [
        row["stage_steps"]["ACTIVE"] - row["stage_steps"]["ADMITTED"]
        for row in lifecycle_rows
        if row["stage_steps"]["ACTIVE"] is not None
        and row["stage_steps"]["ADMITTED"] is not None
    ]
    summary = candidate["summary"]
    checks = {
        "official_1327": candidate_generation["official_version"] == "1.32.7",
        "gpu_trace": candidate_generation["backend"] == "gpu",
        "raw_replay_actions_absent": not candidate_generation[
            "raw_replay_action_payload_stored"
        ],
        "full_719_transition_trace": candidate_generation["steps"] == 719,
        "hard_error_zero": candidate_generation["hard_error_count"] == 0
        and lifecycle["hard_error_count"] == 0,
        "one_place_task_per_filled_commitment": summary[
            "place_task_instance_count"
        ]
        == commitment_count,
        "every_place_task_emitted_terminal_place": summary[
            "place_task_with_terminal_place_action_count"
        ]
        == commitment_count
        and summary["place_task_without_terminal_place_action_count"] == 0,
        "no_route_known_to_cross_day_reset": summary[
            "hand_route_exceeds_day_end_count"
        ]
        == 0,
        "no_day_boundary_task_clear_without_place": summary[
            "hand_cleared_at_day_boundary_without_place_count"
        ]
        == 0,
        "all_commitments_reached_active": len(lifecycle_rows) == commitment_count
        and all(row["stage_steps"]["ACTIVE"] is not None for row in lifecycle_rows),
        "no_commitment_execution_failure": all(
            not row["execution_failure"] for row in lifecycle_rows
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "receipt_id": "M37B_ANIMAL_PIPELINE_ACCEPTANCE_V1",
        "status": status,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "M3.7B purchased-animal BUILD/PICKUP/PLACE pipeline only",
        "explicit_non_claims": [
            "M3.7D gold milestone acceptance",
            "M3.7E economic gate",
            "formal route-search permission",
        ],
        "fixed_seed": candidate_generation["seed"],
        "player": candidate_generation["player"],
        "opponent": candidate_generation["opponent"],
        "filled_animal_commitments": commitments,
        "filled_animal_commitment_count": commitment_count,
        "baseline": {
            "final_bank": baseline_generation["final_bank"],
            **baseline["summary"],
        },
        "candidate": {
            "final_bank_diagnostic_only": candidate_generation["final_bank"],
            **summary,
            "max_admitted_to_active_transitions": max(admitted_to_active, default=0),
        },
        "checks": checks,
        "artifacts": {
            str(path.resolve()): sha256(path)
            for path in (
                args.baseline_generation,
                args.baseline_diagnostic,
                args.candidate_generation,
                args.candidate_diagnostic,
                args.candidate_lifecycle,
                PROJECT / "src" / "project_route_search_v2" / "m3_controller.py",
                PROJECT / "tools" / "analyze_m37b_animal_pipeline_v1.py",
            )
        },
        "reproduce_commands": [
            "wsl.exe -d Ubuntu-24.04 -- env XLA_PYTHON_CLIENT_PREALLOCATE=false /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim/.venv-wsl/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/project_route_search_v2/tools/generate_m36d_controller_trace_v1.py --require-gpu --include-internal-lifecycle --enable-m37-unlock --trace /mnt/e/ai_coding/kaggle/kaggriculture/experiments/project_route_search_v2/artifacts/traces/m37b_day_feasible_animal_trace_v9.npz --output /mnt/e/ai_coding/kaggle/kaggriculture/experiments/project_route_search_v2/receipts/m37b_day_feasible_animal_trace_generation_v9.json",
            "E:\\ai_coding\\kaggle\\kaggriculture\\.venv\\python.exe tools\\analyze_m37b_animal_pipeline_v1.py --trace artifacts\\traces\\m37b_day_feasible_animal_trace_v9.npz --output receipts\\m37b_animal_pipeline_diagnostic_v9.json",
            "E:\\ai_coding\\kaggle\\kaggriculture\\.venv\\python.exe tools\\accept_m37b_animal_pipeline_v1.py",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    print(args.output.resolve())
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
