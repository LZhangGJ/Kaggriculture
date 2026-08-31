"""Measure whether an independently calibrated paired-seat risk veto has coverage."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import run_farmer_augment_union_mpc_v1 as gate
import run_phase_challenger_residual_nested_smoke_v1 as smoke
import run_phase_challenger_residual_paired_seat_two_head_smoke_v0 as paired_smoke
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_paired_seat_two_head_v0 as paired


SCHEMA = "phase-challenger-paired-seat-conformal-risk-diagnostic-v0"
DEFAULT_OUTPUT = smoke.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_paired_seat_conformal_risk_diag_v0_20260829a"
)


def zero_false_negative_cap(
    risk: np.ndarray, nonpositive: np.ndarray,
) -> float:
    values = np.asarray(risk, np.float64)
    labels = np.asarray(nonpositive, np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or not np.any(labels):
        raise ValueError("conformal cap requires aligned nonpositive examples")
    return float(np.nextafter(np.min(values[labels]), -np.inf))


def distribution(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, np.float64)
    if not len(data):
        return {}
    return {
        name: float(value) for name, value in zip(
            ("min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max"),
            np.quantile(data, [0, .05, .10, .25, .50, .75, .90, .95, 1]),
            strict=True,
        )
    }


def candidate_eligibility(
    arrays: Mapping[str, np.ndarray], utility: np.ndarray,
    risk: np.ndarray, cap: float,
) -> dict[str, Any]:
    candidate = np.asarray(arrays["edit"]) != base.KEEP_EDIT
    eligible = candidate & (np.asarray(utility) > 0.0) & (np.asarray(risk) <= cap)
    robust_nonpositive = np.asarray(arrays["target_nonpositive"], np.bool_)
    actual = np.asarray(arrays["delta_margin"], np.float64)
    return {
        "candidate_rows": int(np.count_nonzero(candidate)),
        "eligible_rows": int(np.count_nonzero(eligible)),
        "eligible_robust_nonpositive_rows": int(np.count_nonzero(
            eligible & robust_nonpositive
        )),
        "eligible_actual_negative_rows": int(np.count_nonzero(
            eligible & (actual < 0.0)
        )),
        "eligible_actual_positive_rows": int(np.count_nonzero(
            eligible & (actual > 0.0)
        )),
        "eligible_decisions": len(set(map(
            int, np.asarray(arrays["decision"])[eligible]
        ))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    paired_smoke._FINGERPRINT_ROOT = args.fingerprint_root.resolve()
    arrays, panel, source_indices = paired_smoke.load_arrays_with_groups(
        args.materialized_root.resolve(), "compact1078", args.decision_steps,
    )
    a, a_source = base.r0_arrays(arrays)
    a = paired._attach_group(a, arrays, a_source)
    robust, robust_audit = paired.robustify(a, "delta_margin")
    u_lopo, r_lopo, lopo_audit = paired.group_oof_two_head(
        robust, "opponent", args.trees, args.random_seed + 101,
    )
    u_loso, r_loso, loso_audit = paired.group_oof_two_head(
        robust, "seed", args.trees, args.random_seed + 1101,
    )
    utility, risk = paired.two_head_scores(
        np.concatenate([u_lopo, u_loso], axis=1),
        np.concatenate([r_lopo, r_loso], axis=1),
    )
    target = np.asarray(robust["target_nonpositive"], np.bool_)
    edits = np.asarray(robust["edit"]) != base.KEEP_EDIT
    candidate_target = target[edits]
    candidate_risk = risk[edits]
    fold_by_seed = gate.seed_fold_map(sorted(set(map(int, robust["seed"]))))
    row_folds = np.asarray([
        fold_by_seed[int(seed)] for seed in robust["seed"]
    ], np.int8)
    cross_folds = []
    selected_rows = []
    for fold in range(4):
        calibration = np.flatnonzero((row_folds != fold) & edits)
        held = np.flatnonzero(row_folds == fold)
        cap = zero_false_negative_cap(risk[calibration], target[calibration])
        held_arrays = {
            key: np.asarray(value)[held] for key, value in robust.items()
        }
        eligibility = candidate_eligibility(
            held_arrays, utility[held], risk[held], cap,
        )
        metrics, choices = paired.selection_metrics(
            held_arrays, utility[held], risk[held], {
                "utility_threshold": 0.0,
                "max_nonpositive_risk": cap,
            }, farmer_only=False,
        )
        for decision_rows, choice in zip(
            base.decision_slices(held_arrays["decision"]), choices, strict=True,
        ):
            if int(held_arrays["edit"][choice]) != int(base.KEEP_EDIT):
                selected_rows.append(int(held[int(choice)]))
        cross_folds.append({
            "left_out_seed_fold": fold,
            "calibration_rows": int(len(calibration)),
            "held_rows": int(len(held)),
            "calibration_nonpositive_rows_passing": int(np.count_nonzero(
                target[calibration] & (risk[calibration] <= cap)
            )),
            "conformal_cap": cap,
            "held_eligibility": eligibility,
            "held_selection": metrics,
        })
    selected = np.asarray(selected_rows, np.int64)
    selected_actual = np.asarray(robust["delta_margin"], np.float64)[selected]
    selected_opponents = sorted(set(map(
        int, np.asarray(robust["opponent"])[selected]
    ))) if len(selected) else []
    selected_folds = sorted(set(map(
        int, row_folds[selected]
    ))) if len(selected) else []

    focus = []
    materialized_rows = source_indices[a_source]
    for target_row in (33738, 35603):
        local = np.flatnonzero(materialized_rows == target_row)
        if len(local) == 1:
            row = int(local[0])
            focus.append({
                "materialized_row": target_row,
                "local_A_row": row,
                "actual_delta_margin": float(robust["delta_margin"][row]),
                "paired_worst_value": float(robust["paired_worst_value"][row]),
                "utility_score": float(utility[row]),
                "nonpositive_risk_score": float(risk[row]),
                "seed_fold": int(row_folds[row]),
            })
    fixed_caps = []
    for cap in (0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0):
        fixed_caps.append({
            "cap": cap,
            **candidate_eligibility(robust, utility, risk, cap),
        })
    if len(selected_actual) and np.any(selected_actual < 0.0):
        verdict = "conformal_veto_failed_on_cross_fold_harm"
    elif len(selected_actual) < 4 or len(selected_folds) < 4 or len(selected_opponents) < 2:
        verdict = "conformal_veto_zero_harm_but_coverage_insufficient"
    else:
        verdict = "conformal_veto_minimum_signal_passed_repair_on_dev_only"
    report = {
        "schema": SCHEMA,
        "status": "train_only_group_conformal_risk_diagnostic_complete",
        "verdict": verdict,
        "evidence_boundary": {
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "repair_on_dev": True,
            "phase_model_trained": False,
        },
        "panel": panel,
        "parameters": {
            "decision_steps": list(map(int, args.decision_steps)),
            "trees_per_group_model_per_head": int(args.trees),
            "utility_threshold": 0.0,
            "risk_cap_rule": (
                "nextafter(min calibration risk among paired-worst nonpositive "
                "candidate rows, -infinity)"
            ),
        },
        "paired_robustification": robust_audit,
        "risk_head": {
            "candidate_rows": int(np.count_nonzero(edits)),
            "candidate_nonpositive_prevalence": float(candidate_target.mean()),
            "roc_auc": float(roc_auc_score(candidate_target, candidate_risk)),
            "average_precision": float(average_precision_score(
                candidate_target, candidate_risk,
            )),
            "risk_distribution_nonpositive": distribution(
                candidate_risk[candidate_target]
            ),
            "risk_distribution_positive": distribution(
                candidate_risk[~candidate_target]
            ),
            "fixed_cap_eligibility": fixed_caps,
        },
        "cross_conformal_folds": cross_folds,
        "cross_conformal_selected": {
            "fires": int(len(selected)),
            "beneficial": int(np.count_nonzero(selected_actual > 0.0)),
            "harmful": int(np.count_nonzero(selected_actual < 0.0)),
            "realized_margin_sum": float(selected_actual.sum()),
            "seed_folds": selected_folds,
            "opponents": selected_opponents,
        },
        "focus_paired_rows": focus,
        "OOF_audit": {"opponent": lopo_audit, "seed": loso_audit},
        "models_fit": 2 * (len(lopo_audit) + len(loso_audit)),
        "trees_fit": 2 * args.trees * (len(lopo_audit) + len(loso_audit)),
        "input_contract": {
            "materialized_report_sha256": residual._sha256_file(
                args.materialized_root.resolve() / "FINAL_REPORT.json"
            ),
            "fingerprint_report_sha256": residual._sha256_file(
                args.fingerprint_root.resolve() / "DIAGNOSTIC_REPORT.json"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.md",
        "\n".join((
            "# Paired-seat conformal risk diagnostic v0",
            "", f"Status: `{report['status']}`",
            f"Verdict: `{verdict}`",
            "The risk cap is calibrated independently before utility selection.",
            "No validation or sealed artifact was opened.", "",
        )),
    )
    print(json.dumps({
        "event": report["status"], "verdict": verdict,
        "output": str(output),
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
        "--decision-steps", type=int, nargs="+",
        default=(217, 218, 219, 220),
    )
    result.add_argument("--trees", type=int, default=24)
    result.add_argument("--random-seed", type=int, default=20260829)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
