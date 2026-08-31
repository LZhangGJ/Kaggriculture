"""Run a train-only real-feature smoke for the direct U_phase nested evaluator.

This is plumbing evidence only.  It selects one complete H1 decision per
opponent/seed/seat trajectory from the frozen materialized panel, uses tiny
forests, and never opens validation or sealed data.
"""

from __future__ import annotations

import argparse
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

import run_contract_retrieval_oracle_ablation_v1 as retrieval
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_direct_ranker_v1 as direct


SCHEMA = "phase-challenger-direct-nested-real-feature-smoke-v1"
DEFAULT_MATERIALIZED = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\phase_challenger_compact_et_ab_v1_materialized_20260829a"
)
DEFAULT_OUTPUT = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_direct_nested_real_feature_smoke_v1"
)
REPRESENTATIONS = {"full2844": "full_2844.npy", "compact1078": "compact_1078.npy"}


def _artifact_matches(path: Path, expected: Mapping[str, Any]) -> None:
    if path.stat().st_size != int(expected["bytes"]):
        raise ValueError(f"artifact byte size changed: {path}")
    if residual._sha256_file(path) != str(expected["sha256"]):
        raise ValueError(f"artifact SHA changed: {path}")


def load_smoke_arrays(
    root: Path, representation: str, decision_step: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report_path = root / "FINAL_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "formal_materialization_complete_training_not_requested"
        or report.get("training_executed") is not False
        or report.get("runtime_boundary", {}).get("validation_opened") is not False
        or report.get("runtime_boundary", {}).get("sealed_opened") is not False
    ):
        raise ValueError("materialized panel crossed the train-only smoke boundary")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    storage = report["storage"]
    feature_path = root / REPRESENTATIONS[representation]
    metadata_path = root / "metadata.npz"
    _artifact_matches(feature_path, storage[representation])
    _artifact_matches(metadata_path, storage["metadata"])
    features = np.load(feature_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as raw:
        metadata = {key: raw[key] for key in raw.files}
    row_mask = np.asarray(metadata["step"], np.int64) == int(decision_step)
    indices = np.flatnonzero(row_mask)
    if not len(indices):
        raise ValueError(f"no complete decisions at step {decision_step}")
    selected_decisions = np.asarray(metadata["decision"], np.int64)[indices]
    starts = np.flatnonzero(np.r_[True, selected_decisions[1:] != selected_decisions[:-1]])
    if len(starts) != 96:
        raise ValueError("smoke must retain one decision for every 6x8x2 trajectory")
    arrays = {
        "features": np.asarray(features[indices], np.float32),
        "delta_margin": np.asarray(metadata["delta_margin"], np.float64)[indices],
        "membership": np.asarray(metadata["phase_only"], np.int8)[indices],
        "decision": selected_decisions,
        "opponent": np.asarray(metadata["opponent"], np.int16)[indices],
        "seed": np.asarray(metadata["seed"], np.int64)[indices],
        "edit": np.asarray(metadata["edit"], np.int8)[indices],
    }
    audit = direct.validate(arrays)
    if len(audit["opponents"]) != 6 or len(audit["seeds"]) != 8:
        raise ValueError("real-feature smoke lost opponent or seed coverage")
    return arrays, {
        "representation": representation,
        "decision_step": int(decision_step),
        "rows": int(len(indices)),
        "decisions": int(audit["decisions"]),
        "opponents": audit["opponents"],
        "seeds": audit["seeds"],
        "feature_width": int(audit["feature_width"]),
    }


def result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    nested = result["seed_fold_fixed_threshold"]
    return {
        "status": result["status"],
        "input_audit": result["input_audit"],
        "signal": result["signal"],
        "deployment_calibration": {
            "chosen": result["calibration"]["chosen"],
            "used_for_nested_outer_evaluation": result["calibration"][
                "used_for_nested_outer_evaluation"
            ],
        },
        "nested_outer": {
            key: nested[key]
            for key in (
                "scheme", "global_calibration_used_for_outer",
                "outer_labels_used_to_select_threshold", "complete_four_folds",
                "overall", "by_seed_fold", "by_opponent", "opponents_with_fire",
                "seeds_with_fire", "zero_harm_each_seed_fold",
                "zero_harm_each_opponent", "zero_harm_fixed_threshold",
            )
        },
        "nested_fold_audit": [{
            "left_out_seed_fold": row["left_out_seed_fold"],
            "train_seeds": row["train_seeds"],
            "valid_seeds": row["valid_seeds"],
            "threshold_selection_valid_seed_overlap": row[
                "threshold_selection_valid_seed_overlap"
            ],
            "inner_LOPO_complete": row["inner_LOPO_complete"],
            "inner_chosen": row["inner_chosen"],
            "outer_heldout": row["outer_heldout"],
        } for row in nested["folds"]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    core_results: dict[str, Any] = {}
    panel_audits: dict[str, Any] = {}
    for index, representation in enumerate(REPRESENTATIONS):
        arrays, panel = load_smoke_arrays(
            args.materialized_root.resolve(), representation, args.decision_step,
        )
        fit_started = time.perf_counter()
        core = direct.fit_direct_ranker(
            arrays, trees=args.trees, cv_trees=args.cv_trees,
            random_seed=args.random_seed + index * 100_003,
        )
        if core.get("status") != "fit_complete_train_only_core":
            raise RuntimeError(f"real-feature nested smoke did not fit: {core.get('status')}")
        panel_audits[representation] = panel
        results[representation] = {
            **result_summary(core),
            "elapsed_seconds": time.perf_counter() - fit_started,
        }
        core_results[representation] = core
    comparison = direct.compare_representations(
        core_results["full2844"], core_results["compact1078"],
    )
    report = {
        "schema": SCHEMA,
        "status": "real_feature_nested_plumbing_smoke_complete",
        "evidence_boundary": {
            "smoke_only": True,
            "not_model_effectiveness_evidence": True,
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "candidate_committed": False,
            "direct_union_only_not_R0_residual": True,
        },
        "input": {
            "materialized_root": str(args.materialized_root.resolve()),
            "materialized_report": retrieval._artifact(
                args.materialized_root.resolve() / "FINAL_REPORT.json"
            ),
        },
        "parameters": {
            "decision_step": args.decision_step, "trees": args.trees,
            "cv_trees": args.cv_trees, "random_seed": args.random_seed,
        },
        "panels": panel_audits,
        "representations": results,
        "comparison_diagnostic_only": comparison,
        "elapsed_seconds": time.perf_counter() - started,
    }
    residual._atomic_text(
        output / "SMOKE_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "SMOKE_REPORT.md",
        "\n".join((
            "# PHASE-CHALLENGER-DIRECT-NESTED real-feature smoke",
            "", f"Status: `{report['status']}`",
            f"Decisions per representation: {panel_audits['full2844']['decisions']}",
            f"Rows per representation: {panel_audits['full2844']['rows']}",
            "Nested outer seed folds use inner-LOPO calibration without held-seed overlap.",
            "This is plumbing-only direct-union evidence relative to KEEP, not an R0 residual result.",
            "",
        )),
    )
    print(json.dumps({
        "event": "phase_challenger_direct_nested_real_feature_smoke_complete",
        "output": str(output), "rows": panel_audits["full2844"]["rows"],
        "decisions": panel_audits["full2844"]["decisions"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--decision-step", type=int, default=217)
    result.add_argument("--trees", type=int, default=8)
    result.add_argument("--cv-trees", type=int, default=2)
    result.add_argument("--random-seed", type=int, default=20260829)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
