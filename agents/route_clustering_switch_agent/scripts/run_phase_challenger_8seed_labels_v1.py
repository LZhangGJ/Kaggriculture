#!/usr/bin/env python3
"""Generate eight-seed train-only H1 labels for a phase residual gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_block_mvp_continuation_multitail_v1 as continuation
import run_contract_retrieval_oracle_ablation_v1 as retrieval_v1
import run_multi_farmer_path_residual_v1 as path_v1
import run_phase_breadth_h1_offsets_v1 as h1
import run_phase_challenger_dataset_preflight_v1 as preflight
import run_route_residual_adapter_v0 as residual


SCHEMA = "phase-challenger-8seed-labels-v1"
R2_ROW_SCHEMA = "phase-challenger-r2-candidate-label-v1"
TRAIN_OPPONENTS = tuple(path_v1.TRAIN_OPPONENTS)
TRAIN_SEEDS = tuple(path_v1.TRAIN_SEEDS)
SEATS = (0, 1)
OFFSETS = tuple(h1.OFFSETS)
EXPECTED_FORMAL_STATES = 8064
DEFAULT_OUTPUT_PARENT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_PARENT / "phase_challenger_8seed_labels_v1"


def scenario_contract(
    smoke: bool, requested_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    formal_seeds = tuple(map(int, requested_seeds or TRAIN_SEEDS))
    if len(formal_seeds) != 8 or len(set(formal_seeds)) != 8:
        raise ValueError("label generation requires exactly eight distinct seeds")
    opponents = TRAIN_OPPONENTS[:1] if smoke else TRAIN_OPPONENTS
    seeds = formal_seeds[:1] if smoke else formal_seeds
    anchors = path_v1.ANCHORS[:1] if smoke else path_v1.ANCHORS
    states = len(opponents) * len(seeds) * len(SEATS) * len(anchors) * len(OFFSETS)
    if not smoke and states != EXPECTED_FORMAL_STATES:
        raise AssertionError("formal eight-seed state budget changed")
    if (
        set(opponents) & set(path_v1.HELDOUT_OPPONENTS)
        or set(seeds) & set(path_v1.VALIDATION_SEEDS)
        or set(seeds) & set(residual.SEALED_SEEDS)
    ):
        raise ValueError("train-only scenario contract opened a sealed split")
    return {
        "opponents": opponents,
        "seeds": seeds,
        "seats": SEATS,
        "anchors": anchors,
        "offsets": OFFSETS,
        "states": states,
        "scenarios": tuple(
            (opponent, seed, seat)
            for opponent in opponents for seed in seeds for seat in SEATS
        ),
    }


def seed_fold_map(seeds: Sequence[int]) -> dict[int, int]:
    ordered = tuple(sorted(set(map(int, seeds))))
    if len(ordered) != (1 if len(seeds) == 1 else 8):
        raise ValueError("seed fold map requires smoke-one or formal-eight seeds")
    return {seed: index % 4 for index, seed in enumerate(ordered)}


def experiment_load_args(args: argparse.Namespace) -> argparse.Namespace:
    """Pin both loader lists to train so its empty-list fallback cannot open heldout."""

    return argparse.Namespace(
        frozen_run=args.frozen_run,
        prepared_root=args.prepared_root,
        v1_root=args.v1_root,
        block_library=args.block_library,
        heldout_team=None,
        base_genome_id="NR020",
        composition_genome_id=None,
        train_opponent=list(TRAIN_OPPONENTS),
        validation_opponent=list(TRAIN_OPPONENTS),
    )


def _candidate_panels(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        state_id = str(row["state_id"])
        if int(row["candidate_index"]) != len(result[state_id]):
            raise ValueError("R2 candidate rows lost canonical index order")
        result[state_id].append(row)
    return dict(result)


def summarize_scenario(
    decisions: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    fold_by_seed: Mapping[int, int],
    row_schema: str = SCHEMA,
) -> dict[str, Any]:
    panels = _candidate_panels(candidate_rows)
    counts = preflight._new_counts()
    coverage: dict[str, defaultdict[str, dict[str, Any]]] = {
        name: defaultdict(preflight._new_counts)
        for name in ("opponent", "anchor", "offset", "seed", "seed_fold")
    }
    positive_seed_folds: set[int] = set()
    phase_positive_market_rows = phase_only_tie_states = 0
    canonical_rows: list[dict[str, Any]] = []
    for decision in decisions:
        state = preflight.canonical_union_index(decision)
        if state["split"] != "train_proxy":
            raise ValueError("eight-seed runner received a non-train decision")
        try:
            panel = panels[state["state_id"]]
            expected_r2 = decision["arms"]["R2_all"]
            if (
                [str(row["candidate_sha256"]) for row in panel]
                != list(map(str, expected_r2["candidate_sha256"]))
                or [int(row["market_diff"]) for row in panel]
                != list(map(int, expected_r2["market_diff"]))
            ):
                raise ValueError("physical R2 SHA/market rows changed before discard")
            labels = {
                str(row["candidate_sha256"]): {
                    "outcome": int(row["outcome"]),
                    "margin": float(row["margin"]),
                    "market_diff": int(row["market_diff"]),
                    "path_pure": bool(row["path_pure"]),
                }
                for row in panel
            }
            if len(labels) != len(panel):
                raise ValueError("R2 candidate SHA duplicated within one state")
            rows, audit = preflight.validate_state_labels(
                state, labels,
            )
        except KeyError as exc:
            raise ValueError("decision/candidate state panels do not align") from exc
        phase = [row for row in rows if row["membership"] == "phase_only"]
        positive = [row for row in phase if row["novelty_rank_strict_vs_best_R0"]]
        phase_positive_market_rows += sum(
            int(row["market_diff"]) != 0 for row in positive
        )
        phase_only_tie_states += bool(audit["phase_only_ties_with_best_R0"])
        canonical_rows.extend({**row, "schema": row_schema} for row in rows)
        preflight._add_counts(counts, state, rows)
        fold = int(fold_by_seed[int(state["seed"])])
        if positive:
            positive_seed_folds.add(fold)
        for dimension, key in (
            ("opponent", state["opponent"]),
            ("anchor", state["anchor"]),
            ("offset", state["offset"]),
            ("seed", state["seed"]),
            ("seed_fold", fold),
        ):
            preflight._add_counts(coverage[dimension][str(key)], state, rows)
    if set(panels) != {str(row["state_id"]) for row in decisions}:
        raise ValueError("candidate labels contain a state absent from decisions")
    return {
        "counts": counts,
        "coverage": preflight._plain_coverage(coverage),
        "positive_seed_folds": sorted(positive_seed_folds),
        "phase_positive_market_changing_rows": phase_positive_market_rows,
        "phase_only_tie_states": phase_only_tie_states,
        "parity_failures": 0,
        "canonical_rows": canonical_rows,
        "physical_R2_candidate_rows_audited_then_discarded": len(candidate_rows),
    }


def merge_summary(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key, amount in value["counts"].items():
        target["counts"][key] += int(amount)
    for dimension, cells in value["coverage"].items():
        for cell, counts in cells.items():
            for key, amount in counts.items():
                target["coverage"][dimension][cell][key] += int(amount)
    target["positive_seed_folds"].update(value["positive_seed_folds"])
    target["phase_positive_market_changing_rows"] += int(
        value["phase_positive_market_changing_rows"]
    )
    target["phase_only_tie_states"] += int(value["phase_only_tie_states"])
    target["parity_failures"] += int(value["parity_failures"])


def empty_summary() -> dict[str, Any]:
    return {
        "counts": preflight._new_counts(),
        "coverage": {
            name: defaultdict(preflight._new_counts)
            for name in ("opponent", "anchor", "offset", "seed", "seed_fold")
        },
        "positive_seed_folds": set(),
        "phase_positive_market_changing_rows": 0,
        "phase_only_tie_states": 0,
        "parity_failures": 0,
    }


def _write_rows(handle: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown(report: Mapping[str, Any]) -> str:
    signal = report["phase_only_signal"]
    throughput = report["throughput"]
    return "\n".join([
        "# PHASE-CHALLENGER-8SEED-LABELS-v1",
        "",
        f"- Status: `{report['status']}`",
        f"- Train-only states: {report['scenario_contract']['states_observed']}",
        f"- Physical R2 batches: {throughput['physical_R2_batches']}",
        f"- Physical R2 candidate labels retained: "
        f"{throughput['physical_R2_candidate_rollouts']}",
        f"- Canonical union labels retained: "
        f"{report['scenario_contract']['canonical_union_label_rows']}",
        f"- Positive phase-only decisions: "
        f"{signal['positive_novelty_decisions']}",
        f"- Positive seed folds: {signal['positive_seed_folds']}",
        f"- Signal gate: `{signal['passed']}`; "
        f"reasons: {', '.join(signal['reasons']) or 'none'}",
        "",
        "Only TRAIN_OPPONENTS and TRAIN_SEEDS 2026086300..2026086307 were run. "
        "No heldout, validation, or sealed scenario was opened.",
        "",
        "Every state used one physical R2 batch. R0 and phase were canonical SHA "
        "slices, KEEP reproduced the frozen NR020 trajectory, and market_diff is "
        "retained per canonical union candidate.",
        "",
        "No model was trained and no candidate was committed.",
        "",
    ])


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = scenario_contract(
        args.smoke, getattr(args, "scenario_seeds", None),
    )
    report_schema = str(getattr(args, "report_schema", SCHEMA))
    label_schema = str(getattr(args, "label_schema", SCHEMA))
    r2_row_schema = str(getattr(args, "r2_row_schema", R2_ROW_SCHEMA))
    fold_by_seed = seed_fold_map(contract["seeds"])
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    breadth_report, lopo, phase_mapping, frozen_inputs = h1.load_frozen_phase_mapping(
        args.breadth_root, args.lopo_report,
    )
    load_args = experiment_load_args(args)
    (bundle, baseline_id, _, tapes, genomes, _, active_donors,
     experiment_inputs, source) = path_v1.load_experiment(load_args)
    strict = retrieval_v1.load_strict_library(args.block_library)
    genome = genomes["NR020"]
    for anchor in path_v1.ANCHORS:
        if {donor.block_id for donor in active_donors[anchor]} != {
            donor.block_id for donor in strict.active_by_anchor[anchor]
        }:
            raise ValueError(f"R0 active8 mismatch at {anchor}")

    decisions_path = output / "h1_decisions.jsonl"
    labels_path = output / "canonical_union_labels.jsonl"
    r2_labels_path = output / "R2_candidate_labels.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    temporary = {
        path: path.with_name(path.name + f".tmp.{os.getpid()}")
        for path in (decisions_path, labels_path, r2_labels_path, trajectories_path)
    }
    summary = empty_summary()
    decision_count = canonical_count = physical_r2_rows = 0
    trajectories: list[dict[str, Any]] = []
    with (
        temporary[decisions_path].open("w", encoding="utf-8", newline="\n") as decision_handle,
        temporary[labels_path].open("w", encoding="utf-8", newline="\n") as label_handle,
        temporary[r2_labels_path].open("w", encoding="utf-8", newline="\n") as r2_handle,
        temporary[trajectories_path].open("w", encoding="utf-8", newline="\n") as trajectory_handle,
    ):
        for index, (opponent, seed, seat) in enumerate(contract["scenarios"], 1):
            decisions, r2_rows, trajectory = h1._scenario(
                bundle, baseline_id, tapes, genome, strict, phase_mapping,
                opponent, int(seed), int(seat), "train_proxy",
                contract["anchors"], parity_check=index == 1,
            )
            scenario = summarize_scenario(
                decisions, r2_rows, fold_by_seed, label_schema,
            )
            if (
                int(trajectory["physical_r2_batches"]) != len(decisions)
                or int(trajectory["physical_candidate_rollouts"]) != len(r2_rows)
                or any(int(row["physical_r2_batch_count"]) != 1 for row in decisions)
            ):
                raise AssertionError("one physical R2 batch per state contract failed")
            _write_rows(decision_handle, decisions)
            _write_rows(label_handle, scenario["canonical_rows"])
            _write_rows(r2_handle, (
                {**row, "schema": r2_row_schema} for row in r2_rows
            ))
            _write_rows(trajectory_handle, [trajectory])
            merge_summary(summary, scenario)
            decision_count += len(decisions)
            canonical_count += len(scenario["canonical_rows"])
            physical_r2_rows += len(r2_rows)
            trajectories.append(trajectory)
            del r2_rows, decisions, scenario
            print(json.dumps({
                "event": "phase_challenger_8seed_progress",
                "scenario": index,
                "scenario_total": len(contract["scenarios"]),
                "opponent": opponent,
                "seed": int(seed),
                "seat": int(seat),
                "states": decision_count,
                "canonical_union_rows": canonical_count,
                "physical_R2_rows_retained": physical_r2_rows,
            }, sort_keys=True), flush=True)
    if decision_count != int(contract["states"]):
        raise AssertionError("observed state count changed")
    physical_batches = sum(int(row["physical_r2_batches"]) for row in trajectories)
    physical_rollouts = sum(
        int(row["physical_candidate_rollouts"]) for row in trajectories
    )
    references = sum(int(row["reference_rollouts"]) for row in trajectories)
    native_seconds = sum(float(row["native_candidate_seconds"]) for row in trajectories)
    if (
        physical_batches != decision_count
        or references != decision_count
        or physical_rollouts != physical_r2_rows
        or int(summary["counts"]["decisions"]) != decision_count
    ):
        raise AssertionError("physical/streaming accounting changed")

    plain_coverage = preflight._plain_coverage(summary["coverage"])
    signal = preflight.signal_gate(
        summary["counts"], plain_coverage,
        sorted(summary["positive_seed_folds"]),
    )
    status = (
        "smoke_passed" if args.smoke else (
            "label_signal_ready_for_residual_gate_training"
            if signal["passed"] else "label_signal_data_insufficient_stop"
        )
    )
    for path, temp in temporary.items():
        os.replace(temp, path)
    script_path = Path(__file__).resolve()
    test_path = script_path.parents[1] / "tests" / (
        "test_run_phase_challenger_8seed_labels_v1.py"
    )
    report = {
        "schema": report_schema,
        "status": status,
        "decision": (
            "Stop before model training because the pre-registered four-fold, "
            "opponent, or anchor signal gate is insufficient."
            if not args.smoke and not signal["passed"] else
            "The label panel may proceed to a separately pre-registered model run; "
            "this experiment itself trained nothing."
        ),
        "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020",
            "route_ids": list(genome),
            "sha256": path_v1._sha256(list(genome)),
        },
        "frozen_phase_mapping": {
            str(anchor): arm for anchor, arm in sorted(phase_mapping.items())
        },
        "scenario_contract": {
            "train_opponents": list(contract["opponents"]),
            "train_seeds": list(map(int, contract["seeds"])),
            "seats": list(SEATS),
            "anchors": list(map(int, contract["anchors"])),
            "offsets": list(map(int, OFFSETS)),
            "scenarios": len(contract["scenarios"]),
            "states_expected": int(contract["states"]),
            "states_observed": decision_count,
            "canonical_union_label_rows": canonical_count,
            "physical_R2_candidate_label_rows": physical_r2_rows,
            "heldout_opponents_opened": [],
            "validation_seeds_opened": [],
            "sealed_seeds_opened": [],
        },
        "phase_only_signal": {
            **signal,
            "counts": dict(summary["counts"]),
            "coverage": plain_coverage,
            "seed_fold_map": {
                str(seed): fold for seed, fold in fold_by_seed.items()
            },
            "phase_positive_market_changing_rows": int(
                summary["phase_positive_market_changing_rows"]
            ),
            "phase_only_tie_states": int(summary["phase_only_tie_states"]),
            "R0_prefix_market_oracle_parity_failures": int(
                summary["parity_failures"]
            ),
            "heldout_used_for_signal_or_threshold": False,
        },
        "throughput": {
            "physical_R2_batches": physical_batches,
            "physical_R2_candidate_rollouts": physical_rollouts,
            "physical_R2_candidate_labels_retained": physical_r2_rows,
            "reference_rollouts": references,
            "native_candidate_seconds": native_seconds,
            "candidate_rollouts_per_second": (
                physical_rollouts / native_seconds if native_seconds else 0.0
            ),
            "one_R2_physical_batch_per_state": True,
            "R0_phase_reuse": "canonical SHA slices of the one R2 result",
        },
        "storage_contract": {
            "retained": [
                "h1 decisions", "canonical R0-prefix plus phase-only labels",
                "all physical R2 candidate labels and provenance",
                "one trajectory audit per scenario",
            ],
            "full_R2_candidate_JSONL_retained": True,
            "full_R2_rows": "retained in physical state/candidate_index order",
        },
        "implementation": {
            "runner": retrieval_v1._artifact(script_path),
            "tests": retrieval_v1._artifact(test_path),
            "H1_runtime": retrieval_v1._artifact(Path(h1.__file__).resolve()),
            "preflight_runtime": retrieval_v1._artifact(
                Path(preflight.__file__).resolve()
            ),
            "native_extension": retrieval_v1._artifact(
                Path(path_v1.native_extension.__file__).resolve()
            ),
        },
        "frozen_inputs": {
            **frozen_inputs,
            "breadth_status": breadth_report["status"],
            "lopo_status": lopo["status"],
        },
        "strict_library": strict.inputs,
        "experiment_inputs": experiment_inputs,
        "source": source,
        "evidence_boundary": {
            "train_only": True,
            "heldout_opened": False,
            "validation_opened": False,
            "sealed_opened": False,
            "selector_trained": False,
            "candidate_committed": False,
            "proxy_split_not_strict_baseline_OOD": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            decisions_path.name: retrieval_v1._artifact(
                decisions_path, decision_count
            ),
            labels_path.name: retrieval_v1._artifact(labels_path, canonical_count),
            r2_labels_path.name: retrieval_v1._artifact(
                r2_labels_path, physical_r2_rows
            ),
            trajectories_path.name: retrieval_v1._artifact(
                trajectories_path, len(trajectories)
            ),
        },
    }
    report_path = output / ("SMOKE_REPORT.json" if args.smoke else "FINAL_REPORT.json")
    markdown_path = output / ("SMOKE_REPORT.md" if args.smoke else "FINAL_REPORT.md")
    residual._atomic_text(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "phase_challenger_8seed_complete",
        "status": status,
        "output": str(output),
        "states": decision_count,
        "canonical_union_rows": canonical_count,
        "physical_R2_candidate_rollouts": physical_rollouts,
        "signal_passed": signal["passed"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument(
        "--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT,
    )
    result.add_argument(
        "--block-library", type=Path, default=retrieval_v1.DEFAULT_LIBRARY,
    )
    result.add_argument(
        "--breadth-root", type=Path, default=h1.DEFAULT_BREADTH_ROOT,
    )
    result.add_argument("--lopo-report", type=Path, default=h1.DEFAULT_LOPO)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
