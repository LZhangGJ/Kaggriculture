#!/usr/bin/env python3
"""Audit frozen H1 labels before training a phase-challenger residual gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "phase-challenger-dataset-preflight-v1"
R0 = "R0_active8"
PHASE = "phase_frozen"
FEATURE_WIDTH = 2844
FORMAL_DECISIONS = 4032
FORMAL_CANDIDATES = 170016
EXPECTED_SHA = {
    "formal_report": "9319fe6fa199066538f68d94e881f89e6c3a7531cd96829a6363538b0493f3c6",
    "formal_decisions": "b14beb4507034437fecc27069a02f8f7039ea5b1c013f22ec831c6d66091d971",
    "formal_candidates": "085523699c9301f15b3eeb0b2d74b35bbb974d4d8c6f2c8ca615553f79b4bfa8",
    "union_manifest": "d41f66757e19934213f6fc0bd35cb1d247d2ebf14858b618db32ed009a7af6a6",
}
DEFAULT_FORMAL_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\phase_breadth_h1_offsets_v1_formal_20260829a"
)
DEFAULT_UNION_MANIFEST = (
    DEFAULT_FORMAL_ROOT.parent
    / "h1_offset_router_lopo_v1_derived_20260829a"
    / "H1_OFFSET_ROUTER_LOPO_V1.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_FORMAL_ROOT.parent / "phase_challenger_dataset_preflight_v1"
)
DEFAULT_FAILED_ROOT = DEFAULT_OUTPUT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def verify_inputs(formal_root: Path, union_manifest: Path) -> dict[str, Any]:
    paths = {
        "formal_report": formal_root / "FINAL_REPORT.json",
        "formal_decisions": formal_root / "h1_decisions.jsonl",
        "formal_candidates": formal_root / "h1_candidate_evaluations.jsonl",
        "union_manifest": union_manifest,
    }
    result = {}
    for name, path in paths.items():
        value = artifact(
            path,
            FORMAL_DECISIONS if name == "formal_decisions" else (
                FORMAL_CANDIDATES if name == "formal_candidates" else None
            ),
        )
        if value["sha256"] != EXPECTED_SHA[name]:
            raise ValueError(f"frozen {name} SHA changed: {value['sha256']}")
        result[name] = value
    report = json.loads(paths["formal_report"].read_text(encoding="utf-8"))
    manifest = json.loads(union_manifest.read_text(encoding="utf-8"))
    if report.get("schema") != "phase-breadth-h1-offsets-v1":
        raise ValueError("formal report schema changed")
    if manifest.get("schema") != "h1-offset-router-lopo-v1":
        raise ValueError("union manifest schema changed")
    if manifest.get("union_contract", {}).get("definition") != (
        "ordered canonical SHA union: all R0 candidates, then phase-only candidates"
    ):
        raise ValueError("union ordering contract changed")
    return result


def arm_map(arm: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    shas = list(map(str, arm["candidate_sha256"]))
    markets = list(map(int, arm["market_diff"]))
    if len(shas) != len(markets) or len(shas) != len(set(shas)):
        raise ValueError("arm SHA/market alignment or uniqueness changed")
    return shas, dict(zip(shas, markets, strict=True))


def canonical_union_index(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Build the frozen R0-prefix/phase-only-suffix index for one state."""

    r0_sha, r0_market = arm_map(decision["arms"][R0])
    phase_sha, phase_market = arm_map(decision["arms"][PHASE])
    if not r0_sha or not phase_sha or r0_sha[0] != phase_sha[0]:
        raise ValueError("R0/phase lost their shared KEEP candidate")
    shared = set(r0_sha) & set(phase_sha)
    if any(r0_market[sha] != phase_market[sha] for sha in shared):
        raise ValueError("shared canonical SHA changed market mask")
    r0_set = set(r0_sha)
    phase_only = [sha for sha in phase_sha if sha not in r0_set]
    union = [*r0_sha, *phase_only]
    if union[: len(r0_sha)] != r0_sha or not r0_set.issubset(union):
        raise AssertionError("canonical R0 subset/prefix contract failed")
    market = {**r0_market, **phase_market}
    return {
        "state_id": str(decision["state_id"]),
        "split": str(decision["split"]),
        "opponent": str(decision["opponent"]),
        "seed": int(decision["seed"]),
        "seat": int(decision["seat"]),
        "anchor": int(decision["anchor"]),
        "offset": int(decision["offset"]),
        "step": int(decision["decision_step"]),
        "r0_sha256": r0_sha,
        "phase_sha256": phase_sha,
        "phase_only_sha256": phase_only,
        "union_sha256": union,
        "market_by_sha": market,
        "formal_r0": decision["arms"][R0]["all_candidates_oracle"],
        "formal_phase": decision["arms"][PHASE]["all_candidates_oracle"],
    }


def rank(label: Mapping[str, Any]) -> tuple[int, float]:
    return int(label["outcome"]), float(label["margin"])


def stable_best(shas: Sequence[str], labels: Mapping[str, Mapping[str, Any]]) -> str:
    if not shas:
        raise ValueError("best candidate requires a non-empty arm")
    best = shas[0]
    best_rank = rank(labels[best])
    for sha in shas[1:]:
        value = rank(labels[sha])
        if value > best_rank:
            best, best_rank = sha, value
    return best


def validate_state_labels(
    state: Mapping[str, Any],
    labels: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate labels, arm oracles, and the R0-first union tie-break."""

    union = list(state["union_sha256"])
    if set(union) - set(labels):
        raise ValueError(f"state {state['state_id']} is missing candidate labels")
    for sha in union:
        if int(labels[sha]["market_diff"]) != int(state["market_by_sha"][sha]):
            raise ValueError("candidate label/arm market mask changed")
    best_r0 = stable_best(state["r0_sha256"], labels)
    best_phase = stable_best(state["phase_sha256"], labels)
    for best, formal in (
        (best_r0, state["formal_r0"]),
        (best_phase, state["formal_phase"]),
    ):
        if (
            best != str(formal["best_sha256"])
            or rank(labels[best]) != (
                int(formal["best_outcome"]), float(formal["best_margin"])
            )
        ):
            raise ValueError("streamed labels do not reproduce formal arm oracle")
    best_union = stable_best(union, labels)
    best_r0_rank = rank(labels[best_r0])
    phase_only_ties = [
        sha for sha in state["phase_only_sha256"]
        if rank(labels[sha]) == best_r0_rank
    ]
    if (
        rank(labels[best_union]) == best_r0_rank
        and best_union not in set(state["r0_sha256"])
    ):
        raise AssertionError("phase-only tie displaced the R0-first winner")
    rows = []
    for index, sha in enumerate(union):
        label = labels[sha]
        phase_only = index >= len(state["r0_sha256"])
        rows.append({
            "schema": SCHEMA,
            "state_id": state["state_id"],
            "split": state["split"],
            "opponent": state["opponent"],
            "seed": state["seed"],
            "seat": state["seat"],
            "anchor": state["anchor"],
            "offset": state["offset"],
            "step": state["step"],
            "union_index": index,
            "membership": "phase_only" if phase_only else "R0",
            "candidate_sha256": sha,
            "outcome": int(label["outcome"]),
            "margin": float(label["margin"]),
            "market_diff": int(label["market_diff"]),
            "path_pure": bool(label["path_pure"]),
            "novelty_margin_vs_best_R0": (
                float(label["margin"]) - float(labels[best_r0]["margin"])
                if phase_only else None
            ),
            "novelty_rank_strict_vs_best_R0": (
                rank(label) > best_r0_rank if phase_only else None
            ),
            "deployment_target_ready_after_A_choice": phase_only,
        })
    return rows, {
        "best_R0_sha256": best_r0,
        "best_phase_sha256": best_phase,
        "best_union_sha256": best_union,
        "phase_only_ties_with_best_R0": len(phase_only_ties),
        "tie_break_R0_first_passed": True,
    }


def load_state_indices(path: Path) -> list[dict[str, Any]]:
    states = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            decision = json.loads(line)
            state = canonical_union_index(decision)
            if state["state_id"] in seen:
                raise ValueError("formal decision state_id is not unique")
            seen.add(state["state_id"])
            states.append(state)
    if len(states) != FORMAL_DECISIONS:
        raise ValueError(
            f"expected {FORMAL_DECISIONS} decisions, observed {len(states)}"
        )
    if sum(state["split"] == "train_proxy" for state in states) != 2016:
        raise ValueError("frozen train_proxy decision count changed")
    if any(
        state["split"] != "train_proxy" for state in states[:2016]
    ) or any(
        state["split"] != "heldout_proxy" for state in states[2016:]
    ):
        raise ValueError("formal train/heldout ordering changed")
    return states


def candidate_groups(
    path: Path,
) -> Iterable[tuple[str, dict[str, dict[str, Any]], int]]:
    current_id: str | None = None
    labels: dict[str, dict[str, Any]] = {}
    count = 0
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            state_id = str(row["state_id"])
            if current_id is not None and state_id != current_id:
                yield current_id, labels, count
                labels, count = {}, 0
            current_id = state_id
            sha = str(row["candidate_sha256"])
            if sha in labels:
                raise ValueError("R2 candidate SHA is duplicated within a state")
            labels[sha] = {
                "outcome": int(row["outcome"]),
                "margin": float(row["margin"]),
                "market_diff": int(row["market_diff"]),
                "path_pure": bool(row["path_pure"]),
            }
            count += 1
            total += 1
    if current_id is not None:
        yield current_id, labels, count
    if total != FORMAL_CANDIDATES:
        raise ValueError(
            f"expected {FORMAL_CANDIDATES} candidate labels, observed {total}"
        )


def _new_counts() -> dict[str, Any]:
    return {
        "decisions": 0,
        "r0_rows": 0,
        "phase_only_rows": 0,
        "deployment_target_rows": 0,
        "positive_novelty_rows": 0,
        "positive_novelty_decisions": 0,
        "market_changing_phase_rows": 0,
    }


def _add_counts(
    target: dict[str, Any],
    state: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> None:
    phase = [row for row in label_rows if row["membership"] == "phase_only"]
    positive = [row for row in phase if row["novelty_rank_strict_vs_best_R0"]]
    target["decisions"] += 1
    target["r0_rows"] += len(state["r0_sha256"])
    target["phase_only_rows"] += len(phase)
    target["deployment_target_rows"] += len(phase)
    target["positive_novelty_rows"] += len(positive)
    target["positive_novelty_decisions"] += bool(positive)
    target["market_changing_phase_rows"] += sum(
        int(row["market_diff"]) != 0 for row in phase
    )


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    if not ordered:
        return {key: 0.0 for key in ("mean", "median", "p90", "min", "max")}
    return {
        "mean": sum(ordered) / len(ordered),
        "median": ordered[(len(ordered) - 1) // 2],
        "p90": ordered[int(.9 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def memory_estimate(
    r0_rows: int,
    phase_rows: int,
    max_phase_rows_per_state: int,
    width: int = FEATURE_WIDTH,
) -> dict[str, Any]:
    scalar = 4
    a = int(r0_rows) * width * scalar
    phase = int(phase_rows) * width * scalar
    pair = int(phase_rows) * width * 2 * scalar
    shared = (int(r0_rows) + int(phase_rows)) * width * scalar
    scratch = int(max_phase_rows_per_state) * width * 2 * scalar
    naive = a + phase + pair
    lazy = shared + scratch
    return {
        "feature_width": width,
        "dtype": "float32",
        "dense_materialization": {
            "R0_backbone_bytes": a,
            "phase_base_bytes": phase,
            "phase_pair_5688D_bytes": pair,
            "total_bytes": naive,
        },
        "shared_candidate_encoding_plus_lazy_pair": {
            "candidate_base_memmap_bytes": shared,
            "maximum_pair_chunk_bytes": scratch,
            "total_peak_bytes": lazy,
            "saved_vs_dense_bytes": naive - lazy,
            "contract": (
                "encode each state-candidate row once; build [x_phase, "
                "x_phase-x_A_choice] one state at a time"
            ),
        },
        "no_sampling_required": True,
        "common_state_subvector_not_assumed": True,
    }


def signal_gate(
    train_counts: Mapping[str, Any],
    coverage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    positive_seed_folds: Sequence[int],
) -> dict[str, Any]:
    positive_opponents = [
        key for key, value in coverage["opponent"].items()
        if value["positive_novelty_decisions"] > 0
    ]
    positive_anchors = [
        key for key, value in coverage["anchor"].items()
        if value["positive_novelty_decisions"] > 0
    ]
    deployment_opponents = [
        key for key, value in coverage["opponent"].items()
        if value["deployment_target_rows"] > 0
    ]
    reasons = []
    if int(train_counts["positive_novelty_decisions"]) < 8:
        reasons.append("positive_novelty_decisions_below_8")
    if len(positive_opponents) < 2:
        reasons.append("positive_novelty_opponent_coverage_below_2")
    if len(positive_anchors) < 2:
        reasons.append("positive_novelty_anchor_coverage_below_2")
    if len(deployment_opponents) != 6:
        reasons.append("deployment_target_missing_lopo_opponent")
    if list(sorted(set(map(int, positive_seed_folds)))) != [0, 1, 2, 3]:
        reasons.append("positive_novelty_missing_preregistered_seed_fold")
    passed = not reasons
    return {
        "passed": passed,
        "status": "ready_for_training" if passed else "data_insufficient",
        "reasons": reasons,
        "positive_novelty_decisions": int(
            train_counts["positive_novelty_decisions"]
        ),
        "minimum_positive_novelty_decisions": 8,
        "positive_opponents": sorted(positive_opponents),
        "minimum_positive_opponents": 2,
        "positive_anchors": sorted(map(int, positive_anchors)),
        "minimum_positive_anchors": 2,
        "deployment_target_lopo_opponents": sorted(deployment_opponents),
        "positive_seed_folds": list(sorted(set(map(int, positive_seed_folds)))),
        "required_seed_folds": [0, 1, 2, 3],
        "recommend_train_seeds": None if passed else 8,
        "boundary": (
            "This is label-signal readiness only; it does not prove a zero-harm "
            "LOPO gate can select challengers."
        ),
    }


def _plain_coverage(
    coverage: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        dimension: {
            key: dict(value)
            for key, value in sorted(values.items())
        }
        for dimension, values in coverage.items()
    }


def _markdown(report: Mapping[str, Any]) -> str:
    gate = report["training_signal_gate"]
    counts = report["train_proxy"]
    memory = report["feature_memory_estimate"]
    dense = memory["dense_materialization"]["total_bytes"] / (1 << 20)
    shared = (
        memory["shared_candidate_encoding_plus_lazy_pair"]["total_peak_bytes"]
        / (1 << 20)
    )
    return "\n".join([
        "# PHASE-CHALLENGER-DATASET-PREFLIGHT-v1",
        "",
        f"- Status: `{report['status']}`",
        f"- Frozen decisions/candidate labels: {report['audit']['decisions']} / "
        f"{report['audit']['source_candidate_labels']}",
        f"- Train phase-only rows: {counts['phase_only_rows']}",
        f"- Train positive-novelty decisions: {counts['positive_novelty_decisions']}",
        f"- Deployment-target-ready rows: {counts['deployment_target_rows']}",
        f"- Later-training signal gate: `{gate['passed']}`; reasons: "
        f"{', '.join(gate['reasons']) or 'none'}",
        f"- 2844D dense estimate: {dense:.1f} MiB; shared base + lazy pair: "
        f"{shared:.1f} MiB.",
        "",
        "All 4032 states use an ordered R0 prefix followed only by new phase SHA "
        "candidates. Shared SHA market masks and the R0-first oracle tie-break were "
        "recomputed from the 170016 frozen labels.",
        "",
        "The market-changing candidates remain real joint-dynamics labels. This "
        "preflight does not train a selector, execute a candidate, or establish "
        "zero-harm deployment.",
        "",
    ])


def run(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    union_manifest = args.union_manifest.resolve()
    frozen_inputs = verify_inputs(formal_root, union_manifest)
    states = load_state_indices(formal_root / "h1_decisions.jsonl")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "canonical_state_index.jsonl"
    label_path = output / "canonical_candidate_labels.jsonl"
    state_temp = state_path.with_name(state_path.name + f".tmp.{os.getpid()}")
    label_temp = label_path.with_name(label_path.name + f".tmp.{os.getpid()}")

    coverage: dict[str, defaultdict[str, dict[str, Any]]] = {
        name: defaultdict(_new_counts)
        for name in ("opponent", "anchor", "offset", "seed")
    }
    train_counts = _new_counts()
    all_counts = _new_counts()
    novelty_margins: list[float] = []
    positive_train_seeds: set[int] = set()
    max_phase_per_state = 0
    tie_states = label_count = source_candidate_count = 0
    groups = iter(candidate_groups(formal_root / "h1_candidate_evaluations.jsonl"))

    with (
        state_temp.open("w", encoding="utf-8", newline="\n") as state_handle,
        label_temp.open("w", encoding="utf-8", newline="\n") as label_handle,
    ):
        for state in states:
            try:
                state_id, labels, r2_count = next(groups)
            except StopIteration as exc:
                raise ValueError("candidate labels ended before decision states") from exc
            if state_id != state["state_id"]:
                raise ValueError(
                    f"decision/candidate state order changed: {state['state_id']} != {state_id}"
                )
            source_candidate_count += r2_count
            label_rows, audit = validate_state_labels(state, labels)
            phase_rows = [
                row for row in label_rows if row["membership"] == "phase_only"
            ]
            positive = [
                row for row in phase_rows if row["novelty_rank_strict_vs_best_R0"]
            ]
            max_phase_per_state = max(max_phase_per_state, len(phase_rows))
            tie_states += bool(audit["phase_only_ties_with_best_R0"])
            _add_counts(all_counts, state, label_rows)
            if state["split"] == "train_proxy":
                _add_counts(train_counts, state, label_rows)
                novelty_margins.extend(
                    float(row["novelty_margin_vs_best_R0"]) for row in phase_rows
                )
                if positive:
                    positive_train_seeds.add(int(state["seed"]))
                for dimension in coverage:
                    _add_counts(
                        coverage[dimension][str(state[dimension])], state, label_rows
                    )
            index_row = {
                "schema": SCHEMA,
                **{
                    key: state[key]
                    for key in (
                        "state_id", "split", "opponent", "seed", "seat",
                        "anchor", "offset", "step",
                    )
                },
                "r2_source_candidate_count": r2_count,
                "r0_candidate_count": len(state["r0_sha256"]),
                "phase_candidate_count": len(state["phase_sha256"]),
                "phase_only_candidate_count": len(state["phase_only_sha256"]),
                "union_candidate_count": len(state["union_sha256"]),
                "r0_candidate_sha256": state["r0_sha256"],
                "phase_only_candidate_sha256": state["phase_only_sha256"],
                "union_candidate_sha256": state["union_sha256"],
                "positive_phase_only_rows": len(positive),
                **audit,
            }
            state_handle.write(
                json.dumps(index_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            for row in label_rows:
                label_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                label_count += 1
    try:
        extra = next(groups)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError("candidate labels contain states absent from decisions")
    if source_candidate_count != FORMAL_CANDIDATES:
        raise ValueError("candidate source row accounting changed")
    os.replace(state_temp, state_path)
    os.replace(label_temp, label_path)

    train_seed_values = sorted({
        int(state["seed"]) for state in states if state["split"] == "train_proxy"
    })
    seed_fold = {seed: index % 4 for index, seed in enumerate(train_seed_values)}
    positive_folds = sorted(seed_fold[seed] for seed in positive_train_seeds)
    plain_coverage = _plain_coverage(coverage)
    gate = signal_gate(train_counts, plain_coverage, positive_folds)
    memory = memory_estimate(
        train_counts["r0_rows"], train_counts["phase_only_rows"],
        max_phase_per_state,
    )
    script_path = Path(__file__).resolve()
    test_path = script_path.parents[1] / "tests" / (
        "test_run_phase_challenger_dataset_preflight_v1.py"
    )
    failed_root = args.failed_run_root.resolve()
    failed_non_artifact = {
        "path": str(failed_root),
        "status": "incomplete_atomic_temporary_files_only",
        "excluded_from_artifacts": True,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(failed_root.glob("*.tmp.*"))
        ] if failed_root.is_dir() else [],
    }
    report = {
        "schema": SCHEMA,
        "status": gate["status"],
        "decision": (
            "Proceed to residual-gate training only after the pre-registered "
            "seed-fold coverage is expanded; never tune on proxy-heldout."
            if not gate["passed"] else
            "The frozen label panel meets the pre-registered signal gate."
        ),
        "frozen_inputs": frozen_inputs,
        "audit": {
            "decisions": len(states),
            "source_candidate_labels": source_candidate_count,
            "canonical_union_label_rows": label_count,
            "train_decisions": int(train_counts["decisions"]),
            "heldout_decisions": len(states) - int(train_counts["decisions"]),
            "R0_subset_and_exact_prefix_failures": 0,
            "shared_SHA_market_mask_failures": 0,
            "formal_arm_oracle_parity_failures": 0,
            "R0_first_tie_break_failures": 0,
            "states_with_phase_only_tie_at_best_R0_rank": tie_states,
            "all_rows_streamed_without_sampling": True,
        },
        "train_proxy": {
            **train_counts,
            "phase_only_novelty_margin_vs_best_R0": _quantiles(novelty_margins),
            "market_changing_candidates_included_as_joint_dynamics": True,
            "deployment_target_definition": (
                "phase candidate margin minus the later LOPO A-backbone choice margin"
            ),
            "deployment_target_rows_are_ready_but_not_yet_instantiated": True,
        },
        "all_splits": all_counts,
        "train_coverage": plain_coverage,
        "training_signal_gate": {
            **gate,
            "train_seed_fold_map": {
                str(seed): fold for seed, fold in seed_fold.items()
            },
            "heldout_used_for_gate_or_threshold": False,
        },
        "feature_memory_estimate": memory,
        "implementation": {
            "script": artifact(script_path),
            "tests": artifact(test_path),
        },
        "failed_run_non_artifact": failed_non_artifact,
        "evidence_boundary": {
            "selector_trained": False,
            "candidate_executed": False,
            "closed_loop_executed": False,
            "reward_oracle_called": False,
            "proxy_heldout_not_strict_baseline_OOD": True,
        },
        "artifacts": {
            state_path.name: artifact(state_path, len(states)),
            label_path.name: artifact(label_path, label_count),
        },
    }
    report_path = output / "PREFLIGHT_REPORT.json"
    markdown_path = output / "PREFLIGHT_REPORT.md"
    atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "phase_challenger_dataset_preflight_complete",
        "status": report["status"],
        "output": str(output),
        "decisions": len(states),
        "candidate_labels": source_candidate_count,
        "canonical_union_rows": label_count,
        "signal_passed": gate["passed"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    result.add_argument("--union-manifest", type=Path, default=DEFAULT_UNION_MANIFEST)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--failed-run-root", type=Path, default=DEFAULT_FAILED_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
