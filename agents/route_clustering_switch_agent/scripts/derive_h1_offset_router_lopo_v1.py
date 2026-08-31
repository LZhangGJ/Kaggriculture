#!/usr/bin/env python3
"""Zero-rollout LOPO offset router and R0+phase union audit for H1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "h1-offset-router-lopo-v1"
R0 = "R0_active8"
PHASE = "phase_frozen"
R2 = "R2_all"
UNION = "R0_phase_canonical_union"
PRIMARY_VIEW = "path_pure_oracle"
OFFSETS = (1, 2, 3, 4)
STABILITY_FOLDS_REQUIRED = 5
MEAN_OVERHEAD_RATIO_GUARDRAIL = 1.25
P90_EXTRA_CANDIDATE_GUARDRAIL = 10
EXPECTED_REPORT_SHA256 = (
    "9319fe6fa199066538f68d94e881f89e6c3a7531cd96829a6363538b0493f3c6"
)
EXPECTED_ROWS_SHA256 = (
    "b14beb4507034437fecc27069a02f8f7039ea5b1c013f22ec831c6d66091d971"
)
DEFAULT_FORMAL_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\phase_breadth_h1_offsets_v1_formal_20260829a"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_FORMAL_ROOT.parent / "h1_offset_router_lopo_v1_derived_20260829a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _arm_view(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return row["arms"][arm][PRIMARY_VIEW]


def _rank(value: Mapping[str, Any]) -> tuple[int, float]:
    return int(value["best_outcome"]), float(value["best_margin"])


def _union_view(row: Mapping[str, Any]) -> dict[str, Any]:
    r0 = _arm_view(row, R0)
    phase = _arm_view(row, PHASE)
    winner = phase if _rank(phase) > _rank(r0) else r0
    result = dict(winner)
    result["source_arm"] = PHASE if winner is phase else R0
    result["phase_only_challenger_won"] = bool(
        winner is phase
        and winner["best_sha256"]
        not in set(row["arms"][R0]["candidate_sha256"])
    )
    return result


def _candidate_map(arm: Mapping[str, Any]) -> dict[str, bool]:
    shas = list(map(str, arm["candidate_sha256"]))
    pure = list(map(bool, arm["path_pure_mask"]))
    if len(shas) != len(pure) or len(set(shas)) != len(shas):
        raise ValueError("arm candidate SHA/mask alignment changed")
    return dict(zip(shas, pure, strict=True))


def union_candidate_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    r0 = _candidate_map(row["arms"][R0])
    phase = _candidate_map(row["arms"][PHASE])
    shared = set(r0) & set(phase)
    if any(r0[sha] != phase[sha] for sha in shared):
        raise ValueError("shared canonical SHA changed path-pure mask between arms")
    ordered = list(r0)
    ordered.extend(sha for sha in phase if sha not in r0)
    pure = {**r0, **phase}
    phase_only = [sha for sha in phase if sha not in r0]
    return {
        "candidate_count": len(ordered),
        "path_pure_candidate_count": sum(pure[sha] for sha in ordered),
        "r0_candidate_count": len(r0),
        "phase_candidate_count": len(phase),
        "shared_candidate_count": len(shared),
        "phase_only_candidate_count": len(phase_only),
        "extra_vs_r0": len(phase_only),
        "ratio_vs_r0": len(ordered) / len(r0),
        "r0_subset": set(r0).issubset(ordered),
        "candidate_sha256": ordered,
        "phase_only_sha256": phase_only,
    }


def _pair_stats(rows: Iterable[Mapping[str, Any]], arm: str) -> tuple[int, float]:
    values = [_arm_view(row, arm) for row in rows]
    return (
        sum(bool(value["strict_headroom"]) for value in values),
        sum(float(value["oracle_gain"]) for value in values),
    )


def pareto_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not rows:
        raise ValueError("Pareto choice requires non-empty rows")
    r0 = _pair_stats(rows, R0)
    phase = _pair_stats(rows, PHASE)
    dominates = bool(
        phase[0] >= r0[0]
        and phase[1] >= r0[1]
        and (phase[0] > r0[0] or phase[1] > r0[1])
    )
    return (PHASE if dominates else R0), {
        R0: {"positive_states": r0[0], "oracle_gain_sum": r0[1]},
        PHASE: {"positive_states": phase[0], "oracle_gain_sum": phase[1]},
        "phase_pareto_dominates_r0": dominates,
    }


def _percentile_nearest(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[int(fraction * (len(ordered) - 1))]


def lopo_mapping(
    train_rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Any],
) -> tuple[dict[Any, str], dict[str, Any], dict[str, dict[Any, str]]]:
    opponents = tuple(sorted({str(row["opponent"]) for row in train_rows}))
    if len(opponents) != 6:
        raise ValueError("LOPO requires exactly six train proxy opponents")
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[key_fn(row)].append(row)
    full: dict[Any, str] = {}
    full_audit: dict[Any, dict[str, Any]] = {}
    for key, values in grouped.items():
        full[key], full_audit[key] = pareto_choice(values)

    folds: dict[str, dict[Any, str]] = {}
    for left_out in opponents:
        fold_grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for row in train_rows:
            if str(row["opponent"]) != left_out:
                fold_grouped[key_fn(row)].append(row)
        if set(fold_grouped) != set(grouped):
            raise ValueError("LOPO fold cell coverage changed")
        folds[left_out] = {
            key: pareto_choice(values)[0]
            for key, values in fold_grouped.items()
        }

    frozen: dict[Any, str] = {}
    stability: dict[str, Any] = {}
    for key in sorted(full, key=str):
        choices = {opponent: folds[opponent][key] for opponent in opponents}
        counts = Counter(choices.values())
        phase_votes = int(counts[PHASE])
        stable_phase = bool(
            full[key] == PHASE and phase_votes >= STABILITY_FOLDS_REQUIRED
        )
        frozen[key] = PHASE if stable_phase else R0
        stability[str(key)] = {
            "full_train_choice": full[key],
            "full_train_pareto_audit": full_audit[key],
            "lopo_choice_by_left_out_opponent": choices,
            "lopo_choice_counts": {
                R0: int(counts[R0]), PHASE: phase_votes,
            },
            "phase_same_arm_fold_count": phase_votes,
            "required_phase_folds": STABILITY_FOLDS_REQUIRED,
            "stable_phase": stable_phase,
            "frozen_arm": frozen[key],
            "fallback_reason": (
                "stable_phase"
                if stable_phase
                else (
                    "full_train_phase_not_pareto_dominant"
                    if full[key] == R0
                    else "phase_lopo_below_5_of_6"
                )
            ),
        }
    return frozen, stability, folds


def _selected_view(
    row: Mapping[str, Any],
    selector: str | Callable[[Mapping[str, Any]], str],
) -> Mapping[str, Any]:
    arm = selector(row) if callable(selector) else selector
    return _union_view(row) if arm == UNION else _arm_view(row, arm)


def metrics(
    rows: Sequence[Mapping[str, Any]],
    selector: str | Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    r2_positive = [
        row for row in rows if _arm_view(row, R2)["strict_headroom"]
    ]
    selected = [_selected_view(row, selector) for row in rows]
    r2_gain = sum(float(_arm_view(row, R2)["oracle_gain"]) for row in rows)
    r0_gain = sum(float(_arm_view(row, R0)["oracle_gain"]) for row in rows)
    gain = sum(float(value["oracle_gain"]) for value in selected)
    recalled = sum(
        bool(_selected_view(row, selector)["strict_headroom"])
        for row in r2_positive
    )
    recoveries = sum(
        not bool(_arm_view(row, R0)["strict_headroom"])
        and bool(_selected_view(row, selector)["strict_headroom"])
        for row in r2_positive
    )
    regressions = sum(
        bool(_arm_view(row, R0)["strict_headroom"])
        and not bool(_selected_view(row, selector)["strict_headroom"])
        for row in r2_positive
    )
    return {
        "states": len(rows),
        "r2_positive_states": len(r2_positive),
        "positive_states": sum(bool(value["strict_headroom"]) for value in selected),
        "positive_state_recall": recalled / len(r2_positive) if r2_positive else 1.0,
        "oracle_gain_sum": gain,
        "oracle_gain_retention": gain / r2_gain if r2_gain > 0 else 1.0,
        "oracle_gain_uplift_vs_R0": gain - r0_gain,
        "coverage_recoveries_vs_R0": recoveries,
        "coverage_regressions_vs_R0": regressions,
        "loss_to_win_states": sum(bool(value["loss_to_win"]) for value in selected),
    }


def union_overhead(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [union_candidate_audit(row) for row in rows]
    candidate_counts = [value["candidate_count"] for value in audits]
    extras = [value["extra_vs_r0"] for value in audits]
    ratios = [value["ratio_vs_r0"] for value in audits]
    pure_counts = [value["path_pure_candidate_count"] for value in audits]
    return {
        "states": len(rows),
        "candidate_count": {
            "mean": statistics.mean(candidate_counts),
            "median": statistics.median(candidate_counts),
            "p90_nearest": _percentile_nearest(candidate_counts, 0.9),
            "min": min(candidate_counts),
            "max": max(candidate_counts),
        },
        "path_pure_candidate_count": {
            "mean": statistics.mean(pure_counts),
            "median": statistics.median(pure_counts),
            "p90_nearest": _percentile_nearest(pure_counts, 0.9),
            "min": min(pure_counts),
            "max": max(pure_counts),
        },
        "extra_candidates_vs_R0": {
            "mean": statistics.mean(extras),
            "median": statistics.median(extras),
            "p90_nearest": _percentile_nearest(extras, 0.9),
            "min": min(extras),
            "max": max(extras),
        },
        "candidate_ratio_vs_R0": {
            "mean": statistics.mean(ratios),
            "median": statistics.median(ratios),
            "p90_nearest": _percentile_nearest(ratios, 0.9),
            "min": min(ratios),
            "max": max(ratios),
        },
        "r0_subset_all_states": all(value["r0_subset"] for value in audits),
    }


def union_no_regression_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_regressions: list[str] = []
    gain_regressions: list[str] = []
    recovery_ids: list[str] = []
    phase_only_wins = 0
    for row in rows:
        r0 = _arm_view(row, R0)
        union = _union_view(row)
        if _rank(union) < _rank(r0):
            rank_regressions.append(str(row["state_id"]))
        if float(union["oracle_gain"]) < float(r0["oracle_gain"]):
            gain_regressions.append(str(row["state_id"]))
        if union["phase_only_challenger_won"]:
            phase_only_wins += 1
        if not r0["strict_headroom"] and union["strict_headroom"]:
            recovery_ids.append(str(row["state_id"]))
            if not union["phase_only_challenger_won"]:
                raise AssertionError("union recovery did not come from phase-only SHA")
    return {
        "passed": not rank_regressions and not gain_regressions,
        "rank_regression_count": len(rank_regressions),
        "gain_regression_count": len(gain_regressions),
        "rank_regression_state_ids": rank_regressions,
        "gain_regression_state_ids": gain_regressions,
        "phase_only_best_candidate_states": phase_only_wins,
        "phase_only_positive_recovery_count": len(recovery_ids),
        "phase_only_positive_recovery_state_ids": recovery_ids,
    }


def _fine_key(row: Mapping[str, Any]) -> str:
    return f"{int(row['anchor'])}:{int(row['offset'])}"


def derive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train_proxy"]
    heldout = [row for row in rows if row["split"] == "heldout_proxy"]
    if len(train) != 2016 or len(heldout) != 2016:
        raise ValueError("expected frozen 2016 train + 2016 heldout rows")

    offset_mapping, offset_stability, offset_folds = lopo_mapping(
        train, lambda row: int(row["offset"]),
    )
    if set(offset_mapping) != set(OFFSETS):
        raise ValueError("global offset mapping coverage changed")
    hard_selector = lambda row: offset_mapping[int(row["offset"])]

    fine_mapping, fine_stability, _ = lopo_mapping(train, _fine_key)
    if len(fine_mapping) != 84:
        raise ValueError("fine anchor-phase x offset diagnostic must have 84 cells")
    fine_selector = lambda row: fine_mapping[_fine_key(row)]
    train_cell_metadata: dict[str, tuple[int, int, str]] = {}
    for row in train:
        key = _fine_key(row)
        value = (
            int(row["anchor"]), int(row["offset"]),
            str(row["phase_pool_arm"]),
        )
        if key in train_cell_metadata and train_cell_metadata[key] != value:
            raise ValueError("phase arm changed within fine diagnostic cell")
        train_cell_metadata[key] = value
    for key, value in fine_stability.items():
        anchor, offset, phase_arm = train_cell_metadata[key]
        value.update({
            "anchor": anchor,
            "offset": offset,
            "phase_pool_arm": phase_arm,
            "diagnostic_only": True,
        })

    fold_results: dict[str, Any] = {}
    for left_out, mapping in sorted(offset_folds.items()):
        test = [row for row in train if str(row["opponent"]) == left_out]
        selector = lambda row, selected=mapping: selected[int(row["offset"])]
        fold_results[left_out] = {
            "left_out_opponent": left_out,
            "training_opponents": sorted(
                {str(row["opponent"]) for row in train} - {left_out}
            ),
            "selected_offset_mapping": {
                str(offset): arm for offset, arm in sorted(mapping.items())
            },
            "evaluation": {
                "fold_selected_hard_offset_router": metrics(test, selector),
                R0: metrics(test, R0),
                PHASE: metrics(test, PHASE),
                R2: metrics(test, R2),
            },
        }
    oof_selector = lambda row: offset_folds[str(row["opponent"])][
        int(row["offset"])
    ]
    lopo_cv = {
        "protocol": (
            "for each of six train opponents, select each offset arm with the "
            "other five using the preregistered Pareto rule, then evaluate the left-out opponent"
        ),
        "heldout_proxy_used": False,
        "folds": fold_results,
        "overall_oof": {
            "fold_selected_hard_offset_router": metrics(train, oof_selector),
            R0: metrics(train, R0),
            PHASE: metrics(train, PHASE),
            R2: metrics(train, R2),
        },
    }

    overhead = {
        "train_proxy": union_overhead(train),
        "heldout_proxy": union_overhead(heldout),
        "overall": union_overhead(rows),
    }
    no_regression = {
        "train_proxy": union_no_regression_audit(train),
        "heldout_proxy": union_no_regression_audit(heldout),
        "overall": union_no_regression_audit(rows),
    }
    train_metrics = {
        R0: metrics(train, R0),
        PHASE: metrics(train, PHASE),
        "frozen_hard_offset_router": metrics(train, hard_selector),
        UNION: metrics(train, UNION),
        R2: metrics(train, R2),
    }
    train_union = train_metrics[UNION]
    train_r0 = train_metrics[R0]
    train_union_pareto = bool(
        train_union["positive_state_recall"] >= train_r0["positive_state_recall"]
        and train_union["oracle_gain_retention"] >= train_r0["oracle_gain_retention"]
        and (
            train_union["positive_state_recall"] > train_r0["positive_state_recall"]
            or train_union["oracle_gain_retention"] > train_r0["oracle_gain_retention"]
        )
    )
    train_resource_guardrail = bool(
        overhead["train_proxy"]["candidate_ratio_vs_R0"]["mean"]
        <= MEAN_OVERHEAD_RATIO_GUARDRAIL
        and overhead["train_proxy"]["extra_candidates_vs_R0"]["p90_nearest"]
        <= P90_EXTRA_CANDIDATE_GUARDRAIL
    )
    recommend_union = bool(
        train_union_pareto
        and train_resource_guardrail
        and no_regression["train_proxy"]["passed"]
        and train_union["coverage_recoveries_vs_R0"] > 0
    )

    heldout_comparison = {
        R0: metrics(heldout, R0),
        PHASE: metrics(heldout, PHASE),
        "frozen_hard_offset_router": metrics(heldout, hard_selector),
        UNION: metrics(heldout, UNION),
        "fine_84_cell_router_diagnostic_only": metrics(heldout, fine_selector),
        R2: metrics(heldout, R2),
    }
    heldout_by_offset = {
        str(offset): {
            R0: metrics(
                [row for row in heldout if int(row["offset"]) == offset], R0,
            ),
            "frozen_hard_offset_router": metrics(
                [row for row in heldout if int(row["offset"]) == offset], hard_selector,
            ),
            UNION: metrics(
                [row for row in heldout if int(row["offset"]) == offset], UNION,
            ),
            R2: metrics(
                [row for row in heldout if int(row["offset"]) == offset], R2,
            ),
        }
        for offset in OFFSETS
    }
    hard = heldout_comparison["frozen_hard_offset_router"]
    r0_heldout = heldout_comparison[R0]
    hard_beats_both = bool(
        hard["positive_state_recall"] > r0_heldout["positive_state_recall"]
        and hard["oracle_gain_retention"] > r0_heldout["oracle_gain_retention"]
    )

    status = (
        "recommend_r0_backbone_phase_challenger_gate_training"
        if recommend_union
        else "residual_union_does_not_meet_train_guardrails"
    )
    return {
        "selection_contract": {
            "primary_view": PRIMARY_VIEW,
            "global_cells": "offset in {1,2,3,4}",
            "arms": [R0, PHASE],
            "phase_rule": (
                "phase must be Pareto non-worse than R0 on train positive-state count "
                "and oracle-gain sum, with at least one strict improvement"
            ),
            "tie_or_tradeoff": R0,
            "same_phase_arm_lopo_folds_required": STABILITY_FOLDS_REQUIRED,
            "lopo_fold_count": 6,
            "fallback": R0,
            "heldout_used_for_mapping_threshold_or_resource_guardrail": False,
        },
        "union_contract": {
            "definition": "ordered canonical SHA union: all R0 candidates, then phase-only candidates",
            "shared_sha_market_mask_must_match": True,
            "oracle_tie_break": "R0 candidate order first; phase wins only on strictly greater outcome-margin rank",
            "selector_executed": False,
            "interpretation": "candidate-set oracle upper bound for a future residual gate",
            "train_only_resource_guardrail": {
                "mean_candidate_ratio_vs_R0_max": MEAN_OVERHEAD_RATIO_GUARDRAIL,
                "p90_extra_candidates_max": P90_EXTRA_CANDIDATE_GUARDRAIL,
            },
        },
        "global_offset_arm_stability": offset_stability,
        "frozen_global_offset_mapping": {
            str(offset): arm for offset, arm in sorted(offset_mapping.items())
        },
        "global_offset_lopo_cv": lopo_cv,
        "fine_84_cell_diagnostic": {
            "diagnostic_only": True,
            "not_a_recommendation": True,
            "reason": "84 cells from only six train proxy opponents are high variance",
            "stable_phase_cells": sum(arm == PHASE for arm in fine_mapping.values()),
            "fallback_r0_cells": sum(arm == R0 for arm in fine_mapping.values()),
            "frozen_mapping": dict(sorted(fine_mapping.items())),
            "cell_stability": fine_stability,
        },
        "train_only_architecture_rule": {
            "union_pareto_dominates_r0": train_union_pareto,
            "union_resource_guardrail_passed": train_resource_guardrail,
            "union_no_regression_passed": no_regression["train_proxy"]["passed"],
            "union_has_positive_recoveries": (
                train_union["coverage_recoveries_vs_R0"] > 0
            ),
            "recommend_residual_union_architecture": recommend_union,
            "heldout_used": False,
        },
        "train_comparison": train_metrics,
        "union_candidate_overhead": overhead,
        "union_no_regression_audit": no_regression,
        "heldout_comparison": heldout_comparison,
        "heldout_by_offset": heldout_by_offset,
        "hard_offset_router_heldout_beats_r0_both": hard_beats_both,
        "status": status,
        "decision": (
            "Prefer R0 as the backbone and expose only canonical phase-only challengers "
            "to a future residual gate; the union is an oracle ceiling, not a trained selector."
            if recommend_union
            else "Do not advance the residual union until its train-only guardrails pass."
        ),
        "next_experiment": (
            "train and frozen-evaluate a residual gate over R0 plus phase-only challengers"
            if recommend_union
            else "reduce or rerank phase-only challenger overhead on train proxies"
        ),
    }


def _load_upstream(
    formal_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    report_path = formal_root / "FINAL_REPORT.json"
    rows_path = formal_root / "h1_decisions.jsonl"
    report_sha = _sha256(report_path)
    rows_sha = _sha256(rows_path)
    if report_sha != EXPECTED_REPORT_SHA256 or rows_sha != EXPECTED_ROWS_SHA256:
        raise ValueError("frozen H1 formal report or decision rows SHA changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pinned = report.get("artifacts", {}).get(rows_path.name, {})
    scenario = report.get("scenario_contract", {})
    if (
        report.get("schema") != "phase-breadth-h1-offsets-v1"
        or str(pinned.get("sha256", "")).lower() != rows_sha
        or int(pinned.get("rows", -1)) != 4032
        or int(scenario.get("decisions_observed", -1)) != 4032
        or list(map(int, scenario.get("offsets", ()))) != list(OFFSETS)
        or bool(scenario.get("candidate_committed", True))
    ):
        raise ValueError("upstream formal schema/row/commit contract changed")
    rows = [
        json.loads(line)
        for line in rows_path.open("r", encoding="utf-8")
        if line.strip()
    ]
    if len(rows) != 4032:
        raise ValueError("upstream H1 decision count changed")
    state_ids = {str(row["state_id"]) for row in rows}
    if len(state_ids) != len(rows):
        raise ValueError("upstream state IDs are not unique")

    split_opponents: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, int, int]] = Counter()
    frozen_phase = {
        int(anchor): str(arm)
        for anchor, arm in report["frozen_phase_mapping"].items()
    }
    for row in rows:
        split = str(row["split"])
        opponent = str(row["opponent"])
        anchor = int(row["anchor"])
        offset = int(row["offset"])
        split_opponents[split].add(opponent)
        counts[(opponent, anchor, offset)] += 1
        if (
            row.get("schema") != "phase-breadth-h1-offsets-v1"
            or int(row.get("physical_r2_batch_count", -1)) != 1
            or bool(row.get("selector_executed", True))
            or str(row["phase_pool_arm"]) != frozen_phase[anchor]
            or offset not in OFFSETS
        ):
            raise ValueError("upstream decision fairness contract changed")
        r2 = _candidate_map(row["arms"][R2])
        for arm in (R0, PHASE):
            candidates = _candidate_map(row["arms"][arm])
            if not set(candidates).issubset(r2):
                raise ValueError("online arm is not a canonical R2 subset")
            if any(candidates[sha] != r2[sha] for sha in candidates):
                raise ValueError("online arm market mask differs from R2")
            if row["arms"][arm].get("selector_executed") is not False:
                raise ValueError("upstream row unexpectedly executed a selector")
        union_candidate_audit(row)
    if (
        {key: len(value) for key, value in split_opponents.items()}
        != {"train_proxy": 6, "heldout_proxy": 6}
        or len(frozen_phase) != 21
        or set(counts.values()) != {4}
        or len(counts) != 12 * 21 * 4
    ):
        raise ValueError("upstream 6+6 x 21 x 4 x 2x2 contract changed")
    return report, rows, report_path, rows_path


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# H1-OFFSET-ROUTER-LOPO-v1",
        "",
        f"- Status: {result['status']}",
        f"- Decision: {result['decision']}",
        f"- Next experiment: {result['next_experiment']}",
        "- Zero new rollouts. Primary view is path-pure H1 candidate-set oracle.",
        "- No selector was trained or executed; union results are an upper bound.",
        "- Heldout proxy was not used for arm selection, stability, or resource guardrails.",
        "",
        "## Frozen global offset mapping",
        "",
        "| offset | full-train choice | phase LOPO votes | frozen arm |",
        "|---:|---|---:|---|",
    ]
    for offset, arm in result["frozen_global_offset_mapping"].items():
        value = result["global_offset_arm_stability"][offset]
        lines.append(
            f"| {offset} | {value['full_train_choice']} | "
            f"{value['phase_same_arm_fold_count']}/6 | {arm} |"
        )
    lines.extend([
        "",
        "## Frozen heldout comparison",
        "",
        "| candidate architecture | recall | gain retention | recoveries | regressions | loss-to-win |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in (R0, PHASE, "frozen_hard_offset_router", UNION, R2):
        value = result["heldout_comparison"][name]
        lines.append(
            f"| {name} | {100 * value['positive_state_recall']:.2f}% | "
            f"{100 * value['oracle_gain_retention']:.2f}% | "
            f"{value['coverage_recoveries_vs_R0']} | "
            f"{value['coverage_regressions_vs_R0']} | "
            f"{value['loss_to_win_states']} |"
        )
    overhead = result["union_candidate_overhead"]["heldout_proxy"]
    lines.extend([
        "",
        "## Canonical union overhead",
        "",
        f"- Mean candidates: {overhead['candidate_count']['mean']:.3f}",
        f"- Mean extra vs R0: {overhead['extra_candidates_vs_R0']['mean']:.3f}",
        f"- Mean ratio vs R0: {overhead['candidate_ratio_vs_R0']['mean']:.3f}",
        f"- P90 extra vs R0: {overhead['extra_candidates_vs_R0']['p90_nearest']:.0f}",
        f"- Heldout union no-regression audit: {result['union_no_regression_audit']['heldout_proxy']['passed']}",
        "",
        "## High-variance diagnostic boundary",
        "",
        f"- 84-cell stable-phase cells: {result['fine_84_cell_diagnostic']['stable_phase_cells']}",
        "- The anchor-phase x offset router is diagnostic only and is not the recommendation.",
        "- Proxy-heldout is not strict baseline OOD.",
        "",
    ])
    return "\n".join(lines)


def run(formal_root: Path, output_root: Path) -> dict[str, Any]:
    root = formal_root.resolve()
    report, rows, report_path, rows_path = _load_upstream(root)
    derived = derive(rows)
    script_path = Path(__file__).resolve()
    test_path = (
        script_path.parents[1]
        / "tests"
        / "test_derive_h1_offset_router_lopo_v1.py"
    )
    result = {
        "schema": SCHEMA,
        "upstream": {
            "formal_root": str(root),
            "formal_report": _artifact(report_path),
            "h1_decisions": _artifact(rows_path, len(rows)),
            "upstream_runner": report["implementation"]["runner"],
            "native_extension": report["implementation"]["native_extension"],
        },
        "implementation": {
            "script": _artifact(script_path),
            "tests": _artifact(test_path),
        },
        **derived,
        "evidence_boundary": {
            "zero_additional_rollouts": True,
            "upstream_report_unchanged": True,
            "upstream_decisions_unchanged": True,
            "heldout_used_for_selection_stability_or_resource_guardrail": False,
            "selector_trained": False,
            "selector_executed": False,
            "union_is_candidate_oracle_upper_bound": True,
            "fine_84_cell_router_is_diagnostic_only": True,
            "proxy_split_not_strict_baseline_ood": True,
        },
    }
    destination = output_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "H1_OFFSET_ROUTER_LOPO_V1.json"
    markdown_path = destination / "H1_OFFSET_ROUTER_LOPO_V1.md"
    _atomic_text(json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(markdown_path, _markdown(result))
    print(json.dumps({
        "event": "h1_offset_router_lopo_complete",
        "status": result["status"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "mapping": result["frozen_global_offset_mapping"],
        "heldout_hard": result["heldout_comparison"]["frozen_hard_offset_router"],
        "heldout_union": result["heldout_comparison"][UNION],
    }, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    run(args.formal_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
