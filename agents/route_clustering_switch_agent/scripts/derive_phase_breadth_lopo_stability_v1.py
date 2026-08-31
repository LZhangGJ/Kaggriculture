#!/usr/bin/env python3
"""Zero-rollout LOPO stability audit for the frozen contract breadth sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "phase-breadth-lopo-stability-v1"
ARMS = (
    "R0_active8",
    "K8_contract",
    "K16_contract",
    "K24_contract",
    "K32_contract",
)
R0 = ARMS[0]
GLOBAL_K = "K24_contract"
STABILITY_FOLDS_REQUIRED = 5
EXPECTED_REPORT_SHA256 = "3499ee38c0899fc5763a38a235452eed3bf922774970a264d927a4dbf7d8d054"
EXPECTED_ROWS_SHA256 = "170e98855694a3d6efa782596a3d04b1235fad7d76bdb035c196bbd0e419633b"
DEFAULT_FORMAL_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_breadth_sweep_v2_formal_20260829a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _arm_priority(arm: str) -> int:
    return -ARMS.index(arm)


def choose_anchor_arms(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Select by gain sum; ties prefer R0, then smaller contract K."""
    gains: dict[int, dict[str, float]] = defaultdict(
        lambda: {arm: 0.0 for arm in ARMS},
    )
    for row in rows:
        anchor = int(row["anchor"])
        for arm in ARMS:
            gains[anchor][arm] += float(row["arms"][arm]["oracle_gain"])
    if not gains:
        raise ValueError("arm selection requires non-empty rows")
    choices = {
        anchor: max(
            ARMS,
            key=lambda arm: (gains[anchor][arm], _arm_priority(arm)),
        )
        for anchor in sorted(gains)
    }
    return choices, {anchor: dict(value) for anchor, value in gains.items()}


def lopo_stable_mapping(
    train_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, str], dict[str, Any], dict[str, dict[int, str]]]:
    opponents = tuple(sorted({str(row["opponent"]) for row in train_rows}))
    if len(opponents) != 6:
        raise ValueError("LOPO stability requires exactly six train proxy opponents")
    full_choice, full_gains = choose_anchor_arms(train_rows)
    fold_mapping: dict[str, dict[int, str]] = {}
    for left_out in opponents:
        remaining = [
            row for row in train_rows if str(row["opponent"]) != left_out
        ]
        fold_mapping[left_out], _ = choose_anchor_arms(remaining)
        if set(fold_mapping[left_out]) != set(full_choice):
            raise ValueError("LOPO fold anchor coverage changed")

    frozen: dict[int, str] = {}
    stability: dict[str, Any] = {}
    for anchor in sorted(full_choice):
        candidate = full_choice[anchor]
        fold_choices = {
            opponent: fold_mapping[opponent][anchor] for opponent in opponents
        }
        counts = Counter(fold_choices.values())
        strictly_better = (
            candidate != R0
            and full_gains[anchor][candidate] > full_gains[anchor][R0]
        )
        same_fold_count = int(counts[candidate])
        stable = bool(
            strictly_better and same_fold_count >= STABILITY_FOLDS_REQUIRED
        )
        frozen[anchor] = candidate if stable else R0
        if stable:
            reason = "stable_non_r0"
        elif candidate == R0:
            reason = "full_train_selected_r0"
        elif not strictly_better:
            reason = "non_r0_not_strictly_better_than_r0"
        else:
            reason = "lopo_same_arm_below_5_of_6"
        stability[str(anchor)] = {
            "full_train_gain_sum": full_gains[anchor],
            "full_train_choice": candidate,
            "full_train_non_r0_strictly_better_than_r0": strictly_better,
            "lopo_choice_by_left_out_opponent": fold_choices,
            "lopo_choice_counts": {
                arm: int(counts.get(arm, 0)) for arm in ARMS
            },
            "candidate_same_arm_fold_count": same_fold_count,
            "required_same_arm_folds": STABILITY_FOLDS_REQUIRED,
            "stable": stable,
            "frozen_arm": frozen[anchor],
            "fallback_reason": reason,
        }
    return frozen, stability, fold_mapping


def _selected_arm(
    row: Mapping[str, Any],
    selector: str | Mapping[int, str] | Callable[[Mapping[str, Any]], str],
) -> str:
    if isinstance(selector, str):
        return selector
    if callable(selector):
        return selector(row)
    return selector[int(row["anchor"])]


def metrics(
    rows: Sequence[Mapping[str, Any]],
    selector: str | Mapping[int, str] | Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    r2_positive = [
        row for row in rows if row["arms"]["R2_all"]["strict_headroom"]
    ]
    selected = [
        row["arms"][_selected_arm(row, selector)] for row in rows
    ]
    r2_gain = sum(float(row["arms"]["R2_all"]["oracle_gain"]) for row in rows)
    gain = sum(float(value["oracle_gain"]) for value in selected)
    recalled = sum(
        bool(row["arms"][_selected_arm(row, selector)]["strict_headroom"])
        for row in r2_positive
    )
    recoveries = sum(
        not bool(row["arms"][R0]["strict_headroom"])
        and bool(row["arms"][_selected_arm(row, selector)]["strict_headroom"])
        for row in r2_positive
    )
    regressions = sum(
        bool(row["arms"][R0]["strict_headroom"])
        and not bool(row["arms"][_selected_arm(row, selector)]["strict_headroom"])
        for row in r2_positive
    )
    r0_gain = sum(float(row["arms"][R0]["oracle_gain"]) for row in rows)
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


def derive(
    rows: Sequence[Mapping[str, Any]],
    upstream_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train_proxy"]
    heldout = [row for row in rows if row["split"] == "heldout_proxy"]
    if not train or not heldout:
        raise ValueError("both train_proxy and heldout_proxy rows are required")
    frozen, stability, fold_mapping = lopo_stable_mapping(train)
    full_mapping = {
        int(anchor): str(value["full_train_choice"])
        for anchor, value in stability.items()
    }
    if upstream_report is not None:
        upstream_mapping = {
            int(anchor): str(arm)
            for anchor, arm in upstream_report[
                "per_anchor_router_diagnostic"
            ]["frozen_mapping"].items()
        }
        if upstream_mapping != full_mapping:
            raise ValueError("recomputed full-train router changed from upstream")
        if int(upstream_report["global_k_selection"]["chosen_k"]) != 24:
            raise ValueError("upstream train-selected global K is no longer 24")

    fold_results: dict[str, Any] = {}
    for left_out in sorted(fold_mapping):
        test = [
            row for row in train if str(row["opponent"]) == left_out
        ]
        fold_results[left_out] = {
            "left_out_opponent": left_out,
            "training_opponents": sorted(
                {str(row["opponent"]) for row in train}
                - {left_out}
            ),
            "selected_mapping": {
                str(anchor): arm
                for anchor, arm in sorted(fold_mapping[left_out].items())
            },
            "evaluation": {
                "lopo_router": metrics(test, fold_mapping[left_out]),
                R0: metrics(test, R0),
                GLOBAL_K: metrics(test, GLOBAL_K),
                "R2_all": metrics(test, "R2_all"),
            },
        }
    oof_selector = lambda row: fold_mapping[str(row["opponent"])][int(row["anchor"])]
    train_by_anchor: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train:
        train_by_anchor[int(row["anchor"])].append(row)
    lopo_cv = {
        "protocol": (
            "for each of six train-proxy opponents, select each anchor arm "
            "using only the other five opponents, then evaluate the left-out opponent"
        ),
        "heldout_proxy_used": False,
        "folds": fold_results,
        "overall_oof": {
            "lopo_router": metrics(train, oof_selector),
            R0: metrics(train, R0),
            GLOBAL_K: metrics(train, GLOBAL_K),
            "R2_all": metrics(train, "R2_all"),
        },
        "by_anchor_oof": {
            str(anchor): metrics(values, oof_selector)
            for anchor, values in sorted(train_by_anchor.items())
        },
    }

    heldout_comparison = {
        R0: metrics(heldout, R0),
        GLOBAL_K: metrics(heldout, GLOBAL_K),
        "full_train_high_variance_router": metrics(heldout, full_mapping),
        "stable_lopo_router": metrics(heldout, frozen),
        "R2_all": metrics(heldout, "R2_all"),
    }
    stable_metrics = heldout_comparison["stable_lopo_router"]
    r0_metrics = heldout_comparison[R0]
    beats_r0_both = bool(
        stable_metrics["positive_state_recall"]
        > r0_metrics["positive_state_recall"]
        and stable_metrics["oracle_gain_retention"]
        > r0_metrics["oracle_gain_retention"]
    )
    if beats_r0_both:
        status = "stable_hard_phase_router_survives"
        decision = (
            "The preregistered stable hard phase router exceeds R0 on both "
            "frozen heldout recall and gain retention; it survives this diagnostic."
        )
        next_direction = "validate the frozen hard mapping on a stricter OOD opponent protocol"
    else:
        status = "reject_hard_phase_router"
        decision = (
            "The stable hard phase router does not strictly exceed R0 on both "
            "frozen heldout metrics; reject hard per-anchor routing."
        )
        next_direction = "shared-parameter stage-conditioned learned retriever"

    return {
        "selection_contract": {
            "arms": list(ARMS),
            "objective": "per-anchor train-proxy oracle_gain sum",
            "tie_break": "R0 first, then smaller K",
            "full_train_non_r0_must_strictly_beat_r0": True,
            "same_arm_lopo_folds_required": STABILITY_FOLDS_REQUIRED,
            "lopo_fold_count": 6,
            "fallback": R0,
            "heldout_used_for_rule_or_threshold": False,
        },
        "arm_stability": stability,
        "stability_summary": {
            "anchors": len(frozen),
            "full_train_non_r0_anchors": sum(
                arm != R0 for arm in full_mapping.values()
            ),
            "stable_non_r0_anchors": sum(
                arm != R0 for arm in frozen.values()
            ),
            "fallback_r0_anchors": sum(
                arm == R0 for arm in frozen.values()
            ),
        },
        "full_train_high_variance_mapping": {
            str(anchor): arm for anchor, arm in sorted(full_mapping.items())
        },
        "frozen_stable_mapping": {
            str(anchor): arm for anchor, arm in sorted(frozen.items())
        },
        "lopo_cv": lopo_cv,
        "heldout_comparison": heldout_comparison,
        "heldout_double_metric_beats_r0": beats_r0_both,
        "status": status,
        "decision": decision,
        "next_direction": next_direction,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# PHASE-BREADTH-LOPO-STABILITY-v1",
        "",
        f"- Status: {result['status']}",
        f"- Decision: {result['decision']}",
        f"- Next direction: {result['next_direction']}",
        "- Heldout proxy was not used for arm choice, stability threshold, or fallback.",
        "",
        "## Stability summary",
        "",
        f"- Full-train non-R0 anchors: {result['stability_summary']['full_train_non_r0_anchors']}",
        f"- Stable non-R0 anchors: {result['stability_summary']['stable_non_r0_anchors']}",
        f"- R0 fallback anchors: {result['stability_summary']['fallback_r0_anchors']}",
        "",
        "| anchor | full choice | same-arm folds | frozen arm | reason |",
        "|---:|---|---:|---|---|",
    ]
    for anchor, value in result["arm_stability"].items():
        lines.append(
            f"| {anchor} | {value['full_train_choice']} | "
            f"{value['candidate_same_arm_fold_count']}/6 | "
            f"{value['frozen_arm']} | {value['fallback_reason']} |"
        )
    lines.extend([
        "",
        "## Heldout comparison",
        "",
        "| policy | recall | gain retention | coverage recoveries | coverage regressions | loss-to-win |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, value in result["heldout_comparison"].items():
        lines.append(
            f"| {name} | {100 * value['positive_state_recall']:.2f}% | "
            f"{100 * value['oracle_gain_retention']:.2f}% | "
            f"{value['coverage_recoveries_vs_R0']} | "
            f"{value['coverage_regressions_vs_R0']} | "
            f"{value['loss_to_win_states']} |"
        )
    oof = result["lopo_cv"]["overall_oof"]
    lines.extend([
        "",
        "## Train LOPO out-of-fold",
        "",
        f"- Router recall: {100 * oof['lopo_router']['positive_state_recall']:.2f}%",
        f"- Router gain retention: {100 * oof['lopo_router']['oracle_gain_retention']:.2f}%",
        f"- R0 recall: {100 * oof[R0]['positive_state_recall']:.2f}%",
        f"- R0 gain retention: {100 * oof[R0]['oracle_gain_retention']:.2f}%",
        "",
        "## Evidence boundary",
        "",
        "- Zero additional rollout; all metrics are derived from the frozen breadth formal JSONL.",
        "- Proxy-heldout is not strict baseline OOD.",
        "- The hard router is rejected unless both heldout recall and gain retention strictly exceed R0.",
        "",
    ])
    return "\n".join(lines)


def _load_upstream(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    report_path = root / "FINAL_REPORT.json"
    rows_path = root / "oracle_decisions.jsonl"
    report_sha = _sha256(report_path)
    rows_sha = _sha256(rows_path)
    if report_sha != EXPECTED_REPORT_SHA256 or rows_sha != EXPECTED_ROWS_SHA256:
        raise ValueError("frozen breadth formal SHA changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pinned = report.get("artifacts", {}).get(rows_path.name, {})
    if (
        report.get("schema") != "contract-breadth-sweep-v2"
        or str(pinned.get("sha256", "")).lower() != rows_sha
        or int(pinned.get("rows", -1)) != 1008
    ):
        raise ValueError("upstream report does not pin the expected 1008 decision rows")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != int(report["scenario_contract"]["states_observed"])
        or len(rows) != 1008
    ):
        raise ValueError("upstream row count changed")
    keys = {
        (
            str(row["opponent"]), int(row["seed"]),
            int(row["seat"]), int(row["anchor"]),
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("upstream decision keys are not unique")
    train_opponents = {
        str(row["opponent"]) for row in rows if row["split"] == "train_proxy"
    }
    heldout_opponents = {
        str(row["opponent"]) for row in rows if row["split"] == "heldout_proxy"
    }
    anchors = {int(row["anchor"]) for row in rows}
    if len(train_opponents) != 6 or len(heldout_opponents) != 6 or len(anchors) != 21:
        raise ValueError("upstream 6+6 opponent or 21-anchor contract changed")
    counts: Counter[tuple[str, int]] = Counter(
        (str(row["opponent"]), int(row["anchor"])) for row in rows
    )
    if set(counts.values()) != {4}:
        raise ValueError("each opponent-anchor must contain 2 seeds x 2 seats")
    if any(
        arm not in row["arms"]
        for row in rows for arm in (*ARMS, "R2_all")
    ):
        raise ValueError("upstream arm schema changed")
    return report, rows, report_path, rows_path


def run(formal_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    root = formal_root.resolve()
    report, rows, report_path, rows_path = _load_upstream(root)
    derived = derive(rows, report)
    script_path = Path(__file__).resolve()
    test_path = (
        script_path.parents[1]
        / "tests"
        / "test_derive_phase_breadth_lopo_stability_v1.py"
    )
    result = {
        "schema": SCHEMA,
        "upstream": {
            "formal_root": str(root),
            "formal_report": _artifact(report_path),
            "oracle_decisions": _artifact(rows_path, len(rows)),
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
            "upstream_rows_unchanged": True,
            "heldout_used_for_selection_or_threshold": False,
            "proxy_split_not_strict_baseline_ood": True,
            "sealed_test_executed": False,
        },
    }
    destination = (output_root or root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "PHASE_BREADTH_LOPO_STABILITY.json"
    markdown_path = destination / "PHASE_BREADTH_LOPO_STABILITY.md"
    _atomic_text(json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(markdown_path, _markdown(result))
    print(json.dumps({
        "event": "phase_breadth_lopo_stability_complete",
        "status": result["status"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "stable_non_r0_anchors": result["stability_summary"]["stable_non_r0_anchors"],
        "heldout": result["heldout_comparison"]["stable_lopo_router"],
    }, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    run(args.formal_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
