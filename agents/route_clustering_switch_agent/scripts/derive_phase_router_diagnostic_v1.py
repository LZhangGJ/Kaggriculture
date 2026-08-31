#!/usr/bin/env python3
"""Derive a train-frozen per-anchor R0/R1 router from a completed oracle run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "phase-router-diagnostic-v1"
ARMS = ("R0_active8", "R1_contract8")
DEFAULT_FORMAL_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\contract_retrieval_ablation_v1_formal_20260829b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def frozen_anchor_mapping(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Use train-proxy positive margin-gain sums only; R0 wins exact ties."""

    gains: dict[int, dict[str, float]] = defaultdict(
        lambda: {arm: 0.0 for arm in ARMS},
    )
    for row in rows:
        if row["split"] != "train_proxy":
            continue
        for arm in ARMS:
            gains[int(row["anchor"])][arm] += float(row["arms"][arm]["oracle_gain"])
    if not gains:
        raise ValueError("no train-proxy decisions")
    choice = {
        anchor: max(ARMS, key=lambda arm: (gains[anchor][arm], arm == ARMS[0]))
        for anchor in sorted(gains)
    }
    return choice, {anchor: dict(value) for anchor, value in gains.items()}


def _metrics(
    rows: Sequence[Mapping[str, Any]], arm_for_anchor: Mapping[int, str] | str,
) -> dict[str, Any]:
    r2_positive = [row for row in rows if row["arms"]["R2_all"]["strict_headroom"]]
    selected = [
        row["arms"][
            arm_for_anchor if isinstance(arm_for_anchor, str)
            else arm_for_anchor[int(row["anchor"])]
        ]
        for row in rows
    ]
    r2_gain = sum(float(row["arms"]["R2_all"]["oracle_gain"]) for row in rows)
    gain = sum(float(value["oracle_gain"]) for value in selected)
    recalled = sum(
        bool(row["arms"][
            arm_for_anchor if isinstance(arm_for_anchor, str)
            else arm_for_anchor[int(row["anchor"])]
        ]["strict_headroom"])
        for row in r2_positive
    )
    return {
        "states": len(rows), "r2_positive_states": len(r2_positive),
        "positive_states": sum(bool(value["strict_headroom"]) for value in selected),
        "positive_state_recall": recalled / len(r2_positive) if r2_positive else 1.0,
        "oracle_gain_sum": gain,
        "oracle_gain_retention": gain / r2_gain if r2_gain > 0 else 1.0,
        "loss_to_win_states": sum(bool(value["loss_to_win"]) for value in selected),
    }


def derive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    choice, train_gains = frozen_anchor_mapping(rows)
    splits = {
        "train_proxy": [row for row in rows if row["split"] == "train_proxy"],
        "heldout_proxy": [row for row in rows if row["split"] == "heldout_proxy"],
        "overall": list(rows),
    }
    comparison = {}
    for split, values in splits.items():
        comparison[split] = {
            "R0_active8": _metrics(values, "R0_active8"),
            "R1_contract8": _metrics(values, "R1_contract8"),
            "phase_router": _metrics(values, choice),
            "R2_all": _metrics(values, "R2_all"),
        }
    by_anchor = {}
    for anchor in sorted(choice):
        train = [
            row for row in splits["train_proxy"] if int(row["anchor"]) == anchor
        ]
        heldout = [
            row for row in splits["heldout_proxy"] if int(row["anchor"]) == anchor
        ]
        by_anchor[str(anchor)] = {
            "train_gain_sum": train_gains[anchor],
            "frozen_arm": choice[anchor],
            "train_phase_router": _metrics(train, choice),
            "heldout_phase_router": _metrics(heldout, choice),
        }
    return {
        "selection_rule": (
            "for each anchor, sum path-pure positive oracle margin gain on "
            "train_proxy only; select larger of R0_active8/R1_contract8"
        ),
        "tie_break": "R0_active8",
        "heldout_used_for_selection": False,
        "frozen_anchor_arm": {str(anchor): arm for anchor, arm in choice.items()},
        "comparison": comparison,
        "by_anchor": by_anchor,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = []
    for split in ("train_proxy", "heldout_proxy", "overall"):
        for arm in ("R0_active8", "R1_contract8", "phase_router", "R2_all"):
            row = result["comparison"][split][arm]
            lines.append(
                f"| {split} | {arm} | {100 * row['positive_state_recall']:.2f}% | "
                f"{100 * row['oracle_gain_retention']:.2f}% | "
                f"{row['positive_states']} | {row['loss_to_win_states']} |"
            )
    mapping = ", ".join(
        f"{anchor}:{arm.replace('_active8', '').replace('_contract8', '')}"
        for anchor, arm in result["frozen_anchor_arm"].items()
    )
    return (
        "# PHASE-ROUTER-DIAGNOSTIC-v1\n\n"
        "The arm mapping is selected once from train-proxy gain sums, with R0 "
        "winning ties, then frozen before heldout evaluation. Heldout rows do not "
        "participate in selection.\n\n"
        f"Frozen mapping: `{mapping}`\n\n"
        "| split | arm | R2-positive recall | R2 gain retained | positive states | loss->win states |\n"
        "|---|---|---:|---:|---:|---:|\n" + "\n".join(lines) + "\n\n"
        "Evidence boundary: this is a zero-rollout derivative of the completed "
        "fixed-route proxy oracle. The original proxy split is not a sealed baseline "
        "OOD protocol, so the heldout comparison is diagnostic rather than a final "
        "submitted-agent claim.\n"
    )


def run(formal_root: Path) -> dict[str, Any]:
    root = formal_root.resolve()
    report_path = root / "FINAL_REPORT.json"
    decisions_path = root / "oracle_decisions.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = _sha256(report_path)
    decisions_sha = _sha256(decisions_path)
    pinned = report.get("artifacts", {}).get(decisions_path.name, {})
    if (
        report.get("schema") != "contract-retrieval-oracle-ablation-v1"
        or str(pinned.get("sha256", "")).lower() != decisions_sha
    ):
        raise ValueError("upstream formal report or decisions digest is invalid")
    rows = [
        json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(report["scenario_contract"]["states_observed"]):
        raise ValueError("upstream decision row count changed")
    derived = derive(rows)
    result = {
        "schema": SCHEMA,
        "upstream": {
            "formal_root": str(root),
            "formal_report": {"path": str(report_path), "sha256": report_sha},
            "oracle_decisions": {
                "path": str(decisions_path), "sha256": decisions_sha,
                "rows": len(rows),
            },
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        **derived,
        "evidence_boundary": {
            "zero_additional_rollouts": True,
            "proxy_split_not_strict_baseline_ood": True,
            "heldout_used_for_selection": False,
        },
    }
    json_path = root / "PHASE_ROUTER_DIAGNOSTIC.json"
    markdown_path = root / "PHASE_ROUTER_DIAGNOSTIC.md"
    _atomic_text(json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(markdown_path, _markdown(result))
    print(json.dumps({
        "event": "phase_router_diagnostic_complete",
        "json": str(json_path), "markdown": str(markdown_path),
        "heldout": result["comparison"]["heldout_proxy"],
    }, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    args = parser.parse_args(argv)
    run(args.formal_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
