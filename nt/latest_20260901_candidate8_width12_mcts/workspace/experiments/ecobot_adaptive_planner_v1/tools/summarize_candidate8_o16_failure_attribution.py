#!/usr/bin/env python3
"""Summarize O1.6 headroom attribution for the 45 frozen O1.5 losses.

This audit deliberately separates four questions:

1. Does a wider Candidate8 enumeration rescue the case?
2. Does a wider beam rescue the case?
3. Does the committed sequence show a hard execution failure?
4. Can any frozen complete reference route win the same case?

The final category remains conservative.  A clean committed rollout proves only
that no measured hard failure occurred; it does not prove that the soft plan
compiler economically realizes every potentially useful high-level plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["seat"])


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument(
        "--search-receipt", required=True, action="append", type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_payload = load(args.baseline)
    controls_payload = load(args.controls)
    baseline_rows = {
        key(row): row
        for row in baseline_payload["rows"]["depth3"]
        if float(row["margin"]) <= 0.0
    }
    if len(baseline_rows) != 45:
        raise RuntimeError(
            f"expected 45 frozen O1.5 losses, observed {len(baseline_rows)}"
        )

    execution_rows = {
        key(row): row for row in controls_payload["execution_rows"]
    }
    portfolio_rows = {
        key(row): row for row in controls_payload["portfolio_rows"]
    }
    if set(execution_rows) != set(baseline_rows):
        raise RuntimeError("execution control keys do not match frozen losses")
    if set(portfolio_rows) != set(baseline_rows):
        raise RuntimeError("portfolio control keys do not match frozen losses")

    configs: dict[str, dict] = {}
    search_by_config: dict[str, dict[tuple[str, int, int], dict]] = {}
    for path in args.search_receipt:
        payload = load(path)
        label = str(payload["parameters"]["label"])
        if label in configs:
            raise RuntimeError(f"duplicate search label: {label}")
        configs[label] = {
            "path": str(path),
            "sha256": sha256(path),
            "beam_width": int(payload["parameters"]["beam_width"]),
            "per_node_arms": int(payload["parameters"]["per_node_arms"]),
            "cases": int(payload["summary"]["cases"]),
            "converted_to_win": int(payload["summary"]["converted_to_win"]),
            "mean_margin_gain": float(payload["summary"]["mean_margin_gain"]),
            "simulation_seconds": float(payload["simulation_seconds"]),
        }
        row_map = {key(row): row for row in payload["rows"]}
        if not set(row_map).issubset(baseline_rows):
            raise RuntimeError(f"{label} contains non-frozen cases")
        search_by_config[label] = row_map

    rows: list[dict] = []
    category_counts: Counter[str] = Counter()
    category_by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    rescued_keys: set[tuple[str, int, int]] = set()

    for case_key in sorted(baseline_rows):
        baseline = baseline_rows[case_key]
        execution = execution_rows[case_key]
        portfolio = portfolio_rows[case_key]
        observations = []
        arm_only_win = False
        beam_only_win = False
        joint_win = False
        best_search = None

        for label, row_map in search_by_config.items():
            row = row_map.get(case_key)
            if row is None:
                continue
            cfg = configs[label]
            won = float(row["margin"]) > 0.0
            observations.append({
                "label": label,
                "beam_width": cfg["beam_width"],
                "per_node_arms": cfg["per_node_arms"],
                "own_cash": float(row["own_cash"]),
                "opponent_cash": float(row["opponent_cash"]),
                "margin": float(row["margin"]),
                "margin_gain": float(row["margin_gain"]),
                "won": won,
            })
            if best_search is None or (
                won,
                float(row["margin"]),
                float(row["own_cash"]),
            ) > (
                bool(best_search["won"]),
                float(best_search["margin"]),
                float(best_search["own_cash"]),
            ):
                best_search = observations[-1]
            if won:
                rescued_keys.add(case_key)
                if cfg["beam_width"] == 16 and cfg["per_node_arms"] > 32:
                    arm_only_win = True
                elif cfg["beam_width"] > 16 and cfg["per_node_arms"] == 32:
                    beam_only_win = True
                elif cfg["beam_width"] > 16 and cfg["per_node_arms"] > 32:
                    joint_win = True

        hard_execution_issue = bool(execution["hard_execution_issue"])
        portfolio_wins = float(portfolio["margin"]) > 0.0
        if arm_only_win or beam_only_win or joint_win:
            signals = sum([arm_only_win, beam_only_win, joint_win])
            if signals > 1:
                category = "SEARCH_WIDTH_MIXED"
            elif arm_only_win:
                category = "CANDIDATE_ENUMERATION_TRUNCATION"
            elif beam_only_win:
                category = "BEAM_PRUNING"
            else:
                category = "JOINT_WIDTH_INTERACTION"
        elif hard_execution_issue:
            category = "MEASURED_HARD_EXECUTION_FAILURE"
        elif portfolio_wins:
            category = "SEMANTIC_CANDIDATE_OR_SOFT_COMPILER_GAP"
        else:
            category = "REFERENCE_ROUTE_POOL_OR_BACKBONE_GAP"

        category_counts[category] += 1
        category_by_opponent[case_key[0]][category] += 1
        rows.append({
            "opponent": case_key[0],
            "seed": case_key[1],
            "seat": case_key[2],
            "baseline_own_cash": float(baseline["own_cash"]),
            "baseline_opponent_cash": float(baseline["opponent_cash"]),
            "baseline_margin": float(baseline["margin"]),
            "baseline_selected_ranks": [
                int(value) for value in baseline["selected_ranks"]
            ],
            "baseline_selected_families": list(
                baseline["selected_families"]
            ),
            "hard_execution_issue": hard_execution_issue,
            "end_overflow": int(execution["end_overflow"]),
            "avoidable_crop_losses": int(
                execution["avoidable_crop_losses"]
            ),
            "avoidable_animal_losses": int(
                execution["avoidable_animal_losses"]
            ),
            "best_reference_route": str(portfolio["best_route"]),
            "best_reference_route_margin": float(portfolio["margin"]),
            "winning_reference_routes": int(portfolio["winning_routes"]),
            "arm_only_win": arm_only_win,
            "beam_only_win": beam_only_win,
            "joint_win": joint_win,
            "best_search": best_search,
            "search_observations": observations,
            "category": category,
        })

    portfolio_margins = [
        float(row["margin"]) for row in controls_payload["portfolio_rows"]
    ]
    baseline_margins = [float(row["baseline_margin"]) for row in rows]
    best_search_margins = [
        (
            float(row["best_search"]["margin"])
            if row["best_search"] is not None
            else float(row["baseline_margin"])
        )
        for row in rows
    ]
    reference_gaps_after_width_search = [
        float(row["best_reference_route_margin"]) - best_margin
        for row, best_margin in zip(rows, best_search_margins)
    ]
    output = {
        "schema": "kaggriculture.candidate8-o16-failure-attribution.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_COMPLETE",
        "boundary": (
            "Posthoc exact future is used only for O1.6 diagnosis. A clean "
            "committed sequence excludes measured hard failures but cannot "
            "separate a missing semantic candidate from a legal yet "
            "economically weak soft plan compiler without a forced-plan test."
        ),
        "summary": {
            "frozen_losses": len(rows),
            "search_width_rescued_cases": len(rescued_keys),
            "search_width_rescue_rate": len(rescued_keys) / len(rows),
            "remaining_non_wins": len(rows) - len(rescued_keys),
            "baseline_mean_margin": sum(baseline_margins) / len(rows),
            "best_width_search_mean_margin": (
                sum(best_search_margins) / len(rows)
            ),
            "best_width_search_mean_margin_gain": (
                sum(best_search_margins) / len(rows)
                - sum(baseline_margins) / len(rows)
            ),
            "measured_hard_execution_failure_cases": int(sum(
                row["hard_execution_issue"] for row in rows
            )),
            "reference_portfolio_winning_cases": int(sum(
                row["best_reference_route_margin"] > 0.0 for row in rows
            )),
            "mean_best_reference_margin": (
                sum(portfolio_margins) / len(portfolio_margins)
            ),
            "minimum_best_reference_margin": min(portfolio_margins),
            "maximum_best_reference_margin": max(portfolio_margins),
            "mean_reference_gap_after_best_width_search": (
                sum(reference_gaps_after_width_search)
                / len(reference_gaps_after_width_search)
            ),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "search_configs": configs,
        "category_by_opponent": {
            opponent: dict(sorted(counts.items()))
            for opponent, counts in sorted(category_by_opponent.items())
        },
        "rows": rows,
        "inputs": {
            "baseline": {
                "path": str(args.baseline),
                "sha256": sha256(args.baseline),
            },
            "controls": {
                "path": str(args.controls),
                "sha256": sha256(args.controls),
            },
            "search_receipts": [configs[label] for label in configs],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
