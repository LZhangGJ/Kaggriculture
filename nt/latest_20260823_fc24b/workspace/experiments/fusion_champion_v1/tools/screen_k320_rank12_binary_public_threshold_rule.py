#!/usr/bin/env python3
"""Train-only screen of one-condition FC2B -> two-YARN rescue rules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from screen_k320_rank12_public_threshold_rule import (  # noqa: E402
    FEATURES,
    fit_threshold,
    grouped_oof,
    sha256,
)
from train_k320_route_rescue_tree import _baseline_oracle, _metrics  # noqa: E402
from train_prt_suffix_selector import build_feature_row  # noqa: E402


def load_binary(path: Path, feature_names: list[str] | None = None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_rows = payload.get("decision_features", [])
    if not feature_rows:
        raise AssertionError(f"missing decision features: {path}")
    built = [build_feature_row(row) for row in feature_rows]
    if feature_names is None:
        feature_names = sorted(built[0])
    if any(sorted(row) != feature_names for row in built):
        raise AssertionError("feature schema mismatch")
    keys = [(int(row["seed"]), int(row["candidate_seat"])) for row in feature_rows]
    matrix = np.asarray(
        [[row[name] for name in feature_names] for row in built], dtype=np.float32
    )
    baseline = next(row for row in payload["rows"] if int(row["route_id"]) == -1)
    route = next(row for row in payload["rows"] if int(row["route_id"]) == 4)

    def margins(row):
        lookup = {
            (int(item["seed"]), int(item["candidate_seat"])): int(item["margin"])
            for item in row["per_game"]
        }
        return np.asarray([lookup[key] for key in keys], dtype=np.int64)

    outcomes = np.stack((margins(baseline), margins(route)))
    groups = np.asarray([key[0] for key in keys], dtype=np.int64)
    return payload, matrix, outcomes, groups, feature_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout-a", type=Path, required=True)
    parser.add_argument("--holdout-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, x, margins, groups, names = load_binary(args.train)
    _, ax, amargins, _, anames = load_binary(args.holdout_a, names)
    _, bx, bmargins, _, bnames = load_binary(args.holdout_b, names)
    if anames != names or bnames != names:
        raise AssertionError("holdout feature schema mismatch")
    missing = sorted(set(FEATURES) - set(names))
    if missing:
        raise AssertionError(f"missing features: {missing}")

    candidates = []
    for feature in FEATURES:
        column = names.index(feature)
        for direction in ("le", "gt"):
            oof, folds = grouped_oof(
                x[:, column], margins, groups, direction, arm=1
            )
            candidates.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "oof": oof,
                    "folds": folds,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
        ),
    )
    column = names.index(selected["feature"])
    final = fit_threshold(
        x[:, column], margins, selected["direction"], arm=1
    )

    def evaluate(matrix: np.ndarray, outcomes: np.ndarray) -> dict:
        values = matrix[:, column]
        mask = (
            values <= final["threshold"]
            if selected["direction"] == "le"
            else values > final["threshold"]
        )
        return _metrics(outcomes, mask.astype(np.int64))

    operator = "<=" if selected["direction"] == "le" else ">"
    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-rank12-binary-threshold-rule.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "candidate_stack": train_payload.get("candidate_stack"),
        "future_leakage": False,
        "rule_family": "one public feature threshold; keep FC2B or lock route4_two_yarn",
        "selection_protocol": (
            "feature and direction selected by 5-fold seed-grouped OOF on train; "
            "threshold refit on train only; holdout A/B untouched"
        ),
        "train_samples": int(len(x)),
        "train_unique_seeds": int(len(np.unique(groups))),
        "features_screened": list(FEATURES),
        "sources": [
            {"role": role, "path": str(path.resolve()), "sha256": sha256(path)}
            for role, path in (
                ("train", args.train),
                ("holdout_a", args.holdout_a),
                ("holdout_b", args.holdout_b),
            )
        ],
        "baselines": {
            "train": _baseline_oracle(margins),
            "holdout_a": _baseline_oracle(amargins),
            "holdout_b": _baseline_oracle(bmargins),
        },
        "selected": {
            **selected,
            "threshold": final["threshold"],
            "operator": operator,
            "rule": (
                f"if {selected['feature']} {operator} {final['threshold']}: "
                "lock route4_two_yarn; else keep FC2B"
            ),
            "train_fit": evaluate(x, margins),
            "holdout_a": evaluate(ax, amargins),
            "holdout_b": evaluate(bx, bmargins),
        },
        "top_candidates": sorted(
            candidates,
            key=lambda row: (
                row["oof"]["score_rate"],
                -row["oof"]["harmed_wins"],
                -row["oof"]["switch_rate"],
                row["oof"]["mean_margin"],
            ),
            reverse=True,
        )[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "PASS", "selected": payload["selected"], "output": str(args.output.resolve())},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
