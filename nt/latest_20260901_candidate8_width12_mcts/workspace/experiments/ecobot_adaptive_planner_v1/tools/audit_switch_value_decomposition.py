#!/usr/bin/env python3
"""Audit that W3 score components reconstruct the frozen analytic score."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


POSITIVE_COMPONENTS = ("crop_gross", "animal_gross", "fertilizer_gross")
COST_COMPONENTS = (
    "seed_cost", "feed_cost", "animal_purchase_cost", "action_cost",
    "move_cost", "hire_cost", "land_cost", "lockup_cost",
)
ALL_COMPONENTS = POSITIVE_COMPONENTS + COST_COMPONENTS + (
    "crop_units", "animal_product_units", "fertilizer_used",
    "fertilizer_sellable", "setup_turns",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-feature-dim", type=int, default=310)
    parser.add_argument("--rounding-tolerance", type=float, default=12.0)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        names = [str(value) for value in data["feature_names"]]
    at = {name: index for index, name in enumerate(names)}
    required = [
        "baseline_analytic_score", "candidate_analytic_score",
        *[
            f"{plan}_{component}"
            for plan in ("baseline", "candidate")
            for component in ALL_COMPONENTS
        ],
    ]
    missing = [name for name in required if name not in at]
    plan_audits: dict[str, object] = {}
    max_error = float("inf")
    all_nonnegative = False
    if not missing:
        errors = []
        all_nonnegative = True
        for plan in ("baseline", "candidate"):
            positive = sum(
                features[:, at[f"{plan}_{component}"]]
                for component in POSITIVE_COMPONENTS
            )
            costs = sum(
                features[:, at[f"{plan}_{component}"]]
                for component in COST_COMPONENTS
            )
            reconstructed = positive - costs
            recorded = features[:, at[f"{plan}_analytic_score"]]
            error = reconstructed - recorded
            component_matrix = np.column_stack([
                features[:, at[f"{plan}_{component}"]]
                for component in ALL_COMPONENTS
            ])
            all_nonnegative = all_nonnegative and bool(
                np.all(component_matrix >= 0.0)
            )
            errors.append(np.abs(error))
            plan_audits[plan] = {
                "max_abs_reconstruction_error": float(np.max(np.abs(error))),
                "mean_abs_reconstruction_error": float(np.mean(np.abs(error))),
                "component_minimum": float(np.min(component_matrix)),
                "nonnegative": bool(np.all(component_matrix >= 0.0)),
            }
        max_error = float(np.max(np.concatenate(errors)))

    finite = bool(np.all(np.isfinite(features)))
    status = (
        "PASS"
        if features.shape[1] == args.expected_feature_dim
        and not missing
        and finite
        and all_nonnegative
        and max_error <= args.rounding_tolerance
        else "FAIL"
    )
    payload = {
        "schema": "kaggriculture.switch-value-decomposition-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        },
        "expected_feature_dim": args.expected_feature_dim,
        "rounding_tolerance": args.rounding_tolerance,
        "missing_features": missing,
        "all_features_finite": finite,
        "all_components_nonnegative": all_nonnegative,
        "maximum_abs_reconstruction_error": max_error,
        "plans": plan_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
