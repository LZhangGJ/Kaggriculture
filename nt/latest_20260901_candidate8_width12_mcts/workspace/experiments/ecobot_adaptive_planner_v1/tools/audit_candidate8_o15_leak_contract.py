#!/usr/bin/env python3
"""Verify that O1.5 response previews are independent of true label futures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IDENTITY = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--future-mutation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.base, allow_pickle=False) as base, np.load(
        args.future_mutation, allow_pickle=False
    ) as mutation:
        identity_exact = {
            key: bool(np.array_equal(base[key], mutation[key])) for key in IDENTITY
        }
        canonical_exact = bool(np.array_equal(
            base["consequence_features"], mutation["consequence_features"]
        ))
        scenario_names_exact = bool(np.array_equal(
            base["response_scenario_names"], mutation["response_scenario_names"]
        ))
        scenario_features_exact = bool(np.array_equal(
            base["response_scenario_features"],
            mutation["response_scenario_features"],
        ))
        scenario_outcomes_exact = bool(np.array_equal(
            base["response_scenario_outcomes"],
            mutation["response_scenario_outcomes"],
        ))
        base_own = np.asarray(base["future_own_cash"])
        mutated_own = np.asarray(mutation["future_own_cash"])
        base_rival = np.asarray(base["future_opponent_cash"])
        mutated_rival = np.asarray(mutation["future_opponent_cash"])
        true_labels_changed = bool(
            not np.array_equal(base_own, mutated_own)
            or not np.array_equal(base_rival, mutated_rival)
        )
        pass_matches_canonical = bool(np.array_equal(
            base["response_scenario_features"][:, 0, :],
            base["consequence_features"],
        ))
        scenario = np.asarray(base["response_scenario_features"])
        unique_scenarios_per_arm = np.asarray([
            len(np.unique(row, axis=0)) for row in scenario
        ], dtype=np.int32)
        scenario_names = [str(value) for value in base["response_scenario_names"]]

    gates = {
        "identity_arrays_exact": all(identity_exact.values()),
        "canonical_preview_exact_after_future_mutation": canonical_exact,
        "scenario_names_exact_after_future_mutation": scenario_names_exact,
        "scenario_features_exact_after_future_mutation": scenario_features_exact,
        "scenario_outcomes_exact_after_future_mutation": scenario_outcomes_exact,
        "true_future_labels_changed": true_labels_changed,
        "pass_scenario_matches_canonical_preview": pass_matches_canonical,
        "all_arms_have_multiple_scenario_outcomes": bool(
            np.all(unique_scenarios_per_arm >= 2)
        ),
    }
    payload = {
        "schema": "kaggriculture.candidate8-o15-leak-contract.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "boundary": (
            "This dynamic test proves independence from the true future label "
            "seed. Source-level review separately verifies that route identity, "
            "hidden rival inventory and true future actions are absent from the "
            "response-policy inputs."
        ),
        "identity_arrays_exact": identity_exact,
        "scenario_names": scenario_names,
        "minimum_unique_scenarios_per_arm": int(np.min(unique_scenarios_per_arm)),
        "mean_unique_scenarios_per_arm": float(np.mean(unique_scenarios_per_arm)),
        "gates": gates,
        "inputs": {
            "base": {"path": str(args.base), "sha256": sha256(args.base)},
            "future_mutation": {
                "path": str(args.future_mutation),
                "sha256": sha256(args.future_mutation),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
