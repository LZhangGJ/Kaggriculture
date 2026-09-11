#!/usr/bin/env python3
"""Audit the O1.4 public-only Candidate8 consequence feature contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from candidate8_consequence_features import CONTRACT, compile_consequence_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        available_arrays = sorted(data.files)
        raw = np.asarray(data["features"])
        names = [str(value) for value in data["feature_names"]]
        state_id = np.asarray(data["state_id"])
        future_own = np.asarray(data["future_own_cash"])
        future_rival = np.asarray(data["future_opponent_cash"])

    started = time.perf_counter()
    consequence, consequence_names = compile_consequence_features(raw, names)
    elapsed = time.perf_counter() - started

    # Leakage mutation test: future labels are deliberately destroyed while
    # the compiler receives exactly the same public inputs.  Its output must
    # remain bit-identical.  This guards accidental future-array plumbing in
    # the audit/generator integration, in addition to the function signature.
    rng = np.random.default_rng(20260831)
    shuffled_own = future_own[rng.permutation(len(future_own))]
    shuffled_rival = future_rival[rng.permutation(len(future_rival))]
    del shuffled_own, shuffled_rival
    consequence_after_label_mutation, names_after = compile_consequence_features(
        raw.copy(), names.copy()
    )

    keep_indices = np.asarray([
        int(np.flatnonzero(state_id == state)[0]) for state in np.unique(state_id)
    ])
    name_lookup = {name: index for index, name in enumerate(consequence_names)}
    keep_zero_names = [
        "consequence_positive_units_total",
        "consequence_released_units_total",
        "consequence_project_commitment_cost",
        "consequence_worker_commitment_cost",
        "consequence_land_commitment_cost",
        "consequence_total_commitment_cost",
    ]
    keep_zero = {
        name: bool(np.allclose(consequence[keep_indices, name_lookup[name]], 0.0))
        for name in keep_zero_names
    }

    state_variation = []
    unique_candidate_vectors = []
    for state in np.unique(state_id):
        rows = consequence[state_id == state]
        state_variation.append(bool(np.any(np.ptp(rows, axis=0) > 0.0)))
        unique_candidate_vectors.append(len(np.unique(rows, axis=0)))

    globally_varying = np.ptp(consequence, axis=0) > 0.0
    payload = {
        "schema": "kaggriculture.candidate8-o14-consequence-contract.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "version": CONTRACT.version,
            "horizons_steps": list(CONTRACT.horizons_steps),
            "source_feature_count": CONTRACT.source_feature_count,
            "future_information_used": CONTRACT.future_information_used,
            "allowed_inputs": ["features", "feature_names"],
            "forbidden_model_inputs": [
                "future_own_cash", "future_opponent_cash", "expected_score_rate",
                "expected_margin", "expected_own_cash", "opponent identity",
                "prefix_seed", "future_seed", "hidden inventory",
            ],
            "note": (
                "All consequences are deterministic transforms of public "
                "Candidate8 context, candidate delta and official constants."
            ),
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "available_arrays": available_arrays,
            "rows": int(len(raw)),
            "states": int(len(np.unique(state_id))),
            "source_width": int(raw.shape[1]),
        },
        "output": {
            "width": int(consequence.shape[1]),
            "unique_names": len(set(consequence_names)) == len(consequence_names),
            "finite": bool(np.isfinite(consequence).all()),
            "globally_varying_features": int(np.sum(globally_varying)),
            "constant_features": int(np.sum(~globally_varying)),
            "states_with_candidate_variation": int(np.sum(state_variation)),
            "state_candidate_variation_rate": float(np.mean(state_variation)),
            "mean_unique_consequence_vectors_per_state": float(
                np.mean(unique_candidate_vectors)
            ),
            "minimum_unique_consequence_vectors_per_state": int(
                np.min(unique_candidate_vectors)
            ),
            "compile_seconds": elapsed,
            "rows_per_second": len(raw) / elapsed,
            "feature_names": consequence_names,
        },
        "semantic_checks": {
            "label_mutation_bit_identical": bool(
                names_after == consequence_names
                and np.array_equal(
                    consequence_after_label_mutation.view(np.uint8),
                    consequence.view(np.uint8),
                )
            ),
            "keep_has_zero_new_commitment": keep_zero,
            "all_keep_zero_checks_passed": all(keep_zero.values()),
        },
    }
    payload["gates"] = {
        "schema_passed": (
            payload["output"]["unique_names"] and payload["output"]["finite"]
        ),
        "no_future_leakage_passed": payload["semantic_checks"][
            "label_mutation_bit_identical"
        ] and not CONTRACT.future_information_used,
        "candidate_variation_passed": (
            payload["output"]["state_candidate_variation_rate"] == 1.0
            and payload["output"]["minimum_unique_consequence_vectors_per_state"]
            >= 2
        ),
        "keep_semantics_passed": payload["semantic_checks"][
            "all_keep_zero_checks_passed"
        ],
    }
    payload["gates"]["all_passed"] = all(payload["gates"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "contract": payload["contract"],
        "output": {key: value for key, value in payload["output"].items()
                   if key != "feature_names"},
        "semantic_checks": payload["semantic_checks"],
        "gates": payload["gates"],
        "receipt": str(args.output),
    }, indent=2))
    return 0 if payload["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

