#!/usr/bin/env python3
"""Attach O1.5 rival-response teacher features to frozen Candidate8 labels."""

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


def delta_to_keep(values: np.ndarray, state_id: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for state in np.unique(state_id):
        rows = np.flatnonzero(state_id == state)
        result[rows] = values[rows] - values[rows[0]]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.outcomes, allow_pickle=False) as frozen, np.load(
        args.response, allow_pickle=False
    ) as response:
        exact = {key: bool(np.array_equal(frozen[key], response[key])) for key in IDENTITY}
        if not all(exact.values()):
            raise RuntimeError(f"response rows do not match frozen labels: {exact}")
        canonical_consequence_exact = bool(np.array_equal(
            frozen["consequence_features"], response["consequence_features"]
        ))
        canonical_names_exact = bool(np.array_equal(
            frozen["consequence_feature_names"],
            response["consequence_feature_names"],
        ))
        if not canonical_consequence_exact or not canonical_names_exact:
            raise RuntimeError(
                "O1.5 regenerated a different O1.4 canonical preview: "
                f"values={canonical_consequence_exact}, names={canonical_names_exact}"
            )
        arrays = {key: frozen[key] for key in frozen.files}
        state_id = np.asarray(frozen["state_id"])
        canonical = np.asarray(frozen["consequence_features"], dtype=np.float32)
        response_canonical = np.asarray(
            response["consequence_features"], dtype=np.float32
        )
        canonical_names = [str(value) for value in frozen["consequence_feature_names"]]
        scenario = np.asarray(response["response_scenario_features"], dtype=np.float32)
        outcomes = np.asarray(response["response_scenario_outcomes"], dtype=np.float32)
        scenario_names = [str(value) for value in response["response_scenario_names"]]
        outcome_names = [
            str(value) for value in response["response_scenario_outcome_names"]
        ]

    if scenario.ndim != 3 or scenario.shape[1:] != (5, 228):
        raise RuntimeError(f"unexpected response scenario shape {scenario.shape}")
    if outcomes.shape != (len(state_id), 5, 9):
        raise RuntimeError(f"unexpected response outcome shape {outcomes.shape}")
    pass_scenario_exact = bool(np.array_equal(
        scenario[:, 0, :], response_canonical
    ))
    if not pass_scenario_exact:
        raise RuntimeError(
            "the O1.5 PASS response scenario diverges from the frozen canonical preview"
        )

    scenario_delta = delta_to_keep(scenario, state_id)
    outcome_delta = delta_to_keep(outcomes, state_id)
    # Explicit robust summaries required by the O1.5 contract.  Statistics are
    # calculated over fixed scenario names, never fitted scenario weights.
    summary_blocks = []
    summary_names = []
    for prefix, values in (("absolute", outcomes), ("delta_to_keep", outcome_delta)):
        for stat, block in (
            ("mean", np.mean(values, axis=1)),
            ("q25", np.quantile(values, 0.25, axis=1)),
            ("worst", np.min(values, axis=1)),
            ("best", np.max(values, axis=1)),
            ("std", np.std(values, axis=1)),
        ):
            summary_blocks.append(np.asarray(block, dtype=np.float32))
            summary_names.extend(
                f"response_{prefix}_{stat}_{name}" for name in outcome_names
            )

    scenario_delta_flat = scenario_delta.reshape(len(state_id), -1)
    outcome_flat = outcomes.reshape(len(state_id), -1)
    outcome_delta_flat = outcome_delta.reshape(len(state_id), -1)
    compiled = np.concatenate(
        [canonical, scenario_delta_flat, outcome_flat, outcome_delta_flat,
         *summary_blocks],
        axis=1,
    ).astype(np.float32)
    names = list(canonical_names)
    names.extend(
        f"response_delta_{scenario_name}_{feature_name}"
        for scenario_name in scenario_names
        for feature_name in canonical_names
    )
    names.extend(
        f"response_{scenario_name}_{outcome_name}"
        for scenario_name in scenario_names for outcome_name in outcome_names
    )
    names.extend(
        f"response_delta_{scenario_name}_{outcome_name}"
        for scenario_name in scenario_names for outcome_name in outcome_names
    )
    names.extend(summary_names)
    if compiled.shape[1] != len(names):
        raise AssertionError((compiled.shape, len(names)))

    arrays["consequence_features"] = compiled
    arrays["consequence_feature_names"] = np.asarray(names)
    arrays["response_scenario_names"] = np.asarray(scenario_names)
    # Retain the compact raw scenario teacher targets. They are permitted only
    # during offline training; a deployable distilled selector consumes the
    # original public candidate/context features, not these rollout outcomes.
    arrays["response_scenario_outcomes"] = outcomes.astype(np.float32)
    arrays["response_scenario_outcome_names"] = np.asarray(outcome_names)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_output, **arrays)

    unique_per_state = []
    for state in np.unique(state_id):
        rows = compiled[state_id == state]
        unique_per_state.append(len(np.unique(rows, axis=0)))
    payload = {
        "schema": "kaggriculture.candidate8-o15-response-feature-merge.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "identity_arrays_exact": exact,
        "canonical_consequence_exact": canonical_consequence_exact,
        "canonical_names_exact": canonical_names_exact,
        "pass_scenario_exact": pass_scenario_exact,
        "rows": int(len(state_id)),
        "states": int(len(np.unique(state_id))),
        "scenario_names": scenario_names,
        "compiled_width": int(compiled.shape[1]),
        "canonical_width": int(canonical.shape[1]),
        "scenario_delta_width": int(scenario_delta_flat.shape[1]),
        "scenario_outcome_width": int(outcome_flat.shape[1]),
        "scenario_outcome_delta_width": int(outcome_delta_flat.shape[1]),
        "robust_summary_width": int(sum(block.shape[1] for block in summary_blocks)),
        "minimum_unique_vectors_per_state": int(np.min(unique_per_state)),
        "boundary": (
            "Labels and row identity come unchanged from the frozen O1.4 "
            "dataset. Rival scenarios use fixed synthetic RNG and a public-"
            "history private-state belief; actual route futures are discarded."
        ),
        "inputs": {
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
            "response": {"path": str(args.response), "sha256": sha256(args.response)},
        },
        "dataset": {"path": str(args.dataset_output), "sha256": sha256(args.dataset_output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
