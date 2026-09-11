#!/usr/bin/env python3
"""Evaluate a frozen step-120 public-state route model on an untouched JAX panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from fit_step120_public_margin_router import (
    portable_predict,
    route_frequency,
    select_with_threshold,
    sha256,
    summarize,
)


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--opponents", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    model_path = resolve(args.model)
    features_path = resolve(args.features)
    outcomes_path = resolve(args.outcomes)
    bank_path = resolve(args.bank_receipt)
    receipt_path = resolve(args.receipt)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    candidate_names = [str(value) for value in bank["candidate_names"]]
    bank_opponents = [str(row["name"]) for row in bank["opponents"]]
    selected_opponents = [
        value.strip() for value in args.opponents.split(",") if value.strip()
    ]
    if not selected_opponents:
        raise ValueError("--opponents must not be empty")

    with np.load(features_path, allow_pickle=False) as data:
        features = np.asarray(data["features120"], dtype=np.float32)
        feature_names = [str(value) for value in data["feature_names120"]]
        feature_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
        feature_seeds = np.asarray(data["seeds"], dtype=np.int64)
    with np.load(outcomes_path, allow_pickle=False) as data:
        margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
        outcome_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
        outcome_seeds = np.asarray(data["seeds"], dtype=np.int64)
    if feature_names != [str(value) for value in model["feature_names"]]:
        raise ValueError("feature names do not match the frozen model")
    if not np.array_equal(feature_seeds, outcome_seeds):
        raise ValueError("feature and outcome seeds differ")

    route_names = [str(value) for value in model["route_names"]]
    route_ids = [candidate_names.index(route) for route in route_names]
    x_chunks: list[np.ndarray] = []
    margin_chunks: list[np.ndarray] = []
    opponent_chunks: list[np.ndarray] = []
    for opponent_index, name in enumerate(selected_opponents):
        compiled_id = bank_opponents.index(name)
        feature_match = np.flatnonzero(feature_opponent_ids == compiled_id)
        outcome_match = np.flatnonzero(outcome_opponent_ids == compiled_id)
        if feature_match.size != 1 or outcome_match.size != 1:
            raise ValueError(f"panel is missing exactly one copy of {name}")
        feature_position = int(feature_match[0])
        outcome_position = int(outcome_match[0])
        values = features[:, feature_position].reshape(-1, features.shape[-1])
        route_margins = margins[
            route_ids, :, outcome_position, :
        ].reshape(len(route_names), -1)
        x_chunks.append(values)
        margin_chunks.append(route_margins)
        opponent_chunks.append(
            np.full(values.shape[0], opponent_index, dtype=np.int16)
        )

    x = np.concatenate(x_chunks, axis=0)
    route_margins = np.concatenate(margin_chunks, axis=1)
    opponent_ids = np.concatenate(opponent_chunks)
    predictions = np.stack(
        [portable_predict(row["model"], x) for row in model["models"]], axis=0
    )
    baseline_index = int(model["baseline_route_index"])
    routed_margin, selected = select_with_threshold(
        predictions,
        route_margins,
        baseline_index,
        float(model["switch_threshold"]),
    )
    result = {
        "schema": "kaggriculture-step120-public-margin-router-panel-evaluation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_UNTOUCHED_PANEL_EVALUATED",
        "seed_start": int(feature_seeds[0]),
        "seed_count": int(feature_seeds.size),
        "seat_swapped": True,
        "selected_opponents": selected_opponents,
        "route_count": len(route_names),
        "baseline": summarize(
            route_margins[baseline_index], opponent_ids, selected_opponents
        ),
        "router": summarize(routed_margin, opponent_ids, selected_opponents),
        "hindsight_oracle": summarize(
            np.max(route_margins, axis=0), opponent_ids, selected_opponents
        ),
        "route_frequency": route_frequency(selected, route_names),
        "sources": {
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "features": str(features_path),
            "features_sha256": sha256(features_path),
            "outcomes": str(outcomes_path),
            "outcomes_sha256": sha256(outcomes_path),
            "bank_receipt": str(bank_path),
            "bank_receipt_sha256": sha256(bank_path),
        },
        "truth_boundary": (
            "The frozen model did not train on this seed range. This is JAX "
            "screening against stepwise-exact opponent ports; official Python "
            "1.32.7 remains the final referee."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
