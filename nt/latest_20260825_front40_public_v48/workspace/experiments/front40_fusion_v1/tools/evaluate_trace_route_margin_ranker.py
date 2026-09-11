"""Evaluate a frozen visible-state route ranker on an independent screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_trace_route_margin_ranker import (
    build_context_features,
    build_difference_features,
    planned_route_features,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    context = np.load(args.context, allow_pickle=False)
    bank = np.load(args.route_bank, allow_pickle=False)
    screen = np.load(args.matrix, allow_pickle=False)
    route_ids = np.asarray(artifact["route_ids"], dtype=np.int32)
    step = int(artifact["decision_step"])
    seats, seeds = context["candidate_seat"].shape
    contexts = seats * seeds
    if not np.array_equal(context["seeds"], screen["seeds"]):
        raise ValueError("context and screen seeds differ")

    screen_global = np.asarray(
        screen["route_ids"] if "route_ids" in screen else np.arange(screen["margin"].shape[2]),
        dtype=np.int32,
    )
    screen_lookup = {int(route): index for index, route in enumerate(screen_global)}
    if any(int(route) not in screen_lookup for route in route_ids):
        missing = [int(route) for route in route_ids if int(route) not in screen_lookup]
        raise ValueError(f"screen lacks model routes: {missing}")
    columns = np.asarray([screen_lookup[int(route)] for route in route_ids], np.int32)
    margin = np.asarray(screen["margin"], dtype=np.float32)[:, :, columns].reshape(
        contexts, route_ids.size
    )
    invalid = np.asarray(screen["invalid"], dtype=np.float32)[:, :, columns].reshape(
        contexts, route_ids.size
    )

    context_features, _ = build_context_features(context)
    route_features_full, _ = planned_route_features(bank, step)
    route_features = route_features_full[route_ids]
    difference_full, _ = build_difference_features(context, bank, step)
    difference = difference_full.reshape(contexts, bank["unit_op"].shape[0], -1)[
        :, route_ids
    ].reshape(contexts * route_ids.size, -1)
    features = np.concatenate(
        (
            np.repeat(context_features, route_ids.size, axis=0),
            np.tile(route_features, (contexts, 1)),
            difference,
        ),
        axis=1,
    ).astype(np.float32)

    predicted_margin = artifact["margin_model"].predict(features).reshape(
        contexts, route_ids.size
    )
    predicted_win = artifact["win_model"].predict_proba(features)[:, 1].reshape(
        contexts, route_ids.size
    )
    predicted_rank = artifact["rank_model"].predict(features).reshape(
        contexts, route_ids.size
    )
    score = (
        predicted_margin
        + float(artifact["win_probability_weight"])
        * float(artifact["score_scale"])
        * predicted_win
        + float(artifact["rank_weight"])
        * float(artifact["score_scale"])
        * predicted_rank
        / float(artifact["rank_scale"])
    )
    selected_local = np.argmax(score, axis=1)
    rows = np.arange(contexts)
    selected_margin = margin[rows, selected_local]
    selected_invalid = invalid[rows, selected_local]
    oracle = np.max(margin, axis=1)
    counts = np.bincount(selected_local, minlength=route_ids.size)
    selected_routes = [
        {"route_id": int(route_ids[index]), "contexts": int(counts[index])}
        for index in np.flatnonzero(counts)
    ]
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-route-margin-ranker-eval.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "model": str(args.model),
        "context": str(args.context),
        "matrix": str(args.matrix),
        "runtime_boundary": "current public state plus own private inventory only",
        "games": contexts,
        "wins": int(np.sum(selected_margin > 0)),
        "ties": int(np.sum(selected_margin == 0)),
        "win_rate": float(np.mean(selected_margin > 0)),
        "mean_margin": float(np.mean(selected_margin)),
        "median_margin": float(np.median(selected_margin)),
        "invalid_mean": float(np.mean(selected_invalid)),
        "oracle_win_rate_within_model_routes": float(np.mean(oracle > 0)),
        "selected_routes": selected_routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "games", "wins", "win_rate", "mean_margin")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
