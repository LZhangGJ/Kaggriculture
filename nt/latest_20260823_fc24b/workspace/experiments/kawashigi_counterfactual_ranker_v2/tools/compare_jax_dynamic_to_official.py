#!/usr/bin/env python3
"""Compare a JAX dynamic counterfactual panel with the exact official panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ROUTES = ("10C-4S-75L", "8C-6S-75L", "6C-12S-100L", "6C-8S-75L")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--jax-features", type=Path, required=True)
    parser.add_argument("--jax-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.official.read_text(encoding="utf-8"))
    opponent_names = [row["name"] for row in official["opponent_files"]]
    with np.load(args.jax_features, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        seeds = np.asarray(data["seeds"], dtype=np.int32)
        opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int16)
    with np.load(args.jax_outcomes, allow_pickle=False) as data:
        wins = np.asarray(data["wins"], dtype=bool)[:, :, 0]
        margins = np.asarray(data["margins"], dtype=np.float64)[:, :, 0]
    seed_pos = {int(value): index for index, value in enumerate(seeds)}
    opponent_pos = {opponent_names[int(value)]: index for index, value in enumerate(opponent_ids)}
    rows = {}
    context_features = {}
    for row in official["rows"]:
        key = (str(row["candidate"]), str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        rows[key] = row
        context_features.setdefault(key[1:], np.asarray(row["features"], dtype=np.float64))

    win_match = []
    margin_official = []
    margin_jax = []
    feature_abs = []
    best_match = []
    per_opponent = defaultdict(lambda: {"win_match": [], "margin_official": [], "margin_jax": [], "best_match": []})
    # The official panel may intentionally be a small subset of a larger JAX
    # panel.  Compare exactly the frozen official contexts and look up their
    # matching JAX rows instead of requiring both files to contain identical
    # seed ranges.
    for opponent, seed, seat in sorted(context_features):
        if opponent not in opponent_pos:
            raise KeyError(f"official opponent missing from JAX panel: {opponent}")
        if seed not in seed_pos:
            raise KeyError(f"official seed missing from JAX panel: {seed}")
        op_index = opponent_pos[opponent]
        seed_index = seed_pos[seed]
        off_features = context_features[(opponent, seed, seat)]
        jax_features = features[seat, op_index, seed_index]
        feature_abs.append(np.abs(off_features - jax_features))
        off_route_margin = []
        jax_route_margin = []
        for route_index, route in enumerate(ROUTES):
            row = rows[(route, opponent, seed, seat)]
            off_win = bool(row["win"])
            jax_win = bool(wins[route_index, seat, op_index, seed_index])
            off_margin = float(row["margin"])
            jax_margin = float(margins[route_index, seat, op_index, seed_index])
            match = off_win == jax_win
            win_match.append(match)
            margin_official.append(off_margin)
            margin_jax.append(jax_margin)
            per_opponent[opponent]["win_match"].append(match)
            per_opponent[opponent]["margin_official"].append(off_margin)
            per_opponent[opponent]["margin_jax"].append(jax_margin)
            off_route_margin.append(off_margin)
            jax_route_margin.append(jax_margin)
        same_best = int(np.argmax(off_route_margin)) == int(np.argmax(jax_route_margin))
        best_match.append(same_best)
        per_opponent[opponent]["best_match"].append(same_best)

    official_margin = np.asarray(margin_official)
    jax_margin = np.asarray(margin_jax)
    feature_error = np.stack(feature_abs)
    correlation = float(np.corrcoef(official_margin, jax_margin)[0, 1])
    by_opponent = {}
    for opponent, values in per_opponent.items():
        off = np.asarray(values["margin_official"])
        gpu = np.asarray(values["margin_jax"])
        by_opponent[opponent] = {
            "outcomes": len(values["win_match"]),
            "winner_agreement": float(np.mean(values["win_match"])),
            "best_route_agreement": float(np.mean(values["best_match"])),
            "margin_correlation": float(np.corrcoef(off, gpu)[0, 1]),
            "margin_mae": float(np.mean(np.abs(off - gpu))),
        }
    gates = {
        "winner_agreement_at_least_95pct": float(np.mean(win_match)) >= 0.95,
        "every_opponent_winner_agreement_at_least_90pct": min(value["winner_agreement"] for value in by_opponent.values()) >= 0.90,
        "best_route_agreement_at_least_80pct": float(np.mean(best_match)) >= 0.80,
        "margin_correlation_at_least_0p90": correlation >= 0.90,
    }
    result = {
        "schema": "kawashigi-jax-dynamic-official-parity-v1",
        "status": "PASS_LABEL_ADMISSION" if all(gates.values()) else "FAIL_LABEL_ADMISSION",
        "contexts": len(best_match),
        "outcomes": len(win_match),
        "winner_agreement": float(np.mean(win_match)),
        "best_route_agreement": float(np.mean(best_match)),
        "margin_correlation": correlation,
        "margin_mae": float(np.mean(np.abs(official_margin - jax_margin))),
        "feature_exact_context_rate": float(np.mean(np.max(feature_error, axis=1) == 0.0)),
        "feature_mean_abs_error": float(np.mean(feature_error)),
        "feature_max_abs_error": float(np.max(feature_error)),
        "by_opponent": by_opponent,
        "gates": gates,
        "truth_boundary": "Only PASS_LABEL_ADMISSION permits this JAX opponent panel to supply training labels; official Python remains the final referee.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS_LABEL_ADMISSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
