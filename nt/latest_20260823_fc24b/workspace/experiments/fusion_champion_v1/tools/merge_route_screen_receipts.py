#!/usr/bin/env python3
"""Merge disjoint K320 route-screen receipts without losing seat/seed keys."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    reference = payloads[0]
    immutable = (
        "official_package_version",
        "opponent",
        "prefix",
        "switch_steps",
        "feature_step",
        "candidate_stack",
        "seat_protocol",
    )
    for payload in payloads[1:]:
        for field in immutable:
            if payload.get(field) != reference.get(field):
                raise AssertionError(f"incompatible {field}")

    variants = [row["variant"] for row in reference["rows"]]
    if any([row["variant"] for row in payload["rows"]] != variants for payload in payloads):
        raise AssertionError("variant order mismatch")

    rows = []
    margin_matrix = []
    seen: set[tuple[int, int]] = set()
    for variant_index, variant in enumerate(variants):
        source_rows = [payload["rows"][variant_index] for payload in payloads]
        per_game = [game for row in source_rows for game in row["per_game"]]
        keys = [(int(game["seed"]), int(game["candidate_seat"])) for game in per_game]
        if variant_index == 0:
            if len(keys) != len(set(keys)):
                raise AssertionError("duplicate seed/seat key")
            seen = set(keys)
        elif set(keys) != seen:
            raise AssertionError("arm key mismatch")
        margins = np.asarray([int(game["margin"]) for game in per_game], dtype=np.int64)
        games = len(margins)
        template = {
            key: value
            for key, value in source_rows[0].items()
            if key
            not in {
                "games",
                "wins",
                "ties",
                "losses",
                "score_rate",
                "mean_candidate_cash",
                "mean_opponent_cash",
                "mean_margin",
                "per_game",
            }
        }
        template.update(
            {
                "games": games,
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                "mean_candidate_cash": float(
                    sum(row["mean_candidate_cash"] * row["games"] for row in source_rows)
                    / games
                ),
                "mean_opponent_cash": float(
                    sum(row["mean_opponent_cash"] * row["games"] for row in source_rows)
                    / games
                ),
                "mean_margin": float(np.mean(margins)),
                "per_game": per_game,
            }
        )
        rows.append(template)
        margin_matrix.append(margins)

    features = [row for payload in payloads for row in payload.get("decision_features", [])]
    feature_keys = {(int(row["seed"]), int(row["candidate_seat"])) for row in features}
    if len(features) != len(feature_keys) or feature_keys != seen:
        raise AssertionError("decision feature key mismatch")
    oracle = np.max(np.stack(margin_matrix), axis=0)
    seeds = sorted({key[0] for key in seen})
    result = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"rows", "oracle", "decision_features", "elapsed_seconds"}
        },
        "schema": "kaggriculture.fusion_champion.k320-route-override-screen-merged.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "seed_start": min(seeds),
        "seed_count": len(seeds),
        "seed_ranges": [
            [int(payload["seed_start"]), int(payload["seed_start"] + payload["seed_count"] - 1)]
            for payload in payloads
        ],
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(
                np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)
            ),
            "mean_margin": float(np.mean(oracle)),
        },
        "decision_features": features,
        "source_receipts": [str(path.resolve()) for path in args.inputs],
        "elapsed_seconds": float(sum(payload.get("elapsed_seconds", 0.0) for payload in payloads)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "games_per_arm": rows[0]["games"],
                "oracle": result["oracle"],
                "output": str(args.output.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
