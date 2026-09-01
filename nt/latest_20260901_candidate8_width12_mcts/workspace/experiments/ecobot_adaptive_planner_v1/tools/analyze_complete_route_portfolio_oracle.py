#!/usr/bin/env python3
"""Measure complete-route portfolio coverage over the audited C++ pool."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def greedy_cover(
    score: np.ndarray, candidate_names: list[str], opponent_names: list[str],
    threshold: float,
) -> dict:
    uncovered = set(range(len(opponent_names)))
    chosen: list[dict] = []
    while uncovered:
        best_index = -1
        best_cover: set[int] = set()
        for candidate in range(score.shape[0]):
            covered = {
                opponent for opponent in uncovered
                if score[candidate, opponent] >= threshold
            }
            if len(covered) > len(best_cover):
                best_index = candidate
                best_cover = covered
        if not best_cover:
            break
        chosen.append({
            "route": candidate_names[best_index],
            "newly_covered": len(best_cover),
            "opponents": [opponent_names[index] for index in sorted(best_cover)],
        })
        uncovered -= best_cover
    return {
        "threshold": threshold,
        "selected_routes": chosen,
        "covered": len(opponent_names) - len(uncovered),
        "total": len(opponent_names),
        "uncovered": [opponent_names[index] for index in sorted(uncovered)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.matrix, allow_pickle=False) as saved:
        families = [str(value) for value in saved["families"]]
        raw_score = np.asarray(saved["score"], dtype=np.float64)
        games = np.asarray(saved["games"], dtype=np.int64)
    score = raw_score.copy()
    for index in range(len(families)):
        if games[index, index] == 0:
            score[index, index] = 0.5

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    clean_names = [row["opponent"] for row in merged["opponents"]]
    hard_names = list(merged["route_bands"]["oracle_below_50pct"])
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {row["family"]: row for row in metadata["opponent_routes"]}
    index = {name: offset for offset, name in enumerate(families)}
    missing = [name for name in clean_names if name not in index]
    if missing:
        raise ValueError(f"clean pool routes absent from matrix: {missing}")
    columns = np.asarray([index[name] for name in clean_names], dtype=np.int64)
    clean_score = score[:, columns]

    best_candidate = np.argmax(clean_score, axis=0)
    best_score = clean_score[best_candidate, np.arange(len(clean_names))]
    overall = clean_score.mean(axis=1)
    candidate_order = np.argsort(-overall)

    opponents: list[dict] = []
    for column, name in enumerate(clean_names):
        candidate = int(best_candidate[column])
        opponents.append({
            "opponent": name,
            "hard16": name in hard_names,
            "best_route": families[candidate],
            "best_route_alias": details.get(families[candidate], {}).get("alias"),
            "best_score_rate": float(best_score[column]),
            "routes_at_least_90pct": int((clean_score[:, column] >= 0.90).sum()),
            "routes_at_least_50pct": int((clean_score[:, column] >= 0.50).sum()),
        })

    output = {
        "schema": "kaggriculture.complete-route-portfolio-oracle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Offline identity-aware Oracle. It chooses a complete route after "
            "knowing the exact opponent route and is not deployable."
        ),
        "summary": {
            "candidate_routes": len(families),
            "clean_opponents": len(clean_names),
            "games_per_ordered_pair": int(np.max(games)),
            "oracle_mean_score_rate": float(best_score.mean()),
            "oracle_worst_score_rate": float(best_score.min()),
            "opponents_oracle_at_least_90pct": int((best_score >= 0.90).sum()),
            "opponents_oracle_at_least_50pct": int((best_score >= 0.50).sum()),
            "hard16_oracle_at_least_90pct": sum(
                row["best_score_rate"] >= 0.90 for row in opponents if row["hard16"]
            ),
            "hard16_oracle_at_least_50pct": sum(
                row["best_score_rate"] >= 0.50 for row in opponents if row["hard16"]
            ),
            "best_single_route": families[int(candidate_order[0])],
            "best_single_route_mean_score": float(overall[candidate_order[0]]),
            "best_single_route_worst_score": float(clean_score[candidate_order[0]].min()),
        },
        "top_single_routes": [
            {
                "route": families[int(candidate)],
                "alias": details.get(families[int(candidate)], {}).get("alias"),
                "mean_score": float(overall[candidate]),
                "worst_score": float(clean_score[candidate].min()),
                "opponents_at_least_90pct": int((clean_score[candidate] >= 0.90).sum()),
                "opponents_at_least_50pct": int((clean_score[candidate] >= 0.50).sum()),
            }
            for candidate in candidate_order[:20]
        ],
        "greedy_cover_90pct": greedy_cover(
            clean_score, families, clean_names, 0.90
        ),
        "greedy_cover_50pct": greedy_cover(
            clean_score, families, clean_names, 0.50
        ),
        "hard16": [row for row in opponents if row["hard16"]],
        "opponents": opponents,
        "inputs": {
            "matrix": str(args.matrix),
            "merged_receipt": str(args.merged_receipt),
            "metadata": str(args.metadata),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "summary": output["summary"],
        "hard16": output["hard16"],
        "greedy90": {
            "routes": [row["route"] for row in output["greedy_cover_90pct"]["selected_routes"]],
            "covered": output["greedy_cover_90pct"]["covered"],
            "uncovered": output["greedy_cover_90pct"]["uncovered"],
        },
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
