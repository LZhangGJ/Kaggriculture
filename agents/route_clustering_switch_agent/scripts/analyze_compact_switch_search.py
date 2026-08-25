#!/usr/bin/env python3
"""Summarize a compact native switch search and select fine-search routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-greedy-gain", type=float, default=0.0005,
        help="Stop adding a route when its global paired oracle gain is smaller.",
    )
    args = parser.parse_args()

    with np.load(args.input) as saved:
        # [opening, checkpoint, target, opponent, seed, seat]
        score = saved["outcome"].astype(np.float32) * 0.5
        margin = saved["margin"].astype(np.float32)
        openings = saved["openings"].astype(str).tolist()
        targets = saved["targets"].astype(str).tolist()
        opponents = saved["opponents"].astype(str).tolist()
        checkpoints = saved["checkpoints"].astype(int).tolist()
        seeds = saved["seeds"].astype(int).tolist()

    target_index = {family: index for index, family in enumerate(targets)}
    stay_indices = np.asarray([target_index[value] for value in openings])
    summaries = []
    best_response_counts = np.zeros(len(targets), dtype=np.int64)
    for oi, opening in enumerate(openings):
        for ci, checkpoint in enumerate(checkpoints):
            values = score[oi, ci]  # target, opponent, seed, seat
            margins = margin[oi, ci]
            stay = values[stay_indices[oi]]
            mean_score = values.mean(axis=(1, 2, 3))
            mean_margin = margins.mean(axis=(1, 2, 3))
            # Use paired win/tie/loss first; money margin only breaks equal scores.
            quality = values.astype(np.float64) * 1e12 + margins
            best = np.argmax(quality, axis=0)
            counts = np.bincount(best.ravel(), minlength=len(targets))
            best_response_counts += counts
            oracle = np.take_along_axis(values, best[None, ...], axis=0)[0]
            ranking = np.lexsort((-mean_margin, -mean_score))
            summaries.append({
                "opening": opening,
                "checkpoint": checkpoint,
                "samples": int(stay.size),
                "stay_score": float(stay.mean()),
                "oracle_score": float(oracle.mean()),
                "oracle_improvement": float((oracle - stay).mean()),
                "top_static_targets": [
                    {
                        "family": targets[index],
                        "score": float(mean_score[index]),
                        "improvement": float(mean_score[index] - stay.mean()),
                        "mean_margin": float(mean_margin[index]),
                        "oracle_frequency": int(counts[index]),
                    }
                    for index in ranking[:10]
                ],
            })

    # Greedy oracle compression. Every one of the 175 routes is tested; a route
    # survives only while it adds measurable paired win/tie/loss utility beyond
    # the already selected set. Both Nash openings are always retained.
    sample_blocks = []
    for oi in range(len(openings)):
        block = np.moveaxis(score[oi], 1, -1)  # checkpoint, opponent, seed, seat, target
        sample_blocks.append(block.reshape(-1, len(targets)))
    samples = np.concatenate(sample_blocks, axis=0)
    selected = list(dict.fromkeys(int(value) for value in stay_indices))
    current = np.max(samples[:, selected], axis=1)
    greedy = []
    while True:
        gains = np.maximum(samples, current[:, None]).mean(axis=0) - current.mean()
        gains[selected] = -np.inf
        index = int(np.argmax(gains))
        gain = float(gains[index])
        if not np.isfinite(gain) or gain < args.min_greedy_gain:
            break
        selected.append(index)
        current = np.maximum(current, samples[:, index])
        greedy.append({
            "family": targets[index],
            "marginal_paired_score_gain": gain,
            "cumulative_oracle_score": float(current.mean()),
            "oracle_frequency": int(best_response_counts[index]),
        })

    target_rows = sorted(
        (
            {
                "family": targets[index],
                "oracle_frequency": int(best_response_counts[index]),
                "global_mean_score": float(samples[:, index].mean()),
            }
            for index in range(len(targets))
        ),
        key=lambda row: (-row["oracle_frequency"], -row["global_mean_score"], row["family"]),
    )
    payload = {
        "schema_version": 1,
        "source": str(args.input),
        "search_dimensions": {
            "openings": openings,
            "targets": len(targets),
            "opponents": len(opponents),
            "checkpoints": checkpoints,
            "seeds": len(seeds),
            "seats": 2,
        },
        "screening": {
            "method": "paired greedy oracle compression over all 175 candidates",
            "min_marginal_gain": args.min_greedy_gain,
            "initial_nash_routes": openings,
            "selected_families": [targets[index] for index in selected],
            "added": greedy,
            "baseline_score": float(np.concatenate([
                samples[: len(samples) // len(openings), stay_indices[0]],
                samples[len(samples) // len(openings):, stay_indices[1]],
            ]).mean()) if len(openings) == 2 else None,
            "compressed_oracle_score": float(current.mean()),
        },
        "nodes": summaries,
        "route_oracle_frequency": target_rows,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_families": payload["screening"]["selected_families"],
        "added": greedy,
        "nodes": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
