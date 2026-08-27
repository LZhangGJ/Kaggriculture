#!/usr/bin/env python3
"""Summarize a compact native switch search and select fine-search routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_by_opponent(
    outcome: np.ndarray,
    margin: np.ndarray,
    targets: list[str],
    opponents: list[str],
    stay_index: int,
    limit: int,
    greedy_limit: int = 12,
) -> tuple[list[dict[str, object]], list[str]]:
    """Rank static counter routes separately for every opponent.

    Inputs use the compact node layout ``target, opponent, seed, seat``.
    Raw wins are deliberately ranked before score and money margin because the
    experiment's acceptance gate is a per-opponent raw win rate.
    """
    score = outcome.astype(np.float32) * 0.5
    wins = outcome == 2
    raw_rate = wins.mean(axis=(2, 3))
    both_rate = np.all(wins, axis=3).mean(axis=2)
    score_rate = score.mean(axis=(2, 3))
    mean_margin = margin.mean(axis=(2, 3))
    rows: list[dict[str, object]] = []
    candidate_union: list[str] = []
    for opponent_index, opponent in enumerate(opponents):
        ranking = np.lexsort((
            -mean_margin[:, opponent_index],
            -score_rate[:, opponent_index],
            -both_rate[:, opponent_index],
            -raw_rate[:, opponent_index],
        ))
        quality = (
            outcome[:, opponent_index].astype(np.float64) * 1e12
            + margin[:, opponent_index]
        )
        best = np.argmax(quality, axis=0)
        oracle_outcome = np.take_along_axis(
            outcome[:, opponent_index], best[None, ...], axis=0
        )[0]
        oracle_margin = np.take_along_axis(
            margin[:, opponent_index], best[None, ...], axis=0
        )[0]
        covered = wins[stay_index, opponent_index].copy()
        greedy: list[dict[str, object]] = []
        used = {stay_index}
        for _ in range(greedy_limit):
            choices = []
            for target_index in range(len(targets)):
                if target_index in used:
                    continue
                next_covered = np.logical_or(
                    covered, wins[target_index, opponent_index]
                )
                gain = float(np.mean(next_covered) - np.mean(covered))
                choices.append((
                    gain,
                    float(raw_rate[target_index, opponent_index]),
                    float(both_rate[target_index, opponent_index]),
                    float(score_rate[target_index, opponent_index]),
                    float(mean_margin[target_index, opponent_index]),
                    -target_index,
                    target_index,
                    next_covered,
                ))
            if not choices:
                break
            choice = max(choices, key=lambda row: row[:-1])
            if choice[0] <= 0:
                break
            target_index = int(choice[-2])
            covered = choice[-1]
            used.add(target_index)
            family = targets[target_index]
            if family not in candidate_union:
                candidate_union.append(family)
            greedy.append({
                "family": family,
                "incremental_raw_win_coverage": float(choice[0]),
                "portfolio_raw_win_coverage": float(np.mean(covered)),
                "route_raw_win_rate": float(
                    raw_rate[target_index, opponent_index]
                ),
            })
        top = []
        for target_index in ranking[:limit]:
            family = targets[int(target_index)]
            if family not in candidate_union:
                candidate_union.append(family)
            top.append({
                "family": family,
                "raw_win_rate": float(raw_rate[target_index, opponent_index]),
                "both_seats_win_rate": float(
                    both_rate[target_index, opponent_index]
                ),
                "score_rate": float(score_rate[target_index, opponent_index]),
                "mean_margin": float(mean_margin[target_index, opponent_index]),
            })
        rows.append({
            "opponent": opponent,
            "samples": int(outcome.shape[2] * outcome.shape[3]),
            "baseline": {
                "family": targets[stay_index],
                "raw_win_rate": float(raw_rate[stay_index, opponent_index]),
                "both_seats_win_rate": float(
                    both_rate[stay_index, opponent_index]
                ),
                "score_rate": float(score_rate[stay_index, opponent_index]),
                "mean_margin": float(mean_margin[stay_index, opponent_index]),
            },
            "oracle": {
                "raw_win_rate": float(np.mean(oracle_outcome == 2)),
                "both_seats_win_rate": float(np.mean(np.all(
                    oracle_outcome == 2, axis=1
                ))),
                "score_rate": float(np.mean(oracle_outcome) * 0.5),
                "mean_margin": float(np.mean(oracle_margin)),
            },
            "static_routes_at_or_above_90pct": int(np.count_nonzero(
                raw_rate[:, opponent_index] >= 0.9
            )),
            "top_static_targets": top,
            "greedy_raw_win_portfolio": greedy,
            "greedy_raw_win_coverage": float(np.mean(covered)),
        })
    return rows, candidate_union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-greedy-gain", type=float, default=0.0005,
        help="Stop adding a route when its global paired oracle gain is smaller.",
    )
    parser.add_argument(
        "--per-opponent-limit", type=int, default=5,
        help="Number of static routes retained per opponent.",
    )
    parser.add_argument(
        "--per-opponent-greedy-limit", type=int, default=12,
        help="Maximum routes added to cover complementary wins per opponent.",
    )
    args = parser.parse_args()

    with np.load(args.input) as saved:
        # [opening, checkpoint, target, opponent, seed, seat]
        outcome = saved["outcome"].astype(np.uint8)
        score = outcome.astype(np.float32) * 0.5
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
            opponent_rows, opponent_union = summarize_by_opponent(
                outcome[oi, ci], margins, targets, opponents,
                int(stay_indices[oi]), args.per_opponent_limit,
                args.per_opponent_greedy_limit,
            )
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
                "per_opponent": opponent_rows,
                "per_opponent_candidate_union": opponent_union,
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
            "method": "paired greedy oracle compression over all candidates",
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
