#!/usr/bin/env python3
"""Cluster reconstructed Kaggriculture macro intents with the paper's metric.

The implementation keeps the three distance components separate and writes a
reproducible condensed distance matrix, average-linkage dendrogram, threshold
sensitivity table, provisional family assignments, and family prototypes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


WEIGHTS = {"composition": 0.45, "grid": 0.35, "timing": 0.20}
DEFAULT_THRESHOLDS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument(
        "--reuse-distance",
        action="store_true",
        help="Reuse an existing macro_intent_distance_v1.npy after shape validation.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def condensed_row_offset(n: int, i: int) -> int:
    return n * i - i * (i + 1) // 2


def category_counts(layout: np.ndarray, category_count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.empty((layout.shape[0], layout.shape[1], category_count), dtype=np.int16)
    for category in range(1, category_count + 1):
        counts[:, :, category - 1] = np.count_nonzero(layout == category, axis=2)
    totals = np.count_nonzero(layout, axis=2).astype(np.int16)
    return counts, totals


def compute_condensed_distances(
    layout: np.ndarray,
    cumulative: np.ndarray,
    output_path: Path,
    block_size: int,
) -> np.memmap:
    n = layout.shape[0]
    pair_count = n * (n - 1) // 2
    distances = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(pair_count,)
    )
    counts, totals = category_counts(layout, 10)
    cumulative_i32 = cumulative.astype(np.int32, copy=False)
    action_totals = cumulative_i32.sum(axis=2)

    started = time.perf_counter()
    for start in range(0, n - 1, block_size):
        stop = min(start + block_size, n - 1)
        row_count = stop - start

        composition_num = np.abs(
            counts[start:stop, None, :, :] - counts[None, :, :, :]
        ).sum(axis=3, dtype=np.int32)
        composition_den = np.maximum(
            8,
            np.maximum(totals[start:stop, None, :], totals[None, :, :]),
        )
        composition = (composition_num / composition_den).mean(axis=2)

        grid = np.zeros((row_count, n), dtype=np.float32)
        for anchor in range(layout.shape[1]):
            left = layout[start:stop, None, anchor, :]
            right = layout[None, :, anchor, :]
            union = (left != 0) | (right != 0)
            union_count = union.sum(axis=2, dtype=np.int16)
            mismatch_count = ((left != right) & union).sum(axis=2, dtype=np.int16)
            grid += np.divide(
                mismatch_count,
                union_count,
                out=np.zeros_like(grid),
                where=union_count != 0,
            )
        grid /= layout.shape[1]

        timing_num = np.abs(
            cumulative_i32[start:stop, None, :, :] - cumulative_i32[None, :, :, :]
        ).sum(axis=3, dtype=np.int32)
        timing_den = np.maximum(
            6,
            np.maximum(action_totals[start:stop, None, :], action_totals[None, :, :]),
        )
        timing = (timing_num / timing_den).mean(axis=2)

        combined = (
            WEIGHTS["composition"] * composition
            + WEIGHTS["grid"] * grid
            + WEIGHTS["timing"] * timing
        )
        for local_i, global_i in enumerate(range(start, stop)):
            offset = condensed_row_offset(n, global_i)
            values = combined[local_i, global_i + 1 :]
            distances[offset : offset + values.size] = values.astype(np.float32, copy=False)

        done = stop
        elapsed = time.perf_counter() - started
        print(
            f"distance rows {done}/{n - 1} ({100 * done / (n - 1):.1f}%), "
            f"elapsed {elapsed:.1f}s",
            flush=True,
        )
    distances.flush()
    return distances


def family_order(raw_labels: np.ndarray, rewards: np.ndarray) -> list[int]:
    records = []
    for raw in np.unique(raw_labels):
        idx = np.flatnonzero(raw_labels == raw)
        median_reward = float(np.nanmedian(rewards[idx]))
        records.append((int(raw), int(idx.size), median_reward))
    records.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [raw for raw, _, _ in records]


def label_at_threshold(linkage_matrix: np.ndarray, threshold: float) -> np.ndarray:
    return fcluster(linkage_matrix, t=threshold, criterion="distance").astype(np.int32)


def sensitivity_rows(linkage_matrix: np.ndarray, n: int) -> list[dict]:
    rows = []
    for threshold in DEFAULT_THRESHOLDS:
        labels = label_at_threshold(linkage_matrix, threshold)
        counts = np.bincount(labels)[1:]
        counts = counts[counts > 0]
        descending = np.sort(counts)[::-1]
        rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "family_count": int(counts.size),
                "singleton_count": int(np.count_nonzero(counts == 1)),
                "largest_family": int(descending[0]),
                "top10_coverage": float(descending[:10].sum() / n),
                "top56_coverage": float(descending[:56].sum() / n),
            }
        )
    return rows


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.input_dir / "intent_features_v1.npz"
    seat_path = args.input_dir / "intent_seats_v1.csv"
    features = np.load(feature_path)
    seats = read_csv(seat_path)

    seat_ids = features["seat_id"].astype(str)
    layout = features["layout"]
    stage_actions = features["stage_actions"]
    cumulative = features["cumulative_actions"]
    n = seat_ids.size
    if len(seats) != n:
        raise ValueError(f"seat CSV has {len(seats)} rows, features have {n}")
    if [row["seat_index"] for row in seats] != [str(i) for i in range(n)]:
        raise ValueError("seat_index is not contiguous and aligned")
    if [f'{row["episode_id"]}:{row["player_index"]}:{row["submission_id"]}' for row in seats] != seat_ids.tolist():
        raise ValueError("seat_id order differs between CSV and NPZ")
    if len(set(seat_ids.tolist())) != n:
        raise ValueError("duplicate seat_id values")
    if layout.shape != (n, 5, 100) or cumulative.shape != (n, 10, 12):
        raise ValueError("unexpected feature shapes")

    print(f"validated {n} seats", flush=True)
    condensed_path = args.output_dir / "macro_intent_distance_v1.npy"
    expected_pairs = n * (n - 1) // 2
    if args.reuse_distance and condensed_path.exists():
        condensed = np.load(condensed_path, mmap_mode="r")
        if condensed.shape != (expected_pairs,) or condensed.dtype != np.float32:
            raise ValueError("existing distance matrix has the wrong shape or dtype")
        print("reusing validated distance matrix", flush=True)
    else:
        condensed = compute_condensed_distances(
            layout, cumulative, condensed_path, args.block_size
        )
    if not np.isfinite(condensed).all() or np.any(condensed < 0):
        raise ValueError("distance matrix contains invalid values")
    print(
        f"distance matrix ready: {condensed.size:,} pairs, "
        f"range {float(condensed.min()):.6f}..{float(condensed.max()):.6f}",
        flush=True,
    )

    print("running average linkage", flush=True)
    linkage_matrix = linkage(np.asarray(condensed), method="average")
    if np.any(np.diff(linkage_matrix[:, 2]) < -1e-12):
        raise ValueError("average-linkage merge distances are not monotonic")
    np.save(args.output_dir / "average_linkage_v1.npy", linkage_matrix)

    sensitivity = sensitivity_rows(linkage_matrix, n)
    write_csv(
        args.output_dir / "threshold_sensitivity_v1.csv",
        sensitivity,
        [
            "threshold",
            "family_count",
            "singleton_count",
            "largest_family",
            "top10_coverage",
            "top56_coverage",
        ],
    )

    raw_labels = label_at_threshold(linkage_matrix, args.threshold)
    rewards = np.array([safe_float(row["reward"]) for row in seats], dtype=np.float64)
    ordered_raw = family_order(raw_labels, rewards)
    raw_to_family = {raw: f"N{position:03d}" for position, raw in enumerate(ordered_raw, 1)}
    family_ids = np.array([raw_to_family[int(raw)] for raw in raw_labels], dtype="U8")

    print("building medoids and family profiles", flush=True)
    square = squareform(np.asarray(condensed), checks=False)
    family_count = len(ordered_raw)
    medoid_indices = np.empty(family_count, dtype=np.int32)
    consensus_schedule = np.empty((family_count, 10, 12), dtype=np.uint16)
    family_mode_layout = np.empty((family_count, 5, 100), dtype=np.uint8)
    distance_to_medoid = np.empty(n, dtype=np.float32)
    profiles: list[dict] = []

    for family_position, raw in enumerate(ordered_raw):
        family_id = raw_to_family[raw]
        idx = np.flatnonzero(raw_labels == raw)
        intra = square[np.ix_(idx, idx)]
        medoid_local = int(np.argmin(intra.mean(axis=1)))
        medoid = int(idx[medoid_local])
        medoid_indices[family_position] = medoid
        distance_to_medoid[idx] = square[idx, medoid]
        consensus_schedule[family_position] = np.rint(
            np.median(stage_actions[idx], axis=0)
        ).astype(np.uint16)

        for anchor in range(5):
            for cell in range(100):
                codes = layout[idx, anchor, cell]
                family_mode_layout[family_position, anchor, cell] = np.bincount(
                    codes, minlength=11
                ).argmax()

        cluster_rewards = rewards[idx]
        result_counts = Counter(seats[i]["result"] for i in idx)
        final_counts = []
        for category in range(1, 11):
            per_seat = np.count_nonzero(layout[idx, -1, :] == category, axis=1)
            final_counts.append(float(np.median(per_seat)))
        profile = {
            "family_id": family_id,
            "raw_cluster_label": raw,
            "size": int(idx.size),
            "share": float(idx.size / n),
            "median_reward": float(np.nanmedian(cluster_rewards)),
            "mean_reward": float(np.nanmean(cluster_rewards)),
            "win_rate": float(result_counts["WIN"] / idx.size),
            "tie_rate": float(result_counts["TIE"] / idx.size),
            "median_rank": float(np.median([int(seats[i]["rank"]) for i in idx])),
            "unique_submissions": len({seats[i]["submission_id"] for i in idx}),
            "unique_teams": len({seats[i]["team_id"] for i in idx}),
            "medoid_seat_index": medoid,
            "medoid_seat_id": seat_ids[medoid],
            "medoid_episode_id": seats[medoid]["episode_id"],
            "medoid_player_index": seats[medoid]["player_index"],
            "medoid_submission_id": seats[medoid]["submission_id"],
            "medoid_team": seats[medoid]["team"],
            "medoid_reward": rewards[medoid],
            "mean_distance_to_medoid": float(distance_to_medoid[idx].mean()),
            "max_distance_to_medoid": float(distance_to_medoid[idx].max()),
        }
        for name, value in zip(features["intent_categories"].astype(str), final_counts):
            profile[f"median_final_{name.lower()}"] = value
        profiles.append(profile)

    profile_fields = list(profiles[0].keys())
    write_csv(args.output_dir / "cluster_profiles_v1.csv", profiles, profile_fields)

    assignment_rows = []
    cluster_sizes = {row["family_id"]: row["size"] for row in profiles}
    medoid_set = set(medoid_indices.tolist())
    for i, source in enumerate(seats):
        row = dict(source)
        row.update(
            {
                "provisional_family_id": family_ids[i],
                "cluster_size": cluster_sizes[family_ids[i]],
                "is_medoid": int(i in medoid_set),
                "distance_to_medoid": float(distance_to_medoid[i]),
            }
        )
        assignment_rows.append(row)
    assignment_fields = list(seats[0].keys()) + [
        "provisional_family_id",
        "cluster_size",
        "is_medoid",
        "distance_to_medoid",
    ]
    write_csv(
        args.output_dir / "cluster_assignments_v1.csv",
        assignment_rows,
        assignment_fields,
    )

    submission_rows: list[dict] = []
    by_submission: dict[str, list[int]] = {}
    for i, row in enumerate(seats):
        by_submission.setdefault(row["submission_id"], []).append(i)
    for submission_id, indices_list in by_submission.items():
        idx = np.array(indices_list, dtype=np.int32)
        family_counts = Counter(family_ids[idx].tolist())
        dominant_family, dominant_seats = family_counts.most_common(1)[0]
        top3_seats = sum(count for _, count in family_counts.most_common(3))
        singleton_seats = sum(cluster_sizes[family_ids[i]] == 1 for i in idx)
        first = seats[int(idx[0])]
        submission_rows.append(
            {
                "rank": int(first["rank"]),
                "team": first["team"],
                "team_id": first["team_id"],
                "submission_id": submission_id,
                "seat_count": int(idx.size),
                "family_count": len(family_counts),
                "singleton_seats": int(singleton_seats),
                "singleton_share": float(singleton_seats / idx.size),
                "dominant_family_id": dominant_family,
                "dominant_family_seats": dominant_seats,
                "dominant_family_share": float(dominant_seats / idx.size),
                "top3_family_coverage": float(top3_seats / idx.size),
                "median_distance_to_medoid": float(np.median(distance_to_medoid[idx])),
            }
        )
    submission_rows.sort(key=lambda row: row["rank"])
    write_csv(
        args.output_dir / "submission_cluster_summary_v1.csv",
        submission_rows,
        list(submission_rows[0].keys()),
    )

    np.savez_compressed(
        args.output_dir / "cluster_prototypes_v1.npz",
        seat_id=seat_ids,
        raw_cluster_label=raw_labels,
        provisional_family_id=family_ids,
        medoid_indices=medoid_indices,
        medoid_seat_id=seat_ids[medoid_indices],
        consensus_stage_actions=consensus_schedule,
        mode_layout=family_mode_layout,
        threshold=np.float64(args.threshold),
    )

    main_sensitivity = next(
        row for row in sensitivity if float(row["threshold"]) == args.threshold
    )
    sizes = np.array([row["size"] for row in profiles], dtype=np.int32)
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "provisional_macro_intent_clustering",
        "input_dir": str(args.input_dir.resolve()),
        "input_feature_path": str(feature_path.resolve()),
        "input_seat_path": str(seat_path.resolve()),
        "seat_count": n,
        "pair_count": int(condensed.size),
        "method": {
            "distance": "D = 0.45*C + 0.35*G + 0.20*T",
            "composition": "mean across 5 anchors of L1 category-count difference / max(8, occupied_i, occupied_j)",
            "grid": "mean across 5 anchors of mismatching category cells / union occupied cells",
            "timing": "mean across 10 stages of L1 cumulative-action difference / max(6, cumulative_i, cumulative_j)",
            "linkage": "average",
            "threshold": args.threshold,
            "family_id_policy": "provisional N###, ordered by size desc then median reward desc; not aligned to historical G labels",
            "representative": "exact medoid minimizing mean intra-family distance",
        },
        "distance": {
            "min": float(condensed.min()),
            "median": float(np.median(condensed)),
            "mean": float(condensed.mean(dtype=np.float64)),
            "max": float(condensed.max()),
        },
        "main_cut": main_sensitivity,
        "family_size": {
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "p90": float(np.quantile(sizes, 0.90)),
            "max": int(sizes.max()),
        },
        "threshold_sensitivity": sensitivity,
        "outputs": {
            "distance_matrix": "macro_intent_distance_v1.npy",
            "linkage": "average_linkage_v1.npy",
            "assignments": "cluster_assignments_v1.csv",
            "profiles": "cluster_profiles_v1.csv",
            "prototypes": "cluster_prototypes_v1.npz",
            "sensitivity": "threshold_sensitivity_v1.csv",
            "submission_summary": "submission_cluster_summary_v1.csv",
        },
    }
    with (args.output_dir / "cluster_summary_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"done: {family_count} provisional families at D <= {args.threshold:.2f}; "
        f"{main_sensitivity['singleton_count']} singletons",
        flush=True,
    )


if __name__ == "__main__":
    main()
