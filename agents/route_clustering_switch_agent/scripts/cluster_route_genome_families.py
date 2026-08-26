#!/usr/bin/env python3
"""Cluster unique RouteGenome v2 plans with the repository's intent distance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np


ANCHORS = (168, 288, 432, 576, 719)
CATEGORIES = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "GOOSE", "COW", "SHEEP", "COOP", "PASTURE",
)
SCHEDULE_KEYS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "BUILD_COOP", "BUILD_PASTURE", "GOOSE", "COW", "SHEEP",
    "BUY_LAND", "HIRE",
)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def _load_groups(path: Path, max_genomes: int | None) -> list[dict[str, Any]]:
    rows = []
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if int(row.get("schema_version", 0)) != 2:
                raise ValueError(f"expected RouteGenome v2 aggregate at line {line_number}")
            rows.append(row)
            if max_genomes is not None and len(rows) >= max_genomes:
                break
    if not rows:
        raise ValueError(f"no RouteGenome groups in {path}")
    return rows


def _feature_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    category_id = {name: index + 1 for index, name in enumerate(CATEGORIES)}
    layouts = np.zeros((len(rows), len(ANCHORS), 100), dtype=np.int8)
    schedules = np.zeros((len(rows), 10, len(SCHEDULE_KEYS)), dtype=np.int16)
    for row_index, row in enumerate(rows):
        payload = row["genetic_payload"]
        targets = {int(value["step"]): value for value in payload["anchor_targets"]}
        if tuple(sorted(targets)) != ANCHORS:
            raise ValueError(f"unexpected anchors for {row['genome_id']}: {sorted(targets)}")
        for anchor_index, anchor in enumerate(ANCHORS):
            for placement in targets[anchor].get("placements", []):
                kind = str(placement["kind"])
                x, y = int(placement["x"]), int(placement["y"])
                if kind not in category_id or not (0 <= x < 10 and 0 <= y < 10):
                    raise ValueError(f"invalid placement for {row['genome_id']}: {placement}")
                layouts[row_index, anchor_index, y * 10 + x] = category_id[kind]
        phases = {int(value["phase"]): value for value in payload["phase_macro_counts"]}
        if tuple(sorted(phases)) != tuple(range(10)):
            raise ValueError(f"unexpected phases for {row['genome_id']}: {sorted(phases)}")
        for phase in range(10):
            counts = phases[phase].get("counts", {})
            schedules[row_index, phase] = [int(counts.get(key, 0)) for key in SCHEDULE_KEYS]
    return layouts, schedules


def _compute_distance_matrix(
    layouts: np.ndarray,
    schedules: np.ndarray,
    output: Path,
    block_size: int,
    progress_every: int,
) -> np.memmap:
    size = len(layouts)
    counts = np.stack(
        [(layouts == category).sum(axis=2) for category in range(1, 11)],
        axis=2,
    ).astype(np.int16)
    count_totals = counts.sum(axis=2, dtype=np.int32)
    cumulative = np.cumsum(schedules.astype(np.int32), axis=1)
    cumulative_totals = cumulative.sum(axis=2, dtype=np.int32)
    distances = np.memmap(output, mode="w+", dtype=np.float32, shape=(size, size))
    started = time.time()

    for block_index, start in enumerate(range(0, size, block_size), 1):
        stop = min(size, start + block_size)
        composition = np.zeros((stop - start, size), dtype=np.float32)
        layout = np.zeros_like(composition)
        schedule = np.zeros_like(composition)

        for anchor in range(len(ANCHORS)):
            left_counts = counts[start:stop, anchor]
            right_counts = counts[:, anchor]
            delta = np.abs(
                left_counts[:, None, :].astype(np.int32)
                - right_counts[None, :, :].astype(np.int32)
            ).sum(axis=2)
            denominator = np.maximum(
                8,
                np.maximum(
                    count_totals[start:stop, anchor, None],
                    count_totals[None, :, anchor],
                ),
            )
            composition += delta / denominator

            left_layout = layouts[start:stop, anchor]
            right_layout = layouts[:, anchor]
            active = (left_layout[:, None, :] != 0) | (right_layout[None, :, :] != 0)
            active_count = active.sum(axis=2)
            mismatch = ((left_layout[:, None, :] != right_layout[None, :, :]) & active).sum(axis=2)
            layout += np.divide(
                mismatch,
                active_count,
                out=np.zeros_like(layout),
                where=active_count > 0,
            )

        for phase in range(10):
            left_schedule = cumulative[start:stop, phase]
            right_schedule = cumulative[:, phase]
            delta = np.abs(left_schedule[:, None, :] - right_schedule[None, :, :]).sum(axis=2)
            denominator = np.maximum(
                6,
                np.maximum(
                    cumulative_totals[start:stop, phase, None],
                    cumulative_totals[None, :, phase],
                ),
            )
            schedule += delta / denominator

        distances[start:stop] = (
            0.45 * composition / len(ANCHORS)
            + 0.35 * layout / len(ANCHORS)
            + 0.20 * schedule / 10
        )
        if progress_every > 0 and (
            block_index % progress_every == 0 or stop == size
        ):
            distances.flush()
            print(json.dumps({
                "status": "distance",
                "rows": stop,
                "total": size,
                "elapsed_seconds": time.time() - started,
            }), flush=True)

    diagonal = np.asarray(distances.diagonal())
    if not np.allclose(diagonal, 0.0, atol=1e-7):
        raise ValueError(f"distance diagonal is not zero: max={float(np.max(np.abs(diagonal)))}")
    symmetry_error = 0.0
    for start in range(0, size, block_size):
        stop = min(size, start + block_size)
        error = np.max(np.abs(
            np.asarray(distances[start:stop]) - np.asarray(distances[:, start:stop]).T
        ))
        symmetry_error = max(symmetry_error, float(error))
    if symmetry_error > 1e-6:
        raise ValueError(f"distance matrix is not symmetric: max error={symmetry_error}")
    distances.flush()
    return distances


def _numeric_stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        values = np.zeros(1, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _cluster_labels(linkage_matrix: np.ndarray, threshold: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    return fcluster(linkage_matrix, t=threshold, criterion="distance").astype(np.int32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _family_set_id(
    threshold: float,
    genome_ids: Iterable[str],
    distance_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{threshold:.8f}|unique-genome-unweighted|".encode("ascii"))
    digest.update(distance_sha256.encode("ascii"))
    digest.update(b"|")
    for genome_id in genome_ids:
        digest.update(genome_id.encode("ascii"))
        digest.update(b"\n")
    return f"rgf1_{digest.hexdigest()[:20]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument(
        "--sensitivity",
        default=".04,.06,.08,.10,.12,.14,.16,.18,.20",
    )
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-genomes", type=int)
    parser.add_argument("--reuse-distance", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.max_genomes is not None and args.max_genomes <= 1:
        parser.error("--max-genomes must exceed one")
    input_path = args.groups.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    distance_path = output_root / "intent-distance-v1.f32"
    linkage_path = output_root / "average-linkage-v1.npy"
    mapping_path = output_root / "genome-family-map-v1.jsonl.gz"
    families_path = output_root / "route-families-v1.json"
    summary_path = output_root / "cluster-summary-v1.json"
    outputs = (linkage_path, mapping_path, families_path, summary_path)
    for path in outputs:
        if path.exists() and not args.overwrite:
            parser.error(f"output exists; pass --overwrite: {path}")

    started = time.time()
    rows = _load_groups(input_path, args.max_genomes)
    layouts, schedules = _feature_arrays(rows)
    expected_bytes = len(rows) * len(rows) * np.dtype(np.float32).itemsize
    if args.reuse_distance:
        if not distance_path.is_file() or distance_path.stat().st_size != expected_bytes:
            parser.error(f"cannot reuse distance matrix: {distance_path}")
        distances = np.memmap(
            distance_path, mode="r", dtype=np.float32, shape=(len(rows), len(rows))
        )
    else:
        if distance_path.exists() and not args.overwrite:
            parser.error(f"distance output exists; pass --overwrite: {distance_path}")
        distances = _compute_distance_matrix(
            layouts, schedules, distance_path, args.block_size, args.progress_every
        )

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(np.asarray(distances), checks=False)
    pair_quantiles = {
        "p10": float(np.quantile(condensed, 0.10)),
        "median": float(np.median(condensed)),
        "p90": float(np.quantile(condensed, 0.90)),
    }
    linkage_matrix = linkage(condensed, method="average")
    np.save(linkage_path, linkage_matrix)

    supports = np.asarray([int(row["sample_count"]) for row in rows], dtype=np.int64)
    thresholds = sorted({float(value) for value in args.sensitivity.split(",")} | {args.threshold})
    sensitivity: dict[str, Any] = {}
    for threshold in thresholds:
        trial = _cluster_labels(linkage_matrix, threshold)
        members = defaultdict(list)
        for index, label in enumerate(trial.tolist()):
            members[int(label)].append(index)
        genome_sizes = sorted((len(value) for value in members.values()), reverse=True)
        execution_sizes = sorted(
            (int(supports[value].sum()) for value in members.values()), reverse=True
        )
        sensitivity[str(threshold)] = {
            "families": len(members),
            "genome_singletons": sum(size == 1 for size in genome_sizes),
            "execution_singletons": sum(size == 1 for size in execution_sizes),
            "top56_genome_coverage": sum(genome_sizes[:56]) / len(rows),
            "top56_execution_coverage": sum(execution_sizes[:56]) / int(supports.sum()),
        }

    labels = _cluster_labels(linkage_matrix, args.threshold)
    raw_groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        raw_groups[int(label)].append(index)
    ordered_groups = sorted(
        raw_groups.items(),
        key=lambda value: (
            -int(supports[value[1]].sum()),
            -len(value[1]),
            value[0],
        ),
    )
    distance_sha256 = _sha256_file(distance_path)
    family_set_id = _family_set_id(
        args.threshold,
        (str(row["genome_id"]) for row in rows),
        distance_sha256,
    )
    family_rows = []
    mapping_rows = []
    for rank, (raw_label, member_values) in enumerate(ordered_groups, 1):
        members = np.asarray(member_values, dtype=np.int64)
        member_supports = supports[members].astype(np.float64)
        within = np.asarray(distances[np.ix_(members, members)], dtype=np.float64)
        weighted_mean = np.divide(
            within @ member_supports,
            member_supports.sum(),
        )
        medoid_local = int(np.argmin(weighted_mean))
        medoid_index = int(members[medoid_local])
        medoid_id = str(rows[medoid_index]["genome_id"])
        family = f"V2F{rank:04d}"
        outcomes: Counter[str] = Counter()
        teams: Counter[str] = Counter()
        reward_total = 0.0
        for index in members:
            row = rows[int(index)]
            outcomes.update({key: int(value) for key, value in row["outcome_counts"].items()})
            teams.update({key: int(value) for key, value in row["team_counts"].items()})
            reward_total += (
                float(row["reward_summary"]["own"]["mean"]) * int(row["sample_count"])
            )
        upper = within[np.triu_indices(len(members), 1)] if len(members) > 1 else np.zeros(1)
        family_rows.append({
            "family": family,
            "raw_cluster_label": raw_label,
            "genome_count": len(members),
            "execution_support": int(member_supports.sum()),
            "medoid_genome_id": medoid_id,
            "medoid_weighted_mean_distance": float(weighted_mean[medoid_local]),
            "within_distance": _numeric_stats(upper),
            "historical_outcome_counts": dict(sorted(outcomes.items())),
            "historical_mean_reward": reward_total / member_supports.sum(),
            "top_team_counts": dict(teams.most_common(20)),
        })
        for index in members:
            mapping_rows.append({
                "schema_version": 1,
                "family_set_id": family_set_id,
                "family": family,
                "genome_id": str(rows[int(index)]["genome_id"]),
                "genome_sample_count": int(supports[int(index)]),
                "distance_to_medoid": float(distances[int(index), medoid_index]),
                "is_medoid": int(index) == medoid_index,
            })

    mapping_rows.sort(key=lambda value: value["genome_id"])
    with _open_text(mapping_path, "w") as handle:
        for row in mapping_rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")
    family_payload = {
        "schema_version": 1,
        "kind": "route_genome_v2_intent_families",
        "family_set_id": family_set_id,
        "identity": "unique RouteGenome v2 macro intent; one vote per exact genome",
        "distance": "0.45 composition + 0.35 layout + 0.20 cumulative macro schedule",
        "linkage": "average",
        "threshold": args.threshold,
        "families": family_rows,
    }
    families_path.write_text(
        json.dumps(family_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    family_sizes = [int(row["genome_count"]) for row in family_rows]
    execution_sizes = [int(row["execution_support"]) for row in family_rows]
    summary = {
        "schema_version": 1,
        "kind": "route_genome_v2_intent_cluster_experiment",
        "family_set_id": family_set_id,
        "input": str(input_path),
        "unique_genomes": len(rows),
        "execution_records": int(supports.sum()),
        "distance_matrix": str(distance_path),
        "distance_matrix_bytes": distance_path.stat().st_size,
        "distance_matrix_sha256": distance_sha256,
        "pair_distance": pair_quantiles,
        "linkage": "average",
        "threshold": args.threshold,
        "families": len(family_rows),
        "genome_singletons": sum(size == 1 for size in family_sizes),
        "execution_singletons": sum(size == 1 for size in execution_sizes),
        "family_genome_sizes_desc": sorted(family_sizes, reverse=True),
        "family_execution_sizes_desc": sorted(execution_sizes, reverse=True),
        "sensitivity": sensitivity,
        "outputs": {
            "linkage": str(linkage_path),
            "mapping": str(mapping_path),
            "families": str(families_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "family_set_id": family_set_id,
        "unique_genomes": len(rows),
        "families": len(family_rows),
        "genome_singletons": summary["genome_singletons"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "summary": str(summary_path),
    }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
