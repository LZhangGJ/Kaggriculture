"""Behavior-cloning sampling utilities backed by the 30-agent audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TeacherProfile:
    spec: str
    agent: str
    cluster: int
    audit_weight: float
    matched: bool


def _teacher_aliases(spec: str) -> set[str]:
    path = Path(spec)
    values = {spec, path.name, path.stem}
    if ":" in spec:
        values.add(spec.rsplit(":", 1)[-1])
    return {value.lower() for value in values if value}


def load_teacher_profiles(
    path: str | Path | None,
    teacher_specs: Sequence[str],
) -> list[TeacherProfile]:
    """Join command-line teacher specs to the interaction-clustering report."""

    rows: list[dict[str, object]] = []
    if path is not None and Path(path).is_file():
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("schema") != "kaggriculture.behavior-response-clustering.v1":
            raise ValueError(f"unsupported teacher clustering schema in {path}")
        rows = list(report.get("agents", []))
    lookup = {
        str(row["agent"]).lower(): row
        for row in rows
        if row.get("agent") is not None
    }
    next_cluster = 1 + max(
        (int(row.get("cluster", -1)) for row in rows), default=-1
    )
    profiles: list[TeacherProfile] = []
    for spec in teacher_specs:
        match = next(
            (lookup[alias] for alias in _teacher_aliases(spec) if alias in lookup),
            None,
        )
        if match is None:
            profiles.append(
                TeacherProfile(spec, Path(spec).stem, next_cluster, 1.0, False)
            )
            next_cluster += 1
        else:
            profiles.append(
                TeacherProfile(
                    spec=spec,
                    agent=str(match["agent"]),
                    cluster=int(match["cluster"]),
                    audit_weight=float(match.get("balanced_bc_weight", 1.0)),
                    matched=True,
                )
            )
    return profiles


def cluster_balanced_indices(
    cluster_ids: Sequence[int],
    rng: np.random.Generator,
    *,
    size: int | None = None,
) -> np.ndarray:
    """Sample equal numbers from each observed strategy cluster per epoch."""

    labels = np.asarray(cluster_ids, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("cluster_ids must be a non-empty one-dimensional sequence")
    target = len(labels) if size is None else int(size)
    if target <= 0:
        raise ValueError("size must be positive")
    clusters = np.unique(labels)
    pools = {cluster: np.flatnonzero(labels == cluster) for cluster in clusters}
    cursors = {cluster: len(pool) for cluster, pool in pools.items()}
    shuffled = {cluster: pool.copy() for cluster, pool in pools.items()}
    output: list[int] = []
    cluster_schedule = np.resize(clusters, target)
    rng.shuffle(cluster_schedule)
    for cluster in cluster_schedule:
        pool = shuffled[int(cluster)]
        cursor = cursors[int(cluster)]
        if cursor >= len(pool):
            pool = pools[int(cluster)].copy()
            rng.shuffle(pool)
            shuffled[int(cluster)] = pool
            cursor = 0
        output.append(int(pool[cursor]))
        cursors[int(cluster)] = cursor + 1
    return np.asarray(output, dtype=np.int64)


def robust_cluster_indices(
    cluster_ids: Sequence[int],
    outcomes: Sequence[float],
    rng: np.random.Generator,
    *,
    robust_cluster_ids: Sequence[int] | None = None,
    size: int | None = None,
    worst_fraction: float = 0.35,
    cvar_fraction: float = 0.25,
) -> np.ndarray:
    """Mix cluster balance with CVaR emphasis on the weakest matchups."""

    labels = np.asarray(cluster_ids, dtype=np.int64)
    robust_labels = np.asarray(
        cluster_ids if robust_cluster_ids is None else robust_cluster_ids,
        dtype=np.int64,
    )
    scores = np.asarray(outcomes, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or robust_labels.shape != labels.shape
        or not len(labels)
    ):
        raise ValueError("cluster_ids and outcomes must be aligned vectors")
    if not 0.0 <= worst_fraction <= 1.0 or not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("worst_fraction and cvar_fraction must be in range")
    target = len(labels) if size is None else int(size)
    if target <= 0:
        raise ValueError("size must be positive")
    robust_count = int(round(target * worst_fraction))
    balanced_count = target - robust_count
    output = (
        cluster_balanced_indices(labels, rng, size=balanced_count).tolist()
        if balanced_count
        else []
    )
    clusters = np.unique(robust_labels)
    cvar_values: list[float] = []
    pools: list[np.ndarray] = []
    for cluster in clusters:
        pool = np.flatnonzero(robust_labels == cluster)
        pools.append(pool)
        ordered = np.sort(scores[pool])
        tail_count = max(1, int(np.ceil(len(ordered) * cvar_fraction)))
        cvar_values.append(float(ordered[:tail_count].mean()))
    weakness = np.max(cvar_values) - np.asarray(cvar_values)
    probabilities = np.exp(weakness - weakness.max())
    probabilities /= probabilities.sum()
    for _ in range(robust_count):
        cluster_slot = int(rng.choice(len(clusters), p=probabilities))
        output.append(int(rng.choice(pools[cluster_slot])))
    rng.shuffle(output)
    return np.asarray(output, dtype=np.int64)


def cluster_sampling_report(
    teacher_names: Sequence[str], cluster_ids: Sequence[int]
) -> dict[str, object]:
    teacher_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    for teacher, cluster in zip(teacher_names, cluster_ids, strict=True):
        teacher_counts[teacher] = teacher_counts.get(teacher, 0) + 1
        key = str(int(cluster))
        cluster_counts[key] = cluster_counts.get(key, 0) + 1
    return {
        "schema": "kaggriculture.bc-cluster-sampling.v1",
        "teachers": dict(sorted(teacher_counts.items())),
        "clusters_before_balancing": dict(sorted(cluster_counts.items())),
    }
