"""Deterministic interaction-profile clustering for expert-agent audits.

The public expert pool is dominated by stateful JAX/Python agents that cannot
all be imported through the lightweight training environment.  A common-opponent
response profile is therefore the strongest already-available, apples-to-apples
behavioral evidence: for every agent and opponent it records score rate, cash,
and margin under the same swapped-seat protocol.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ResponseRecord:
    agent: str
    opponent: str
    score_rate: float
    mean_cash: float
    mean_margin: float
    games: int
    source: str


@dataclass(frozen=True)
class BehaviorClustering:
    agents: tuple[str, ...]
    opponents: tuple[str, ...]
    feature_names: tuple[str, ...]
    raw_features: np.ndarray
    normalized_features: np.ndarray
    clusters: np.ndarray
    cluster_count: int
    silhouette: float
    silhouette_by_k: tuple[tuple[int, float], ...]
    nearest_agents: tuple[str, ...]
    nearest_distances: np.ndarray
    balanced_weights: np.ndarray
    imputed_cells: np.ndarray


def load_round_robin_csv(path: str | Path) -> list[ResponseRecord]:
    """Load both orientations from the exact round-robin pair table."""

    source = str(Path(path).resolve())
    records: list[ResponseRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            games = int(row["games"])
            score = float(row["a_score_rate"])
            a_cash = float(row["a_mean_cash"])
            b_cash = float(row["b_mean_cash"])
            margin = float(row["a_mean_margin"])
            a = str(row["agent_a"])
            b = str(row["agent_b"])
            records.append(
                ResponseRecord(a, b, score, a_cash, margin, games, source)
            )
            records.append(
                ResponseRecord(b, a, 1.0 - score, b_cash, -margin, games, source)
            )
    return records


def load_panel_json(
    path: str | Path,
    *,
    agent_name: str | None = None,
) -> tuple[list[ResponseRecord], dict[str, Any]]:
    """Load a fusion-panel receipt and retain its provenance metadata."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    name = agent_name or str(payload["candidate"])
    records = [
        ResponseRecord(
            agent=name,
            opponent=str(row["opponent"]),
            score_rate=float(row["score_rate"]),
            mean_cash=float(row["mean_candidate_cash"]),
            mean_margin=float(row["mean_margin"]),
            games=int(row["games"]),
            source=str(source_path),
        )
        for row in payload["rows"]
    ]
    provenance = {
        "agent": name,
        "receipt_candidate": payload.get("candidate"),
        "status": payload.get("status"),
        "games_per_opponent": payload.get("games_per_opponent"),
        "seed_start": payload.get("seed_start"),
        "seed_count": payload.get("seed_count"),
        "source": str(source_path),
    }
    return records, provenance


def common_opponents(
    records: Sequence[ResponseRecord],
    agents: Sequence[str],
) -> tuple[str, ...]:
    """Return opponents observed for every non-self agent.

    Round-robin tables intentionally omit self matches, so an agent does not
    disqualify itself from being a common reference opponent.
    """

    by_agent: dict[str, set[str]] = {name: set() for name in agents}
    for record in records:
        if record.agent in by_agent and record.opponent != record.agent:
            by_agent[record.agent].add(record.opponent)
    universe = set().union(*(values for values in by_agent.values()))
    shared = {
        opponent
        for opponent in universe
        if all(agent == opponent or opponent in by_agent[agent] for agent in agents)
    }
    return tuple(sorted(shared))


def build_response_features(
    records: Sequence[ResponseRecord],
    agents: Sequence[str],
    opponents: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Build [score, cash, margin] features against common opponents.

    Self matches do not exist in a round robin.  They are left missing and then
    median-imputed per feature column; the returned mask makes this explicit in
    the report.
    """

    lookup = {(row.agent, row.opponent): row for row in records}
    width = 3 * len(opponents)
    values = np.full((len(agents), width), np.nan, dtype=np.float64)
    names: list[str] = []
    for opponent_index, opponent in enumerate(opponents):
        names.extend(
            (
                f"vs:{opponent}:score_rate",
                f"vs:{opponent}:mean_cash",
                f"vs:{opponent}:mean_margin",
            )
        )
        for agent_index, agent in enumerate(agents):
            row = lookup.get((agent, opponent))
            if row is None:
                continue
            offset = 3 * opponent_index
            values[agent_index, offset : offset + 3] = (
                row.score_rate,
                row.mean_cash,
                row.mean_margin,
            )
    missing = ~np.isfinite(values)
    for column in range(width):
        finite = values[np.isfinite(values[:, column]), column]
        if not len(finite):
            raise ValueError(f"feature {names[column]} has no observations")
        values[missing[:, column], column] = float(np.median(finite))
    return values, tuple(names), missing


def robust_standardize(values: np.ndarray) -> np.ndarray:
    """Median/IQR standardization, with constant columns safely zeroed."""

    median = np.median(values, axis=0)
    lower, upper = np.percentile(values, (25.0, 75.0), axis=0)
    scale = upper - lower
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-9, scale, standard)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - median) / scale


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def _farthest_first(values: np.ndarray, clusters: int) -> np.ndarray:
    norms = np.square(values).sum(axis=1)
    chosen = [int(np.argmax(norms))]
    nearest = np.square(values - values[chosen[0]]).sum(axis=1)
    while len(chosen) < clusters:
        candidate = int(np.argmax(nearest))
        if candidate in chosen:
            candidate = next(index for index in range(len(values)) if index not in chosen)
        chosen.append(candidate)
        nearest = np.minimum(
            nearest, np.square(values - values[candidate]).sum(axis=1)
        )
    return values[np.asarray(chosen)].copy()


def deterministic_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= clusters <= len(values):
        raise ValueError("clusters must be between one and the sample count")
    centers = _farthest_first(values, clusters)
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(iterations):
        distances = _squared_distances(values, centers)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    # Stable labels: strongest cluster (mean score dimensions) first.
    score_columns = np.arange(0, values.shape[1], 3)
    ordering = sorted(
        range(clusters),
        key=lambda cluster: (-float(centers[cluster, score_columns].mean()), cluster),
    )
    remap = np.empty(clusters, dtype=np.int64)
    for new, old in enumerate(ordering):
        remap[old] = new
    return remap[labels], centers[np.asarray(ordering)]


def silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(values):
        return -1.0
    distances = np.sqrt(
        np.maximum(
            ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )
    scores = np.zeros(len(values), dtype=np.float64)
    for index, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        own = own[own != index]
        if not len(own):
            scores[index] = 0.0
            continue
        a = float(distances[index, own].mean())
        b = min(
            float(distances[index, labels == other].mean())
            for other in unique
            if other != label
        )
        scores[index] = (b - a) / max(a, b, 1e-12)
    return float(scores.mean())


def cluster_behavior_profiles(
    agents: Sequence[str],
    opponents: Sequence[str],
    raw_features: np.ndarray,
    feature_names: Sequence[str],
    imputed_cells: np.ndarray | None = None,
    *,
    max_clusters: int = 8,
) -> BehaviorClustering:
    """Select K by silhouette and derive diversity-balanced BC weights."""

    if len(agents) != len(raw_features):
        raise ValueError("agent and feature row counts differ")
    normalized = robust_standardize(np.asarray(raw_features, dtype=np.float64))
    upper = min(max_clusters, len(agents) - 1)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for count in range(2, upper + 1):
        labels, _ = deterministic_kmeans(normalized, count)
        score = silhouette_score(normalized, labels)
        candidates.append((score, -count, labels))
    if candidates:
        score, neg_count, labels = max(candidates, key=lambda row: (row[0], row[1]))
        count = -neg_count
    else:
        score, count = -1.0, 1
        labels = np.zeros(len(agents), dtype=np.int64)

    pair_distances = np.sqrt(
        np.maximum(
            ((normalized[:, None, :] - normalized[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )
    np.fill_diagonal(pair_distances, np.inf)
    nearest_index = np.argmin(pair_distances, axis=1)
    nearest_distance = pair_distances[np.arange(len(agents)), nearest_index]
    cluster_sizes = np.bincount(labels, minlength=count)
    # Square-root tempering retains diversity pressure without letting a noisy
    # singleton dominate a minibatch.
    weights = 1.0 / np.sqrt(cluster_sizes[labels])
    weights /= weights.mean()
    missing = (
        np.zeros_like(raw_features, dtype=np.bool_)
        if imputed_cells is None
        else np.asarray(imputed_cells, dtype=np.bool_)
    )
    return BehaviorClustering(
        agents=tuple(agents),
        opponents=tuple(opponents),
        feature_names=tuple(feature_names),
        raw_features=np.asarray(raw_features, dtype=np.float64),
        normalized_features=normalized,
        clusters=labels,
        cluster_count=count,
        silhouette=score,
        silhouette_by_k=tuple(
            (int(-negative_count), float(candidate_score))
            for candidate_score, negative_count, _ in candidates
        ),
        nearest_agents=tuple(agents[index] for index in nearest_index),
        nearest_distances=nearest_distance,
        balanced_weights=weights,
        imputed_cells=missing,
    )


def clustering_report(
    result: BehaviorClustering,
    records: Sequence[ResponseRecord],
    provenance: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    lookup = {(row.agent, row.opponent): row for row in records}
    rows: list[dict[str, Any]] = []
    for index, agent in enumerate(result.agents):
        evidence = [
            lookup[(agent, opponent)]
            for opponent in result.opponents
            if (agent, opponent) in lookup
        ]
        rows.append(
            {
                "agent": agent,
                "cluster": int(result.clusters[index]),
                "nearest_agent": result.nearest_agents[index],
                "nearest_distance": float(result.nearest_distances[index]),
                "balanced_bc_weight": float(result.balanced_weights[index]),
                "common_opponents_observed": len(evidence),
                "imputed_feature_cells": int(result.imputed_cells[index].sum()),
                "games": int(sum(row.games for row in evidence)),
                "mean_score_rate": float(np.mean([row.score_rate for row in evidence])),
                "mean_cash": float(np.mean([row.mean_cash for row in evidence])),
                "mean_margin": float(np.mean([row.mean_margin for row in evidence])),
                "sources": sorted({row.source for row in evidence}),
            }
        )
    clusters = [
        {
            "cluster": cluster,
            "size": int(np.sum(result.clusters == cluster)),
            "agents": [
                agent
                for agent, label in zip(result.agents, result.clusters, strict=True)
                if int(label) == cluster
            ],
        }
        for cluster in range(result.cluster_count)
    ]
    return {
        "schema": "kaggriculture.behavior-response-clustering.v1",
        "method": {
            "profile": "common-opponent [score_rate, mean_cash, mean_margin]",
            "normalization": "per-feature median/IQR",
            "clustering": "deterministic farthest-first k-means",
            "k_selection": (
                "maximum silhouette over K=2.."
                f"{max((count for count, _ in result.silhouette_by_k), default=1)}; "
                "smaller K wins ties"
            ),
            "bc_weight": "inverse sqrt(cluster size), normalized to mean 1",
            "important_limit": (
                "This diagnoses interaction-response behavior, not primitive action "
                "or route identity. Action-level trajectories remain a separate audit."
            ),
        },
        "agent_count": len(result.agents),
        "common_opponents": list(result.opponents),
        "feature_count": len(result.feature_names),
        "cluster_count": result.cluster_count,
        "silhouette": result.silhouette,
        "silhouette_by_k": {
            str(count): score for count, score in result.silhouette_by_k
        },
        "k_at_search_boundary": bool(
            result.silhouette_by_k
            and result.cluster_count == max(count for count, _ in result.silhouette_by_k)
        ),
        "imputed_feature_cells": int(result.imputed_cells.sum()),
        "provenance": list(provenance),
        "clusters": clusters,
        "agents": rows,
    }


def write_clustering_outputs(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "behavior_clusters.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    agent_rows = list(report["agents"])
    with (output / "behavior_clusters.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [key for key in agent_rows[0] if key != "sources"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in agent_rows)
    lines = [
        "# 30-agent interaction-response clustering",
        "",
        f"Agents: **{report['agent_count']}**; common opponents: "
        f"**{len(report['common_opponents'])}**; selected clusters: "
        f"**{report['cluster_count']}**; silhouette: **{report['silhouette']:.4f}**.",
        "",
        "This is a common-opponent response audit, not proof that two agents use "
        "identical primitive routes. Missing self-match cells are median-imputed and "
        "counted explicitly.",
        "",
        "Silhouette by K: "
        + ", ".join(
            f"K={key}: {value:.4f}"
            for key, value in report["silhouette_by_k"].items()
        )
        + ".",
        (
            "The best K is the configured search ceiling, so eight clusters should "
            "be read as at least eight response regimes, not a proven natural K."
            if report["k_at_search_boundary"]
            else "The selected K is inside the configured search range."
        ),
        "",
        "## Clusters",
        "",
    ]
    for cluster in report["clusters"]:
        lines.append(
            f"- C{cluster['cluster']} ({cluster['size']}): "
            + ", ".join(cluster["agents"])
        )
    lines.extend(
        (
            "",
            "## Per-agent diagnostics",
            "",
            "| agent | cluster | nearest | distance | score | margin | BC weight | evidence games | imputed |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in agent_rows:
        lines.append(
            f"| {row['agent']} | C{row['cluster']} | {row['nearest_agent']} | "
            f"{row['nearest_distance']:.3f} | {row['mean_score_rate']:.3f} | "
            f"{row['mean_margin']:.1f} | {row['balanced_bc_weight']:.3f} | "
            f"{row['games']} | {row['imputed_feature_cells']} |"
        )
    (output / "behavior_clusters.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
