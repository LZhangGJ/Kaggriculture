"""Pairwise meta-game diagnostics, robust sampling, and PSRO mixtures."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MatchOutcome:
    policy: str
    opponent: str
    policy_cluster: int
    opponent_cluster: int
    outcome: float
    margin: float = 0.0
    games: int = 1


@dataclass(frozen=True)
class PayoffMatrix:
    labels: tuple[int, ...]
    payoff: np.ndarray
    games: np.ndarray
    observed: np.ndarray


def build_payoff_matrix(records: Sequence[MatchOutcome]) -> PayoffMatrix:
    if not records:
        raise ValueError("at least one match outcome is required")
    labels = tuple(
        sorted(
            {
                int(cluster)
                for row in records
                for cluster in (row.policy_cluster, row.opponent_cluster)
            }
        )
    )
    lookup = {cluster: index for index, cluster in enumerate(labels)}
    payoff_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    games = np.zeros_like(payoff_sum, dtype=np.int64)
    for row in records:
        left = lookup[int(row.policy_cluster)]
        right = lookup[int(row.opponent_cluster)]
        count = max(int(row.games), 1)
        payoff_sum[left, right] += float(row.outcome) * count
        games[left, right] += count
    observed = games > 0
    payoff = np.zeros_like(payoff_sum)
    payoff[observed] = payoff_sum[observed] / games[observed]
    # A swapped-seat observation is exact evidence for the missing reverse cell.
    for left in range(len(labels)):
        for right in range(len(labels)):
            if left == right:
                observed[left, right] = True
                continue
            if not observed[left, right] and observed[right, left]:
                payoff[left, right] = -payoff[right, left]
    return PayoffMatrix(labels, payoff, games, observed)


def regret_matching_mixture(
    payoff: np.ndarray,
    *,
    iterations: int = 10_000,
) -> np.ndarray:
    """Approximate a zero-sum meta-strategy with two-sided regret matching."""

    values = np.asarray(payoff, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values):
        raise ValueError("payoff must be a non-empty square matrix")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    count = values.shape[0]
    row_regret = np.zeros(count, dtype=np.float64)
    column_regret = np.zeros(count, dtype=np.float64)
    row_average = np.zeros(count, dtype=np.float64)
    column_average = np.zeros(count, dtype=np.float64)

    def strategy(regret: np.ndarray) -> np.ndarray:
        positive = np.maximum(regret, 0.0)
        total = float(positive.sum())
        return positive / total if total > 1e-12 else np.full(count, 1.0 / count)

    for _ in range(iterations):
        row = strategy(row_regret)
        column = strategy(column_regret)
        row_average += row
        column_average += column
        pure_row = values @ column
        row_value = float(row @ pure_row)
        pure_column = -(row @ values)
        column_value = float(column @ pure_column)
        row_regret += pure_row - row_value
        column_regret += pure_column - column_value
    mixture = 0.5 * (row_average + column_average) / iterations
    mixture = np.maximum(mixture, 0.0)
    return mixture / max(float(mixture.sum()), 1e-12)


def robust_opponent_weights(
    payoff: np.ndarray,
    learner_mixture: Sequence[float],
    *,
    worst_fraction: float = 0.35,
    temperature: float = 0.20,
) -> np.ndarray:
    """Blend PSRO coverage with emphasis on low-payoff opponent policies."""

    matrix = np.asarray(payoff, dtype=np.float64)
    learner = np.asarray(learner_mixture, dtype=np.float64)
    if matrix.shape != (len(learner), len(learner)):
        raise ValueError("learner mixture must align with the payoff matrix")
    if not 0.0 <= worst_fraction <= 1.0 or temperature <= 0.0:
        raise ValueError("invalid robust opponent sampling parameters")
    learner = np.maximum(learner, 0.0)
    learner /= max(float(learner.sum()), 1e-12)
    opponent_payoffs = learner @ matrix
    logits = -(opponent_payoffs - opponent_payoffs.max()) / temperature
    hard = np.exp(logits)
    hard /= max(float(hard.sum()), 1e-12)
    return (1.0 - worst_fraction) * learner + worst_fraction * hard


def payoff_records_from_official_manifest(
    manifest_path: str | Path,
) -> list[MatchOutcome]:
    """Read one outcome per selected replay without loading raw observations."""

    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records: list[MatchOutcome] = []
    for episode in list(manifest.get("episodes", []) or []):
        label_path = path.parent / str(episode["label_file"])
        with gzip.open(label_path, "rt", encoding="utf-8") as stream:
            bundle = json.load(stream)
        actors = list(bundle.get("actors", []) or [])
        if len(actors) != 2:
            continue
        for player in range(2):
            own = actors[player]
            opponent = actors[1 - player]
            records.append(
                MatchOutcome(
                    policy=str(own["teacher"]),
                    opponent=str(opponent["teacher"]),
                    policy_cluster=int(own["teacher_cluster"]),
                    opponent_cluster=int(opponent["teacher_cluster"]),
                    outcome=float(own.get("outcome", 0.0)),
                    margin=float(own.get("reward", 0.0))
                    - float(opponent.get("reward", 0.0)),
                )
            )
    if not records:
        raise ValueError(f"no match outcomes found in {path}")
    return records


def payoff_matrix_report(matrix: PayoffMatrix) -> dict[str, Any]:
    mixture = regret_matching_mixture(matrix.payoff)
    robust = robust_opponent_weights(matrix.payoff, mixture)
    expected = matrix.payoff @ robust
    return {
        "schema": "kaggriculture.cluster-payoff-matrix.v1",
        "cluster_labels": list(matrix.labels),
        "payoff": matrix.payoff.tolist(),
        "games": matrix.games.tolist(),
        "observed": matrix.observed.tolist(),
        "coverage": float(matrix.observed.mean()),
        "psro_mixture": mixture.tolist(),
        "robust_opponent_weights": robust.tolist(),
        "mixture_value_by_policy": expected.tolist(),
        "worst_cluster_value": float(expected.min()),
    }


def write_payoff_report(report: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
