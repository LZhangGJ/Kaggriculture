"""Small, auditable league/PSRO utilities for hierarchical PPO."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def league_mixture_from_payoff(
    report: Mapping[str, Any] | None,
    members: int,
) -> np.ndarray:
    """Blend compatible PSRO and worst-opponent mixtures, else use uniform."""

    if members <= 0:
        raise ValueError("members must be positive")
    candidates: list[np.ndarray] = []
    if report is not None:
        for key in ("psro_mixture", "robust_opponent_weights"):
            values = np.asarray(report.get(key, []), dtype=np.float64)
            if values.shape == (members,) and np.isfinite(values).all():
                values = np.maximum(values, 0.0)
                if values.sum() > 0:
                    candidates.append(values / values.sum())
    if not candidates:
        return np.full(members, 1.0 / members, dtype=np.float64)
    mixture = np.stack(candidates).mean(axis=0)
    return mixture / mixture.sum()


def sample_league_member(
    mixture: Sequence[float], rng: np.random.Generator
) -> int:
    probabilities = np.asarray(mixture, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("mixture must be a non-empty vector")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("mixture must contain finite non-negative weights")
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("mixture must have positive mass")
    return int(rng.choice(len(probabilities), p=probabilities / total))


def assemble_seat_actions(
    learner_actions: Sequence[Mapping[str, Any]],
    opponent_actions: Sequence[Mapping[str, Any]],
    learner_seat: int,
) -> list[list[Mapping[str, Any]]]:
    """Put independently decoded actor actions back into environment pairs."""

    if learner_seat not in (0, 1):
        raise ValueError("learner_seat must be zero or one")
    if len(learner_actions) != len(opponent_actions):
        raise ValueError("learner and opponent actions must align")
    return [
        [learner, opponent] if learner_seat == 0 else [opponent, learner]
        for learner, opponent in zip(
            learner_actions, opponent_actions, strict=True
        )
    ]
