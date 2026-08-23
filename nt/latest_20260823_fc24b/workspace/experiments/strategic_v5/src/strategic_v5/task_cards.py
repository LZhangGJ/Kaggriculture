"""Fixed-shape runtime program produced by the offline replay-card compiler."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS


MARKET_CARD_SLOTS_V1 = 21
TASK_CARD_SCHEMA_V1 = "replay_task_cards_v1"
TASK_CARD_PROFILE_ENV_V1 = "KAGGRI_TASK_CARD_PROFILE"


class ReplayTaskCardProgramV1(NamedTuple):
    """Dense JAX program; every tensor shape stays constant across profiles."""

    enabled: jax.Array
    support: jax.Array
    consensus: jax.Array
    market_quantity: jax.Array
    market_priority: jax.Array
    build_animal_id: jax.Array
    build_priority: jax.Array


def empty_replay_task_card_program_v1() -> ReplayTaskCardProgramV1:
    return ReplayTaskCardProgramV1(
        enabled=jnp.zeros((EPISODE_STEPS,), dtype=jnp.bool_),
        support=jnp.zeros((EPISODE_STEPS,), dtype=jnp.int16),
        consensus=jnp.zeros((EPISODE_STEPS,), dtype=jnp.float32),
        market_quantity=jnp.zeros(
            (EPISODE_STEPS, MARKET_CARD_SLOTS_V1), dtype=jnp.int16
        ),
        market_priority=jnp.zeros(
            (EPISODE_STEPS, MARKET_CARD_SLOTS_V1), dtype=jnp.float32
        ),
        build_animal_id=jnp.full((EPISODE_STEPS,), -1, dtype=jnp.int8),
        build_priority=jnp.zeros((EPISODE_STEPS,), dtype=jnp.float32),
    )


def _runtime_payload(document: dict) -> dict:
    if document.get("schema_version") != TASK_CARD_SCHEMA_V1:
        raise ValueError(
            f"unsupported task-card schema: {document.get('schema_version')!r}"
        )
    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("task-card profile has no runtime section")
    return runtime


def load_replay_task_card_program_v1(
    path: str | os.PathLike[str],
) -> ReplayTaskCardProgramV1:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime = _runtime_payload(document)
    program = ReplayTaskCardProgramV1(
        enabled=jnp.asarray(runtime["enabled"], dtype=jnp.bool_),
        support=jnp.asarray(runtime["support"], dtype=jnp.int16),
        consensus=jnp.asarray(runtime["consensus"], dtype=jnp.float32),
        market_quantity=jnp.asarray(runtime["market_quantity"], dtype=jnp.int16),
        market_priority=jnp.asarray(runtime["market_priority"], dtype=jnp.float32),
        build_animal_id=jnp.asarray(runtime["build_animal_id"], dtype=jnp.int8),
        build_priority=jnp.asarray(runtime["build_priority"], dtype=jnp.float32),
    )
    expected = empty_replay_task_card_program_v1()
    for actual, wanted in zip(program, expected, strict=True):
        if actual.shape != wanted.shape:
            raise ValueError(
                f"invalid task-card tensor shape {actual.shape}; expected {wanted.shape}"
            )
    return program


def default_replay_task_card_profile_path_v1() -> Path | None:
    configured = os.environ.get(TASK_CARD_PROFILE_ENV_V1)
    if configured:
        return Path(configured).expanduser().resolve()
    project = Path(__file__).resolve().parents[2]
    active = project / "artifacts" / "task_cards" / "active_profile.json"
    return active if active.exists() else None


def _load_default_program() -> ReplayTaskCardProgramV1:
    path = default_replay_task_card_profile_path_v1()
    if path is None:
        return empty_replay_task_card_program_v1()
    return load_replay_task_card_program_v1(path)


DEFAULT_REPLAY_TASK_CARD_PROGRAM_V1 = _load_default_program()
