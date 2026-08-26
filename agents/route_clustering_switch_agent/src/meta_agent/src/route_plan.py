"""Executable plans anchored to a replay carrier.

The first route compiler deliberately keeps the carrier's mature micro tape and
describes only the state-aware detours needed by a mutated RouteGenome. This is
a bridge toward a fully generative planner: every compiled mutation is
explicit, auditable, and measurable independently of its source replay.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_HORIZON = 719


def normalized_action(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the stable Kaggriculture player-action shape."""

    raw = copy.deepcopy(dict(value or {}))
    return {
        "farmer": list(raw.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in raw.get("hands", []) or []],
        "market": [list(order) for order in raw.get("market", []) or []],
    }


def _position(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CarrierRoute:
    """One replay side plus the observations needed to compile its tape."""

    replay_path: Path
    player_index: int
    episode_id: int
    seed: int
    actions: tuple[Mapping[str, Any], ...]
    opponent_actions: tuple[Mapping[str, Any], ...]
    actor_positions: tuple[tuple[tuple[int, int] | None, ...], ...]
    configuration: Mapping[str, Any]
    info: Mapping[str, Any]
    rewards: tuple[float, float]

    @classmethod
    def from_replay(
        cls,
        replay_path: str | Path,
        player_index: int,
        *,
        horizon: int = DEFAULT_HORIZON,
    ) -> "CarrierRoute":
        path = Path(replay_path).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        player = int(player_index)
        if player not in (0, 1):
            raise ValueError("carrier player_index must be zero or one")
        steps = list(raw.get("steps", []) or [])
        if len(steps) < 2:
            raise ValueError(f"carrier replay has no turns: {path}")
        turns = min(int(horizon), len(steps) - 1)

        def tape(seat: int) -> tuple[Mapping[str, Any], ...]:
            values = [
                normalized_action(steps[turn + 1][seat].get("action") or {})
                for turn in range(turns)
            ]
            while len(values) < horizon:
                values.append(normalized_action(None))
            return tuple(values)

        positions = []
        for turn in range(turns):
            observation = steps[turn][player].get("observation") or {}
            farms = list(observation.get("farms", []) or [])
            farm = farms[player] if player < len(farms) else {}
            actors = [farm.get("farmer"), *(farm.get("hands", []) or [])]
            positions.append(tuple(_position(value) for value in actors))
        while len(positions) < horizon:
            positions.append(tuple())

        info = dict(raw.get("info") or {})
        rewards = list(raw.get("rewards", []) or [])
        if len(rewards) < 2:
            rewards = [
                float(steps[-1][seat].get("reward") or 0.0) for seat in (0, 1)
            ]
        return cls(
            replay_path=path,
            player_index=player,
            episode_id=int(info.get("EpisodeId") or raw.get("id") or 0),
            seed=int(info.get("seed") or 0),
            actions=tape(player),
            opponent_actions=tape(1 - player),
            actor_positions=tuple(positions),
            configuration=dict(raw.get("configuration") or {}),
            info=info,
            rewards=(float(rewards[0] or 0.0), float(rewards[1] or 0.0)),
        )

    def actor_position(self, step: int, actor_index: int) -> tuple[int, int] | None:
        if not 0 <= int(step) < len(self.actor_positions):
            return None
        actors = self.actor_positions[int(step)]
        return actors[int(actor_index)] if 0 <= int(actor_index) < len(actors) else None


@dataclass(frozen=True)
class ProductionTransform:
    """Transform one carrier production location into a new target."""

    source_xy: tuple[int, int]
    source_kind: str
    target_xy: tuple[int, int]
    target_kind: str


@dataclass(frozen=True)
class RouteTask:
    """One state-aware macro action scheduled on top of the carrier tape."""

    task_id: str
    due_step: int
    activation_step: int
    operation: str
    item: str | None
    target_xy: tuple[int, int]
    actor_index: int | None = None
    return_xy: tuple[int, int] | None = None
    source_step: int | None = None
    grace_steps: int = 36


@dataclass
class CompileReport:
    """Machine-readable compiler coverage and risk report."""

    parent_genome_id: str
    target_genome_id: str
    static_market_quantity_delta: dict[str, int] = field(default_factory=dict)
    prerequisite_market_delta: dict[str, int] = field(default_factory=dict)
    production_transforms: list[dict[str, Any]] = field(default_factory=list)
    event_shifts: list[dict[str, Any]] = field(default_factory=list)
    static_patches: int = 0
    tasks: int = 0
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.unsupported:
            return "partial"
        if self.warnings:
            return "compiled_with_warnings"
        return "compiled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_genome_id": self.parent_genome_id,
            "target_genome_id": self.target_genome_id,
            "status": self.status,
            "static_market_quantity_delta": dict(self.static_market_quantity_delta),
            "prerequisite_market_delta": dict(self.prerequisite_market_delta),
            "production_transforms": list(self.production_transforms),
            "event_shifts": list(self.event_shifts),
            "static_patches": int(self.static_patches),
            "tasks": int(self.tasks),
            "warnings": list(self.warnings),
            "unsupported": list(self.unsupported),
        }


@dataclass(frozen=True)
class CompiledRoutePlan:
    """Patched carrier tape and dynamic tasks emitted by the compiler."""

    carrier: CarrierRoute
    actions: tuple[Mapping[str, Any], ...]
    tasks: tuple[RouteTask, ...]
    report: CompileReport
