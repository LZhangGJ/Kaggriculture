"""Mutation-ready macro route genomes recovered from Kaggriculture replays.

The genome intentionally records goals and economic behavior rather than the
719-turn movement tape.  A later route compiler can therefore mutate the plan
without inheriting every movement or execution failure from the source replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
STRUCTURES = ("COOP", "PASTURE")
PRODUCTION_KINDS = (*CROPS, *ANIMALS, *STRUCTURES)
MACRO_KEYS = (
    *CROPS,
    "BUILD_COOP",
    "BUILD_PASTURE",
    *ANIMALS,
    "BUY_LAND",
    "HIRE",
)
DEFAULT_ANCHORS = (168, 288, 432, 576, 719)
DEFAULT_PHASE_WIDTH = 72
DEFAULT_HORIZON = 719
ROUTE_GENOME_SCHEMA_VERSION = 2


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _market_actions(market_profile: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return market decisions without replay-specific observed prices."""

    return [
        {
            "operation": str(value.get("operation", "")),
            "item": value.get("item"),
            "orders": int(value.get("orders", 0)),
            "quantity": int(value.get("quantity", 0)),
        }
        for value in market_profile
    ]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _position(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if 0 <= x < 10 and 0 <= y < 10 else None


def _own_farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = list(observation.get("farms", []) or [])
    return farms[player] if player < len(farms) and isinstance(farms[player], Mapping) else {}


def _team_names(replay: Mapping[str, Any]) -> list[str]:
    info = replay.get("info") or {}
    values = list(info.get("TeamNames", []) or [])
    if len(values) >= 2:
        return [str(values[0] or ""), str(values[1] or "")]
    agents = list(info.get("Agents", []) or [])
    names = [
        str(value.get("Name", "") if isinstance(value, Mapping) else "")
        for value in agents[:2]
    ]
    return (names + ["", ""])[:2]


@dataclass(frozen=True)
class RouteGenome:
    """Serializable route seed whose ID depends only on genetic content."""

    genome_id: str
    source: Mapping[str, Any]
    anchor_targets: tuple[Mapping[str, Any], ...]
    phase_macro_counts: tuple[Mapping[str, Any], ...]
    structural_events: tuple[Mapping[str, Any], ...]
    market_profile: tuple[Mapping[str, Any], ...]
    cash_profile: Mapping[str, Any]
    schema_version: int = ROUTE_GENOME_SCHEMA_VERSION

    def genetic_payload(self) -> dict[str, Any]:
        payload = {
            "anchor_targets": [dict(value) for value in self.anchor_targets],
            "phase_macro_counts": [dict(value) for value in self.phase_macro_counts],
            "structural_events": [dict(value) for value in self.structural_events],
        }
        if self.schema_version == 1:
            # Backward-compatible validation for the first extracted artifact.
            payload["market_profile"] = [dict(value) for value in self.market_profile]
        else:
            payload["market_actions"] = _market_actions(self.market_profile)
        return payload

    @staticmethod
    def content_id(
        payload: Mapping[str, Any],
        schema_version: int = ROUTE_GENOME_SCHEMA_VERSION,
    ) -> str:
        digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        return f"rg{schema_version}_{digest[:24]}"

    @classmethod
    def create(
        cls,
        *,
        source: Mapping[str, Any],
        anchor_targets: Sequence[Mapping[str, Any]],
        phase_macro_counts: Sequence[Mapping[str, Any]],
        structural_events: Sequence[Mapping[str, Any]],
        market_profile: Sequence[Mapping[str, Any]],
        cash_profile: Mapping[str, Any],
    ) -> "RouteGenome":
        profile = [dict(value) for value in market_profile]
        genetic = {
            "anchor_targets": [dict(value) for value in anchor_targets],
            "phase_macro_counts": [dict(value) for value in phase_macro_counts],
            "structural_events": [dict(value) for value in structural_events],
            "market_actions": _market_actions(profile),
        }
        value = cls(
            genome_id=cls.content_id(genetic, ROUTE_GENOME_SCHEMA_VERSION),
            source=dict(source),
            anchor_targets=tuple(genetic["anchor_targets"]),
            phase_macro_counts=tuple(genetic["phase_macro_counts"]),
            structural_events=tuple(genetic["structural_events"]),
            market_profile=tuple(profile),
            cash_profile=dict(cash_profile),
            schema_version=ROUTE_GENOME_SCHEMA_VERSION,
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.schema_version not in (1, ROUTE_GENOME_SCHEMA_VERSION):
            raise ValueError(f"unsupported RouteGenome schema: {self.schema_version}")
        if self.genome_id != self.content_id(self.genetic_payload(), self.schema_version):
            raise ValueError("genome_id does not match genetic content")
        source_id = str(self.source.get("source_id", ""))
        if not source_id or ":" not in source_id:
            raise ValueError("source.source_id must be '<episode_id>:<player_index>'")
        anchor_steps = [int(value["step"]) for value in self.anchor_targets]
        if anchor_steps != sorted(set(anchor_steps)):
            raise ValueError("anchor steps must be sorted and unique")
        for target in self.anchor_targets:
            seen: set[tuple[int, int]] = set()
            for placement in target.get("placements", []):
                kind = str(placement.get("kind", ""))
                xy = (int(placement.get("x", -1)), int(placement.get("y", -1)))
                if kind not in PRODUCTION_KINDS or not (0 <= xy[0] < 10 and 0 <= xy[1] < 10):
                    raise ValueError(f"invalid production placement: {placement!r}")
                if xy in seen:
                    raise ValueError(f"duplicate production tile at anchor: {xy}")
                seen.add(xy)
        for phase in self.phase_macro_counts:
            counts = phase.get("counts", {})
            if any(int(value) < 0 for value in counts.values()):
                raise ValueError("macro counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "genome_id": self.genome_id,
            "source": dict(self.source),
            "anchor_targets": [dict(value) for value in self.anchor_targets],
            "phase_macro_counts": [dict(value) for value in self.phase_macro_counts],
            "structural_events": [dict(value) for value in self.structural_events],
            "market_profile": [dict(value) for value in self.market_profile],
            "cash_profile": dict(self.cash_profile),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteGenome":
        value = cls(
            schema_version=int(payload.get("schema_version", 1)),
            genome_id=str(payload["genome_id"]),
            source=dict(payload["source"]),
            anchor_targets=tuple(dict(row) for row in payload["anchor_targets"]),
            phase_macro_counts=tuple(dict(row) for row in payload["phase_macro_counts"]),
            structural_events=tuple(dict(row) for row in payload["structural_events"]),
            market_profile=tuple(dict(row) for row in payload["market_profile"]),
            cash_profile=dict(payload["cash_profile"]),
        )
        value.validate()
        return value


def extract_route_genome(
    replay: Mapping[str, Any],
    player: int,
    *,
    provenance: Mapping[str, Any] | None = None,
    anchor_steps: Sequence[int] = DEFAULT_ANCHORS,
    phase_width: int = DEFAULT_PHASE_WIDTH,
    horizon: int = DEFAULT_HORIZON,
) -> RouteGenome:
    """Extract one player's macro genome from a replay.

    Actions are read from ``steps[t + 1]`` against the state in ``steps[t]``.
    Intended production is updated even if the official interpreter rejected
    the action, matching the route-clustering identity used by this project.
    """

    if player not in (0, 1):
        raise ValueError("Kaggriculture RouteGenome expects player 0 or 1")
    if phase_width <= 0 or horizon <= 0:
        raise ValueError("phase_width and horizon must be positive")
    anchors = tuple(sorted({int(value) for value in anchor_steps if int(value) > 0}))
    steps = list(replay.get("steps", []) or [])
    if len(steps) < 2:
        raise ValueError("replay has no executable turns")
    if any(player >= len(list(step or [])) for step in steps[:2]):
        raise ValueError(f"replay is missing player {player}")

    action_turns = min(horizon, len(steps) - 1)
    phase_count = int(math.ceil(horizon / phase_width))
    phase_rows = [Counter() for _ in range(phase_count)]
    intended: dict[tuple[int, int], str] = {}
    anchor_targets: list[dict[str, Any]] = []
    structural_events: list[dict[str, Any]] = []
    market_rows: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"orders": 0, "quantity": 0, "prices": []}
    )
    cash_values: list[float] = []

    for recorded_step in range(1, action_turns + 1):
        turn = recorded_step - 1
        previous = steps[recorded_step - 1][player]
        current = steps[recorded_step][player]
        observation = previous.get("observation") or {}
        action = current.get("action") or {}
        farm = _own_farm(observation, player)
        cash_values.append(float(farm.get("money", 0.0) or 0.0))
        actors = [farm.get("farmer"), *(farm.get("hands", []) or [])]
        orders = [action.get("farmer"), *(action.get("hands", []) or [])]
        phase = min(phase_count - 1, turn // phase_width)

        for actor_index, raw_order in enumerate(orders):
            order = list(raw_order or [])
            if not order:
                continue
            operation = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            xy = _position(actors[actor_index]) if actor_index < len(actors) else None
            macro_key = ""
            production_kind = ""
            if operation == "PLANT" and item in CROPS:
                macro_key = production_kind = item
            elif operation == "BUILD_COOP":
                macro_key, production_kind = "BUILD_COOP", "COOP"
            elif operation == "BUILD_PASTURE":
                macro_key, production_kind = "BUILD_PASTURE", "PASTURE"
            elif operation == "PLACE" and item in ANIMALS:
                macro_key = production_kind = item
            if macro_key:
                phase_rows[phase][macro_key] += 1
                if xy is not None:
                    intended[xy] = production_kind
                if operation != "PLANT":
                    event = {
                        "step": turn,
                        "operation": operation,
                        "item": item or None,
                    }
                    if xy is not None:
                        event["x"], event["y"] = xy
                    structural_events.append(event)

        prices = dict((observation.get("market") or {}).get("prices") or {})
        for raw_order in action.get("market", []) or []:
            order = list(raw_order or [])
            if not order:
                continue
            operation = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            try:
                quantity = max(0, int(order[2])) if len(order) >= 3 else 1
            except (TypeError, ValueError):
                quantity = 0
            row = market_rows[(operation, item)]
            row["orders"] += 1
            row["quantity"] += quantity
            if item in prices:
                row["prices"].append(float(prices[item] or 0.0))
            if operation in ("BUY_LAND", "HIRE"):
                phase_rows[phase][operation] += 1
                structural_events.append({
                    "step": turn,
                    "operation": operation,
                    "item": None,
                })

        if recorded_step in anchors:
            counts = Counter(intended.values())
            placements = [
                {"x": x, "y": y, "kind": kind}
                for (x, y), kind in sorted(
                    intended.items(), key=lambda value: (value[0][1], value[0][0])
                )
            ]
            anchor_targets.append({
                "step": recorded_step,
                "counts": {kind: int(counts.get(kind, 0)) for kind in PRODUCTION_KINDS},
                "placements": placements,
            })

    terminal_observation = steps[min(action_turns, len(steps) - 1)][player].get("observation") or {}
    terminal_farm = _own_farm(terminal_observation, player)
    cash_values.append(float(terminal_farm.get("money", 0.0) or 0.0))

    phases = [
        {
            "phase": index,
            "start_step": index * phase_width,
            "end_step": min(horizon, (index + 1) * phase_width) - 1,
            "counts": {key: int(row.get(key, 0)) for key in MACRO_KEYS},
        }
        for index, row in enumerate(phase_rows)
    ]
    market_profile = []
    for (operation, item), row in sorted(market_rows.items()):
        values = list(row["prices"])
        entry: dict[str, Any] = {
            "operation": operation,
            "item": item or None,
            "orders": int(row["orders"]),
            "quantity": int(row["quantity"]),
        }
        if values:
            entry["observed_price"] = {
                "min": min(values),
                "p25": _percentile(values, 0.25),
                "median": float(statistics.median(values)),
                "p75": _percentile(values, 0.75),
                "max": max(values),
            }
        market_profile.append(entry)

    info = replay.get("info") or {}
    episode_id = int(
        info.get("EpisodeId")
        or replay.get("id")
        or (provenance or {}).get("episode_id")
        or 0
    )
    rewards = list(replay.get("rewards", []) or [])
    if len(rewards) < 2:
        rewards = [
            steps[-1][index].get("reward") if index < len(steps[-1]) else 0.0
            for index in range(2)
        ]
    own_reward = float(rewards[player] or 0.0)
    opponent_reward = float(rewards[1 - player] or 0.0)
    teams = _team_names(replay)
    source = {
        "source_id": f"{episode_id}:{player}",
        "episode_id": episode_id,
        "player_index": player,
        "team_name": teams[player],
        "opponent_team_name": teams[1 - player],
        "final_reward": own_reward,
        "opponent_reward": opponent_reward,
        "result": "win" if own_reward > opponent_reward else "loss" if own_reward < opponent_reward else "tie",
        "replay_steps": len(steps),
        **dict(provenance or {}),
    }
    source["source_id"] = f"{episode_id}:{player}"
    source["episode_id"] = episode_id
    source["player_index"] = player

    cash_profile = {
        "initial": cash_values[0] if cash_values else 0.0,
        "minimum": min(cash_values, default=0.0),
        "p10": _percentile(cash_values, 0.10),
        "median": float(statistics.median(cash_values)) if cash_values else 0.0,
        "final": cash_values[-1] if cash_values else 0.0,
        "negative_steps": sum(value < 0 for value in cash_values),
    }
    return RouteGenome.create(
        source=source,
        anchor_targets=anchor_targets,
        phase_macro_counts=phases,
        structural_events=structural_events,
        market_profile=market_profile,
        cash_profile=cash_profile,
    )
