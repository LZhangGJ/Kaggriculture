"""Checkpoint-only causal route selector with family priors and hysteresis."""

from __future__ import annotations

import math
import pickle
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from pathlib import Path

from .fingerprints import fingerprint_distance


@dataclass(frozen=True, slots=True)
class RouteDecision:
    checkpoint: int
    next_checkpoint: int
    route_id: str
    family_hash: str
    changed_route: bool
    score: float
    distance: dict[str, float]
    support: int
    median_reward: float


def _next_checkpoint(checkpoints: list[int], checkpoint: int) -> int:
    return next((value for value in checkpoints if value > checkpoint), checkpoint)


def _family_rows(
    records: Iterable[dict[str, Any]], next_checkpoint: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = str(next_checkpoint)
    for record in records:
        groups[str(record["semantic_prefix_hashes"][key])].append(record)
    return groups


def rank_route_families(
    fingerprint: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    checkpoint: int,
    checkpoints: list[int],
    quality_weight: float = 0.10,
    support_weight: float = 0.005,
    land_weight: float = 4.0,
) -> list[dict[str, Any]]:
    """Rank next-prefix families using only current state and library priors."""
    next_checkpoint = _next_checkpoint(checkpoints, checkpoint)
    reference_key = str(checkpoint)
    ranked = []
    for family_hash, rows in _family_rows(records, next_checkpoint).items():
        distances = [
            (
                fingerprint_distance(
                    fingerprint,
                    row["fingerprints"][reference_key],
                    land_weight=land_weight,
                ),
                row,
            )
            for row in rows
        ]
        distances.sort(key=lambda item: (item[0]["total"], item[1]["route_id"]))
        best_distance, representative = distances[0]
        median_reward = statistics.median(float(row["final_reward"]) for row in rows)
        normalized_quality = max(0.0, min(1.0, median_reward / 150_000.0))
        score = (
            best_distance["total"]
            - quality_weight * normalized_quality
            - support_weight * math.log1p(len(rows))
        )
        ranked.append(
            {
                "family_hash": family_hash,
                "records": rows,
                "representative": representative,
                "distance": best_distance,
                "score": score,
                "support": len(rows),
                "median_reward": median_reward,
                "next_checkpoint": next_checkpoint,
            }
        )
    return sorted(ranked, key=lambda row: (row["score"], -row["support"], row["family_hash"]))


class CausalRouteSelector:
    """Keep one route between checkpoints and switch only across shared prefixes."""

    def __init__(
        self,
        library: dict[str, Any],
        *,
        switch_margin: float = 0.015,
        quality_weight: float = 0.10,
        support_weight: float = 0.005,
        land_weight: float = 4.0,
    ) -> None:
        self.records = list(library.get("records", []))
        self.checkpoints = sorted(int(value) for value in library.get("checkpoints", []))
        self.switch_margin = float(switch_margin)
        self.quality_weight = float(quality_weight)
        self.support_weight = float(support_weight)
        self.land_weight = float(land_weight)
        self.current: dict[str, Any] | None = None

    def reset(self) -> None:
        self.current = None

    def action(self, step: int) -> dict[str, Any]:
        if self.current is None:
            raise RuntimeError("select() must choose a route before action().")
        actions = self.current["actions"]
        if not 0 <= step < len(actions):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return actions[step]

    def select(self, fingerprint: dict[str, Any], checkpoint: int) -> RouteDecision:
        if checkpoint not in self.checkpoints:
            raise ValueError(f"{checkpoint} is not a configured route checkpoint.")
        candidates = self.records
        if self.current is not None:
            prefix = self.current["prefix_hashes"][str(checkpoint)]
            candidates = [
                row for row in candidates if row["prefix_hashes"][str(checkpoint)] == prefix
            ]
        if not candidates:
            raise RuntimeError("No route shares the executed prefix at this checkpoint.")
        ranked = rank_route_families(
            fingerprint,
            candidates,
            checkpoint=checkpoint,
            checkpoints=self.checkpoints,
            quality_weight=self.quality_weight,
            support_weight=self.support_weight,
            land_weight=self.land_weight,
        )
        best = ranked[0]
        if self.current is not None:
            next_key = str(best["next_checkpoint"])
            current_family = self.current["semantic_prefix_hashes"][next_key]
            incumbent = next(
                (row for row in ranked if row["family_hash"] == current_family), None
            )
            if incumbent is not None and incumbent["score"] <= best["score"] + self.switch_margin:
                best = incumbent
        previous_id = self.current["route_id"] if self.current is not None else None
        # Preserve the incumbent representative when its family remains selected.
        family_ids = {row["route_id"] for row in best["records"]}
        if self.current is None or self.current["route_id"] not in family_ids:
            self.current = best["representative"]
        assert self.current is not None
        return RouteDecision(
            checkpoint=checkpoint,
            next_checkpoint=int(best["next_checkpoint"]),
            route_id=str(self.current["route_id"]),
            family_hash=str(best["family_hash"]),
            changed_route=previous_id is not None and previous_id != self.current["route_id"],
            score=float(best["score"]),
            distance=dict(best["distance"]),
            support=int(best["support"]),
            median_reward=float(best["median_reward"]),
        )


class ShardedCausalRouteSelector:
    """Lazy checkpoint selector for the complete submission route library."""

    def __init__(
        self,
        root: str | Path,
        *,
        switch_margin: float = 0.015,
        quality_weight: float = 0.10,
        support_weight: float = 0.005,
        land_weight: float = 4.0,
    ) -> None:
        self.root = Path(root)
        with (self.root / "manifest.pkl").open("rb") as handle:
            manifest = pickle.load(handle)
        self.checkpoints = sorted(int(value) for value in manifest["checkpoints"])
        self.route_offsets = {
            str(route_id): (int(value[0]), int(value[1]))
            for route_id, value in manifest["route_offsets"].items()
        }
        self.switch_margin = float(switch_margin)
        self.quality_weight = float(quality_weight)
        self.support_weight = float(support_weight)
        self.land_weight = float(land_weight)
        self.current_route_id: str | None = None
        self.current_actions: list[dict[str, Any]] | None = None

    @property
    def current(self) -> dict[str, Any] | None:
        if self.current_route_id is None:
            return None
        return {"route_id": self.current_route_id}

    @current.setter
    def current(self, value: None) -> None:
        if value is not None:
            raise TypeError("Sharded selector current may only be cleared externally")
        self.reset()

    def reset(self) -> None:
        self.current_route_id = None
        self.current_actions = None

    def action(self, step: int) -> dict[str, Any]:
        if self.current_actions is None:
            raise RuntimeError("select() must choose a route before action().")
        if not 0 <= step < len(self.current_actions):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return self.current_actions[step]

    def _rows(self, checkpoint: int) -> list[tuple[Any, ...]]:
        with (self.root / f"checkpoint-{checkpoint:03d}.pkl").open("rb") as handle:
            return pickle.load(handle)

    def _load_actions(self, route_id: str) -> None:
        offset, _ = self.route_offsets[route_id]
        with (self.root / "actions.bin").open("rb") as handle:
            handle.seek(offset)
            self.current_actions = pickle.load(handle)
        self.current_route_id = route_id

    def select(self, fingerprint: dict[str, Any], checkpoint: int) -> RouteDecision:
        if checkpoint not in self.checkpoints:
            raise ValueError(f"{checkpoint} is not a configured route checkpoint.")
        rows = self._rows(checkpoint)
        by_id = {str(row[0]): row for row in rows}
        if self.current_route_id is not None:
            incumbent_row = by_id[self.current_route_id]
            rows = [row for row in rows if row[2] == incumbent_row[2]]
        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            groups[str(row[3])].append(row)
        ranked = []
        next_checkpoint = _next_checkpoint(self.checkpoints, checkpoint)
        for family_hash, family_rows in groups.items():
            distances = [
                (
                    fingerprint_distance(fingerprint, row[4], land_weight=self.land_weight),
                    row,
                )
                for row in family_rows
            ]
            distances.sort(key=lambda value: (value[0]["total"], str(value[1][0])))
            distance, representative = distances[0]
            median_reward = statistics.median(float(row[1]) for row in family_rows)
            normalized_quality = max(0.0, min(1.0, median_reward / 150_000.0))
            score = (
                distance["total"]
                - self.quality_weight * normalized_quality
                - self.support_weight * math.log1p(len(family_rows))
            )
            ranked.append(
                {
                    "family_hash": family_hash,
                    "records": family_rows,
                    "representative": representative,
                    "distance": distance,
                    "score": score,
                    "support": len(family_rows),
                    "median_reward": median_reward,
                }
            )
        ranked.sort(key=lambda row: (row["score"], -row["support"], row["family_hash"]))
        best = ranked[0]
        if self.current_route_id is not None:
            incumbent_family = str(by_id[self.current_route_id][3])
            incumbent = next(
                (row for row in ranked if row["family_hash"] == incumbent_family),
                None,
            )
            if incumbent is not None and incumbent["score"] <= best["score"] + self.switch_margin:
                best = incumbent
        previous_id = self.current_route_id
        family_ids = {str(row[0]) for row in best["records"]}
        if self.current_route_id is None or self.current_route_id not in family_ids:
            self._load_actions(str(best["representative"][0]))
        assert self.current_route_id is not None
        return RouteDecision(
            checkpoint=checkpoint,
            next_checkpoint=next_checkpoint,
            route_id=self.current_route_id,
            family_hash=str(best["family_hash"]),
            changed_route=previous_id is not None and previous_id != self.current_route_id,
            score=float(best["score"]),
            distance=dict(best["distance"]),
            support=int(best["support"]),
            median_reward=float(best["median_reward"]),
        )
