"""Empirical maximin opening wrapper for the recurrent route selector."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from .recurrent_meta import NumpyRecurrentTreeQSelector, RecurrentTreeQSelector
from .selector import RouteDecision


class EquilibriumOpeningSelector(RecurrentTreeQSelector):
    """Choose a robust opening mixture, then delegate later decisions.

    The draw is a deterministic hash of the causal initial fingerprint and a
    user-provided salt.  It therefore varies across public initial states while
    remaining reproducible and requiring no hidden environment information.
    """

    def __init__(
        self, *args, mixture_path: str, mixture_salt: int | None = 0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        payload = json.loads(Path(mixture_path).read_text(encoding="utf-8"))
        support = list(payload.get("support", []))
        if not support:
            raise ValueError("opening mixture has empty support")
        total = sum(max(0.0, float(row["weight"])) for row in support)
        if total <= 0.0:
            raise ValueError("opening mixture weights must have positive mass")
        self.opening_support = [
            (str(row["route_id"]), max(0.0, float(row["weight"])) / total)
            for row in support
        ]
        missing = [route_id for route_id, _ in self.opening_support if route_id not in self.selector.route_offsets]
        if missing:
            raise ValueError(f"opening mixture routes missing from runtime: {missing}")
        self.fixed_mixture_salt = (
            None if mixture_salt is None else int(mixture_salt)
        )
        self.mixture_salt = (
            secrets.randbits(64)
            if self.fixed_mixture_salt is None else self.fixed_mixture_salt
        )
        self.opening_route_id: str | None = None

    def reset(self) -> None:
        super().reset()
        if getattr(self, "fixed_mixture_salt", 0) is None:
            self.mixture_salt = secrets.randbits(64)
        self.opening_route_id = None

    def _opening_draw(self, fingerprint: dict[str, Any]) -> float:
        payload = json.dumps(
            {"salt": self.mixture_salt, "fingerprint": fingerprint},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        return value / float(2**64)

    def _choose_opening(self, fingerprint: dict[str, Any]) -> str:
        draw = self._opening_draw(fingerprint)
        cumulative = 0.0
        for route_id, probability in self.opening_support:
            cumulative += probability
            if draw < cumulative:
                return route_id
        return self.opening_support[-1][0]

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        if int(checkpoint) != 0:
            return super().select(fingerprint, checkpoint)
        route_id = self._choose_opening(fingerprint)
        self.selector._load_actions(route_id)
        self.opening_route_id = route_id
        return RouteDecision(
            checkpoint=0,
            next_checkpoint=int(self.selector.checkpoints[1]),
            route_id=route_id,
            family_hash=f"equilibrium:{route_id}",
            changed_route=False,
            score=0.0,
            distance={},
            support=1,
            median_reward=0.0,
        )


class NumpyEquilibriumOpeningSelector(NumpyRecurrentTreeQSelector):
    """Submission equivalent of :class:`EquilibriumOpeningSelector`."""

    def __init__(
        self, *args, mixture_path: str, mixture_salt: int | None = 0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        payload = json.loads(Path(mixture_path).read_text(encoding="utf-8"))
        support = list(payload.get("support", []))
        total = sum(max(0.0, float(row["weight"])) for row in support)
        if not support or total <= 0.0:
            raise ValueError("opening mixture must contain positive support")
        self.opening_support = [
            (str(row["route_id"]), max(0.0, float(row["weight"])) / total)
            for row in support
        ]
        missing = [
            route_id for route_id, _ in self.opening_support
            if route_id not in self.selector.route_offsets
        ]
        if missing:
            raise ValueError(f"opening mixture routes missing from runtime: {missing}")
        self.fixed_mixture_salt = (
            None if mixture_salt is None else int(mixture_salt)
        )
        self.mixture_salt = (
            secrets.randbits(64)
            if self.fixed_mixture_salt is None else self.fixed_mixture_salt
        )
        self.opening_route_id: str | None = None

    def reset(self) -> None:
        super().reset()
        if getattr(self, "fixed_mixture_salt", 0) is None:
            self.mixture_salt = secrets.randbits(64)
        self.opening_route_id = None

    def _choose_opening(self, fingerprint: dict[str, Any]) -> str:
        payload = json.dumps(
            {"salt": self.mixture_salt, "fingerprint": fingerprint},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        draw = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        ) / float(2**64)
        cumulative = 0.0
        for route_id, probability in self.opening_support:
            cumulative += probability
            if draw < cumulative:
                return route_id
        return self.opening_support[-1][0]

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        if int(checkpoint) != 0:
            return super().select(fingerprint, checkpoint)
        route_id = self._choose_opening(fingerprint)
        self.selector._load_actions(route_id)
        self.opening_route_id = route_id
        return RouteDecision(
            checkpoint=0,
            next_checkpoint=int(self.selector.checkpoints[1]),
            route_id=route_id,
            family_hash=f"equilibrium:{route_id}",
            changed_route=False,
            score=0.0,
            distance={},
            support=1,
            median_reward=0.0,
        )
