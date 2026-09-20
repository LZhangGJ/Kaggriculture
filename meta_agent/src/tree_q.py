"""Shared state-branch value features and an exploratory sharded-tree selector."""

from __future__ import annotations

import math
import pickle
import random
import statistics
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .fingerprints import fingerprint_distance
from .selector import RouteDecision, _next_checkpoint


FINGERPRINT_DIM = 128
BRANCH_DIM = 128
DISTANCE_KEYS = (
    "total", "own", "town", "opponent", "market", "land",
    "land_phase", "land_layout", "land_work",
)
SCALAR_DIM = len(DISTANCE_KEYS) + 7
FEATURE_DIM = 3 * FINGERPRINT_DIM + BRANCH_DIM + SCALAR_DIM

_MANIFEST_CACHE: dict[str, dict[str, Any]] = {}
_DESCRIPTOR_CACHE: dict[str, dict[int, dict[str, np.ndarray]]] = {}
_MODEL_CACHE: dict[tuple[str, int], dict[str, Any] | None] = {}
_ROWS_CACHE: dict[tuple[str, int], list[tuple[Any, ...]]] = {}
_ACTIONS_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _hash_add(vector: np.ndarray, key: str, value: float) -> None:
    payload = key.encode("utf-8")
    code = zlib.crc32(payload) & 0xFFFFFFFF
    index = code % len(vector)
    sign = -1.0 if code & 0x80000000 else 1.0
    vector[index] += sign * float(value)


def fingerprint_vector(fingerprint: dict[str, Any]) -> np.ndarray:
    """Deterministically hash the sparse causal fingerprint into a dense vector."""
    result = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    _hash_add(result, "step", float(fingerprint.get("step", 0) or 0) / 719.0)
    for key, value in dict(fingerprint.get("numeric", {}) or {}).items():
        _hash_add(result, f"numeric:{key}", float(value or 0.0))
    for name, values in dict(fingerprint.get("lands", {}) or {}).items():
        for key, value in dict(values or {}).items():
            _hash_add(result, f"land:{name}:{key}", float(value or 0.0))
    for index, shop in enumerate(fingerprint.get("shops", ()) or ()):
        _hash_add(result, f"shop:{index}:{shop}", 1.0)
        _hash_add(result, f"shop-count:{shop}", 1.0)
    return np.tanh(result).astype(np.float32, copy=False)


def _action_tokens(action: dict[str, Any], relative_step: int) -> Iterable[tuple[str, float]]:
    bucket = min(3, max(0, relative_step) // 6)
    rows = [("farmer", action.get("farmer", ["PASS"]) or ["PASS"])]
    rows.extend(("hand", row or ["PASS"]) for row in (action.get("hands", []) or []))
    rows.extend(("market", row or ["NONE"]) for row in (action.get("market", []) or []))
    for actor, row in rows:
        op = str(row[0] if row else "PASS")
        item = str(row[1]) if len(row) >= 2 else ""
        try:
            quantity = max(0.0, float(row[2])) if len(row) >= 3 else 1.0
        except (TypeError, ValueError):
            quantity = 1.0
        yield f"{actor}:op:{op}", 1.0
        yield f"{actor}:item:{op}:{item}", 1.0
        yield f"bucket:{bucket}:{actor}:{op}:{item}", 1.0
        yield f"qty:{actor}:{op}:{item}", math.log1p(quantity) / math.log(1001.0)


def branch_descriptor(actions: list[dict[str, Any]], start: int, stop: int) -> np.ndarray:
    result = np.zeros(BRANCH_DIM, dtype=np.float32)
    segment = actions[max(0, start) : max(start, stop)]
    for relative, action in enumerate(segment):
        for key, value in _action_tokens(action or {}, relative):
            _hash_add(result, key, value / max(1, len(segment)))
    _hash_add(result, "segment:length", len(segment) / 48.0)
    return np.tanh(result).astype(np.float32, copy=False)


def build_branch_descriptors(root: str | Path, output: str | Path) -> dict[str, int]:
    """Compile semantic next-segment vectors for every checkpoint family."""
    root = Path(root)
    with (root / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    offsets = {str(k): tuple(v) for k, v in manifest["route_offsets"].items()}
    actions: dict[str, list[dict[str, Any]]] = {}
    with (root / "actions.bin").open("rb") as handle:
        for route_id, (offset, _) in offsets.items():
            handle.seek(int(offset))
            actions[route_id] = pickle.load(handle)
    checkpoints = [int(value) for value in manifest["checkpoints"]]
    payload: dict[int, dict[str, np.ndarray]] = {}
    families = 0
    for checkpoint in checkpoints:
        with (root / f"checkpoint-{checkpoint:03d}.pkl").open("rb") as handle:
            rows = pickle.load(handle)
        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            groups[str(row[3])].append(row)
        next_checkpoint = _next_checkpoint(checkpoints, checkpoint)
        current: dict[str, np.ndarray] = {}
        for family_hash, family_rows in groups.items():
            representative = max(family_rows, key=lambda row: (float(row[1]), str(row[0])))
            current[family_hash] = branch_descriptor(
                actions[str(representative[0])], checkpoint, next_checkpoint
            ).astype(np.float16)
        payload[checkpoint] = current
        families += len(current)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pickle.dumps(payload, protocol=5))
    return {"checkpoints": len(payload), "families": families, "bytes": output.stat().st_size}


def _load_manifest(root: Path) -> dict[str, Any]:
    key = str(root.resolve())
    if key not in _MANIFEST_CACHE:
        with (root / "manifest.pkl").open("rb") as handle:
            _MANIFEST_CACHE[key] = pickle.load(handle)
    return _MANIFEST_CACHE[key]


def _load_descriptors(path: Path) -> dict[int, dict[str, np.ndarray]]:
    key = str(path.resolve())
    if key not in _DESCRIPTOR_CACHE:
        with path.open("rb") as handle:
            _DESCRIPTOR_CACHE[key] = pickle.load(handle)
    return _DESCRIPTOR_CACHE[key]


def load_numpy_model(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    key = (str(source.resolve()), source.stat().st_mtime_ns)
    if key not in _MODEL_CACHE:
        with source.open("rb") as handle:
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = pickle.load(handle)
    return _MODEL_CACHE[key]


def numpy_forward(model: dict[str, Any] | None, features: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(features), dtype=np.float32)
    value = np.asarray(features, dtype=np.float32)
    for index, (weight, bias) in enumerate(model["layers"]):
        value = value @ np.asarray(weight, dtype=np.float32).T + np.asarray(bias, dtype=np.float32)
        if index + 1 < len(model["layers"]):
            value = value * (1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0))))
    return value.reshape(-1)


def candidate_feature(
    fingerprint: dict[str, Any],
    reference: dict[str, Any],
    descriptor: np.ndarray,
    distance: dict[str, float],
    *,
    checkpoint: int,
    next_checkpoint: int,
    median_reward: float,
    support: int,
    baseline_score: float,
    rank: int,
    candidates: int,
) -> np.ndarray:
    current = fingerprint_vector(fingerprint)
    target = fingerprint_vector(reference)
    scalars = [float(distance.get(key, 0.0)) for key in DISTANCE_KEYS]
    scalars.extend(
        [
            checkpoint / 719.0,
            next_checkpoint / 719.0,
            max(-1.0, min(1.0, median_reward / 150_000.0)),
            min(1.0, math.log1p(support) / 8.0),
            max(-2.0, min(2.0, baseline_score)),
            rank / max(1, candidates - 1),
            min(1.0, candidates / 16.0),
        ]
    )
    result = np.concatenate(
        [current, target, np.abs(current - target), np.asarray(descriptor, dtype=np.float32), scalars]
    )
    assert result.shape == (FEATURE_DIM,)
    return result.astype(np.float32, copy=False)


@dataclass(slots=True)
class TreeQTrainingDecision:
    checkpoint: int
    route_id: str
    family_hash: str
    feature: np.ndarray
    selected_rank: int
    candidates: int
    q_value: float
    explored: bool


class ExploringShardedRouteSelector:
    """Shared-Q selector with exact-prefix legality and top-k exploration."""

    def __init__(
        self,
        root: str | Path,
        descriptor_path: str | Path,
        *,
        model_path: str | Path | None = None,
        epsilon: float = 0.20,
        top_k: int = 8,
        prior_weight: float = 0.20,
        seed: int = 0,
        land_weight: float = 4.0,
        quality_weight: float = 0.10,
        support_weight: float = 0.005,
    ) -> None:
        self.root = Path(root)
        manifest = _load_manifest(self.root)
        self.checkpoints = sorted(int(value) for value in manifest["checkpoints"])
        self.route_offsets = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in manifest["route_offsets"].items()
        }
        self.descriptors = _load_descriptors(Path(descriptor_path))
        self.model_path = Path(model_path) if model_path is not None else None
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.top_k = max(1, int(top_k))
        self.prior_weight = max(0.0, float(prior_weight))
        self.land_weight = float(land_weight)
        self.quality_weight = float(quality_weight)
        self.support_weight = float(support_weight)
        self.rng = random.Random(int(seed))
        self.current_route_id: str | None = None
        self.current_actions: list[dict[str, Any]] | None = None
        self.training_decisions: list[TreeQTrainingDecision] = []
        self.candidate_family_hashes: tuple[str, ...] = ()

    @property
    def current(self) -> dict[str, Any] | None:
        return None if self.current_route_id is None else {"route_id": self.current_route_id}

    @current.setter
    def current(self, value: None) -> None:
        if value is not None:
            raise TypeError("Exploring selector current may only be cleared")
        self.reset()

    def reset(self) -> None:
        self.current_route_id = None
        self.current_actions = None
        self.training_decisions.clear()
        self.candidate_family_hashes = ()

    def action(self, step: int) -> dict[str, Any]:
        if self.current_actions is None:
            raise RuntimeError("select() must choose a route before action()")
        if not 0 <= step < len(self.current_actions):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return self.current_actions[step]

    def route_adjustment(self, features: np.ndarray) -> np.ndarray:
        """Optional stateful residual supplied by a history-aware subclass."""
        return np.zeros(len(features), dtype=np.float32)

    def choose_candidate(self, scores: np.ndarray) -> tuple[int, bool]:
        """Choose among legal candidates; stateful policies may override this hook."""
        explored = len(scores) > 1 and self.rng.random() < self.epsilon
        selected = self.rng.randrange(len(scores)) if explored else int(np.argmax(scores))
        return selected, explored

    def _rows(self, checkpoint: int) -> list[tuple[Any, ...]]:
        key = (str(self.root.resolve()), int(checkpoint))
        if key not in _ROWS_CACHE:
            with (self.root / f"checkpoint-{checkpoint:03d}.pkl").open("rb") as handle:
                _ROWS_CACHE[key] = pickle.load(handle)
        return _ROWS_CACHE[key]

    def _load_actions(self, route_id: str) -> None:
        key = (str(self.root.resolve()), str(route_id))
        if key not in _ACTIONS_CACHE:
            offset, _ = self.route_offsets[route_id]
            with (self.root / "actions.bin").open("rb") as handle:
                handle.seek(offset)
                _ACTIONS_CACHE[key] = pickle.load(handle)
        self.current_actions = _ACTIONS_CACHE[key]
        self.current_route_id = route_id

    def select(self, fingerprint: dict[str, Any], checkpoint: int) -> RouteDecision:
        rows = self._rows(checkpoint)
        by_id = {str(row[0]): row for row in rows}
        if self.current_route_id is not None:
            incumbent = by_id[self.current_route_id]
            rows = [row for row in rows if row[2] == incumbent[2]]
        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            groups[str(row[3])].append(row)
        next_checkpoint = _next_checkpoint(self.checkpoints, checkpoint)
        ranked: list[dict[str, Any]] = []
        for family_hash, family_rows in groups.items():
            distances = [
                (fingerprint_distance(fingerprint, row[4], land_weight=self.land_weight), row)
                for row in family_rows
            ]
            distances.sort(key=lambda item: (item[0]["total"], str(item[1][0])))
            distance, representative = distances[0]
            median_reward = statistics.median(float(row[1]) for row in family_rows)
            normalized_quality = max(0.0, min(1.0, median_reward / 150_000.0))
            baseline_score = (
                distance["total"]
                - self.quality_weight * normalized_quality
                - self.support_weight * math.log1p(len(family_rows))
            )
            ranked.append(
                {
                    "family_hash": family_hash,
                    "representative": representative,
                    "distance": distance,
                    "median_reward": median_reward,
                    "support": len(family_rows),
                    "baseline_score": baseline_score,
                }
            )
        ranked.sort(key=lambda row: (row["baseline_score"], -row["support"], row["family_hash"]))
        candidates = ranked[: self.top_k]
        self.candidate_family_hashes = tuple(
            str(row["family_hash"]) for row in candidates
        )
        features = np.stack(
            [
                candidate_feature(
                    fingerprint,
                    row["representative"][4],
                    self.descriptors[checkpoint][row["family_hash"]],
                    row["distance"],
                    checkpoint=checkpoint,
                    next_checkpoint=next_checkpoint,
                    median_reward=row["median_reward"],
                    support=row["support"],
                    baseline_score=row["baseline_score"],
                    rank=index,
                    candidates=len(candidates),
                )
                for index, row in enumerate(candidates)
            ]
        )
        q_values = (
            numpy_forward(load_numpy_model(self.model_path), features)
            if self.model_path is not None
            else np.zeros(len(features), dtype=np.float32)
        )
        q_values = q_values + np.asarray(self.route_adjustment(features), dtype=np.float32)
        prior = np.linspace(0.0, -self.prior_weight, len(candidates), dtype=np.float32)
        selected_index, explored = self.choose_candidate(q_values + prior)
        selected = candidates[selected_index]
        route_id = str(selected["representative"][0])
        previous_id = self.current_route_id
        if self.current_route_id != route_id:
            self._load_actions(route_id)
        self.training_decisions.append(
            TreeQTrainingDecision(
                checkpoint=checkpoint,
                route_id=route_id,
                family_hash=str(selected["family_hash"]),
                feature=features[selected_index],
                selected_rank=selected_index,
                candidates=len(candidates),
                q_value=float(q_values[selected_index]),
                explored=explored,
            )
        )
        return RouteDecision(
            checkpoint=checkpoint,
            next_checkpoint=next_checkpoint,
            route_id=route_id,
            family_hash=str(selected["family_hash"]),
            changed_route=previous_id is not None and previous_id != route_id,
            score=float(selected["baseline_score"]),
            distance=dict(selected["distance"]),
            support=int(selected["support"]),
            median_reward=float(selected["median_reward"]),
        )
