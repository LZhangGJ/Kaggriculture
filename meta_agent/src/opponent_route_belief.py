"""NumPy-only probabilistic recognition of an opponent's global route family."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CATEGORIES = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "GOOSE", "COW", "SHEEP", "COOP", "PASTURE",
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _step(observation: Any) -> int:
    explicit = _get(observation, "step", None)
    if explicit is not None:
        return int(explicit)
    return int(_get(observation, "day", 0) or 0) * 24 + int(
        _get(observation, "hour", 0) or 0
    )


def public_route_features(observation: Any, farm_index: int) -> dict[str, float]:
    """Match the public feature names used by the replay training extractor.

    ``own_`` here means "the farm whose route is being classified".  During
    online inference that farm is the acting player's opponent.
    """
    farms = list(_get(observation, "farms", []) or [])
    farm = farms[farm_index] if 0 <= farm_index < len(farms) else {}
    counts = {name: 0 for name in CATEGORIES}
    for row in list(_get(farm, "tiles", []) or [])[:10]:
        for tile in list(row or [])[:10]:
            if not isinstance(tile, Mapping):
                continue
            crop = str(tile.get("crop") or "")
            animal = str(tile.get("animal") or "")
            kind = str(tile.get("kind") or "")
            if crop in counts:
                counts[crop] += 1
            elif animal in counts:
                counts[animal] += 1
            elif kind in counts:
                counts[kind] += 1
    result = {
        f"own_{name.lower()}": float(counts[name]) for name in CATEGORIES
    }
    result.update({
        "own_land": float(len(_get(farm, "unlocked_quadrants", []) or [])),
        "own_hands": float(len(_get(farm, "hands", []) or [])),
        "own_money": float(_get(farm, "money", 0.0) or 0.0),
    })
    town = _get(observation, "town", {}) or {}
    for shop in _get(town, "unlocked_shops", []) or []:
        result[f"shop_{str(shop).lower()}"] = 1.0
    market = _get(observation, "market", {}) or {}
    for product, value in dict(_get(market, "inventory", {}) or {}).items():
        result[f"market_inventory_{str(product).lower()}"] = float(value)
    for product, value in dict(_get(market, "prices", {}) or {}).items():
        result[f"market_price_{str(product).lower()}"] = float(value)
    result["route_checkpoint"] = _step(observation) / 719.0
    return result


class NumpyRouteForest:
    """Compact exported ExtraTrees/RandomForest probability evaluator."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported opponent route belief model")
        self.feature_names = tuple(str(value) for value in payload["feature_names"])
        self.feature_schema = str(payload.get("feature_schema", "simple_public_v1"))
        self.classes = tuple(str(value) for value in payload["classes"])
        self.trees = list(payload["trees"])
        self.priors = np.asarray(payload.get("priors"), dtype=np.float32)
        if self.priors.shape != (len(self.classes),):
            self.priors = np.full(len(self.classes), 1.0 / len(self.classes), dtype=np.float32)
        self.priors /= max(float(self.priors.sum()), 1e-12)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyRouteForest":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(values.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)

    def predict_vector(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float32)
        if x.shape != (len(self.feature_names),):
            raise ValueError(f"expected feature shape {(len(self.feature_names),)}, got {x.shape}")
        probabilities = np.zeros(len(self.classes), dtype=np.float64)
        for tree in self.trees:
            left = tree["left"]
            right = tree["right"]
            feature = tree["feature"]
            threshold = tree["threshold"]
            node = 0
            while int(left[node]) >= 0:
                node = int(left[node]) if x[int(feature[node])] <= float(threshold[node]) else int(right[node])
            leaf = np.asarray(tree["value"][node], dtype=np.float64)
            probabilities += leaf / max(float(leaf.sum()), 1e-12)
        probabilities /= max(1, len(self.trees))
        # A small empirical-prior floor prevents false certainty for rare G tails.
        probabilities = 0.98 * probabilities + 0.02 * self.priors
        return (probabilities / max(float(probabilities.sum()), 1e-12)).astype(np.float32)

    def predict_observation(self, observation: Any, farm_index: int) -> np.ndarray:
        if self.feature_schema == "public_market_v1":
            from .recurrent_meta import market_town_vector, public_farm_vector

            farms = list(_get(observation, "farms", []) or [])
            farm = farms[farm_index] if 0 <= farm_index < len(farms) else {}
            vector = np.concatenate([
                public_farm_vector(farm),
                market_town_vector(observation),
                np.asarray([_step(observation) / 719.0], dtype=np.float32),
            ])
            return self.predict_vector(vector)
        return self.predict_vector(self.vector(public_route_features(observation, farm_index)))


class OpponentRouteBelief:
    """Causal filtered belief over all global route families."""

    def __init__(self, model: str | Path | NumpyRouteForest, history_weight: float = 0.25) -> None:
        self.model = model if isinstance(model, NumpyRouteForest) else NumpyRouteForest.load(model)
        self.history_weight = max(0.0, min(0.95, float(history_weight)))
        self.posterior = self.model.priors.copy()
        self.last_step = -1

    @property
    def classes(self) -> tuple[str, ...]:
        return self.model.classes

    def reset(self) -> None:
        self.posterior = self.model.priors.copy()
        self.last_step = -1

    def observe(self, observation: Any) -> np.ndarray:
        step = _step(observation)
        if step == 0 or step < self.last_step:
            self.reset()
        player = int(_get(observation, "player", 0) or 0)
        current = self.model.predict_observation(observation, 1 - player)
        if self.last_step < 0:
            self.posterior = current
        else:
            self.posterior = (
                (1.0 - self.history_weight) * current
                + self.history_weight * self.posterior
            )
            self.posterior /= max(float(self.posterior.sum()), 1e-12)
        self.last_step = step
        return self.posterior.copy()

    def top(self, count: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(-self.posterior)[: max(1, int(count))]
        return [(self.classes[index], float(self.posterior[index])) for index in order]
