"""Small public-state trees that override recurrent route candidate ranks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .equilibrium_selector import EquilibriumOpeningSelector


CROP_COUNT_INDICES = np.asarray((6, 12, 18, 24, 30), dtype=np.int64)
ANIMAL_COUNT_INDICES = np.asarray((36, 42, 48), dtype=np.int64)
OPPONENT_BASIC_INDICES = np.asarray((0, 1, 2, 3), dtype=np.int64)
SHOP_HASH_SLICE = slice(18, 34)


def opponent_route_features(
    opponent_public: np.ndarray,
    market_town: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Return explicit, causally public features for a shallow route tree."""
    opponent = np.asarray(opponent_public, dtype=np.float32)
    market = np.asarray(market_town, dtype=np.float32)
    counts = opponent[np.concatenate((CROP_COUNT_INDICES, ANIMAL_COUNT_INDICES))]
    if feature_set == "opponent_counts":
        return counts.astype(np.float32, copy=False)
    if feature_set == "opponent_context":
        return np.concatenate(
            (counts, opponent[OPPONENT_BASIC_INDICES], market[SHOP_HASH_SLICE])
        ).astype(np.float32, copy=False)
    raise ValueError(f"unknown simple route feature set: {feature_set!r}")


class SimpleOpponentTreeSelector(EquilibriumOpeningSelector):
    """Use one shallow rank classifier per checkpoint, with PPO fallback."""

    def __init__(self, *args: Any, tree_model_path: str, **kwargs: Any) -> None:
        with Path(tree_model_path).open("rb") as handle:
            payload = pickle.load(handle)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported simple route tree model")
        self.feature_set = str(payload["feature_set"])
        self.trees = {int(key): value for key, value in payload["trees"].items()}
        self._tree_checkpoint = -1
        self.tree_predictions: list[dict[str, int | bool]] = []
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._tree_checkpoint = -1
        self.tree_predictions.clear()

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        self._tree_checkpoint = int(checkpoint)
        try:
            return super().select(fingerprint, checkpoint)
        finally:
            self._tree_checkpoint = -1

    def _choose_candidate(self, scores: np.ndarray) -> tuple[int, bool]:
        tree = self.trees.get(self._tree_checkpoint)
        if tree is None or not self.encoded_history:
            return int(np.argmax(scores)), False
        current = self.encoded_history[-1]
        features = opponent_route_features(
            current.opponent_public, current.market_town, self.feature_set
        )
        predicted = int(tree.predict(features.reshape(1, -1))[0])
        valid = 0 <= predicted < len(scores)
        selected = predicted if valid else int(np.argmax(scores))
        self.tree_predictions.append(
            {
                "checkpoint": self._tree_checkpoint,
                "predicted_rank": predicted,
                "selected_rank": selected,
                "candidates": len(scores),
                "valid": valid,
            }
        )
        return selected, False


class DirectOpponentTreeSelector(EquilibriumOpeningSelector):
    """Choose a legal next-prefix family learned from paired route outcomes."""

    def __init__(self, *args: Any, tree_model_path: str, **kwargs: Any) -> None:
        with Path(tree_model_path).open("rb") as handle:
            payload = pickle.load(handle)
        if (
            int(payload.get("schema_version", 0)) != 1
            or payload.get("model_kind") != "direct_family"
        ):
            raise ValueError("unsupported direct route tree model")
        self.feature_set = str(payload["feature_set"])
        self.trees = {str(key): value for key, value in payload["trees"].items()}
        self._tree_node = ""
        self.tree_predictions: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._tree_node = ""
        self.tree_predictions.clear()

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        checkpoint = int(checkpoint)
        self._tree_node = ""
        if checkpoint > 0 and self.selector.current_route_id is not None:
            rows = self.selector._rows(checkpoint)
            current = next(
                row for row in rows
                if str(row[0]) == str(self.selector.current_route_id)
            )
            self._tree_node = f"{checkpoint}:{current[2]}"
        try:
            return super().select(fingerprint, checkpoint)
        finally:
            self._tree_node = ""

    def _choose_candidate(self, scores: np.ndarray) -> tuple[int, bool]:
        tree = self.trees.get(self._tree_node)
        families = tuple(self.selector.candidate_family_hashes)
        if tree is None or not self.encoded_history or not families:
            return int(np.argmax(scores)), False
        current = self.encoded_history[-1]
        features = opponent_route_features(
            current.opponent_public, current.market_town, self.feature_set
        )
        predicted = str(tree.predict(features.reshape(1, -1))[0])
        valid = predicted in families
        selected = families.index(predicted) if valid else int(np.argmax(scores))
        self.tree_predictions.append(
            {
                "node": self._tree_node,
                "predicted_family": predicted,
                "selected_rank": selected,
                "valid": valid,
            }
        )
        return selected, False


class CommittedOpponentRouteSelector(EquilibriumOpeningSelector):
    """Select a full compatible continuation from paired outcome regressors."""

    def __init__(self, *args: Any, tree_model_path: str, **kwargs: Any) -> None:
        with Path(tree_model_path).open("rb") as handle:
            payload = pickle.load(handle)
        if (
            int(payload.get("schema_version", 0)) != 1
            or payload.get("model_kind") != "committed_route"
        ):
            raise ValueError("unsupported committed route tree model")
        self.feature_set = str(payload["feature_set"])
        self.route_models = {
            str(node): {str(route): model for route, model in models.items()}
            for node, models in payload["route_models"].items()
        }
        self.commit_checkpoint = int(payload["checkpoint"])
        self.committed_route_id: str | None = None
        self.commit_predictions: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.committed_route_id = None
        self.commit_predictions.clear()

    def _fixed_decision(self, checkpoint: int, changed: bool = False) -> Any:
        route_id = str(self.selector.current_route_id)
        next_checkpoint = next(
            (value for value in self.selector.checkpoints if value > checkpoint),
            checkpoint,
        )
        from .selector import RouteDecision
        return RouteDecision(
            checkpoint=checkpoint, next_checkpoint=int(next_checkpoint),
            route_id=route_id, family_hash=f"committed:{route_id}",
            changed_route=changed, score=0.0, distance={}, support=1,
            median_reward=0.0,
        )

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        checkpoint = int(checkpoint)
        if self.committed_route_id is not None:
            return self._fixed_decision(checkpoint)
        if checkpoint != self.commit_checkpoint or self.selector.current_route_id is None:
            return super().select(fingerprint, checkpoint)
        rows = self.selector._rows(checkpoint)
        current = next(
            row for row in rows
            if str(row[0]) == str(self.selector.current_route_id)
        )
        node = f"{checkpoint}:{current[2]}"
        models = self.route_models.get(node)
        if not models or not self.encoded_history:
            return super().select(fingerprint, checkpoint)
        current_state = self.encoded_history[-1]
        features = opponent_route_features(
            current_state.opponent_public,
            current_state.market_town,
            self.feature_set,
        ).reshape(1, -1)
        compatible = {str(row[0]) for row in rows if row[2] == current[2]}
        scores = {
            route: float(model.predict(features)[0])
            for route, model in models.items() if route in compatible
        }
        if not scores:
            return super().select(fingerprint, checkpoint)
        previous = str(self.selector.current_route_id)
        selected = max(scores, key=lambda route: (scores[route], route))
        self.selector._load_actions(selected)
        self.committed_route_id = selected
        self.commit_predictions.append({
            "node": node, "route_id": selected, "score": scores[selected],
            "candidates": len(scores),
        })
        return self._fixed_decision(checkpoint, changed=selected != previous)
