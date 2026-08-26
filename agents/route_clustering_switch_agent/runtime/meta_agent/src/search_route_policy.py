"""Pure-Python route controller distilled from counterfactual C++ search."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .route_switch_features import RouteSwitchHistory, route_switch_vector


class NumpySearchTree:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.classes = tuple(str(value) for value in payload["classes"])
        self.left = np.asarray(payload["left"], dtype=np.int32)
        self.right = np.asarray(payload["right"], dtype=np.int32)
        self.feature = np.asarray(payload["feature"], dtype=np.int16)
        self.threshold = np.asarray(payload["threshold"], dtype=np.float32)
        self.value = np.asarray(payload["value"], dtype=np.float32)

    def predict(self, vector: np.ndarray) -> str:
        node = 0
        while int(self.left[node]) >= 0:
            node = (
                int(self.left[node])
                if float(vector[int(self.feature[node])]) <= float(self.threshold[node])
                else int(self.right[node])
            )
        return self.classes[int(np.argmax(self.value[node]))]


class SearchRouteController:
    """Choose a searched Nash opening and at most one learned route switch."""

    def __init__(
        self,
        policy: str | Path | Mapping[str, Any],
        route_by_family: Mapping[str, str],
        opening_weights: Sequence[tuple[str, float]],
        rng_seed: int | None = None,
    ) -> None:
        payload = (
            json.loads(Path(policy).read_text(encoding="utf-8"))
            if isinstance(policy, (str, Path))
            else dict(policy)
        )
        self.route_by_family = {str(key): str(value) for key, value in route_by_family.items()}
        self.feature_schema = str(payload.get("feature_schema", "semantic_route_switch_v1"))
        if self.feature_schema != "semantic_route_switch_v1":
            raise ValueError(f"unsupported route feature schema: {self.feature_schema}")
        total = sum(max(0.0, float(weight)) for _, weight in opening_weights)
        if total <= 0:
            raise ValueError("opening weights must have positive mass")
        self.opening_weights = tuple(
            (str(family), max(0.0, float(weight)) / total)
            for family, weight in opening_weights
            if float(weight) > 0
        )
        self.rng = random.Random(rng_seed)
        self.forced_opening: str | None = None
        grouped_nodes: dict[str, list[tuple[int, NumpySearchTree]]] = {}
        for value in payload["nodes"]:
            selected = value["selected"]
            if not bool(selected.get("enabled", True)):
                continue
            grouped_nodes.setdefault(str(selected["opening"]), []).append((
                int(selected["checkpoint"]), NumpySearchTree(selected["tree"])
            ))
        self.nodes = {
            opening: tuple(sorted(nodes, key=lambda row: row[0]))
            for opening, nodes in grouped_nodes.items()
        }
        self.last_step = -1
        self.opening = ""
        self.current = ""
        self.switched = False
        self.history = RouteSwitchHistory()

    def _choose_opening(self, observation: Mapping[str, Any]) -> str:
        if self.forced_opening is not None:
            if self.forced_opening not in self.route_by_family:
                raise KeyError(self.forced_opening)
            return self.forced_opening
        value = self.rng.random()
        cumulative = 0.0
        for family, weight in self.opening_weights:
            cumulative += weight
            if value < cumulative:
                return family
        return self.opening_weights[-1][0]

    def observe(
        self,
        observation: Mapping[str, Any],
        action_tapes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> tuple[str, bool]:
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step < self.last_step:
            self.opening = self._choose_opening(observation)
            self.current = self.opening
            self.switched = False
            self.history.reset()
        self.history.update(observation)
        changed = False
        node = next(
            (value for value in self.nodes.get(self.opening, ()) if step == value[0]),
            None,
        )
        if not self.switched and node is not None:
            route_id = self.route_by_family[self.opening]
            route_actions = (action_tapes or {}).get(route_id)
            vector = route_switch_vector(observation, self.history, route_actions)
            prediction = node[1].predict(vector)
            targets = getattr(self, "targets", ())
            if prediction in self.route_by_family:
                target = prediction
            else:
                # Schema v1 stored the target-array index as its class.
                target_index = int(prediction)
                target = str(targets[target_index]) if 0 <= target_index < len(targets) else ""
            if target in self.route_by_family and target != self.current:
                self.current = target
                changed = True
                self.switched = True
        self.last_step = step
        return self.route_by_family[self.current], changed


class SearchRoutedTeammateAgent:
    def __init__(self, expanded_agent: Any, controller: SearchRouteController, targets: Sequence[str]) -> None:
        self.expanded_agent = expanded_agent
        self.controller = controller
        self.controller.targets = tuple(str(value) for value in targets)

    def __call__(self, observation: Mapping[str, Any], configuration: Any = None):
        step = int(observation.get("step", 0) or 0)
        route_id, changed = self.controller.observe(
            observation, self.expanded_agent.action_tapes
        )
        if step == 0:
            self.expanded_agent.select(route_id)
        elif changed:
            self.expanded_agent.select_schedule(
                (*self.expanded_agent.schedule, (step, route_id))
            )
        return self.expanded_agent(observation, configuration)
