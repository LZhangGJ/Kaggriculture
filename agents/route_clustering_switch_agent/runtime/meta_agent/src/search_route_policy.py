"""Pure-Python route controller distilled from counterfactual C++ search."""

from __future__ import annotations

import base64
import io
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


class QuantizedExtraTreesRouteQ:
    """Compact, sklearn-free inference for a multi-output ExtraTrees route-Q model."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        packed = base64.b64decode(str(payload["q_model_npz_base64"]))
        with np.load(io.BytesIO(packed), allow_pickle=False) as saved:
            self.targets = tuple(saved["targets"].astype(str).tolist())
            self.checkpoints = tuple(int(value) for value in saved["checkpoints"])
            self.checkpoint_tree_offsets = saved["checkpoint_tree_offsets"].astype(
                np.int32
            )
            self.tree_node_offsets = saved["tree_node_offsets"].astype(np.int32)
            self.left = saved["left"].astype(np.int32)
            self.right = saved["right"].astype(np.int32)
            self.feature = saved["feature"].astype(np.int16)
            # sklearn trees compare float32 inputs against float64 thresholds.
            # Retaining the threshold precision makes the exported runtime
            # decision-identical at values close to a split boundary.
            self.threshold = saved["threshold"].astype(np.float64)
            self.leaf_values = saved["leaf_values"]
        self.checkpoint_index = {
            checkpoint: index for index, checkpoint in enumerate(self.checkpoints)
        }

    def predict(self, checkpoint: int, vector: np.ndarray) -> str:
        checkpoint_index = self.checkpoint_index[int(checkpoint)]
        tree_start = int(self.checkpoint_tree_offsets[checkpoint_index])
        tree_stop = int(self.checkpoint_tree_offsets[checkpoint_index + 1])
        scores = np.zeros(len(self.targets), dtype=np.uint64)
        for tree_index in range(tree_start, tree_stop):
            node = int(self.tree_node_offsets[tree_index])
            while int(self.left[node]) >= 0:
                node = (
                    int(self.left[node])
                    if float(vector[int(self.feature[node])]) <= float(self.threshold[node])
                    else int(self.right[node])
                )
            scores += self.leaf_values[node].astype(np.uint64)
        return self.targets[int(np.argmax(scores))]


class QuantizedRouteQNode:
    def __init__(self, model: QuantizedExtraTreesRouteQ, checkpoint: int) -> None:
        self.model = model
        self.checkpoint = int(checkpoint)

    def predict(self, vector: np.ndarray) -> str:
        return self.model.predict(self.checkpoint, vector)


class ExactStateRouteSelector:
    """Small public-state lookup with a safe opening-route fallback."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        packed = base64.b64decode(str(payload["exact_state_model_npz_base64"]))
        with np.load(io.BytesIO(packed), allow_pickle=False) as saved:
            self.opening = str(saved["opening"])
            self.lookups = {}
            for stage in ("early", "late"):
                checkpoint = int(saved[f"{stage}_checkpoint"])
                states = saved[f"{stage}_states"].astype(np.float32)
                actions = saved[f"{stage}_actions"].astype(np.int32)
                targets = saved[f"{stage}_targets"].astype(str)
                self.lookups[checkpoint] = {
                    state.tobytes(): str(targets[int(action)])
                    for state, action in zip(states, actions)
                }

    def predict(self, checkpoint: int, vector: np.ndarray) -> str:
        lookup = self.lookups[int(checkpoint)]
        state = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
        return lookup.get(state, self.opening)


class ExactStateRouteNode:
    def __init__(self, model: ExactStateRouteSelector, checkpoint: int) -> None:
        self.model = model
        self.checkpoint = int(checkpoint)

    def predict(self, vector: np.ndarray) -> str:
        return self.model.predict(self.checkpoint, vector)


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
        grouped_nodes: dict[str, list[tuple[int, Any]]] = {}
        route_q = (
            QuantizedExtraTreesRouteQ(payload)
            if "q_model_npz_base64" in payload else None
        )
        exact_state = (
            ExactStateRouteSelector(payload)
            if "exact_state_model_npz_base64" in payload else None
        )
        for value in payload["nodes"]:
            selected = value["selected"]
            if not bool(selected.get("enabled", True)):
                continue
            checkpoint = int(selected["checkpoint"])
            if exact_state is not None:
                predictor = ExactStateRouteNode(exact_state, checkpoint)
            elif route_q is not None:
                predictor = QuantizedRouteQNode(route_q, checkpoint)
            else:
                predictor = NumpySearchTree(selected["tree"])
            grouped_nodes.setdefault(str(selected["opening"]), []).append((
                checkpoint, predictor
            ))
        self.nodes = {
            opening: tuple(sorted(nodes, key=lambda row: row[0]))
            for opening, nodes in grouped_nodes.items()
        }
        fallback_nodes: dict[str, list[tuple[int, Any]]] = {}
        for value in payload.get("fallback_nodes", ()):
            selected = value["selected"]
            if not bool(selected.get("enabled", True)):
                continue
            fallback_nodes.setdefault(str(selected["opening"]), []).append((
                int(selected["checkpoint"]), NumpySearchTree(selected["tree"])
            ))
        self.fallback_nodes = {
            opening: tuple(sorted(nodes, key=lambda row: row[0]))
            for opening, nodes in fallback_nodes.items()
        }
        self.mode_nodes: dict[str, dict[str, tuple[tuple[int, Any], ...]]] = {
            "fallback": self.fallback_nodes
        }
        for mode, wrapped_nodes in dict(payload.get("mode_nodes", {})).items():
            grouped: dict[str, list[tuple[int, Any]]] = {}
            for value in wrapped_nodes:
                selected = value["selected"]
                if not bool(selected.get("enabled", True)):
                    continue
                grouped.setdefault(str(selected["opening"]), []).append((
                    int(selected["checkpoint"]), NumpySearchTree(selected["tree"])
                ))
            self.mode_nodes[str(mode)] = {
                opening: tuple(sorted(nodes, key=lambda row: row[0]))
                for opening, nodes in grouped.items()
            }
        gate_payload = payload.get("gate")
        self.gate = (
            NumpySearchTree(gate_payload["tree"])
            if isinstance(gate_payload, Mapping) else None
        )
        self.gate_checkpoint = int(gate_payload.get("checkpoint", -1)) if gate_payload else -1
        self.gate_positive_label = (
            str(gate_payload.get("positive_label", "current")) if gate_payload else ""
        )
        self.gate_mode_by_label = (
            {str(key): str(value) for key, value in dict(
                gate_payload.get("mode_by_label", {})
            ).items()}
            if gate_payload else {}
        )
        self.mode = "q" if self.gate is None else ""
        self.last_step = -1
        self.opening = ""
        self.current = ""
        self.switched = False
        self.history = RouteSwitchHistory()
        self.decision_trace: list[dict[str, Any]] = []

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
            self.decision_trace.clear()
            self.mode = "q" if self.gate is None else ""
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
            gate_prediction = ""
            if self.gate is not None and not self.mode and step == self.gate_checkpoint:
                gate_prediction = self.gate.predict(vector)
                self.mode = self.gate_mode_by_label.get(
                    str(gate_prediction),
                    "q" if gate_prediction == self.gate_positive_label else "fallback",
                )
                if self.mode != "q":
                    node = next(
                        (
                            value for value in self.mode_nodes.get(
                                self.mode, {}
                            ).get(self.opening, ())
                            if step == value[0]
                        ),
                        None,
                    )
            elif self.mode != "q":
                node = next(
                    (
                        value for value in self.mode_nodes.get(
                            self.mode, {}
                        ).get(self.opening, ())
                        if step == value[0]
                    ),
                    None,
                )
            if node is None:
                self.last_step = step
                return self.route_by_family[self.current], False
            prediction = node[1].predict(vector)
            self.decision_trace.append({
                "checkpoint": step,
                "gate_prediction": str(gate_prediction),
                "mode": self.mode,
                "prediction": str(prediction),
                "vector": vector.tolist(),
            })
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
