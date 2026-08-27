from __future__ import annotations

import base64
import io

import numpy as np

from meta_agent.src.search_route_policy import SearchRouteController


def _exact_model_payload() -> str:
    packed = io.BytesIO()
    np.savez_compressed(
        packed,
        opening=np.asarray("BASE"),
        early_checkpoint=np.asarray(48),
        early_states=np.asarray([[1.0, 2.0]], dtype=np.float32),
        early_actions=np.asarray([1], dtype=np.int32),
        early_targets=np.asarray(["BASE", "EARLY"]),
        late_checkpoint=np.asarray(96),
        late_states=np.asarray([[3.0, 4.0]], dtype=np.float32),
        late_actions=np.asarray([1], dtype=np.int32),
        late_targets=np.asarray(["BASE", "LATE"]),
    )
    return base64.b64encode(packed.getvalue()).decode("ascii")


def test_exact_state_nodes_use_safe_fallback_and_both_checkpoints() -> None:
    controller = SearchRouteController(
        {
            "feature_schema": "semantic_route_switch_v1",
            "exact_state_model_npz_base64": _exact_model_payload(),
            "nodes": [
                {"selected": {"opening": "BASE", "checkpoint": 48}},
                {"selected": {"opening": "BASE", "checkpoint": 96}},
            ],
        },
        {"BASE": "base", "EARLY": "early", "LATE": "late"},
        [("BASE", 1.0)],
        rng_seed=0,
    )
    early = controller.nodes["BASE"][0][1]
    late = controller.nodes["BASE"][1][1]
    assert early.predict(np.asarray([1.0, 2.0], dtype=np.float32)) == "EARLY"
    assert late.predict(np.asarray([3.0, 4.0], dtype=np.float32)) == "LATE"
    assert early.predict(np.asarray([9.0, 9.0], dtype=np.float32)) == "BASE"


def _leaf(label: str) -> dict:
    return {
        "classes": [label],
        "left": [-1],
        "right": [-1],
        "feature": [-2],
        "threshold": [-2.0],
        "value": [[1.0]],
    }


def _observation(step: int) -> dict:
    return {
        "step": step,
        "player": 0,
        "farms": [{}, {}],
        "private": {},
        "market": {},
        "shops": [],
    }


def test_search_controller_can_defer_but_switches_at_most_once() -> None:
    payload = {
        "feature_schema": "semantic_route_switch_v1",
        "nodes": [
            {"selected": {"opening": "A", "checkpoint": 10, "enabled": True, "tree": _leaf("A")}},
            {"selected": {"opening": "A", "checkpoint": 20, "enabled": True, "tree": _leaf("B")}},
            {"selected": {"opening": "A", "checkpoint": 30, "enabled": True, "tree": _leaf("C")}},
        ],
    }
    controller = SearchRouteController(
        payload, {"A": "a", "B": "b", "C": "c"}, [("A", 1.0)], rng_seed=0
    )
    tapes = {"a": [{} for _ in range(31)]}

    assert controller.observe(_observation(0), tapes) == ("a", False)
    assert controller.observe(_observation(10), tapes) == ("a", False)
    assert not controller.switched
    assert controller.observe(_observation(20), tapes) == ("b", True)
    assert controller.switched
    assert controller.observe(_observation(30), tapes) == ("b", False)


def test_search_controller_uses_fallback_tree_when_observable_gate_rejects_q() -> None:
    payload = {
        "feature_schema": "semantic_route_switch_v1",
        "nodes": [
            {"selected": {
                "opening": "A", "checkpoint": 10, "enabled": True,
                "tree": _leaf("B"),
            }},
        ],
        "fallback_nodes": [
            {"selected": {
                "opening": "A", "checkpoint": 10, "enabled": True,
                "tree": _leaf("C"),
            }},
        ],
        "gate": {
            "checkpoint": 10,
            "positive_label": "current",
            "tree": _leaf("other"),
        },
    }
    controller = SearchRouteController(
        payload, {"A": "a", "B": "b", "C": "c"}, [("A", 1.0)], rng_seed=0
    )
    tapes = {"a": [{} for _ in range(11)]}

    assert controller.observe(_observation(0), tapes) == ("a", False)
    assert controller.observe(_observation(10), tapes) == ("c", True)
    assert controller.mode == "fallback"
    assert controller.decision_trace[-1]["gate_prediction"] == "other"

