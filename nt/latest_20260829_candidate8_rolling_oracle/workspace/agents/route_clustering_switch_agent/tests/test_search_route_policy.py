from __future__ import annotations

from meta_agent.src.search_route_policy import SearchRouteController


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
