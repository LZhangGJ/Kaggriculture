#!/usr/bin/env python3
"""Regression check for configured replay-route target fallbacks."""

from meta_agent.src.search_route_policy import SearchRouteController


def main():
    policy = {
        "feature_schema": "recurrent_meta_v1",
        "target_fallbacks": {"bad": "safe"},
        "nodes": [{
            "selected": {"enabled": True, "opening": "open", "checkpoint": 1,
                         "tree": {"classes": ["bad"], "left": [-1], "right": [-1],
                                  "feature": [-2], "threshold": [-2], "value": [[1]]}},
        }],
    }
    controller = SearchRouteController(
        policy, {"open": "open-tape", "bad": "bad-tape", "safe": "safe-tape"},
        [("open", 1.0)],
    )
    assert controller.observe({"step": 0}) == ("open-tape", False)
    assert controller.observe({"step": 1}) == ("safe-tape", True)
    policy["target_fallbacks"] = {"missing": "safe"}
    try:
        SearchRouteController(policy, {"open": "open-tape", "safe": "safe-tape"}, [("open", 1)])
        raise AssertionError("invalid route fallback accepted")
    except ValueError:
        pass
    print("route target fallback: PASS")


if __name__ == "__main__":
    main()
