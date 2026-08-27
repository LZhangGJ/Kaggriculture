from __future__ import annotations

import sys

from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent


def test_dynamic_source_is_registered_for_dataclasses() -> None:
    name = "test_dynamic_teammate_module"
    source = """
from dataclasses import dataclass

@dataclass
class Marker:
    value: int = 1

class K320:
    pass

_META_K320 = K320()

def set_meta_route_policy(_policy):
    pass

def agent(_observation):
    return {}
"""
    try:
        expanded = TeammateExpandedRouteAgent(source, {"r": [{}]}, name)
        assert sys.modules[name].Marker().value == 1
        assert expanded.namespace is sys.modules[name].__dict__
    finally:
        sys.modules.pop(name, None)
