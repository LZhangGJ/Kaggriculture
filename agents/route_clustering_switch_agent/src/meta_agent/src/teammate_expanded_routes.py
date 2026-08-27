"""Route-tape adapter for the frozen teammate execution stack."""

from __future__ import annotations

import inspect
import json
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_action_tapes(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(zlib.decompress(Path(path).read_bytes()))
    return {str(key): list(value) for key, value in payload.items()}


class TeammateExpandedRouteAgent:
    """Run one selected replay tape through every teammate action overlay.

    Two instances must be used for the two seats so their frozen module globals
    remain isolated.  ``select`` may be called between games; step zero resets
    the teammate state machines.
    """

    def __init__(
        self,
        source: str,
        action_tapes: Mapping[str, Sequence[Mapping[str, Any]]],
        name: str,
    ) -> None:
        module = types.ModuleType(name)
        namespace = module.__dict__
        sys.modules[name] = module
        exec(compile(source, name, "exec"), namespace)
        self.namespace = namespace
        self.action_tapes = action_tapes
        self.route_id = next(iter(action_tapes))
        self.schedule: tuple[tuple[int, str], ...] = ((0, self.route_id),)
        self.agent = namespace["agent"]
        parameters = inspect.signature(self.agent).parameters
        self.takes_configuration = len(parameters) >= 2

        def route_only_policy(**kwargs: Any) -> Any:
            if kwargs.get("decision") == "opening_pressure":
                return False
            if kwargs.get("decision") == "late_prt":
                return False
            return kwargs.get("default")

        namespace["set_meta_route_policy"](route_only_policy)
        k320 = namespace["_META_K320"]

        def selected_actions(_observation: Any) -> Sequence[Mapping[str, Any]]:
            step = int(
                _observation.get("step", 0)
                if isinstance(_observation, Mapping)
                else getattr(_observation, "step", 0)
            )
            route_id = self.schedule[0][1]
            for start, candidate in self.schedule:
                if step < start:
                    break
                route_id = candidate
            return self.action_tapes[route_id]

        k320.__dict__["_kawa_actions"] = selected_actions

    def select(self, route_id: str) -> None:
        if route_id not in self.action_tapes:
            raise KeyError(route_id)
        self.route_id = route_id
        self.schedule = ((0, route_id),)

    def select_schedule(self, values: Sequence[tuple[int, str]]) -> None:
        schedule = tuple(sorted((int(step), str(route)) for step, route in values))
        if not schedule or schedule[0][0] != 0:
            raise ValueError("a route schedule must start at step zero")
        missing = [route for _, route in schedule if route not in self.action_tapes]
        if missing:
            raise KeyError(missing[0])
        self.route_id = schedule[0][1]
        self.schedule = schedule

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        if self.takes_configuration:
            return self.agent(observation, configuration)
        return self.agent(observation)
