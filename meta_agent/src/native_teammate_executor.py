"""Build the offline C++ teammate executor from the frozen Python payload.

Python is used once to unpack route assets and construct the native object.
Every turn of every counterfactual match then stays inside C++.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fast_kaggriculture import NativeTeammateExecutor

from .teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes


_MOON_LABELS = (
    "10C4S_3Q",
    "8C6S_3Q",
    "6C8S_3Q",
    "6C12S_4Q_FIRST_YARN",
    "6C12S_4Q_SECOND_YARN",
)


def _market_tape(values: Any) -> list[dict[str, Any]]:
    """Turn a market-only reference into normal action dictionaries."""
    return [
        {"farmer": ["PASS"], "hands": [], "market": list(market or [])}
        for market in list(values or [])
    ]


class NativeTeammateBundle:
    """Route/family mapping plus the compiled native match executor."""

    def __init__(
        self,
        source_path: str | Path,
        actions_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        source = Path(source_path).read_text(encoding="utf-8")
        actions = load_action_tapes(actions_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        entries = list(metadata["opponent_routes"])
        self.families = tuple(str(value["family"]) for value in entries)
        self.route_ids = tuple(str(value["route_id"]) for value in entries)
        self.family_index = {family: index for index, family in enumerate(self.families)}

        # Executing the frozen payload here only exposes its immutable reference
        # arrays.  It is not called during native matches.
        probe = TeammateExpandedRouteAgent(source, actions, "native_bundle_probe")
        k320 = probe.namespace["_BASE"].__dict__["_K320"].__dict__
        moon_module = probe.namespace["_MOON"].__dict__
        r5 = _market_tape(k320["_V17_R5_MARKETS"])
        md = _market_tape(k320["_V17_MD_MARKETS"])
        moon = [moon_module[f"_ACTIONS_{label}"] for label in _MOON_LABELS]
        moon_legacy = [
            moon_module[f"_LEGACY_ACTIONS_{label}"] for label in _MOON_LABELS
        ]
        routes = [actions[route_id] for route_id in self.route_ids]
        self.executor = NativeTeammateExecutor(routes, r5, md, moon, moon_legacy)

    def index(self, family: str) -> int:
        return self.family_index[family]

    def play(
        self,
        left: str,
        right: str,
        seed: int,
        *,
        switch_step: int = -1,
        switch_target: str | None = None,
        seat: int = 0,
        capture_trace: bool = False,
        neutral_special_economy: bool = False,
        experimental_general_takeover: bool = False,
        experimental_general_takeover_player: int = -1,
        experimental_market_arm: int = 0,
        experimental_market_player: int = -1,
        stop_after_steps: int = -1,
        evaluation_compaction_minimum_step: int = -1,
        evaluation_compaction_evidence_mode: int = 0,
    ) -> dict[str, Any]:
        target = -1 if switch_target is None else self.index(switch_target)
        switch0 = (int(switch_step), target) if seat == 0 else (-1, -1)
        switch1 = (int(switch_step), target) if seat == 1 else (-1, -1)
        return self.executor.play(
            self.index(left), self.index(right), int(seed),
            switch0[0], switch0[1], switch1[0], switch1[1], capture_trace,
            neutral_special_economy, False, -1, 7, experimental_general_takeover,
            int(experimental_general_takeover_player),
            int(experimental_market_arm), int(experimental_market_player),
            int(stop_after_steps),
            int(evaluation_compaction_minimum_step),
            int(evaluation_compaction_evidence_mode),
        )
