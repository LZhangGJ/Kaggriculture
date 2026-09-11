"""Build the offline C++ teammate executor from the frozen Python payload.

Python is used once to unpack route assets and construct the native object.
Every turn of every counterfactual match then stays inside C++.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fast_kaggriculture import NativeAdaptiveExecutor, NativeTeammateExecutor

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
        backbone_path: str | Path | None = None,
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
        backbone = None
        if backbone_path is not None:
            with np.load(Path(backbone_path), allow_pickle=False) as payload:
                if "branch_labels" in payload.files:
                    expected = (
                        "cow_mixed__carrot",
                        "cow_mixed__wheat",
                        "sheep_heavy__carrot",
                        "sheep_heavy__wheat",
                    )
                    observed = tuple(str(value) for value in payload["branch_labels"])
                    if observed != expected:
                        raise ValueError(
                            f"adaptive branch family order mismatch: {observed!r}"
                        )
                backbone_names = (
                        "hand_target_by_day",
                        "unlocked_target_by_day",
                        "animal_owned_target_by_day",
                        "animal_service_target_by_day",
                        "crop_target_by_day",
                        "crop_plant_by_day",
                        "crop_water_by_day",
                        "crop_harvest_by_day",
                        "crop_fertilize_by_day",
                        "crop_harvest_min_yield_by_day",
                        "crop_clear_by_day",
                        "animal_feed_by_day",
                        "animal_care_by_day",
                        "animal_product_by_day",
                        "animal_fertilizer_by_day",
                        "wheat_buffer_by_day",
                        "sell_first_hour_by_day",
                        "sell_last_hour_by_day",
                        "buy_first_hour_by_day",
                        "buy_last_hour_by_day",
                        "terminal_liquidation_start_step",
                    )
                # Older semantic fixtures may contain one plan and omit the
                # richer daily-flow arrays.  Latest-submission families contain
                # four plans in the validated order above.  Preserve both
                # forms while making every available field visible natively.
                backbone = {
                    name: np.asarray(payload[name]).tolist()
                    for name in backbone_names
                    if name in payload.files
                }
        self.adaptive_executor = NativeAdaptiveExecutor(
            routes, r5, md, moon, moon_legacy, backbone
        )

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
    ) -> dict[str, Any]:
        target = -1 if switch_target is None else self.index(switch_target)
        switch0 = (int(switch_step), target) if seat == 0 else (-1, -1)
        switch1 = (int(switch_step), target) if seat == 1 else (-1, -1)
        return self.executor.play(
            self.index(left), self.index(right), int(seed),
            switch0[0], switch0[1], switch1[0], switch1[1], capture_trace,
        )
