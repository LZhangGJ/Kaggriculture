"""Causal replay-trie policy used by the F03 meta-agent prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .deviation_repair import WeedTransactionRepair
from .fingerprints import LandStateTracker, observation_fingerprint
from .market_manager import CausalLeadSaleManager, LeadSaleConfig, StorageMarketManager
from .route_library import load_library
from .selector import CausalRouteSelector, RouteDecision, ShardedCausalRouteSelector


class ReplayTrieAgent:
    """Execute a shared route prefix and choose suffixes at causal checkpoints."""

    def __init__(
        self,
        library: str | Path | dict[str, Any],
        *,
        switch_margin: float = 0.0,
        quality_weight: float = 0.10,
        support_weight: float = 0.005,
        land_weight: float = 4.0,
        reanchor_at_checkpoints: bool = False,
        reanchor_checkpoints: tuple[int, ...] | list[int] | set[int] | None = None,
        manage_sells: bool = False,
        lead_sells: bool = False,
        lead_turns: int = 5,
        lead_batch: int = 20,
        lead_max_distance: int = 8,
        repair_weeds: bool = False,
        weed_replay_steps: int = 8,
        weed_stop_on_match: bool = False,
        selector_override: Any | None = None,
    ) -> None:
        source = Path(library) if isinstance(library, (str, Path)) else None
        if selector_override is not None:
            self.selector = selector_override
        elif source is not None and source.is_dir() and (source / "manifest.pkl").exists():
            self.selector = ShardedCausalRouteSelector(
                source,
                switch_margin=switch_margin,
                quality_weight=quality_weight,
                support_weight=support_weight,
                land_weight=land_weight,
            )
        else:
            payload = load_library(library) if source is not None else library
            self.selector = CausalRouteSelector(
                payload,
                switch_margin=switch_margin,
                quality_weight=quality_weight,
                support_weight=support_weight,
                land_weight=land_weight,
            )
        self.checkpoints = frozenset(self.selector.checkpoints)
        if reanchor_checkpoints is not None:
            self.reanchor_checkpoints = frozenset(int(value) for value in reanchor_checkpoints)
        elif reanchor_at_checkpoints:
            self.reanchor_checkpoints = frozenset(
                int(value) for value in self.checkpoints if int(value) > 0
            )
        else:
            self.reanchor_checkpoints = frozenset()
        self.decisions: list[RouteDecision] = []
        self.land_tracker = LandStateTracker()
        self.market_manager = StorageMarketManager() if manage_sells else None
        self.lead_sale_manager = (
            CausalLeadSaleManager(
                LeadSaleConfig(
                    lead_turns=lead_turns,
                    max_batch=lead_batch,
                    max_public_distance=lead_max_distance,
                )
            )
            if lead_sells
            else None
        )
        self.weed_repair = (
            WeedTransactionRepair(
                self.selector.action,
                replay_steps=weed_replay_steps,
                stop_on_match=weed_stop_on_match,
            )
            if repair_weeds
            else None
        )

    def reset(self) -> None:
        self.selector.reset()
        self.decisions.clear()
        self.land_tracker.reset()
        if self.lead_sale_manager is not None:
            self.lead_sale_manager.reset()
        if self.weed_repair is not None:
            self.weed_repair.reset()

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        step = _step(observation)
        if step == 0:
            self.reset()
        self.land_tracker.update(observation)
        observe = getattr(self.selector, "observe", None)
        if observe is not None:
            observe(observation, configuration)
        if step in self.checkpoints:
            if step in self.reanchor_checkpoints:
                # Replay variants contain state-driven repair actions that make
                # their literal prefixes diverge before the next shop reveal.
                # At a checkpoint we may re-anchor only by the *current causal
                # state*; own-state distance remains the dominant term.
                self.selector.current = None
            decision = self.selector.select(
                observation_fingerprint(
                    observation,
                    unlock_days=self.land_tracker.unlock_days,
                ),
                step,
            )
            self.decisions.append(decision)
        action = _copy_action(self.selector.action(step))
        _align_hands(action, observation)
        if self.weed_repair is not None:
            action = self.weed_repair.apply(observation, action)
            _align_hands(action, observation)
        if self.lead_sale_manager is not None:
            action = self.lead_sale_manager.repay(step, action)
        if self.market_manager is not None:
            action = self.market_manager.apply(observation, action, configuration)
        if self.lead_sale_manager is not None:
            future_action = _copy_action(
                self.selector.action(step + self.lead_sale_manager.config.lead_turns)
            )
            action = self.lead_sale_manager.advance(
                observation,
                action,
                future_action,
                configuration,
            )
        adjust_market = getattr(self.selector, "adjust_market", None)
        if adjust_market is not None:
            action = adjust_market(observation, action, configuration)
        return action


def _step(observation: Any) -> int:
    try:
        explicit = observation.get("step")
        if explicit is not None:
            return int(explicit)
        return int(observation.get("day", 0) or 0) * 24 + int(
            observation.get("hour", 0) or 0
        )
    except (AttributeError, TypeError, ValueError):
        return 0


def _align_hands(action: dict[str, Any], observation: Any) -> None:
    try:
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", []) or []
        count = len(farms[player].get("hands", []) or [])
    except (AttributeError, IndexError, TypeError, ValueError):
        return
    hands = list(action.get("hands", []) or [])
    action["hands"] = (hands + [["PASS"]] * count)[:count]


def _copy_action(action: Any) -> dict[str, Any]:
    value = action if isinstance(action, dict) else {}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(row or ["PASS"]) for row in (value.get("hands") or [])],
        "market": [list(row or ["PASS"]) for row in (value.get("market") or [])],
    }
