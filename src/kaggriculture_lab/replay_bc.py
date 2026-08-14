"""Replay extraction helpers for behavior-cloning experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .gpu_policy import (
    MARKET_ACTIONS,
    MARKET_INDEX,
    PRODUCTS,
    UNIT_INDEX,
    action_masks,
    action_targets,
    encode_batch,
)


def _unit_key(raw: Any) -> tuple[str, str | None]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        return ("PASS", None)
    return (str(raw[0]), raw[1] if len(raw) > 1 else None)


def _market_key(raw: Any) -> tuple[str, str | None, int] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        return None
    op = str(raw[0])
    item = raw[1] if len(raw) > 1 else None
    quantity = int(raw[2]) if len(raw) > 2 else 1
    if op == "SELL" and item in PRODUCTS:
        return (op, item, 100)
    if op in ("HIRE", "BUY_LAND"):
        return (op, None, 1)
    return (op, item, quantity)


def _audit_action(action: Mapping[str, Any], stats: Counter[str]) -> None:
    units = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]
    for raw in units:
        stats["unit_labels"] += 1
        if _unit_key(raw) not in UNIT_INDEX:
            stats["unit_unrepresentable"] += 1
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) > 2:
            stats["unit_quantity_dropped"] += 1

    market = list(action.get("market", []) or [])
    stats["market_orders"] += len(market)
    stats["market_orders_dropped"] += max(0, len(market) - 1)
    stats["market_nonempty_turns"] += bool(market)
    stats["market_multi_turns"] += len(market) > 1
    if market:
        key = _market_key(market[0])
        if key not in MARKET_INDEX:
            stats["market_first_unrepresentable"] += 1
        else:
            op, item, quantity = MARKET_ACTIONS[MARKET_INDEX[key]]
            decoded = [op] if item is None else [op, item, quantity]
            if list(market[0]) != decoded:
                stats["market_first_lossy"] += 1


def encode_replay(replay: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Turn one official replay into compact BC arrays.

    Kaggle stores the action for observation step ``t`` on replay step ``t + 1``.
    Only active observations with a mapping action are retained.
    """

    steps = list(replay.get("steps", []) or [])
    rewards = list(replay.get("rewards", []) or [0.0, 0.0])
    while len(rewards) < 2:
        rewards.append(0.0)
    outcome = float((rewards[0] > rewards[1]) - (rewards[0] < rewards[1]))
    observations: list[Any] = []
    actions: list[Mapping[str, Any]] = []
    values: list[float] = []
    stats: Counter[str] = Counter()

    for step in range(max(0, len(steps) - 1)):
        current = steps[step]
        following = steps[step + 1]
        for player in range(min(2, len(current), len(following))):
            record = current[player]
            next_record = following[player]
            if record.get("status") not in ("ACTIVE", "INACTIVE"):
                continue
            observation = record.get("observation")
            action = next_record.get("action")
            if observation is None or not isinstance(action, Mapping):
                stats["skipped_examples"] += 1
                continue
            observations.append(observation)
            actions.append(action)
            values.append(outcome if player == 0 else -outcome)
            _audit_action(action, stats)

    if not observations:
        raise ValueError("Replay contains no trainable observation/action pairs")

    features, unit_context, active = encode_batch(observations)
    unit_masks, market_masks = action_masks(observations)
    unit_targets, market_targets, target_active = action_targets(observations, actions)
    active &= target_active
    unit_valid = active & np.take_along_axis(unit_masks, unit_targets[..., None], axis=2).squeeze(2)
    market_valid = np.take_along_axis(market_masks, market_targets[:, None], axis=1).squeeze(1)
    stats["examples"] = len(observations)
    stats["unit_mask_invalid"] = int((active & ~unit_valid).sum())
    stats["market_mask_invalid"] = int((~market_valid).sum())

    arrays = {
        "features": features.astype(np.float16),
        "unit_context": unit_context.astype(np.float16),
        "unit_masks": unit_masks,
        "market_masks": market_masks,
        "unit_targets": unit_targets.astype(np.int16),
        "market_targets": market_targets.astype(np.int16),
        "unit_valid": unit_valid,
        "market_valid": market_valid,
        "value_targets": np.asarray(values, dtype=np.float16),
    }
    return arrays, dict(stats)
