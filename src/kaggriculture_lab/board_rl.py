"""On-policy helpers for the structured dual-board PyTorch policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .board_policy import BoardKaggriculturePolicy, encode_board_batch
from .gpu_policy import (
    PRODUCTS,
    _get,
    _mapping,
    _masked,
    action_masks,
    decode_actions,
)


_LIQUIDATION_BASE = dict(
    zip(PRODUCTS, (25, 35, 60, 120, 250, 50, 160, 200, 100), strict=True)
)
_LIQUIDATION_T = dict(
    zip(PRODUCTS, (400, 450, 200, 100, 300, 332, 122, 105, 200), strict=True)
)
_LIQUIDATION_BELOW_FUNC = dict(
    zip(PRODUCTS, (2, 3, 0, 2, 3, 0, 2, 3, 0), strict=True)
)
_LIQUIDATION_BELOW_TARGET = dict(
    zip(PRODUCTS, (0.80, 0.20, 0.40, 0.70, 0.20, 0.40, 0.60, 0.20, 0.40), strict=True)
)
_LIQUIDATION_ABOVE_FUNC = dict(
    zip(PRODUCTS, (3, 2, 2, 0, 1, 3, 0, 1, 0), strict=True)
)
_LIQUIDATION_ABOVE_TARGET = dict(
    zip(PRODUCTS, (0.20, 0.70, 0.60, 1.60, 3.60, 0.20, 1.60, 3.20, 0.40), strict=True)
)
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
RESOURCE_GROWTH_HORIZONS = (24, 72, 168)
_SEED_BOOK_VALUE = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
_ANIMAL_BOOK_VALUE = {"GOOSE": 300, "COW": 400, "SHEEP": 500}


@dataclass
class ActorCriticBatch:
    """A sampled factorized action with its differentiable policy statistics."""

    actions: list[dict[str, Any]]
    unit_indices: torch.Tensor
    market_indices: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor


def relative_money_potential(observation: Any) -> float:
    """Return an antisymmetric, actor-relative public wealth potential.

    Consecutive differences of this value provide a dense zero-sum reward while
    still telescoping to the final relative-money outcome when ``gamma == 1``.
    """

    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    if player not in (0, 1) or len(farms) < 2:
        return 0.0
    own_money = max(float(_get(farms[player], "money", 0.0)), 0.0)
    opponent_money = max(float(_get(farms[1 - player], "money", 0.0)), 0.0)
    scale = np.log1p(2_000_000.0)
    return float((np.log1p(own_money) - np.log1p(opponent_money)) / scale)


def _market_shape(code: int, value: float) -> float:
    value = max(value, 0.0)
    if code == 0:
        return value
    if code == 1:
        return value * value
    if code == 2:
        return value**0.5
    return float(np.log1p(value))


def _liquidation_quote(item: str, inventory: int) -> float:
    base = float(_LIQUIDATION_BASE[item])
    below = inventory < 10_000
    code = (
        _LIQUIDATION_BELOW_FUNC[item]
        if below
        else _LIQUIDATION_ABOVE_FUNC[item]
    )
    target = (
        _LIQUIDATION_BELOW_TARGET[item]
        if below
        else _LIQUIDATION_ABOVE_TARGET[item]
    )
    scale = _market_shape(code, float(_LIQUIDATION_T[item]))
    amplitude = target * base / max(scale, 1e-12)
    delta = amplitude * _market_shape(code, abs(float(inventory - 10_000)))
    price = base + delta if below else base - delta
    return max(float(np.rint(price)), 1.0)


def _sequential_sale_value(item: str, quantity: int, inventory: int) -> float:
    value = 0.0
    for offset in range(max(int(quantity), 0)):
        quote = _liquidation_quote(item, inventory + offset)
        value += quote
        if quote <= 1:
            break
    return value


def liquidatable_net_asset(observation: Any) -> float:
    """Cash plus executable, market-impact-adjusted product liquidation value.

    Seeds, land, workers, structures, and animals have no sell action and are
    intentionally excluded.  Products already in the shed receive the smallest
    execution haircut; carried and on-board yields are discounted for the extra
    deposit/harvest steps required before sale.
    """

    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    if player not in (0, 1) or len(farms) < 2:
        return 0.0
    own = farms[player]
    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    inventories = list(_get(private, "inventories", []) or [])
    held = {item: max(int(shed.get(item, 0)), 0) for item in PRODUCTS}
    carried = {item: 0 for item in PRODUCTS}
    for inventory in inventories:
        mapping = _mapping(inventory)
        for item in PRODUCTS:
            carried[item] += max(int(mapping.get(item, 0)), 0)
    field = {item: 0 for item in PRODUCTS}
    for row in list(_get(own, "tiles", []) or []):
        for tile in list(row or []):
            mapping = _mapping(tile)
            yield_units = max(int(mapping.get("yield_units", 0)), 0)
            crop = str(mapping.get("crop") or "")
            animal = str(mapping.get("animal") or "")
            if crop in field:
                field[crop] += yield_units
            product = _ANIMAL_PRODUCT.get(animal)
            if product is not None:
                field[product] += yield_units
            if bool(mapping.get("fertilizer_available", False)):
                field["FERTILIZER"] += 1

    market = _get(observation, "market", {}) or {}
    market_inventory = _mapping(_get(market, "inventory", {}))
    step = int(_get(observation, "step", 0) or 0)
    remaining = max(719 - step, 0)
    shed_haircut = 0.98 if remaining >= 1 else 0.0
    carried_haircut = 0.90 if remaining >= 2 else 0.0
    field_haircut = 0.75 if remaining >= 3 else 0.0
    liquidation = 0.0
    for item in PRODUCTS:
        quantities = (held[item], carried[item], field[item])
        total = sum(quantities)
        if not total:
            continue
        proceeds = _sequential_sale_value(
            item, total, int(market_inventory.get(item, 10_000))
        )
        weighted_haircut = (
            shed_haircut * quantities[0]
            + carried_haircut * quantities[1]
            + field_haircut * quantities[2]
        ) / total
        liquidation += proceeds * weighted_haircut
    return max(float(_get(own, "money", 0.0)), 0.0) + liquidation


def productive_resource_value(observation: Any) -> float:
    """Conservative book value for resources that can create future cash.

    This is used only by auxiliary critics.  The PPO reward remains the
    liquidatable-net-asset transition reward, so the heuristic cannot change
    the objective or grant credit directly to the policy.
    """

    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    if player not in (0, 1) or len(farms) < 2:
        return 0.0
    own = farms[player]
    private = _get(observation, "private", {}) or {}
    seeds = _mapping(_get(private, "seeds", {}))
    value = liquidatable_net_asset(observation)
    value += sum(
        max(int(seeds.get(item, 0) or 0), 0) * price
        for item, price in _SEED_BOOK_VALUE.items()
    )
    value += 180.0 * len(_get(own, "hands", []) or [])
    value += 700.0 * max(len(_get(own, "unlocked_quadrants", []) or []) - 1, 0)
    for row in list(_get(own, "tiles", []) or []):
        for raw_tile in list(row or []):
            tile = _mapping(raw_tile)
            kind = str(tile.get("kind") or "")
            crop = str(tile.get("crop") or "")
            animal = str(tile.get("animal") or "")
            if crop in _SEED_BOOK_VALUE:
                value += 0.75 * _SEED_BOOK_VALUE[crop]
            if animal in _ANIMAL_BOOK_VALUE:
                value += 0.75 * _ANIMAL_BOOK_VALUE[animal]
            if kind in ("COOP", "PASTURE"):
                value += 200.0
    return max(float(value), 0.0)


def productive_resource_potentials(observations: Sequence[Any]) -> np.ndarray:
    """Log-scaled absolute productive values for an actor batch."""

    values = np.asarray(
        [productive_resource_value(row) for row in observations], dtype=np.float64
    )
    return (np.log1p(values) / np.log1p(2_000_000.0)).astype(np.float32)


def multi_horizon_resource_growth_targets(
    observations: Sequence[Any],
    *,
    final_observation: Any | None = None,
    horizons: Sequence[int] = RESOURCE_GROWTH_HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact future 1/3/7-day productive-resource growth labels."""

    if not observations:
        raise ValueError("observations must be non-empty")
    horizon_values = tuple(int(value) for value in horizons)
    if not horizon_values or min(horizon_values) <= 0:
        raise ValueError("horizons must be positive")
    states = list(observations)
    if final_observation is not None:
        states.append(final_observation)
    book = np.asarray(
        [productive_resource_value(row) for row in states], dtype=np.float64
    )
    transformed = np.log1p(book) / np.log1p(2_000_000.0)
    targets = np.zeros((len(observations), len(horizon_values)), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=np.bool_)
    for step in range(len(observations)):
        for slot, horizon in enumerate(horizon_values):
            future = step + horizon
            if future < len(states):
                targets[step, slot] = float(transformed[future] - transformed[step])
                mask[step, slot] = True
    return targets, mask


def multi_horizon_rollout_targets(
    potentials: torch.Tensor,
    dones: torch.Tensor,
    *,
    horizons: Sequence[int] = RESOURCE_GROWTH_HORIZONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rollout state potentials ``[T+1,B]`` into growth targets."""

    if potentials.ndim != 2 or dones.ndim != 2:
        raise ValueError("potentials and dones must be rank-two tensors")
    if potentials.shape[0] != dones.shape[0] + 1 or potentials.shape[1] != dones.shape[1]:
        raise ValueError("potentials must be [T+1,B] and dones [T,B]")
    horizon_values = tuple(int(value) for value in horizons)
    if not horizon_values or min(horizon_values) <= 0:
        raise ValueError("horizons must be positive")
    steps, batch = dones.shape
    targets = potentials.new_zeros((steps, batch, len(horizon_values)))
    mask = torch.zeros(
        (steps, batch, len(horizon_values)),
        dtype=torch.bool,
        device=potentials.device,
    )
    for step in range(steps):
        for slot, horizon in enumerate(horizon_values):
            future = step + horizon
            if future > steps:
                continue
            # A terminal transition at the horizon endpoint is valid; a reset
            # before it would mix two episodes and is therefore masked out.
            valid = ~dones[step : max(future - 1, step)].any(dim=0)
            targets[step, :, slot] = potentials[future] - potentials[step]
            mask[step, :, slot] = valid
    return targets, mask


def multi_horizon_transition_targets(
    potentials: torch.Tensor,
    next_potentials: torch.Tensor,
    dones: torch.Tensor,
    *,
    horizons: Sequence[int] = RESOURCE_GROWTH_HORIZONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Growth labels from aligned before/after rollout states, safe across resets."""

    if potentials.ndim != 2 or next_potentials.shape != potentials.shape:
        raise ValueError("potentials and next_potentials must share shape [T,B]")
    if dones.shape != potentials.shape:
        raise ValueError("dones must share shape [T,B]")
    horizon_values = tuple(int(value) for value in horizons)
    if not horizon_values or min(horizon_values) <= 0:
        raise ValueError("horizons must be positive")
    steps, batch = potentials.shape
    targets = potentials.new_zeros((steps, batch, len(horizon_values)))
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for step in range(steps):
        for slot, horizon in enumerate(horizon_values):
            endpoint = step + horizon - 1
            if endpoint >= steps:
                continue
            valid = ~dones[step:endpoint].any(dim=0)
            targets[step, :, slot] = next_potentials[endpoint] - potentials[step]
            mask[step, :, slot] = valid
    return targets, mask


def paired_liquidatable_net_asset_potentials(
    observations: Sequence[Any],
) -> np.ndarray:
    """Return exactly antisymmetric relative net-asset potentials for actor pairs."""

    if len(observations) % 2:
        raise ValueError("observations must contain consecutive two-player pairs")
    assets = np.asarray(
        [liquidatable_net_asset(observation) for observation in observations],
        dtype=np.float64,
    )
    scale = np.log1p(2_000_000.0)
    output = np.zeros(len(assets), dtype=np.float32)
    for start in range(0, len(assets), 2):
        difference = (
            np.log1p(max(assets[start], 0.0))
            - np.log1p(max(assets[start + 1], 0.0))
        ) / scale
        output[start] = difference
        output[start + 1] = -difference
    return output


def potential_shaped_value_targets(
    potentials: Sequence[float],
    *,
    gamma: float = 0.997,
    reward_scale: float = 10.0,
    terminal_outcome: float = 0.0,
    win_bonus: float = 1.0,
) -> np.ndarray:
    """Convert T+1 state potentials into T per-step discounted return targets.

    This matches the self-play reward definition: every transition receives the
    change in liquidatable relative net assets, and the final transition also
    receives the signed terminal outcome.  Unlike copying one final value onto
    every observation, it preserves when an expert created or destroyed value.
    """

    values = np.asarray(potentials, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("potentials must be a one-dimensional T+1 sequence")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    rewards = np.diff(values) * float(reward_scale)
    rewards[-1] += float(win_bonus) * float(terminal_outcome)
    returns = np.empty_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = running
    return returns.astype(np.float32)


def actor_critic_batch(
    model: BoardKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
) -> ActorCriticBatch:
    """Sample legal joint actions and retain statistics for an RL update."""

    boards_np, global_np, units_np, active_np = encode_board_batch(observations)
    unit_masks_np, market_masks_np = action_masks(observations)
    boards = torch.as_tensor(boards_np, device=device)
    global_features = torch.as_tensor(global_np, device=device)
    units = torch.as_tensor(units_np, device=device)
    active = torch.as_tensor(active_np, device=device)
    unit_masks = torch.as_tensor(unit_masks_np, device=device)
    market_masks = torch.as_tensor(market_masks_np, device=device)

    unit_logits, market_logits, values = model(
        boards, global_features, units, active
    )
    unit_distribution = torch.distributions.Categorical(
        logits=_masked(unit_logits, unit_masks)
    )
    market_distribution = torch.distributions.Categorical(
        logits=_masked(market_logits, market_masks)
    )
    unit_indices = unit_distribution.sample()
    market_indices = market_distribution.sample()

    own_active = active[:, 0].to(unit_logits.dtype)
    log_probs = (
        (unit_distribution.log_prob(unit_indices) * own_active).sum(dim=-1)
        + market_distribution.log_prob(market_indices)
    )
    active_factors = own_active.sum(dim=-1) + 1.0
    entropies = (
        (unit_distribution.entropy() * own_active).sum(dim=-1)
        + market_distribution.entropy()
    ) / active_factors
    actions = decode_actions(
        unit_indices.detach().cpu().numpy(),
        market_indices.detach().cpu().numpy(),
        observations,
    )
    return ActorCriticBatch(
        actions=actions,
        unit_indices=unit_indices,
        market_indices=market_indices,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
    )


@torch.no_grad()
def board_values(
    model: BoardKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Evaluate only the value head for bootstrap targets."""

    boards_np, global_np, units_np, active_np = encode_board_batch(observations)
    _, _, values = model(
        torch.as_tensor(boards_np, device=device),
        torch.as_tensor(global_np, device=device),
        torch.as_tensor(units_np, device=device),
        torch.as_tensor(active_np, device=device),
    )
    return values


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets for tensors shaped ``[T, B]``."""

    if rewards.shape != values.shape or dones.shape != values.shape:
        raise ValueError("rewards, dones, and values must share shape [T, B]")
    if bootstrap_values.shape != values.shape[1:]:
        raise ValueError("bootstrap_values must have shape [B]")
    advantages = torch.zeros_like(values)
    next_advantage = torch.zeros_like(bootstrap_values)
    next_value = bootstrap_values
    for step in range(values.shape[0] - 1, -1, -1):
        continuation = (~dones[step]).to(values.dtype)
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        next_advantage = (
            delta + gamma * gae_lambda * continuation * next_advantage
        )
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages, advantages + values
