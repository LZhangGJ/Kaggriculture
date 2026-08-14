"""Structured BC/PPO policy with lossless multi-order action heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .gpu_policy import (
    ANIMALS,
    CROPS,
    FEATURE_DIM,
    ITEMS,
    MAX_UNITS,
    PRODUCTS,
    UNIT_ACTIONS,
    UNIT_INDEX,
    _get,
    action_masks,
    encode_batch,
)

MAX_MARKET_ORDERS = 10
MAX_QUANTITY = 100
QUANTITY_CLASSES = MAX_QUANTITY + 1

MARKET_TOKENS: tuple[tuple[str, str | None], ...] = (
    ("NONE", None),
    ("HIRE", None),
    ("BUY_LAND", None),
    *(("BUY_SEED", crop) for crop in CROPS),
    *(("BUY_PRODUCT", item) for item in ("WHEAT", "FERTILIZER")),
    *(("BUY_ANIMAL", animal) for animal in ANIMALS),
    *(("SELL", item) for item in PRODUCTS),
)
MARKET_TOKEN_INDEX = {token: index for index, token in enumerate(MARKET_TOKENS)}


def _quantity(raw: Any, default: int = 0) -> int:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        return default
    try:
        return min(MAX_QUANTITY, max(0, int(raw[2])))
    except (TypeError, ValueError):
        return default


def structured_action_targets(
    observations: Sequence[Any],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Encode full unit quantities and all ordered market slots."""

    size = len(actions)
    unit_tokens = np.zeros((size, MAX_UNITS), dtype=np.int16)
    unit_quantities = np.zeros((size, MAX_UNITS), dtype=np.int16)
    unit_quantity_active = np.zeros((size, MAX_UNITS), dtype=np.bool_)
    unit_active = np.zeros((size, MAX_UNITS), dtype=np.bool_)
    market_tokens = np.zeros((size, MAX_MARKET_ORDERS), dtype=np.int16)
    market_quantities = np.zeros((size, MAX_MARKET_ORDERS), dtype=np.int16)
    market_quantity_active = np.zeros((size, MAX_MARKET_ORDERS), dtype=np.bool_)
    market_order_active = np.zeros((size, MAX_MARKET_ORDERS), dtype=np.bool_)

    for row, (observation, action) in enumerate(zip(observations, actions, strict=True)):
        player = int(_get(observation, "player", 0))
        farms = list(_get(observation, "farms", []) or [])
        hand_count = len(_get(farms[player], "hands", []) or []) if player < len(farms) else 0
        raw_units = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])[:hand_count]]
        for slot, raw in enumerate(raw_units[:MAX_UNITS]):
            unit_active[row, slot] = True
            op = str(raw[0]) if isinstance(raw, Sequence) and raw else "PASS"
            item = raw[1] if isinstance(raw, Sequence) and len(raw) > 1 else None
            unit_tokens[row, slot] = UNIT_INDEX.get((op, item), UNIT_INDEX[("PASS", None)])
            if op in ("PICKUP", "PLACE"):
                unit_quantity_active[row, slot] = True
                unit_quantities[row, slot] = _quantity(raw, default=1)

        for slot, raw in enumerate(list(action.get("market", []) or [])[:MAX_MARKET_ORDERS]):
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
                continue
            op = str(raw[0])
            item = raw[1] if len(raw) > 1 else None
            token = (op, None if op in ("HIRE", "BUY_LAND") else item)
            token_index = MARKET_TOKEN_INDEX.get(token)
            if token_index is None:
                continue
            market_tokens[row, slot] = token_index
            market_order_active[row, slot] = True
            if op not in ("HIRE", "BUY_LAND"):
                market_quantity_active[row, slot] = True
                market_quantities[row, slot] = _quantity(raw, default=1)

    return {
        "unit_targets": unit_tokens,
        "unit_quantity_targets": unit_quantities,
        "unit_quantity_active": unit_quantity_active,
        "unit_active": unit_active,
        "market_targets": market_tokens,
        "market_quantity_targets": market_quantities,
        "market_quantity_active": market_quantity_active,
        "market_order_active": market_order_active,
    }


def decode_structured_actions(
    unit_indices: np.ndarray,
    unit_quantities: np.ndarray,
    market_indices: np.ndarray,
    market_quantities: np.ndarray,
    observations: Sequence[Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row, observation in enumerate(observations):
        player = int(_get(observation, "player", 0))
        farms = list(_get(observation, "farms", []) or [])
        hand_count = len(_get(farms[player], "hands", []) or []) if player < len(farms) else 0

        def unit_action(slot: int) -> list[Any]:
            op, item = UNIT_ACTIONS[int(unit_indices[row, slot])]
            result: list[Any] = [op] if item is None else [op, item]
            if op in ("PICKUP", "PLACE"):
                result.append(max(1, int(unit_quantities[row, slot])))
            return result

        market: list[list[Any]] = []
        for slot in range(MAX_MARKET_ORDERS):
            op, item = MARKET_TOKENS[int(market_indices[row, slot])]
            if op == "NONE":
                continue
            if item is None:
                market.append([op])
            else:
                market.append([op, item, max(1, int(market_quantities[row, slot]))])
        actions.append(
            {
                "farmer": unit_action(0),
                "hands": [unit_action(slot + 1) for slot in range(min(hand_count, MAX_UNITS - 1))],
                "market": market,
            }
        )
    return actions


class StructuredKaggriculturePolicy(nn.Module):
    """Step-aware actor/value model used for replay BC and later PPO."""

    def __init__(self, hidden_size: int = 768, *, route_prior: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.route_prior = route_prior
        self.step_embedding = nn.Embedding(720, 128)
        self.seat_embedding = nn.Embedding(2, 32)
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM + 160, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.unit_slot_embedding = nn.Embedding(MAX_UNITS, 32)
        unit_width = hidden_size + 35
        self.unit_token_head = nn.Sequential(
            nn.Linear(unit_width, 256), nn.SiLU(), nn.Linear(256, len(UNIT_ACTIONS))
        )
        self.unit_quantity_head = nn.Sequential(
            nn.Linear(unit_width, 256), nn.SiLU(), nn.Linear(256, QUANTITY_CLASSES)
        )
        self.market_slot_embedding = nn.Embedding(MAX_MARKET_ORDERS, 32)
        market_width = hidden_size + 32
        self.market_token_head = nn.Sequential(
            nn.Linear(market_width, 256), nn.SiLU(), nn.Linear(256, len(MARKET_TOKENS))
        )
        self.market_quantity_head = nn.Sequential(
            nn.Linear(market_width, 256), nn.SiLU(), nn.Linear(256, QUANTITY_CLASSES)
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.SiLU(), nn.Linear(256, 1))
        if route_prior:
            # Strong Kaggriculture agents follow a mostly deterministic 720-step route.
            # These zero-initialized tables learn that route directly; the observation
            # trunk remains a residual that handles weeds, market changes, and recovery.
            self.unit_route_logits = nn.Parameter(
                torch.zeros(720, 2, MAX_UNITS, len(UNIT_ACTIONS))
            )
            self.unit_quantity_route_logits = nn.Parameter(
                torch.zeros(720, 2, MAX_UNITS, QUANTITY_CLASSES)
            )
            self.market_route_logits = nn.Parameter(
                torch.zeros(720, 2, MAX_MARKET_ORDERS, len(MARKET_TOKENS))
            )
            self.market_quantity_route_logits = nn.Parameter(
                torch.zeros(720, 2, MAX_MARKET_ORDERS, QUANTITY_CLASSES)
            )

    def forward(
        self, features: torch.Tensor, unit_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step = torch.clamp(torch.round(features[:, 0].float() * 720.0).long(), 0, 719)
        seat = torch.clamp(torch.round(features[:, 3].float()).long(), 0, 1)
        latent = self.trunk(
            torch.cat((features, self.step_embedding(step), self.seat_embedding(seat)), dim=-1)
        )
        batch_size = latent.shape[0]

        unit_slots = self.unit_slot_embedding(
            torch.arange(MAX_UNITS, device=latent.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        unit_latent = latent.unsqueeze(1).expand(-1, MAX_UNITS, -1)
        unit_input = torch.cat((unit_latent, unit_slots, unit_context), dim=-1)

        market_slots = self.market_slot_embedding(
            torch.arange(MAX_MARKET_ORDERS, device=latent.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        market_latent = latent.unsqueeze(1).expand(-1, MAX_MARKET_ORDERS, -1)
        market_input = torch.cat((market_latent, market_slots), dim=-1)
        unit_logits = self.unit_token_head(unit_input)
        unit_quantity_logits = self.unit_quantity_head(unit_input)
        market_logits = self.market_token_head(market_input)
        market_quantity_logits = self.market_quantity_head(market_input)
        if self.route_prior:
            unit_logits = unit_logits + self.unit_route_logits[step, seat]
            unit_quantity_logits = (
                unit_quantity_logits + self.unit_quantity_route_logits[step, seat]
            )
            market_logits = market_logits + self.market_route_logits[step, seat]
            market_quantity_logits = (
                market_quantity_logits + self.market_quantity_route_logits[step, seat]
            )
        return (
            unit_logits,
            unit_quantity_logits,
            market_logits,
            market_quantity_logits,
            self.value_head(latent).squeeze(-1),
        )


def policy_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device | str
) -> StructuredKaggriculturePolicy:
    """Construct a PolicyV2 model while preserving old checkpoint compatibility."""

    state_dict = checkpoint["model"]
    route_prior = bool(checkpoint.get("route_prior", "unit_route_logits" in state_dict))
    model = StructuredKaggriculturePolicy(
        int(checkpoint["hidden_size"]), route_prior=route_prior
    ).to(device)
    model.load_state_dict(state_dict)
    return model


@dataclass
class StructuredPolicyBatch:
    actions: list[dict[str, Any]]
    unit_indices: torch.Tensor
    unit_quantities: torch.Tensor
    market_indices: torch.Tensor
    market_quantities: torch.Tensor
    values: torch.Tensor


@torch.no_grad()
def structured_policy_batch(
    model: StructuredKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    *,
    deterministic: bool = True,
    mask_unit_actions: bool = True,
) -> StructuredPolicyBatch:
    features_np, unit_context_np, _ = encode_batch(observations)
    features = torch.as_tensor(features_np, device=device)
    unit_context = torch.as_tensor(unit_context_np, device=device)
    unit_logits, unit_quantity_logits, market_logits, market_quantity_logits, values = model(
        features, unit_context
    )
    if mask_unit_actions:
        unit_masks_np, _ = action_masks(observations)
        unit_masks = torch.as_tensor(unit_masks_np, device=device)
        unit_logits = unit_logits.masked_fill(~unit_masks, torch.finfo(unit_logits.dtype).min)

    if deterministic:
        unit_indices = unit_logits.argmax(dim=-1)
        unit_quantities = unit_quantity_logits.argmax(dim=-1)
        market_indices = market_logits.argmax(dim=-1)
        market_quantities = market_quantity_logits.argmax(dim=-1)
    else:
        unit_indices = torch.distributions.Categorical(logits=unit_logits).sample()
        unit_quantities = torch.distributions.Categorical(logits=unit_quantity_logits).sample()
        market_indices = torch.distributions.Categorical(logits=market_logits).sample()
        market_quantities = torch.distributions.Categorical(logits=market_quantity_logits).sample()

    actions = decode_structured_actions(
        unit_indices.cpu().numpy(),
        unit_quantities.cpu().numpy(),
        market_indices.cpu().numpy(),
        market_quantities.cpu().numpy(),
        observations,
    )
    return StructuredPolicyBatch(
        actions,
        unit_indices,
        unit_quantities,
        market_indices,
        market_quantities,
        values,
    )
