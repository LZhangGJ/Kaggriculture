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
    FARM_FEATURES,
    FEATURE_DIM,
    ITEMS,
    MAX_UNITS,
    PRODUCTS,
    SHOP_NAMES,
    TILE_FEATURES,
    UNIT_CONTEXT_BASE_DIM,
    UNIT_CONTEXT_INVENTORY_DIM,
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

FARM_FEATURE_OFFSET = 4 + 2 * len(PRODUCTS) + len(SHOP_NAMES)


def canonicalize_seat_features(features: torch.Tensor) -> torch.Tensor:
    """Put the acting player's public farm first and remove the absolute seat bit."""

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected [batch, {FEATURE_DIM}] features, got {tuple(features.shape)}")
    farm0_start = FARM_FEATURE_OFFSET
    farm1_start = farm0_start + FARM_FEATURES
    suffix_start = farm1_start + FARM_FEATURES
    seat1 = features[:, 3:4] >= 0.5

    prefix = torch.cat((features[:, :3], torch.zeros_like(features[:, 3:4]), features[:, 4:farm0_start]), dim=1)
    farm0 = features[:, farm0_start:farm1_start]
    farm1 = features[:, farm1_start:suffix_start]
    own_farm = torch.where(seat1, farm1, farm0)
    opponent_farm = torch.where(seat1, farm0, farm1)
    return torch.cat((prefix, own_farm, opponent_farm, features[:, suffix_start:]), dim=1)


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

    def __init__(
        self,
        hidden_size: int = 768,
        *,
        route_prior: bool = False,
        canonical_seat: bool = False,
        autoregressive_market: bool = False,
        unit_inventory_context: bool = False,
        contextual_unit_inventory: bool = False,
        contextual_unit_local: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.route_prior = route_prior
        self.canonical_seat = canonical_seat
        self.autoregressive_market = autoregressive_market
        self.contextual_unit_inventory = contextual_unit_inventory
        self.contextual_unit_local = contextual_unit_local
        self.unit_inventory_context = (
            unit_inventory_context
            or contextual_unit_inventory
            or contextual_unit_local
        )
        self.unit_context_dim = (
            UNIT_CONTEXT_INVENTORY_DIM
            if self.unit_inventory_context
            else UNIT_CONTEXT_BASE_DIM
        )
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
        unit_width = hidden_size + 32 + UNIT_CONTEXT_BASE_DIM
        self.unit_token_head = nn.Sequential(
            nn.Linear(unit_width, 256), nn.SiLU(), nn.Linear(256, len(UNIT_ACTIONS))
        )
        self.unit_quantity_head = nn.Sequential(
            nn.Linear(unit_width, 256), nn.SiLU(), nn.Linear(256, QUANTITY_CLASSES)
        )
        if self.unit_inventory_context:
            inventory_dim = UNIT_CONTEXT_INVENTORY_DIM - UNIT_CONTEXT_BASE_DIM
            self.unit_inventory_token_residual = nn.Sequential(
                nn.Linear(inventory_dim, 64, bias=False),
                nn.SiLU(),
                nn.Linear(64, len(UNIT_ACTIONS), bias=False),
            )
            self.unit_inventory_quantity_residual = nn.Sequential(
                nn.Linear(inventory_dim, 64, bias=False),
                nn.SiLU(),
                nn.Linear(64, QUANTITY_CLASSES, bias=False),
            )
            nn.init.zeros_(self.unit_inventory_token_residual[-1].weight)
            nn.init.zeros_(self.unit_inventory_quantity_residual[-1].weight)
        if contextual_unit_inventory:
            inventory_dim = UNIT_CONTEXT_INVENTORY_DIM - UNIT_CONTEXT_BASE_DIM
            self.unit_contextual_inventory_embedding = nn.Linear(
                inventory_dim,
                64,
                bias=False,
            )
            self.unit_contextual_inventory_trunk = nn.Sequential(
                nn.Linear(unit_width + 64, 256),
                nn.SiLU(),
                nn.LayerNorm(256),
            )
            self.unit_contextual_inventory_token_head = nn.Linear(
                256,
                len(UNIT_ACTIONS),
                bias=False,
            )
            self.unit_contextual_inventory_quantity_head = nn.Linear(
                256,
                QUANTITY_CLASSES,
                bias=False,
            )
            nn.init.zeros_(self.unit_contextual_inventory_token_head.weight)
            nn.init.zeros_(self.unit_contextual_inventory_quantity_head.weight)
        if contextual_unit_local:
            inventory_dim = UNIT_CONTEXT_INVENTORY_DIM - UNIT_CONTEXT_BASE_DIM
            local_width = unit_width + inventory_dim + TILE_FEATURES
            self.unit_contextual_local_trunk = nn.Sequential(
                nn.Linear(local_width, 256),
                nn.SiLU(),
                nn.LayerNorm(256),
            )
            self.unit_contextual_local_token_head = nn.Linear(
                256,
                len(UNIT_ACTIONS),
                bias=False,
            )
            self.unit_contextual_local_quantity_head = nn.Linear(
                256,
                QUANTITY_CLASSES,
                bias=False,
            )
            nn.init.zeros_(self.unit_contextual_local_token_head.weight)
            nn.init.zeros_(self.unit_contextual_local_quantity_head.weight)
        self.market_slot_embedding = nn.Embedding(MAX_MARKET_ORDERS, 32)
        market_width = hidden_size + 32
        self.market_token_head = nn.Sequential(
            nn.Linear(market_width, 256), nn.SiLU(), nn.Linear(256, len(MARKET_TOKENS))
        )
        self.market_quantity_head = nn.Sequential(
            nn.Linear(market_width, 256), nn.SiLU(), nn.Linear(256, QUANTITY_CLASSES)
        )
        if autoregressive_market:
            self.market_ar_token_embedding = nn.Embedding(len(MARKET_TOKENS) + 1, 32)
            self.market_ar_quantity_embedding = nn.Embedding(QUANTITY_CLASSES, 16)
            self.market_ar_state = nn.GRUCell(hidden_size + 80, 256)
            self.market_ar_token_head = nn.Linear(256, len(MARKET_TOKENS))
            self.market_ar_quantity_head = nn.Sequential(
                nn.Linear(288, 256), nn.SiLU(), nn.Linear(256, QUANTITY_CLASSES)
            )
            self.market_ar_initial = nn.Linear(hidden_size, 256)
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
        self,
        features: torch.Tensor,
        unit_context: torch.Tensor,
        market_teacher_tokens: torch.Tensor | None = None,
        market_teacher_quantities: torch.Tensor | None = None,
        *,
        sample_market: bool = False,
        return_market_choices: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        step = torch.clamp(torch.round(features[:, 0].float() * 720.0).long(), 0, 719)
        seat = torch.clamp(torch.round(features[:, 3].float()).long(), 0, 1)
        seat_context = self.seat_embedding(seat)
        if self.canonical_seat:
            features = canonicalize_seat_features(features)
            # Keep the trunk shape checkpoint-compatible while preventing the
            # absolute seat embedding from becoming a shortcut.
            seat_context = torch.zeros_like(seat_context)
        latent = self.trunk(
            torch.cat((features, self.step_embedding(step), seat_context), dim=-1)
        )
        batch_size = latent.shape[0]

        unit_slots = self.unit_slot_embedding(
            torch.arange(MAX_UNITS, device=latent.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        unit_latent = latent.unsqueeze(1).expand(-1, MAX_UNITS, -1)
        unit_input = torch.cat(
            (
                unit_latent,
                unit_slots,
                unit_context[..., :UNIT_CONTEXT_BASE_DIM],
            ),
            dim=-1,
        )

        unit_logits = self.unit_token_head(unit_input)
        unit_quantity_logits = self.unit_quantity_head(unit_input)
        if self.unit_inventory_context:
            # Per-unit inventories are stored on the same /100 scale as aggregate
            # inventory features.  Most carried stacks are only 1-10 items, so a
            # local rescale keeps the residual signal numerically useful while the
            # zero-inventory invariant remains exact.
            inventory_context = unit_context[..., UNIT_CONTEXT_BASE_DIM:] * 10.0
            unit_logits = unit_logits + self.unit_inventory_token_residual(
                inventory_context
            )
            unit_quantity_logits = (
                unit_quantity_logits
                + self.unit_inventory_quantity_residual(inventory_context)
            )
            if self.contextual_unit_inventory:
                inventory_gate = (inventory_context.abs().sum(dim=-1, keepdim=True) > 0).to(
                    unit_logits.dtype
                )
                contextual_input = torch.cat(
                    (
                        unit_input,
                        self.unit_contextual_inventory_embedding(inventory_context),
                    ),
                    dim=-1,
                )
                contextual_hidden = self.unit_contextual_inventory_trunk(
                    contextual_input
                )
                unit_logits = unit_logits + inventory_gate * (
                    self.unit_contextual_inventory_token_head(contextual_hidden)
                )
                unit_quantity_logits = unit_quantity_logits + inventory_gate * (
                    self.unit_contextual_inventory_quantity_head(contextual_hidden)
                )
            if self.contextual_unit_local:
                tile_start = FARM_FEATURE_OFFSET + 9
                own_tiles = features[
                    :, tile_start : tile_start + 100 * TILE_FEATURES
                ].view(batch_size, 100, TILE_FEATURES)
                unit_x = torch.clamp(
                    torch.round(unit_context[..., 0] * 9.0).long(),
                    0,
                    9,
                )
                unit_y = torch.clamp(
                    torch.round(unit_context[..., 1] * 9.0).long(),
                    0,
                    9,
                )
                tile_indices = unit_y * 10 + unit_x
                local_tiles = torch.gather(
                    own_tiles,
                    1,
                    tile_indices.unsqueeze(-1).expand(-1, -1, TILE_FEATURES),
                )
                local_hidden = self.unit_contextual_local_trunk(
                    torch.cat((unit_input, inventory_context, local_tiles), dim=-1)
                )
                active_gate = unit_context[..., 2:3].to(unit_logits.dtype)
                unit_logits = unit_logits + active_gate * (
                    self.unit_contextual_local_token_head(local_hidden)
                )
                unit_quantity_logits = unit_quantity_logits + active_gate * (
                    self.unit_contextual_local_quantity_head(local_hidden)
                )
        market_slots = self.market_slot_embedding(
            torch.arange(MAX_MARKET_ORDERS, device=latent.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        if self.autoregressive_market:
            (
                market_logits,
                market_quantity_logits,
                market_choices,
                market_quantity_choices,
            ) = self._autoregressive_market(
                latent,
                market_slots,
                market_teacher_tokens,
                market_teacher_quantities,
                sample=sample_market,
            )
        else:
            if sample_market or return_market_choices:
                raise ValueError("market sampling choices require autoregressive_market")
            market_latent = latent.unsqueeze(1).expand(-1, MAX_MARKET_ORDERS, -1)
            market_input = torch.cat((market_latent, market_slots), dim=-1)
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
        outputs = (
            unit_logits,
            unit_quantity_logits,
            market_logits,
            market_quantity_logits,
            self.value_head(latent).squeeze(-1),
        )
        if return_market_choices:
            return (*outputs, market_choices, market_quantity_choices)
        return outputs

    def _autoregressive_market(
        self,
        latent: torch.Tensor,
        market_slots: torch.Tensor,
        teacher_tokens: torch.Tensor | None,
        teacher_quantities: torch.Tensor | None,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (teacher_tokens is None) != (teacher_quantities is None):
            raise ValueError("market teacher tokens and quantities must be provided together")
        if sample and teacher_tokens is not None:
            raise ValueError("cannot sample market actions while teacher forcing")
        batch_size = latent.shape[0]
        state = torch.tanh(self.market_ar_initial(latent))
        previous_token = torch.full(
            (batch_size,), len(MARKET_TOKENS), device=latent.device, dtype=torch.long
        )
        previous_quantity = torch.zeros(
            batch_size, device=latent.device, dtype=torch.long
        )
        token_outputs = []
        quantity_outputs = []
        token_choices = []
        quantity_choices = []
        for slot in range(MAX_MARKET_ORDERS):
            recurrent_input = torch.cat(
                (
                    latent,
                    market_slots[:, slot],
                    self.market_ar_token_embedding(previous_token),
                    self.market_ar_quantity_embedding(previous_quantity),
                ),
                dim=-1,
            )
            state = self.market_ar_state(recurrent_input, state)
            token_logits = self.market_ar_token_head(state)
            current_token = (
                teacher_tokens[:, slot].long()
                if teacher_tokens is not None
                else (
                    torch.distributions.Categorical(logits=token_logits).sample()
                    if sample
                    else token_logits.argmax(dim=-1)
                )
            )
            quantity_logits = self.market_ar_quantity_head(
                torch.cat((state, self.market_ar_token_embedding(current_token)), dim=-1)
            )
            current_quantity = (
                teacher_quantities[:, slot].long()
                if teacher_quantities is not None
                else (
                    torch.distributions.Categorical(logits=quantity_logits).sample()
                    if sample
                    else quantity_logits.argmax(dim=-1)
                )
            )
            token_outputs.append(token_logits)
            quantity_outputs.append(quantity_logits)
            token_choices.append(current_token)
            quantity_choices.append(current_quantity)
            previous_token = current_token
            previous_quantity = current_quantity.clamp(0, MAX_QUANTITY)
        return (
            torch.stack(token_outputs, dim=1),
            torch.stack(quantity_outputs, dim=1),
            torch.stack(token_choices, dim=1),
            torch.stack(quantity_choices, dim=1),
        )


def policy_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device | str
) -> StructuredKaggriculturePolicy:
    """Construct a PolicyV2 model while preserving old checkpoint compatibility."""

    state_dict = checkpoint["model"]
    route_prior = bool(checkpoint.get("route_prior", "unit_route_logits" in state_dict))
    canonical_seat = bool(checkpoint.get("canonical_seat", False))
    autoregressive_market = bool(checkpoint.get("autoregressive_market", False))
    unit_inventory_context = bool(checkpoint.get("unit_inventory_context", False))
    contextual_unit_inventory = bool(
        checkpoint.get("contextual_unit_inventory", False)
    )
    contextual_unit_local = bool(checkpoint.get("contextual_unit_local", False))
    model = StructuredKaggriculturePolicy(
        int(checkpoint["hidden_size"]),
        route_prior=route_prior,
        canonical_seat=canonical_seat,
        autoregressive_market=autoregressive_market,
        unit_inventory_context=unit_inventory_context,
        contextual_unit_inventory=contextual_unit_inventory,
        contextual_unit_local=contextual_unit_local,
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
    features_np, unit_context_np, _ = encode_batch(
        observations,
        include_unit_inventory=model.unit_inventory_context,
    )
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
