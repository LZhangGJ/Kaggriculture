"""Structured dual-board PyTorch policy for Kaggriculture.

The original GPU policy flattens both farms into one feature vector.  Board V1
keeps the two 10 x 10 farms spatial, applies one shared CNN to both sides, and
combines those embeddings with public unit entities and global economy/market
features.  Legal action generation and decoding continue to use gpu_policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .gpu_policy import (
    ANIMALS,
    CROPS,
    ITEMS,
    MARKET_ACTIONS,
    MAX_HANDS,
    MAX_UNITS,
    PRODUCTS,
    SHOP_NAMES,
    TILE_FEATURES,
    UNIT_ACTIONS,
    PolicyBatch,
    _get,
    _mapping,
    _masked,
    _tile_features,
    action_masks,
    decode_actions,
)


BOARD_SIZE = 10
BOARD_UNIT_CHANNELS = 2  # main farmer and hired-hand occupancy
BOARD_GEOMETRY_CHANNELS = 1  # normalized Manhattan distance to shed access
BOARD_CHANNELS = TILE_FEATURES + BOARD_UNIT_CHANNELS + BOARD_GEOMETRY_CHANNELS

UNIT_PRIVATE_ITEMS = len(ITEMS)
UNIT_FEATURES = 2 + 1 + 1 + UNIT_PRIVATE_ITEMS + 1

GLOBAL_FEATURES = (
    5  # step, day, hour, remaining season, acting seat
    + 2  # canonical self/opponent money
    + 2  # active hand count
    + 2  # hires today
    + 8  # unlocked quadrant flags for both farms
    + len(ITEMS)  # own private shed
    + len(CROPS)  # own private seeds
    + len(ITEMS)  # own carried inventory total
    + len(PRODUCTS)  # shared market inventory
    + len(PRODUCTS)  # shared market prices
    + 8 * len(SHOP_NAMES)  # ordered shop slots, including duplicates
    + 1  # active shop count
)

_QUADRANTS = ("NW", "NE", "SW", "SE")
_BASE_PRICES = np.asarray((25, 35, 60, 120, 250, 50, 160, 200, 100), dtype=np.float32)
_SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))


def _canonical_farms(observation: Any) -> tuple[Any, Any]:
    farms = list(_get(observation, "farms", []) or [])
    while len(farms) < 2:
        farms.append({})
    player = int(_get(observation, "player", 0) or 0)
    player = 0 if player not in (0, 1) else player
    return farms[player], farms[1 - player]


def _positions(farm: Any) -> list[list[int]]:
    farmer = list(_get(farm, "farmer", [0, 0]) or [0, 0])
    hands = [list(value) for value in list(_get(farm, "hands", []) or [])[:MAX_HANDS]]
    return [farmer, *hands]


def _distance_plane() -> np.ndarray:
    y, x = np.indices((BOARD_SIZE, BOARD_SIZE))
    distance = np.full((BOARD_SIZE, BOARD_SIZE), 2 * (BOARD_SIZE - 1), dtype=np.float32)
    for shed_x, shed_y in _SHED_ACCESS:
        distance = np.minimum(distance, np.abs(x - shed_x) + np.abs(y - shed_y))
    return distance / float(2 * (BOARD_SIZE - 1))


_DISTANCE_PLANE = _distance_plane()


def _encode_board(farm: Any, day: int) -> np.ndarray:
    board = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tiles = list(_get(farm, "tiles", []) or [])
    for y in range(BOARD_SIZE):
        row = list(tiles[y]) if y < len(tiles) else []
        for x in range(BOARD_SIZE):
            tile = row[x] if x < len(row) else "LOCKED"
            board[:TILE_FEATURES, y, x] = _tile_features(tile, day)

    positions = _positions(farm)
    if positions:
        x, y = positions[0]
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            board[TILE_FEATURES, y, x] = 1.0
    for x, y in positions[1:]:
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            board[TILE_FEATURES + 1, y, x] += 1.0 / float(MAX_HANDS)
    board[-1] = _DISTANCE_PLANE
    return board


def _encode_units(
    farm: Any,
    inventories: Sequence[Any] | None,
    *,
    private_known: bool,
) -> tuple[np.ndarray, np.ndarray]:
    units = np.zeros((MAX_UNITS, UNIT_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_UNITS, dtype=np.bool_)
    positions = _positions(farm)
    inventories = list(inventories or [])
    denominator = float(BOARD_SIZE - 1)
    for index, position in enumerate(positions[:MAX_UNITS]):
        x, y = position
        units[index, :4] = (
            float(x) / denominator,
            float(y) / denominator,
            1.0,
            float(index == 0),
        )
        if private_known and index < len(inventories):
            inventory = _mapping(inventories[index])
            units[index, 4 : 4 + len(ITEMS)] = [
                min(max(float(inventory.get(item, 0)), 0.0) / 100.0, 2.0)
                for item in ITEMS
            ]
        units[index, -1] = float(private_known)
        mask[index] = True
    return units, mask


def _global_features(observation: Any, own: Any, opponent: Any) -> np.ndarray:
    player = int(_get(observation, "player", 0) or 0)
    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    hour = int(_get(observation, "hour", step % 24) or 0)
    values: list[float] = [
        step / 719.0,
        day / 29.0,
        hour / 23.0,
        (719 - min(max(step, 0), 719)) / 719.0,
        float(player),
    ]
    values.extend(
        np.log1p(max(float(_get(farm, "money", 0)), 0.0)) / np.log1p(2_000_000.0)
        for farm in (own, opponent)
    )
    values.extend(
        min(len(_get(farm, "hands", []) or []) / float(MAX_HANDS), 1.0)
        for farm in (own, opponent)
    )
    values.extend(
        min(max(float(_get(farm, "hires_today", 0)), 0.0) / float(MAX_HANDS), 1.0)
        for farm in (own, opponent)
    )
    for farm in (own, opponent):
        unlocked = set(_get(farm, "unlocked_quadrants", []) or [])
        values.extend(float(quadrant in unlocked) for quadrant in _QUADRANTS)

    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))
    inventories = list(_get(private, "inventories", []) or [])
    carried = {item: 0 for item in ITEMS}
    for inventory in inventories:
        mapping = _mapping(inventory)
        for item in ITEMS:
            carried[item] += int(mapping.get(item, 0))
    values.extend(min(max(float(shed.get(item, 0)), 0.0) / 100.0, 2.0) for item in ITEMS)
    values.extend(min(max(float(seeds.get(crop, 0)), 0.0) / 100.0, 2.0) for crop in CROPS)
    values.extend(min(max(float(carried[item]), 0.0) / 100.0, 2.0) for item in ITEMS)

    market = _get(observation, "market", {}) or {}
    inventory = _mapping(_get(market, "inventory", {}))
    prices = _mapping(_get(market, "prices", {}))
    values.extend((float(inventory.get(item, 10_000)) - 10_000.0) / 10_000.0 for item in PRODUCTS)
    values.extend(float(prices.get(item, 0)) / float(_BASE_PRICES[index]) for index, item in enumerate(PRODUCTS))

    town = _get(observation, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])[:8]
    shop_index = {name: index for index, name in enumerate(SHOP_NAMES)}
    for slot in range(8):
        one_hot = [0.0] * len(SHOP_NAMES)
        if slot < len(shops) and shops[slot] in shop_index:
            one_hot[shop_index[shops[slot]]] = 1.0
        values.extend(one_hot)
    values.append(len(shops) / 8.0)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (GLOBAL_FEATURES,):
        raise AssertionError(
            f"global feature width drifted: expected {GLOBAL_FEATURES}, got {result.shape}"
        )
    return result


def encode_board_observation(
    observation: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode one actor-visible observation in canonical self/opponent order."""

    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    own, opponent = _canonical_farms(observation)
    private = _get(observation, "private", {}) or {}
    own_inventories = list(_get(private, "inventories", []) or [])
    own_units, own_mask = _encode_units(
        own, own_inventories, private_known=True
    )
    opponent_units, opponent_mask = _encode_units(
        opponent, None, private_known=False
    )
    boards = np.stack((_encode_board(own, day), _encode_board(opponent, day)))
    units = np.stack((own_units, opponent_units))
    masks = np.stack((own_mask, opponent_mask))
    return boards, _global_features(observation, own, opponent), units, masks


def encode_board_batch(
    observations: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoded = [encode_board_observation(observation) for observation in observations]
    return tuple(
        np.stack([row[index] for row in encoded]) for index in range(4)
    )  # type: ignore[return-value]


class BoardKaggriculturePolicy(nn.Module):
    """Shared-CNN actor critic using factorized unit and market heads."""

    def __init__(self, hidden_size: int = 384, board_width: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.board_encoder = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, board_width // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(board_width // 2, board_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(board_width, board_width, 3, padding=1),
            nn.SiLU(),
        )
        self.board_projection = nn.Sequential(
            nn.Linear(2 * board_width, 128), nn.SiLU()
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(UNIT_FEATURES, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.unit_projection = nn.Sequential(nn.Linear(128, 64), nn.SiLU())
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.SiLU(),
        )
        fusion_width = 4 * 128 + 2 * 64 + 128
        self.trunk = nn.Sequential(
            nn.Linear(fusion_width, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.slot_embedding = nn.Embedding(MAX_UNITS, 32)
        self.unit_head = nn.Sequential(
            nn.Linear(hidden_size + 64 + 32, 256),
            nn.SiLU(),
            nn.Linear(256, len(UNIT_ACTIONS)),
        )
        self.market_head = nn.Linear(hidden_size, len(MARKET_ACTIONS))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, 1)
        )

    def _boards(self, boards: torch.Tensor) -> torch.Tensor:
        batch = boards.shape[0]
        encoded = self.board_encoder(boards.flatten(0, 1))
        pooled = torch.cat(
            (encoded.mean(dim=(-2, -1)), encoded.amax(dim=(-2, -1))), dim=-1
        )
        return self.board_projection(pooled).view(batch, 2, -1)

    def _units(
        self, units: torch.Tensor, unit_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.unit_encoder(units)
        weights = unit_mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        maximum = encoded.masked_fill(~unit_mask.unsqueeze(-1), -torch.inf).amax(dim=2)
        maximum = torch.where(unit_mask.any(dim=2, keepdim=True), maximum, 0.0)
        summary = self.unit_projection(torch.cat((mean, maximum), dim=-1))
        return summary, encoded

    def forward(
        self,
        boards: torch.Tensor,
        global_features: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self._boards(boards)
        unit_summary, unit_tokens = self._units(unit_features, unit_mask)
        global_hidden = self.global_encoder(global_features)
        fused = torch.cat(
            (
                board[:, 0],
                board[:, 1],
                board[:, 0] - board[:, 1],
                (board[:, 0] - board[:, 1]).abs(),
                unit_summary[:, 0],
                unit_summary[:, 1],
                global_hidden,
            ),
            dim=-1,
        )
        latent = self.trunk(fused)
        slots = self.slot_embedding(
            torch.arange(MAX_UNITS, device=latent.device)
        ).unsqueeze(0).expand(latent.shape[0], -1, -1)
        expanded = latent.unsqueeze(1).expand(-1, MAX_UNITS, -1)
        unit_logits = self.unit_head(
            torch.cat((expanded, unit_tokens[:, 0], slots), dim=-1)
        )
        return (
            unit_logits,
            self.market_head(latent),
            self.value_head(latent).squeeze(-1),
        )


@torch.no_grad()
def board_policy_batch(
    model: BoardKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    *,
    deterministic: bool = False,
) -> PolicyBatch:
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
    unit_logits = _masked(unit_logits, unit_masks)
    market_logits = _masked(market_logits, market_masks)
    if deterministic:
        unit_indices = unit_logits.argmax(dim=-1)
        market_indices = market_logits.argmax(dim=-1)
    else:
        unit_indices = torch.distributions.Categorical(logits=unit_logits).sample()
        market_indices = torch.distributions.Categorical(logits=market_logits).sample()
    actions = decode_actions(
        unit_indices.cpu().numpy(), market_indices.cpu().numpy(), observations
    )
    return PolicyBatch(actions, unit_indices, market_indices, values)


@dataclass(frozen=True)
class BoardBatchShapes:
    boards: tuple[int, int, int, int]
    global_features: tuple[int]
    units: tuple[int, int, int]
    unit_mask: tuple[int, int]


BOARD_BATCH_SHAPES = BoardBatchShapes(
    boards=(2, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE),
    global_features=(GLOBAL_FEATURES,),
    units=(2, MAX_UNITS, UNIT_FEATURES),
    unit_mask=(2, MAX_UNITS),
)
