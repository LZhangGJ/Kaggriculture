"""Observation-to-entity-token conversion for Kaggriculture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .constants import (
    CONT_DIM,
    CROPS,
    ITEMS,
    ITEM_TO_ID,
    NUM_POSITION_IDS,
    OWNER_OPPONENT,
    OWNER_SELF,
    OWNER_SHARED,
    PRODUCTS,
    TOKEN_FARM,
    TOKEN_GLOBAL,
    TOKEN_MARKET,
    TOKEN_TILE,
    TOKEN_TOWN,
    TOKEN_UNIT,
)
from .structures import EncodedObservation


SHOP_NAMES = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
SHOP_TO_ID = {name: index + 1 for index, name in enumerate(SHOP_NAMES)}

TILE_EMPTY = 1
TILE_LOCKED = 2
TILE_WEED = 3
TILE_PLANT = 4
TILE_COOP = 5
TILE_PASTURE = 6


def _get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position_id(value: Any) -> int:
    try:
        return min(NUM_POSITION_IDS - 1, max(1, int(value) + 1))
    except (TypeError, ValueError):
        return 0


def _ratio(value: Any, scale: float) -> float:
    return max(-10.0, min(10.0, _number(value) / scale))


class ObservationTokenizer:
    """Encode public board state plus the acting player's private state.

    The output intentionally keeps all 200 board tiles: the opposing board is
    public and is essential to the competitive market signal.  Private shed and
    unit inventories are added only for the acting player.
    """

    def encode(self, observation: dict[str, Any] | Any) -> EncodedObservation:
        obs = observation
        player = int(_get(obs, "player", 0) or 0)
        farms = list(_get(obs, "farms", []) or [])
        private = _mapping(_get(obs, "private", {}))
        market = _mapping(_get(obs, "market", {}))
        town = _mapping(_get(obs, "town", {}))
        day = _number(_get(obs, "day", 0))
        hour = _number(_get(obs, "hour", 0))
        # The competition horizon is fixed at 720 turns.  Day/hour contain
        # this information indirectly, but terminal liquidation decisions are
        # easier to learn from an explicit monotone remaining-time signal.
        step = _number(_get(obs, "step", day * 24 + hour))
        horizon_steps = 720.0
        step_fraction = max(0.0, min(1.0, step / horizon_steps))
        remaining_turn_fraction = max(0.0, min(1.0, (horizon_steps - 1.0 - step) / horizon_steps))

        continuous: list[list[float]] = []
        token_type: list[int] = []
        category_a: list[int] = []
        category_b: list[int] = []
        category_c: list[int] = []
        xs: list[int] = []
        ys: list[int] = []
        owners: list[int] = []
        own_unit_token_indices: list[int] = []

        own_farm = _mapping(farms[player]) if 0 <= player < len(farms) else {}
        opponent_money = sum(
            _number(_get(farm, "money", 0))
            for index, farm in enumerate(farms)
            if index != player
        )
        own_money = _number(_get(own_farm, "money", 0))

        def add(
            kind: int,
            values: Sequence[float] = (),
            *,
            cat_a: int = 0,
            cat_b: int = 0,
            cat_c: int = 0,
            x: int = 0,
            y: int = 0,
            owner: int = OWNER_SHARED,
        ) -> int:
            row = [0.0] * CONT_DIM
            for index, value in enumerate(values[:CONT_DIM]):
                row[index] = float(value)
            continuous.append(row)
            token_type.append(kind)
            category_a.append(cat_a)
            category_b.append(cat_b)
            category_c.append(cat_c)
            xs.append(x)
            ys.append(y)
            owners.append(owner)
            return len(continuous) - 1

        add(
            TOKEN_GLOBAL,
            (
                _ratio(day, 30),
                _ratio(hour, 24),
                _ratio(own_money, 10_000),
                _ratio(opponent_money, 10_000),
                _ratio(len(farms), 2),
                _ratio(len(town.get("unlocked_shops", [])), 8),
                step_fraction,
                remaining_turn_fraction,
            ),
        )

        inventories = list(private.get("inventories", []) or [])
        own_shed = _mapping(private.get("shed", {}))
        own_seeds = _mapping(private.get("seeds", {}))

        for farm_index, farm_value in enumerate(farms):
            farm = _mapping(farm_value)
            is_self = farm_index == player
            owner = OWNER_SELF if is_self else OWNER_OPPONENT
            tiles = list(farm.get("tiles", []) or [])
            board_size = max(1, len(tiles))
            hands = list(farm.get("hands", []) or [])
            unlocked = list(farm.get("unlocked_quadrants", []) or [])
            add(
                TOKEN_FARM,
                (
                    _ratio(farm.get("money", 0), 10_000),
                    _ratio(len(hands), 10),
                    _ratio(len(unlocked), 4),
                    _ratio(farm.get("hires_today", 0), 10),
                    _ratio(sum(1 for row in tiles for tile in row if tile is None), board_size * board_size),
                ),
                cat_a=min(31, len(unlocked) + 1),
                owner=owner,
            )

            positions = [farm.get("farmer", [0, 0]), *hands]
            for unit_index, position in enumerate(positions):
                px, py = _safe_position(position)
                inventory = _mapping(inventories[unit_index]) if is_self and unit_index < len(inventories) else {}
                unit_values = [
                    float(unit_index == 0),
                    _ratio(unit_index, 10),
                    _ratio(sum(_number(v) for v in inventory.values()), 100),
                    *[_ratio(inventory.get(item, 0), 20) for item in ITEMS[1:]],
                ]
                token_index = add(
                    TOKEN_UNIT,
                    unit_values,
                    cat_a=1 if unit_index == 0 else 2,
                    x=_position_id(px),
                    y=_position_id(py),
                    owner=owner,
                )
                if is_self:
                    own_unit_token_indices.append(token_index)

            for y, row_value in enumerate(tiles):
                row = list(row_value or [])
                for x, tile in enumerate(row):
                    tile_mapping = _mapping(tile)
                    kind, item_id, tile_values = _tile_features(tile, tile_mapping, day)
                    add(
                        TOKEN_TILE,
                        tile_values,
                        cat_a=kind,
                        cat_b=item_id,
                        cat_c=1 if tile != "LOCKED" else 2,
                        x=_position_id(x),
                        y=_position_id(y),
                        owner=owner,
                    )

        inventory = _mapping(market.get("inventory", {}))
        prices = _mapping(market.get("prices", {}))
        for item in PRODUCTS:
            add(
                TOKEN_MARKET,
                (
                    _ratio(inventory.get(item, 0), 10_000),
                    _ratio(prices.get(item, 0), 300),
                    _ratio(own_shed.get(item, 0), 100),
                    _ratio(own_seeds.get(item, 0) if item in CROPS else 0, 100),
                    _ratio(own_money, 10_000),
                ),
                cat_a=ITEM_TO_ID[item],
            )

        shop_counts = {name: 0 for name in SHOP_NAMES}
        for shop in town.get("unlocked_shops", []) or []:
            if shop in shop_counts:
                shop_counts[shop] += 1
        for shop, count in shop_counts.items():
            add(TOKEN_TOWN, (_ratio(count, 8),), cat_a=SHOP_TO_ID[shop])

        return EncodedObservation(
            continuous=torch.tensor(continuous, dtype=torch.float32),
            token_type=torch.tensor(token_type, dtype=torch.long),
            category_a=torch.tensor(category_a, dtype=torch.long),
            category_b=torch.tensor(category_b, dtype=torch.long),
            category_c=torch.tensor(category_c, dtype=torch.long),
            x=torch.tensor(xs, dtype=torch.long),
            y=torch.tensor(ys, dtype=torch.long),
            owner=torch.tensor(owners, dtype=torch.long),
            own_unit_token_indices=tuple(own_unit_token_indices),
        )


def _safe_position(position: Any) -> tuple[int, int]:
    try:
        return int(position[0]), int(position[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0, 0


def _tile_features(tile: Any, value: Mapping[str, Any], day: float) -> tuple[int, int, list[float]]:
    if tile is None:
        return TILE_EMPTY, 0, []
    if tile == "LOCKED":
        return TILE_LOCKED, 0, []
    kind = value.get("kind")
    if kind == "WEED":
        return TILE_WEED, 0, []
    if kind == "PLANT":
        crop = str(value.get("crop", ""))
        return (
            TILE_PLANT,
            ITEM_TO_ID.get(crop, 0),
            [
                _ratio(day - _number(value.get("planted_day", day)), 30),
                _ratio(value.get("yield_units", 0), 6),
                float(bool(value.get("watered_today", False))),
                _ratio(value.get("consecutive_unwatered", 0), 2),
                _ratio(_number(value.get("fertilized_until_day", -1)) - day, 3),
            ],
        )
    animal = value.get("animal")
    if animal:
        return (
            TILE_COOP if kind == "COOP" else TILE_PASTURE,
            ITEM_TO_ID.get(str(animal), 0),
            [
                _ratio(day - _number(value.get("placed_day", day)), 30),
                _ratio(value.get("yield_units", 0), 6),
                float(bool(value.get("fed_today", False))),
                _ratio(value.get("consecutive_unfed", 0), 2),
                float(bool(value.get("cared_today", False))),
                float(bool(value.get("fertilizer_available", False))),
                _ratio(value.get("pending_care_bonus", 0), 6),
            ],
        )
    if kind == "COOP":
        return TILE_COOP, 0, []
    if kind == "PASTURE":
        return TILE_PASTURE, 0, []
    return TILE_EMPTY, 0, []


def collate_encoded(
    encoded_list: Sequence[EncodedObservation], *, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[tuple[int, ...]]]:
    if not encoded_list:
        raise ValueError("Cannot collate an empty observation batch.")
    max_tokens = max(encoded.num_tokens for encoded in encoded_list)
    batch_size = len(encoded_list)
    continuous = torch.zeros((batch_size, max_tokens, CONT_DIM), dtype=torch.float32)
    categorical_names = ("token_type", "category_a", "category_b", "category_c", "x", "y", "owner")
    categorical = {
        name: torch.zeros((batch_size, max_tokens), dtype=torch.long) for name in categorical_names
    }
    padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool)
    unit_indices: list[tuple[int, ...]] = []
    for batch_index, encoded in enumerate(encoded_list):
        count = encoded.num_tokens
        continuous[batch_index, :count] = encoded.continuous
        for name in categorical_names:
            value = getattr(encoded, name)
            if value.shape[0] != count:
                raise ValueError(f"Encoded observation field {name} has a mismatched token count.")
            categorical[name][batch_index, :count] = value
        padding_mask[batch_index, :count] = False
        unit_indices.append(encoded.own_unit_token_indices)
    batch = {
        "continuous": continuous.to(device=device, non_blocking=True),
        **{name: value.to(device=device, non_blocking=True) for name, value in categorical.items()},
        "padding_mask": padding_mask.to(device=device, non_blocking=True),
    }
    return batch, unit_indices
