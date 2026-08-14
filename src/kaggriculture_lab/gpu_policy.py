"""GPU-batched policy scaffold for Kaggriculture self-play.

The official game transition stays on CPU.  Observations from many trusted fast
environments are encoded together, scored in one CUDA batch, and decoded back to
legal Kaggriculture actions.  The fixed action vocabulary deliberately covers the
full unit operation set and a useful one-order-per-turn market subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without gpu extra
    raise ImportError("Install the GPU extra with `pip install -e '.[gpu]'`") from exc


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = PRODUCTS + ANIMALS
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
MAX_HANDS = 16
MAX_UNITS = MAX_HANDS + 1

UNIT_ACTIONS: tuple[tuple[str, str | None], ...] = (
    *((op, None) for op in ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "DIG", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE")),
    *(("PLANT", crop) for crop in CROPS),
    *(("PICKUP", item) for item in ITEMS),
    *(("PLACE", item) for item in ITEMS),
)

_MARKET_QUANTITIES = (1, 4, 8, 16)
MARKET_ACTIONS: tuple[tuple[str, str | None, int], ...] = (
    ("NONE", None, 0),
    ("HIRE", None, 1),
    ("BUY_LAND", None, 1),
    *(("BUY_SEED", crop, quantity) for crop in CROPS for quantity in _MARKET_QUANTITIES),
    *(("BUY_PRODUCT", item, quantity) for item in ("WHEAT", "FERTILIZER") for quantity in _MARKET_QUANTITIES),
    *(("BUY_ANIMAL", animal, 1) for animal in ANIMALS),
    *(("SELL", item, 100) for item in PRODUCTS),
)

UNIT_INDEX = {action: index for index, action in enumerate(UNIT_ACTIONS)}
MARKET_INDEX = {action: index for index, action in enumerate(MARKET_ACTIONS)}

_TILE_KINDS = (
    "EMPTY",
    "LOCKED",
    "WEED",
    *CROPS,
    "COOP_EMPTY",
    "PASTURE_EMPTY",
    *ANIMALS,
    "OTHER",
)
_TILE_INDEX = {kind: index for index, kind in enumerate(_TILE_KINDS)}
TILE_FEATURES = len(_TILE_KINDS) + 10
FARM_FEATURES = 9 + 100 * TILE_FEATURES + MAX_UNITS * 3
FEATURE_DIM = 4 + len(PRODUCTS) * 2 + len(SHOP_NAMES) + 2 * FARM_FEATURES + len(ITEMS) * 2 + len(CROPS)

_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
_ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_LAND_PRICES = (1000, 2000, 4000)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _tile_kind(tile: Any) -> str:
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    if not isinstance(tile, Mapping):
        return "OTHER"
    kind = tile.get("kind")
    if kind == "WEED":
        return "WEED"
    if kind == "PLANT":
        return tile.get("crop", "OTHER")
    if kind in ("COOP", "PASTURE"):
        return tile.get("animal") or f"{kind}_EMPTY"
    return "OTHER"


def _tile_features(tile: Any, day: int) -> list[float]:
    values = [0.0] * TILE_FEATURES
    values[_TILE_INDEX[_tile_kind(tile)]] = 1.0
    if isinstance(tile, Mapping):
        offset = len(_TILE_KINDS)
        planted_day = int(tile.get("planted_day", day))
        values[offset : offset + 10] = [
            max(0, day - planted_day) / 30.0,
            float(bool(tile.get("watered_today", False))),
            min(float(tile.get("consecutive_unwatered", 0)) / 2.0, 1.0),
            min(float(tile.get("yield_units", 0)) / 6.0, 2.0),
            max(0.0, float(tile.get("fertilized_until_day", -1) - day + 1) / 3.0),
            float(bool(tile.get("fed_today", False))),
            min(float(tile.get("consecutive_unfed", 0)) / 2.0, 1.0),
            float(bool(tile.get("cared_today", False))),
            float(bool(tile.get("fertilizer_available", False))),
            min(float(tile.get("pending_care_bonus", 0)) / 6.0, 2.0),
        ]
    return values


def _positions(farm: Any, board_size: int) -> list[float]:
    result: list[float] = []
    farmer = _get(farm, "farmer", [0, 0])
    hands = list(_get(farm, "hands", []) or [])[:MAX_HANDS]
    positions = [farmer, *hands]
    denominator = max(1, board_size - 1)
    for index in range(MAX_UNITS):
        if index < len(positions):
            result.extend((float(positions[index][0]) / denominator, float(positions[index][1]) / denominator, 1.0))
        else:
            result.extend((0.0, 0.0, 0.0))
    return result


def _farm_features(farm: Any, day: int) -> list[float]:
    tiles = list(_get(farm, "tiles", []) or [])
    board_size = len(tiles) or 10
    unlocked = set(_get(farm, "unlocked_quadrants", []) or [])
    values = [
        float(_get(farm, "money", 0.0)) / 10_000.0,
        float(_get(farm, "farmer", [0, 0])[0]) / max(1, board_size - 1),
        float(_get(farm, "farmer", [0, 0])[1]) / max(1, board_size - 1),
        min(len(_get(farm, "hands", []) or []) / MAX_HANDS, 1.0),
        min(float(_get(farm, "hires_today", 0)) / MAX_HANDS, 1.0),
        *(float(quadrant in unlocked) for quadrant in ("NW", "NE", "SW", "SE")),
    ]
    for y in range(10):
        for x in range(10):
            tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else "LOCKED"
            values.extend(_tile_features(tile, day))
    values.extend(_positions(farm, board_size))
    assert len(values) == FARM_FEATURES
    return values


def encode_observation(observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one player observation plus its active-unit context."""
    player = int(_get(observation, "player", 0))
    step = int(_get(observation, "step", 0))
    day = int(_get(observation, "day", 0))
    hour = int(_get(observation, "hour", 0))
    values = [step / 720.0, day / 30.0, hour / 24.0, float(player)]

    market = _get(observation, "market", {})
    inventory = _mapping(_get(market, "inventory", {}))
    prices = _mapping(_get(market, "prices", {}))
    values.extend(float(inventory.get(item, 0)) / 10_000.0 for item in PRODUCTS)
    values.extend(float(prices.get(item, 0)) / 250.0 for item in PRODUCTS)

    town = _get(observation, "town", {})
    shops = list(_get(town, "unlocked_shops", []) or [])
    values.extend(shops.count(shop) / 8.0 for shop in SHOP_NAMES)

    farms = list(_get(observation, "farms", []) or [])
    while len(farms) < 2:
        farms.append({})
    values.extend(_farm_features(farms[0], day))
    values.extend(_farm_features(farms[1], day))

    private = _get(observation, "private", {})
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))
    inventories = list(_get(private, "inventories", []) or [])
    carried = {item: 0 for item in ITEMS}
    for unit_inventory in inventories:
        for item in ITEMS:
            carried[item] += int(_mapping(unit_inventory).get(item, 0))
    values.extend(min(float(shed.get(item, 0)) / 100.0, 2.0) for item in ITEMS)
    values.extend(min(float(carried[item]) / 100.0, 2.0) for item in ITEMS)
    values.extend(min(float(seeds.get(crop, 0)) / 100.0, 2.0) for crop in CROPS)

    own_farm = farms[player] if 0 <= player < len(farms) else farms[0]
    board_size = len(_get(own_farm, "tiles", []) or []) or 10
    positions = [list(_get(own_farm, "farmer", [0, 0])), *list(_get(own_farm, "hands", []) or [])[:MAX_HANDS]]
    unit_context = np.zeros((MAX_UNITS, 3), dtype=np.float32)
    active = np.zeros(MAX_UNITS, dtype=np.bool_)
    for index, position in enumerate(positions):
        unit_context[index] = (position[0] / max(1, board_size - 1), position[1] / max(1, board_size - 1), 1.0)
        active[index] = True

    vector = np.asarray(values, dtype=np.float32)
    assert vector.shape == (FEATURE_DIM,)
    return vector, unit_context, active


def encode_batch(observations: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = [encode_observation(observation) for observation in observations]
    return (
        np.stack([item[0] for item in encoded]),
        np.stack([item[1] for item in encoded]),
        np.stack([item[2] for item in encoded]),
    )


def _fib(index: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, index)):
        a, b = b, a + b
    return a


def _unit_mask(observation: Any, unit_index: int) -> np.ndarray:
    mask = np.zeros(len(UNIT_ACTIONS), dtype=np.bool_)
    mask[UNIT_INDEX[("PASS", None)]] = True
    player = int(_get(observation, "player", 0))
    farms = list(_get(observation, "farms", []) or [])
    if player >= len(farms):
        return mask
    farm = farms[player]
    positions = [list(_get(farm, "farmer", [0, 0])), *list(_get(farm, "hands", []) or [])[:MAX_HANDS]]
    if unit_index >= len(positions):
        return mask
    x, y = positions[unit_index]
    tiles = list(_get(farm, "tiles", []) or [])
    board_size = len(tiles)
    if y >= board_size or x >= len(tiles[y]):
        return mask
    tile = tiles[y][x]
    private = _get(observation, "private", {})
    inventories = list(_get(private, "inventories", []) or [])
    carried = _mapping(inventories[unit_index]) if unit_index < len(inventories) else {}
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))

    for op, (dx, dy) in {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}.items():
        if 0 <= x + dx < board_size and 0 <= y + dy < board_size:
            mask[UNIT_INDEX[(op, None)]] = True

    half = board_size // 2
    shed_adjacent = (x, y) in {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    if shed_adjacent:
        for item in ITEMS:
            if int(shed.get(item, 0)) > 0:
                mask[UNIT_INDEX[("PICKUP", item)]] = True
            if int(carried.get(item, 0)) > 0:
                mask[UNIT_INDEX[("PLACE", item)]] = True
        if any(int(value) > 0 for value in carried.values()):
            mask[UNIT_INDEX[("DROP", None)]] = True

    if tile == "LOCKED":
        return mask
    if tile is None:
        for crop in CROPS:
            if int(seeds.get(crop, 0)) > 0:
                mask[UNIT_INDEX[("PLANT", crop)]] = True
        mask[UNIT_INDEX[("BUILD_COOP", None)]] = True
        mask[UNIT_INDEX[("BUILD_PASTURE", None)]] = True
        return mask
    if not isinstance(tile, Mapping):
        return mask

    if "animal" not in tile:
        mask[UNIT_INDEX[("DIG", None)]] = True
    kind = tile.get("kind")
    if kind == "PLANT":
        if not tile.get("watered_today", False):
            mask[UNIT_INDEX[("WATER", None)]] = True
        if int(tile.get("yield_units", 0)) > 0:
            mask[UNIT_INDEX[("HARVEST", None)]] = True
        if int(carried.get("FERTILIZER", 0)) > 0:
            mask[UNIT_INDEX[("FERTILIZE", None)]] = True
    if "animal" in tile:
        if int(tile.get("yield_units", 0)) > 0:
            mask[UNIT_INDEX[("HARVEST", None)]] = True
        if not tile.get("fed_today", False) and int(carried.get("WHEAT", 0)) > 0:
            mask[UNIT_INDEX[("FEED", None)]] = True
        if not tile.get("cared_today", False):
            mask[UNIT_INDEX[("CARE", None)]] = True
        if tile.get("fertilizer_available", False):
            mask[UNIT_INDEX[("COLLECT_FERTILIZER", None)]] = True
    else:
        for animal in ANIMALS:
            required = "COOP" if animal == "GOOSE" else "PASTURE"
            if kind == required and int(carried.get(animal, 0)) > 0:
                mask[UNIT_INDEX[("PLACE", animal)]] = True
    return mask


def _market_mask(observation: Any) -> np.ndarray:
    mask = np.zeros(len(MARKET_ACTIONS), dtype=np.bool_)
    mask[0] = True
    player = int(_get(observation, "player", 0))
    farms = list(_get(observation, "farms", []) or [])
    if player >= len(farms):
        return mask
    farm = farms[player]
    money = float(_get(farm, "money", 0.0))
    private = _get(observation, "private", {})
    shed = _mapping(_get(private, "shed", {}))
    capacity = max(0, 100 - sum(int(value) for value in shed.values()))

    hands = len(_get(farm, "hands", []) or [])
    hires_today = int(_get(farm, "hires_today", 0))
    if hands < MAX_HANDS and money >= _fib(hires_today):
        mask[MARKET_INDEX[("HIRE", None, 1)]] = True
    unlocked = len(_get(farm, "unlocked_quadrants", []) or [])
    if 1 <= unlocked <= 3 and money >= _LAND_PRICES[unlocked - 1]:
        mask[MARKET_INDEX[("BUY_LAND", None, 1)]] = True
    for crop in CROPS:
        for quantity in _MARKET_QUANTITIES:
            if money >= _SEED_COST[crop] * quantity:
                mask[MARKET_INDEX[("BUY_SEED", crop, quantity)]] = True
    market = _get(observation, "market", {})
    prices = _mapping(_get(market, "prices", {}))
    for item in ("WHEAT", "FERTILIZER"):
        for quantity in _MARKET_QUANTITIES:
            if capacity >= quantity and money >= float(prices.get(item, 0)) * quantity:
                mask[MARKET_INDEX[("BUY_PRODUCT", item, quantity)]] = True
    for animal in ANIMALS:
        if capacity >= 1 and money >= _ANIMAL_COST[animal]:
            mask[MARKET_INDEX[("BUY_ANIMAL", animal, 1)]] = True
    for item in PRODUCTS:
        if int(shed.get(item, 0)) > 0:
            mask[MARKET_INDEX[("SELL", item, 100)]] = True
    return mask


def action_masks(observations: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    unit_masks = np.stack([
        np.stack([_unit_mask(observation, unit) for unit in range(MAX_UNITS)])
        for observation in observations
    ])
    market_masks = np.stack([_market_mask(observation) for observation in observations])
    return unit_masks, market_masks


class KaggriculturePolicy(nn.Module):
    """Compact shared policy/value network suitable for large CUDA batches."""

    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.slot_embedding = nn.Embedding(MAX_UNITS, 32)
        self.unit_head = nn.Sequential(
            nn.Linear(hidden_size + 32 + 3, 256),
            nn.SiLU(),
            nn.Linear(256, len(UNIT_ACTIONS)),
        )
        self.market_head = nn.Linear(hidden_size, len(MARKET_ACTIONS))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, features: torch.Tensor, unit_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.trunk(features)
        batch_size = latent.shape[0]
        slots = self.slot_embedding(torch.arange(MAX_UNITS, device=latent.device)).unsqueeze(0).expand(batch_size, -1, -1)
        expanded = latent.unsqueeze(1).expand(-1, MAX_UNITS, -1)
        unit_logits = self.unit_head(torch.cat((expanded, slots, unit_context), dim=-1))
        return unit_logits, self.market_head(latent), self.value_head(latent).squeeze(-1)


def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def decode_actions(unit_indices: np.ndarray, market_indices: np.ndarray, observations: Sequence[Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row, observation in enumerate(observations):
        player = int(_get(observation, "player", 0))
        farms = list(_get(observation, "farms", []) or [])
        hand_count = len(_get(farms[player], "hands", []) or []) if player < len(farms) else 0

        def unit_action(index: int) -> list[Any]:
            op, item = UNIT_ACTIONS[int(index)]
            return [op] if item is None else [op, item]

        market_op, market_item, quantity = MARKET_ACTIONS[int(market_indices[row])]
        if market_op == "NONE":
            market: list[list[Any]] = []
        elif market_item is None:
            market = [[market_op]]
        else:
            market = [[market_op, market_item, quantity]]
        actions.append({
            "farmer": unit_action(unit_indices[row, 0]),
            "hands": [unit_action(unit_indices[row, index + 1]) for index in range(min(hand_count, MAX_HANDS))],
            "market": market,
        })
    return actions


@dataclass
class PolicyBatch:
    actions: list[dict[str, Any]]
    unit_indices: torch.Tensor
    market_indices: torch.Tensor
    values: torch.Tensor


@torch.no_grad()
def policy_batch(
    model: KaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    *,
    deterministic: bool = False,
    mask_actions: bool = True,
) -> PolicyBatch:
    features_np, unit_context_np, _ = encode_batch(observations)
    features = torch.as_tensor(features_np, device=device)
    unit_context = torch.as_tensor(unit_context_np, device=device)
    unit_logits, market_logits, values = model(features, unit_context)
    if mask_actions:
        unit_masks_np, market_masks_np = action_masks(observations)
        unit_masks = torch.as_tensor(unit_masks_np, device=device)
        market_masks = torch.as_tensor(market_masks_np, device=device)
        unit_logits = _masked(unit_logits, unit_masks)
        market_logits = _masked(market_logits, market_masks)
    if deterministic:
        unit_indices = unit_logits.argmax(dim=-1)
        market_indices = market_logits.argmax(dim=-1)
    else:
        unit_indices = torch.distributions.Categorical(logits=unit_logits).sample()
        market_indices = torch.distributions.Categorical(logits=market_logits).sample()
    actions = decode_actions(unit_indices.cpu().numpy(), market_indices.cpu().numpy(), observations)
    return PolicyBatch(actions, unit_indices, market_indices, values)


def action_targets(observations: Sequence[Any], actions: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map scripted actions to vocabulary indices for behavior cloning."""
    unit_targets = np.zeros((len(actions), MAX_UNITS), dtype=np.int64)
    market_targets = np.zeros(len(actions), dtype=np.int64)
    active = np.zeros((len(actions), MAX_UNITS), dtype=np.bool_)
    for row, (observation, action) in enumerate(zip(observations, actions, strict=True)):
        player = int(_get(observation, "player", 0))
        farms = list(_get(observation, "farms", []) or [])
        hand_count = len(_get(farms[player], "hands", []) or []) if player < len(farms) else 0
        unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])[:hand_count]]
        for unit, raw in enumerate(unit_actions[:MAX_UNITS]):
            active[row, unit] = True
            op = raw[0] if isinstance(raw, Sequence) and raw else "PASS"
            item = raw[1] if isinstance(raw, Sequence) and len(raw) > 1 else None
            unit_targets[row, unit] = UNIT_INDEX.get((str(op), item), UNIT_INDEX[("PASS", None)])
        market = list(action.get("market", []) or [])
        if market:
            raw = market[0]
            op = raw[0]
            item = raw[1] if len(raw) > 1 else None
            quantity = int(raw[2]) if len(raw) > 2 else 1
            if op == "SELL" and item in PRODUCTS:
                key = ("SELL", item, 100)
            elif op in ("HIRE", "BUY_LAND"):
                key = (op, None, 1)
            else:
                choices = [q for q in _MARKET_QUANTITIES if (op, item, q) in MARKET_INDEX]
                nearest = min(choices, key=lambda q: abs(q - quantity)) if choices else quantity
                key = (op, item, nearest)
            market_targets[row] = MARKET_INDEX.get(key, 0)
    return unit_targets, market_targets, active


def flatten_environment_observations(pairs: Iterable[tuple[Any, Any]]) -> list[Any]:
    return [observation for pair in pairs for observation in pair]
