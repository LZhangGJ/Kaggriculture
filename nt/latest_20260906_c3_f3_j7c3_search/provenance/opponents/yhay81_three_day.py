"""Kaggle entrypoint for the frozen hybrid_shopforge_3day_frontier_state_router_r5 native policy."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Mapping
from pathlib import Path

_ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
_PRODUCTS = _ITEMS[:9]
_SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)
_SHOP_ID = {name: index for index, name in enumerate(_SHOPS)}
_KIND_ID = {
    None: 0, "EMPTY": 0, "SOIL": 0, "LOCKED": 1, "WEED": 2,
    "COOP": 3, "PASTURE": 4, "PLANT": 5,
}
_UNIT_OPS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
)
_MARKET_OPS = ("PASS", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")
_BOARD = 10
_MAX_UNITS = 40
_MAX_SHOPS = 8
_LIBRARY = None


def _read(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


class _PackedTile(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("kind", ctypes.c_uint8)]


class _PackedFarm(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("money", ctypes.c_double),
        ("tiles", _PackedTile * _BOARD * _BOARD),
        ("n_units", ctypes.c_int32),
        ("n_quadrants", ctypes.c_int32),
        ("hires_today", ctypes.c_int32),
        ("shed", ctypes.c_int16 * len(_ITEMS)),
        ("inv", ctypes.c_int16 * len(_ITEMS) * _MAX_UNITS),
    ]


class _PackedObservation(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("step", ctypes.c_int32),
        ("n_shops", ctypes.c_int32),
        ("market_inventory", ctypes.c_int32 * len(_PRODUCTS)),
        ("market_prices", ctypes.c_int32 * len(_PRODUCTS)),
        ("shops", ctypes.c_uint8 * _MAX_SHOPS),
        ("farms", _PackedFarm * 2),
    ]


class _PackedAction(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("unit_ops", ctypes.c_uint8 * _MAX_UNITS),
        ("unit_args", ctypes.c_uint8 * _MAX_UNITS),
        ("unit_ns", ctypes.c_int16 * _MAX_UNITS),
        ("n_units", ctypes.c_int32),
        ("order_ops", ctypes.c_uint8 * 16),
        ("order_items", ctypes.c_uint8 * 16),
        ("order_ns", ctypes.c_int32 * 16),
        ("n_orders", ctypes.c_int32),
    ]


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        extension = "dylib" if sys.platform == "darwin" else "so"
        # Kaggle's source loader does not define ``__file__``.  The compiled
        # function filename still points at the extracted top-level main.py.
        path = Path(_library.__code__.co_filename).resolve().parent / f"agent.{extension}"
        library = ctypes.CDLL(str(path))
        library.kag_submission_abi_version.restype = ctypes.c_uint32
        library.kag_submission_act.argtypes = [
            ctypes.POINTER(_PackedObservation),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_PackedAction),
        ]
        library.kag_submission_act.restype = ctypes.c_int
        if int(library.kag_submission_abi_version()) != 1:
            raise RuntimeError("ShopForge SixDay Guard submission ABI mismatch")
        _LIBRARY = library
    return _LIBRARY


def _fill_counts(target, mapping, names):
    for index, name in enumerate(names):
        target[index] = int(_read(mapping, name, 0) or 0) if mapping else 0


def _fill_tile(dst, tile):
    if tile is None:
        return
    if isinstance(tile, str):
        dst.kind = _KIND_ID.get(tile, 0)
        return
    kind = _read(tile, "kind")
    if kind == "PLANT" or _read(tile, "crop"):
        dst.kind = _KIND_ID["PLANT"]
        return
    dst.kind = _KIND_ID.get(kind, 0)


def _pack_observation(observation, seat):
    packed = _PackedObservation()
    step = int(_read(observation, "step", 0) or 0)
    packed.step = step
    market = _read(observation, "market", {}) or {}
    _fill_counts(packed.market_inventory, _read(market, "inventory", {}) or {}, _PRODUCTS)
    _fill_counts(packed.market_prices, _read(market, "prices", {}) or {}, _PRODUCTS)
    shops = list(_read(_read(observation, "town", {}) or {}, "unlocked_shops", []) or [])
    packed.n_shops = min(len(shops), _MAX_SHOPS)
    for index, shop in enumerate(shops[:_MAX_SHOPS]):
        packed.shops[index] = _SHOP_ID[str(shop)]

    farms = list(_read(observation, "farms", []) or [])
    if len(farms) != 2:
        raise ValueError("ShopForge needs exactly two public farms")
    for player, farm in enumerate(farms):
        dest = packed.farms[player]
        dest.money = float(_read(farm, "money", 0) or 0)
        positions = [_read(farm, "farmer", [0, 0]), *list(_read(farm, "hands", []) or [])]
        dest.n_units = max(1, min(len(positions), _MAX_UNITS))
        dest.n_quadrants = len(list(_read(farm, "unlocked_quadrants", []) or []))
        dest.hires_today = int(_read(farm, "hires_today", 0) or 0)
        tiles = list(_read(farm, "tiles", []) or [])
        for y, row in enumerate(tiles[:_BOARD]):
            for x, tile in enumerate(list(row or [])[:_BOARD]):
                _fill_tile(dest.tiles[y][x], tile)

    private = _read(observation, "private", {}) or {}
    own = packed.farms[seat]
    _fill_counts(own.shed, _read(private, "shed", {}) or {}, _ITEMS)
    inventories = list(_read(private, "inventories", []) or [])
    for unit, carried in enumerate(inventories[:_MAX_UNITS]):
        _fill_counts(own.inv[unit], carried or {}, _ITEMS)
    return packed


def _unit_order(op, arg, quantity):
    name = _UNIT_OPS[op] if 0 <= op < len(_UNIT_OPS) else "PASS"
    if name in {"PLANT", "PICKUP", "PLACE"}:
        item = _ITEMS[arg] if 0 <= arg < len(_ITEMS) else _ITEMS[0]
        return [name, item] if quantity == 1 else [name, item, int(quantity)]
    return [name]


def _market_order(op, item, quantity):
    name = _MARKET_OPS[op] if 0 <= op < len(_MARKET_OPS) else "PASS"
    if name == "PASS":
        return None
    if name in {"HIRE", "BUY_LAND"}:
        return [name]
    item_name = _ITEMS[item] if 0 <= item < len(_ITEMS) else _ITEMS[0]
    return [name, item_name, int(quantity)]


def _unpack_action(packed):
    n_units = max(1, min(int(packed.n_units), _MAX_UNITS))
    farmer = _unit_order(packed.unit_ops[0], packed.unit_args[0], packed.unit_ns[0])
    hands = [
        _unit_order(packed.unit_ops[index], packed.unit_args[index], packed.unit_ns[index])
        for index in range(1, n_units)
    ]
    market = []
    for index in range(max(0, min(int(packed.n_orders), 16))):
        order = _market_order(packed.order_ops[index], packed.order_items[index], packed.order_ns[index])
        if order is not None:
            market.append(order)
    return {"farmer": farmer, "hands": hands, "market": market}


def agent(observation, configuration=None):
    seat = int(_read(observation, "player", 0) or 0)
    packed = _pack_observation(observation, seat)
    episode_steps = int(_read(configuration or {}, "episodeSteps", 720) or 720)
    output = _PackedAction()
    status = _library().kag_submission_act(
        ctypes.byref(packed), seat, episode_steps, ctypes.byref(output)
    )
    if status != 0:
        raise RuntimeError("ShopForge native policy failed")
    return _unpack_action(output)
