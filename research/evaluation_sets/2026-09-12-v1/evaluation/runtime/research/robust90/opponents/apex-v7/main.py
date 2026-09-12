"""Kaggle entrypoint for Apex V7 God Emperor (Margin Multiplier Architecture)."""

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
        parent_dir = Path(_library.__code__.co_filename).resolve().parent
        library = None
        for ext in (["dylib", "so"] if sys.platform == "darwin" else ["so", "dylib"]):
            p = parent_dir / f"agent.{ext}"
            if p.exists():
                try:
                    library = ctypes.CDLL(str(p))
                    if library is not None:
                        break
                except OSError:
                    library = None
        if library is None:
            extension = "dylib" if sys.platform == "darwin" else "so"
            path = parent_dir / f"agent.{extension}"
            import subprocess
            cmd = [
                "g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                f"-I{parent_dir}/source/include", "-o", str(path),
                str(parent_dir / "source/policy.cpp"), str(parent_dir / "submission_bridge.cpp")
            ]
            subprocess.run(cmd, check=True)
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
    step = int(_read(observation, "step", None) if _read(observation, "step", None) is not None else (int(_read(observation, "day", 0) or 0) * 24 + int(_read(observation, "hour", 0) or 0)))
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


_WEED_STATE = {0: {}, 1: {}}
_AGENT_STATE = {0: {}, 1: {}}


def _tile_at(farm, pos):
    if not pos or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = _read(farm, 'tiles', []) or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def agent(observation, configuration=None):
    seat = int(_read(observation, 'player', 0) or 0)
    packed = _pack_observation(observation, seat)
    episode_steps = int(_read(configuration or {}, 'episodeSteps', 720) or 720)
    output = _PackedAction()
    status = _library().kag_submission_act(
        ctypes.byref(packed), seat, episode_steps, ctypes.byref(output)
    )
    if status != 0:
        raise RuntimeError('ShopForge native policy failed')
    action = _unpack_action(output)

    try:
        step = int(_read(observation, 'step', None) if _read(observation, 'step', None) is not None else (int(_read(observation, 'day', 0) or 0) * 24 + int(_read(observation, 'hour', 0) or 0)))
        farms = _read(observation, 'farms', [{}, {}]) or [{}, {}]
        farm = farms[seat] if seat < len(farms) else {}
        opp_farm = farms[1 - seat] if (1 - seat) < len(farms) else {}

        state = _AGENT_STATE.setdefault(seat, {})
        if step == 0:
            state.clear()
            state['is_clone'] = False

        # 1. Weed Auto-Dig Shield (Prevents worker paralysis on random weed spawns)
        farmer_pos = _read(farm, 'farmer', None)
        hands_pos = list(_read(farm, 'hands', []) or [])
        positions = [farmer_pos, *hands_pos]
        unit_actions = [
            list(action.get('farmer', ['PASS']) or ['PASS']),
            *[list(h or ['PASS']) for h in (action.get('hands') or [])]
        ]

        game = _WEED_STATE.setdefault(seat, {})
        if step == 0:
            game.clear()
        active = game.setdefault('active', {})

        for actor, trans in list(active.items()):
            idx = 0 if actor == 'farmer' else int(actor) + 1
            if idx < len(unit_actions) and step - trans.get('start', -1) == 1:
                unit_actions[idx] = list(trans.get('intended', ['PASS']))
            else:
                active.pop(actor, None)

        for idx, (pos, intended) in enumerate(zip(positions, unit_actions)):
            actor = 'farmer' if idx == 0 else idx - 1
            if actor in active or not isinstance(intended, list) or not intended:
                continue
            if intended[0] in ('PLANT', 'BUILD_PASTURE', 'BUILD_COOP'):
                tile = _tile_at(farm, pos)
                if tile is not None and _read(tile, 'kind') == 'WEED':
                    active[actor] = {'start': step, 'intended': list(intended)}
                    unit_actions[idx] = ['DIG']

# 1B. Field clearance handled by C++ tape

        action['farmer'] = unit_actions[0] if unit_actions else ['PASS']
        action['hands'] = unit_actions[1:]

        # 2. Market Orders Shield (Apex V7 God Emperor)
        market = list(action.get('market') or [])
        private = _read(observation, 'private', {}) or {}
        shed = _read(private, 'shed', {}) or {}
        seeds = _read(private, 'seeds', {}) or {}
        market_info = _read(observation, 'market', {}) or {}
        prices = _read(market_info, 'prices', {}) or {}
        my_money = float(_read(farm, 'money', 0) or 0)

        # Clone Detection Radar:
        # Check opponent hires and animal structures by Step 10
        if 2 <= step <= 10 and not state.get('is_clone', False):
            opp_hands = len(list(_read(opp_farm, 'hands', []) or []))
            opp_tiles = _read(opp_farm, 'tiles', []) or []
            opp_structures = sum(1 for r in opp_tiles for t in (r if isinstance(r, list) else []) if isinstance(t, dict) and t.get('kind') in ('PASTURE', 'COOP'))
            if opp_hands >= 3 and opp_structures >= 1:
                state['is_clone'] = True

        is_clone = state.get('is_clone', False)

        # 2A. Anti-Clone Desynchronization (The Front-Running Ambush):
        # Steals peak prices 1 turn before the clone's scheduled dump!
        strawberry_count = int(_read(shed, 'STRAWBERRY', 0) or 0)
        melon_count = int(_read(shed, 'MELON', 0) or 0)
        fert = int(_read(shed, 'FERTILIZER', 0) or 0)
        wheat_count = int(_read(shed, 'WHEAT', 0) or 0)
        wool_count = int(_read(shed, 'WOOL', 0) or 0)
        milk_count = int(_read(shed, 'MILK', 0) or 0)

        if is_clone:
            # Front-run Step 250 Melon dump at Step 249:
            if step == 249 and melon_count >= 6 and len(market) < 10:
                if not any(o[0] == 'SELL' and o[1] == 'MELON' for o in market):
                    market.append(['SELL', 'MELON', min(melon_count, 12)])
            # Front-run Step 382 Strawberry dump at Step 381:
            elif step == 381 and strawberry_count >= 6 and len(market) < 10:
                if not any(o[0] == 'SELL' and o[1] == 'STRAWBERRY' for o in market):
                    market.append(['SELL', 'STRAWBERRY', min(strawberry_count, 8)])
            # Front-run Step 404 Strawberry dump at Step 403:
            elif step == 403 and strawberry_count >= 6 and len(market) < 10:
                if not any(o[0] == 'SELL' and o[1] == 'STRAWBERRY' for o in market):
                    market.append(['SELL', 'STRAWBERRY', min(strawberry_count, 8)])
            # Front-run Step 503 wave at Steps 499-501:
            elif 499 <= step <= 501 and strawberry_count >= 8 and len(market) < 10:
                if not any(o[0] == 'SELL' and o[1] == 'STRAWBERRY' for o in market):
                    market.append(['SELL', 'STRAWBERRY', min(strawberry_count, 8)])
            # Front-run Step 523 Fertilizer/Wheat dump at Step 522:
            elif step == 522 and fert >= 18 and len(market) < 10:
                if not any(o[0] == 'SELL' and o[1] == 'FERTILIZER' for o in market):
                    market.append(['SELL', 'FERTILIZER', min(fert - 16, 4)])

        # Pre-extract all product prices safely upfront
        wheat_price = float(_read(prices, 'WHEAT', 0) or 0)
        carrot_price = float(_read(prices, 'CARROT', 0) or 0)
        tomato_price = float(_read(prices, 'TOMATO', 0) or 0)
        strawberry_price = float(_read(prices, 'STRAWBERRY', 0) or 0)
        melon_price = float(_read(prices, 'MELON', 0) or 0)
        egg_price = float(_read(prices, 'EGG', 0) or 0)
        milk_price = float(_read(prices, 'MILK', 0) or 0)
        wool_price = float(_read(prices, 'WOOL', 0) or 0)
        fert_price = float(_read(prices, 'FERTILIZER', 0) or 0)

        # 2B. Protected Fertilizer Monetization:
        if step < 718 and fert_price >= 18.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'FERTILIZER' for order in market):
                if step < 350 or step >= 524:
                    if fert >= 3:
                        market.append(['SELL', 'FERTILIZER', min(fert, 2)])
                else:
                    if fert >= 20:
                        market.append(['SELL', 'FERTILIZER', min(fert - 16, 2)])

        # 2C. Early Cash Emergency Buffer:
        # Guarantees never missing a worker hire or animal purchase while strictly protecting 35 animal wheat feed
        if step <= 300 and my_money < 180.0 and len(market) < 10:
            if fert >= 4 and not any(order[0] == 'SELL' and order[1] == 'FERTILIZER' for order in market):
                market.append(['SELL', 'FERTILIZER', 2])
            elif wheat_count > 38 and not any(order[0] == 'SELL' and order[1] == 'WHEAT' for order in market):
                market.append(['SELL', 'WHEAT', min(wheat_count - 35, 2)])

        # 2D. Universal Fluid Shed Capacity Guard (Projected Shed Accounting):
        # Keeps shed safely below the 100-unit hard cap with generous buffer so worker drops never discard
        shed_total = sum(int(_read(shed, p, 0) or 0) for p in _PRODUCTS)
        pending_sales = sum(int(order[2]) for order in market if len(order) >= 3 and order[0] == 'SELL')
        projected_shed = shed_total - pending_sales
        if projected_shed >= 75 and len(market) < 10:
            excess = min(projected_shed - 65, 8)
            # Priority 1: High-value animal goods and surplus fertilizer
            if fert >= 4 and not any(order[0] in ('SELL', 'BUY_PRODUCT') and order[1] == 'FERTILIZER' for order in market):
                market.append(['SELL', 'FERTILIZER', min(fert, excess)])
            elif milk_count >= 4 and milk_price >= 80.0 and not any(order[0] in ('SELL', 'BUY_PRODUCT') and order[1] == 'MILK' for order in market):
                market.append(['SELL', 'MILK', min(milk_count, 4)])
            elif wool_count >= 4 and wool_price >= 25.0 and not any(order[0] in ('SELL', 'BUY_PRODUCT') and order[1] == 'WOOL' for order in market):
                w_batch = 4 if wool_price >= 120.0 else 2
                market.append(['SELL', 'WOOL', min(wool_count, w_batch)])
            elif wheat_count > 38 and not any(order[0] in ('SELL', 'BUY_PRODUCT') and order[1] == 'WHEAT' for order in market):
                market.append(['SELL', 'WHEAT', min(wheat_count - 35, excess)])
            else:
                # Priority 2: Universal relief across any crop surplus to guarantee zero harvest loss
                surplus_candidates = []
                for p, cnt, pr in [
                    ('STRAWBERRY', strawberry_count, strawberry_price),
                    ('MELON', melon_count, melon_price),
                    ('CARROT', int(_read(shed, 'CARROT', 0) or 0), carrot_price),
                    ('TOMATO', int(_read(shed, 'TOMATO', 0) or 0), tomato_price),
                    ('EGG', int(_read(shed, 'EGG', 0) or 0), egg_price),
                ]:
                    if cnt >= 2 and not any(order[0] in ('SELL', 'BUY_PRODUCT') and order[1] == p for order in market):
                        surplus_candidates.append((cnt * pr, p, cnt))
                if surplus_candidates:
                    surplus_candidates.sort(reverse=True)
                    best_item = surplus_candidates[0][1]
                    best_cnt = surplus_candidates[0][2]
                    market.append(['SELL', best_item, min(best_cnt, excess)])

        # 2E. Town Shop Synchronized High-Margin Monetization (V7 Precision Micro-Batching):
        town_info = _read(observation, 'town', {}) or {}
        unlocked_shops = [str(s) for s in (_read(town_info, 'unlocked_shops', []) or [])]
        has_yarn_store = any('YARN' in s for s in unlocked_shops)
        has_dairy_shop = any('SMOOTHIE' in s or 'PIZZA' in s or 'ICE_CREAM' in s for s in unlocked_shops)
        has_berry_shop = any('SMOOTHIE' in s or 'BRUNCH' in s or 'ICE_CREAM' in s or 'FARMERS' in s for s in unlocked_shops)
        has_pet_cafe = any('PET_CAFE' in s for s in unlocked_shops)
        has_farmers_market = any('FARMERS' in s for s in unlocked_shops)

        # Wool Micro-Batching: Prevents Quadratic Decay (above_func: "sq")!
        # Town Yarn Store consumes 2 wool every 4 steps (multiplier = 2).
        if has_yarn_store and (step % 4 == 1) and wool_count >= 2 and wool_price >= 25.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'WOOL' for order in market):
                batch = 4 if wool_price >= 150.0 else 2
                market.append(['SELL', 'WOOL', min(wool_count, batch)])

        # Milk Monetization:
        if has_dairy_shop and step >= 250 and milk_count >= 4 and milk_price >= 80.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'MILK' for order in market):
                market.append(['SELL', 'MILK', min(milk_count, 4)])

        # Strawberry Monetization (Lower threshold to 70.0 when shop exists):
        if has_berry_shop and (step % 4 == 1) and strawberry_count >= 3 and strawberry_price >= 70.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'STRAWBERRY' for order in market):
                market.append(['SELL', 'STRAWBERRY', min(strawberry_count, 3)])

        # Carrot Surge (Monetizes Pet Cafe & Farmers Market consumption):
        carrot_count = int(_read(shed, 'CARROT', 0) or 0)
        if (has_pet_cafe or has_farmers_market) and (step % 4 == 1) and carrot_count >= 2 and carrot_price >= 25.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'CARROT' for order in market):
                market.append(['SELL', 'CARROT', min(carrot_count, 3)])

        # 2F. Predator Market Ambush:
        opp_tiles = _read(opp_farm, 'tiles', []) or []
        opp_maturing_strawberries = 0
        opp_maturing_melons = 0
        current_day = int(_read(observation, 'day', 0) or 0)
        for row in opp_tiles:
            for t in (row if isinstance(row, list) else []):
                if isinstance(t, dict) and t.get('kind') == 'PLANT':
                    p_day = int(t.get('planted_day', 0) or 0)
                    y_units = int(t.get('yield_units', 0) or 0)
                    if t.get('crop') == 'STRAWBERRY' and ((current_day - p_day) >= 8 or y_units > 0):
                        opp_maturing_strawberries += 1
                    elif t.get('crop') == 'MELON' and ((current_day - p_day) >= 9 or y_units > 0):
                        opp_maturing_melons += 1

        if opp_maturing_strawberries >= 3 and strawberry_count >= 2 and strawberry_price >= 85.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'STRAWBERRY' for order in market):
                market.append(['SELL', 'STRAWBERRY', min(strawberry_count, 4)])

        melon_price = float(_read(prices, 'MELON', 0) or 0)
        if opp_maturing_melons >= 2 and melon_count >= 2 and melon_price >= 95.0 and len(market) < 10:
            if not any(order[0] == 'SELL' and order[1] == 'MELON' for order in market):
                market.append(['SELL', 'MELON', min(melon_count, 4)])

        # 2G. Step 622 Precision Carrot Seed Guarantee:
        if step == 622 and len(market) < 10:
            c_seeds = int(_read(seeds, 'CARROT', 0) or 0)
            if c_seeds < 4 and my_money >= 200.0:
                needed = 4 - c_seeds
                if not any(o[0] == 'BUY_SEED' and o[1] == 'CARROT' for o in market):
                    market.append(['BUY_SEED', 'CARROT', needed])

        # 2H. Universal Order Expansion in ENDGAME (Steps 700 to 718):
        if step >= 700:
            for order in market:
                if order[0] == 'SELL' and len(order) >= 3:
                    item = order[1]
                    available = int(_read(shed, item, 0) or 0)
                    if available > order[2]:
                        order[2] = available

        # 2I. 4-Stage Liquidation (Steps 716-719):
        if step in (718, 719):
            # Strip out any non-SELL orders so all 10 order slots are 100% dedicated to revenue
            market = [o for o in market if o[0] == 'SELL']

        if step == 716:
            for item in ['STRAWBERRY', 'MELON', 'FERTILIZER']:
                count = int(_read(shed, item, 0) or 0)
                if count >= 2:
                    existing = next((o for o in market if o[0] == 'SELL' and o[1] == item), None)
                    if existing is not None:
                        existing[2] = max(existing[2], count)
                    elif len(market) < 10:
                        market.append(['SELL', item, count])

        if step == 717:
            for item in ['MILK', 'WOOL', 'STRAWBERRY', 'MELON', 'FERTILIZER', 'EGG', 'CARROT', 'TOMATO']:
                count = int(_read(shed, item, 0) or 0)
                if count > 0:
                    existing = next((o for o in market if o[0] == 'SELL' and o[1] == item), None)
                    if existing is not None:
                        existing[2] = max(existing[2], count)
                    elif len(market) < 10:
                        market.append(['SELL', item, count])

        if step == 718:
            for item in ['MILK', 'WOOL', 'STRAWBERRY', 'MELON', 'TOMATO', 'EGG', 'CARROT', 'FERTILIZER', 'WHEAT']:
                count = int(_read(shed, item, 0) or 0)
                if count > 0:
                    existing = next((o for o in market if o[0] == 'SELL' and o[1] == item), None)
                    if existing is not None:
                        existing[2] = max(existing[2], count)
                    elif len(market) < 10:
                        market.append(['SELL', item, count])

        if step == 719:
            # Complete and utter liquidation of all 9 items in shed
            for item in ['FERTILIZER', 'WHEAT', 'MILK', 'WOOL', 'STRAWBERRY', 'MELON', 'TOMATO', 'EGG', 'CARROT']:
                count = int(_read(shed, item, 0) or 0)
                if count > 0:
                    existing = next((o for o in market if o[0] == 'SELL' and o[1] == item), None)
                    if existing is not None:
                        existing[2] = max(existing[2], count)
                    elif len(market) < 10:
                        market.append(['SELL', item, count])

        action['market'] = market[:10]
    except Exception:
        pass

    return action
