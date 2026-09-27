# arena-in-gpu-v1 (source-v26): hash-pinned in ppo/arena_gpu.py; mounted read-only into the arena sandbox.
"""gpu_sim State (host numpy copy) -> official Kaggriculture observation dict, byte-identical in JSON to what the
official kaggle-environments 1.32.7 engine hands an arena bot (key order included), for the default configuration.

Inverse of gpu_sim/tests/reference_assertions.py plus the dict key orders of the official engine:
  obs keys: remainingOverageTime, player, private, farms, market, town, day, hour, step
  plant tile keys: kind, crop, planted_day, watered_today, consecutive_unwatered, yield_units, max_lifespan_step, fertilized_until_day
  animal tile keys: kind, animal, placed_day, yield_units, consecutive_unfed, fed_today, cared_today, fertilizer_available, pending_care_bonus
  inventories: dict insertion order = gpu_sim unit_inventory_order (first-add order).
"""
import numpy as np

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
SHOP_NAMES = tuple(sorted(("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")))
QUADS = ("NW", "NE", "SW", "SE")  # official LAND_ORDER = NE, SW, SE after NW
EMPTY, LOCKED, WEED, PLANT, COOP, PASTURE = range(6)
KIND_NAME = {WEED: "WEED", PLANT: "PLANT", COOP: "COOP", PASTURE: "PASTURE"}
F_WATERED, F_FED, F_CARED, F_FERT = 1, 2, 4, 8
FIELDS = ('step', 'money', 'tile_kind', 'tile_crop', 'tile_animal', 'tile_origin_day', 'tile_yield', 'tile_neglect',
          'tile_max_lifespan', 'tile_fertilized_until', 'tile_pending_care', 'tile_flags', 'unit_pos', 'unit_active',
          'unit_inventory', 'unit_inventory_order', 'shed', 'seeds', 'hires_today', 'unlocked_count',
          'market_inventory', 'market_price', 'town_shops', 'town_count')


def host_state(state, games=None):
    """Device State (jax or torch views) -> dict of numpy arrays (optionally only some game rows)."""
    out = {}
    for name in FIELDS:
        v = state[name] if isinstance(state, dict) else getattr(state, name)
        if hasattr(v, 'detach'):
            v = v[games] if games is not None else v
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
            v = v[games] if games is not None else v
        out[name] = v
    return out


def _tile(k, crop, animal, origin, yld, neglect, mls, fert, care, flags):
    if k == EMPTY:
        return None
    if k == LOCKED:
        return "LOCKED"
    if k == WEED:
        return {"kind": "WEED"}
    if k == PLANT:
        return {"kind": "PLANT", "crop": CROPS[crop], "planted_day": origin, "watered_today": bool(flags & F_WATERED),
                "consecutive_unwatered": neglect, "yield_units": yld, "max_lifespan_step": mls, "fertilized_until_day": fert}
    if animal < 0:
        return {"kind": KIND_NAME[k]}
    return {"kind": KIND_NAME[k], "animal": ANIMALS[animal], "placed_day": origin, "yield_units": yld,
            "consecutive_unfed": neglect, "fed_today": bool(flags & F_FED), "cared_today": bool(flags & F_CARED),
            "fertilizer_available": bool(flags & F_FERT), "pending_care_bonus": care}


def _farm(s, g, p):
    cols = [s[n][g, p].tolist() for n in ('tile_kind', 'tile_crop', 'tile_animal', 'tile_origin_day', 'tile_yield', 'tile_neglect',
                                          'tile_max_lifespan', 'tile_fertilized_until', 'tile_pending_care', 'tile_flags')]
    tiles = [[_tile(*(c[y][x] for c in cols)) for x in range(10)] for y in range(10)]
    active = s['unit_active'][g, p]
    n = int(active.sum())
    pos = s['unit_pos'][g, p].tolist()
    return {"money": float(s['money'][g, p]), "tiles": tiles, "farmer": pos[0], "hands": pos[1:n],
            "unlocked_quadrants": list(QUADS[:int(s['unlocked_count'][g, p])]), "hires_today": int(s['hires_today'][g, p])}, n


def observation(s, g, seat):
    """Official observation of `seat` in game row g of host state s (as the production path sends it: with 'step')."""
    farms, counts = zip(*(_farm(s, g, p) for p in (0, 1)))
    inv = s['unit_inventory'][g, seat].tolist()
    order = s['unit_inventory_order'][g, seat].tolist()
    inventories = []
    for u in range(counts[seat]):
        present = sorted((o, i) for i, o in enumerate(order[u]) if o >= 0)
        inventories.append({SHED_ITEMS[i]: inv[u][i] for _, i in present})
    shed = s['shed'][g, seat].tolist()
    seeds = s['seeds'][g, seat].tolist()
    step = int(s['step'][g])
    return {"remainingOverageTime": 60, "player": seat,
            "private": {"shed": dict(zip(SHED_ITEMS, shed)), "seeds": dict(zip(CROPS, seeds)), "inventories": inventories},
            "farms": list(farms),
            "market": {"inventory": dict(zip(PRODUCTS, s['market_inventory'][g].tolist())), "prices": dict(zip(PRODUCTS, s['market_price'][g].tolist()))},
            "town": {"unlocked_shops": [SHOP_NAMES[i] for i in s['town_shops'][g][:int(s['town_count'][g])].tolist()]},
            "day": step // 24, "hour": step % 24, "step": step}


def first_diff(a, b, path=''):
    """Human-readable first difference between two JSON-like values (order-sensitive for dict keys)."""
    if type(a) is not type(b):
        return f'{path}: type {type(a).__name__} {a!r:.80} != {type(b).__name__} {b!r:.80}'
    if isinstance(a, dict):
        if list(a) != list(b):
            return f'{path}: keys {list(a)} != {list(b)}'
        for k in a:
            d = first_diff(a[k], b[k], f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}: len {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if a == b else f'{path}: {a!r:.80} != {b!r:.80}'
