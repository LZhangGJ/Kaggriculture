"""Public-only economic features for an isolated student-input experiment.

Known town demand is exact. Rival asset counts are *possibilities*, not a
prediction that the opponent will harvest or sell them.
"""

from __future__ import annotations

import numpy as np


# Official simulator shop IDs and per-four-step consumption, products 0..8.
SHOP_DEMAND = np.asarray([
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Bakery
    [1, 0, 0, 1, 0, 1, 0, 0, 0],  # Brunch Spot
    [1, 1, 1, 1, 0, 0, 0, 0, 0],  # Farmers Market
    [1, 0, 0, 1, 0, 0, 1, 0, 0],  # Ice Cream Shop
    [0, 2, 0, 0, 0, 0, 0, 0, 0],  # Pet Cafe
    [1, 0, 1, 0, 0, 0, 1, 0, 0],  # Pizza Shop
    [0, 0, 0, 1, 0, 0, 1, 0, 0],  # Smoothie Shop
    [0, 0, 0, 0, 0, 0, 0, 2, 0],  # Yarn Store
], dtype=np.int16)
FIRST_CROP_DAY = (2, 2, 8, 10, 10)
FIRST_ANIMAL_DAY = (4, 8, 6)
HORIZON_DAYS = (2, 6, 30)
FEATURE_WIDTH = 8 + 3 * 9 + 3 * 9 + 9
PREFIX_WIDTH = 3 * 9
SHOP_RESOURCE_WIDTH = 9


def _integer(raw: np.ndarray, at: int) -> int:
    value = float(raw[at])
    rounded = round(value)
    if not np.isfinite(value) or abs(value - rounded) > .01:
        raise ValueError(f"non-integer packed observation at {at}: {value}")
    return int(rounded)


def _unpack(raw: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Read only fields needed here; native actor uses own farm first."""
    step, day = _integer(raw, 0), _integer(raw, 1)
    at = 4
    rival_tiles = None
    for farm in range(2):
        hands = _integer(raw, at + 3)
        at += 4 + 2 * hands + 2
        tiles = raw[at:at + 1500].reshape(100, 15)
        if farm == 1:
            rival_tiles = tiles
        at += 1500
    at += 12 + 5
    bags = _integer(raw, at)
    at += 1
    for _ in range(bags):
        at += 12
        order = _integer(raw, at)
        at += 1 + order
    inventory = np.rint(raw[at:at + 9]).astype(np.int32)
    at += 18  # Inventory and prices; prices are already in the raw actor input.
    count = _integer(raw, at)
    at += 1
    shops = np.rint(raw[at:at + count]).astype(np.int32)
    if (at + count != len(raw) or not 0 <= count <= 8 or
            np.any((shops < 0) | (shops >= 8))):
        raise ValueError("malformed packed shop sequence")
    return step, day, inventory, shops, rival_tiles


def economic_features(raw: np.ndarray, *, mature_stored: bool = False) -> np.ndarray:
    """71 fixed features: shops, no-new-supply pressure, rival asset windows."""
    step, day, inventory, shops, tiles = _unpack(np.asarray(raw))
    shop_counts = np.bincount(shops, minlength=8)
    rate = shop_counts @ SHOP_DEMAND
    pressure, rival_ready = [], []
    stored = np.zeros(9, dtype=np.float32)
    assets = []
    for tile in tiles:
        kind, crop, animal = (int(round(tile[i])) for i in (0, 1, 2))
        if kind == 3 and 0 <= crop < 5:
            product, first = crop, int(round(tile[3])) + FIRST_CROP_DAY[crop]
        elif kind == 6 and 9 <= animal < 12:
            product, first = animal - 4, int(round(tile[4])) + FIRST_ANIMAL_DAY[animal - 9]
        else:
            continue
        # Normalized float32 observations can reconstruct an integer zero as
        # a tiny positive number; use the simulator's integer tile ABI.
        yield_units = max(0, int(round(float(tile[5]))))
        saleable = yield_units if not mature_stored or first <= day else 0
        stored[product] += saleable
        assets.append((product, first, saleable))
    for days in HORIZON_DAYS:
        end = min(720, step + 24 * days)
        town_ticks = sum(t % 4 == 0 for t in range(step, end))
        center_ticks = sum(t % 24 == 0 for t in range(step, end))
        demand = rate * town_ticks
        demand[:8] += center_ticks
        pressure.extend((inventory - 10000 - demand) / 500.0)
        ready = np.zeros(9, dtype=np.float32)
        for product, first, yield_units in assets:
            if first < end // 24 or yield_units > 0:
                ready[product] += 1
        rival_ready.extend(ready / 25.0)
    result = np.concatenate((shop_counts / 8.0, pressure, rival_ready,
                             stored / 100.0)).astype(np.float32)
    if result.shape != (FEATURE_WIDTH,) or not np.isfinite(result).all():
        raise ValueError("invalid economic feature vector")
    return result


def prefix_flow_features(resource: np.ndarray, day: int) -> np.ndarray:
    """Summarize the existing, *planned* 30-day portfolio flow by horizon."""
    if not 0 <= day < 30:
        raise ValueError("day outside game")
    flow = np.asarray(resource, dtype=np.float32)[17:347].reshape(30, 11)[:, :9]
    result = np.concatenate([
        flow[day:min(30, day + days)].sum(axis=0) / 500.0
        for days in HORIZON_DAYS
    ]).astype(np.float32)
    if result.shape != (PREFIX_WIDTH,) or not np.isfinite(result).all():
        raise ValueError("invalid prefix flow summary")
    return result


def shop_rate_features(raw: np.ndarray) -> np.ndarray:
    """Exact, public town consumption per market tick, normalized by eight."""
    shops = _unpack(np.asarray(raw))[3]
    result = (np.bincount(shops, minlength=8) @ SHOP_DEMAND / 8.0).astype(np.float32)
    if result.shape != (SHOP_RESOURCE_WIDTH,):
        raise ValueError("invalid shop rate vector")
    return result


def demo() -> None:
    # The new shop changes exactly 2 carrots per town tick, no future RNG.
    assert np.array_equal(SHOP_DEMAND[4], [0, 2, 0, 0, 0, 0, 0, 0, 0])
    assert FEATURE_WIDTH == 71
    raw = [288, 12, 0, 0]
    for farm in range(2):
        raw.extend([0, 0, 0, 0, 1, 0])
        tiles = np.zeros((100, 15), dtype=np.float32)
        if farm == 1:
            tiles[0, [0, 1, 3, 5]] = [3, 0, 12, 1]
        raw.extend(tiles.ravel())
    raw.extend([0] * 17 + [1] + [0] * 12 + [0] + [10000] * 9 +
               [0] * 9 + [0])
    old = economic_features(np.asarray(raw, dtype=np.float32))
    corrected = economic_features(np.asarray(raw, dtype=np.float32),
                                  mature_stored=True)
    assert old[62] == .01 and corrected[62] == 0
    assert old[35] == 1 / 25 and corrected[35] == 0
    resource = np.zeros(347, dtype=np.float32)
    resource[17 + 11 * 12 + 1] = 5
    assert np.all(prefix_flow_features(resource, 12)[[1, 10, 19]] == .01)
    assert np.array_equal(shop_rate_features(np.asarray(raw, dtype=np.float32)),
                          np.zeros(SHOP_RESOURCE_WIDTH, dtype=np.float32))
    with_shop = np.asarray(raw[:-1] + [1, 4], dtype=np.float32)
    assert shop_rate_features(with_shop)[1] == .25


if __name__ == "__main__":
    demo()
