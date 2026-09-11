"""Exact host assertions against canonical official trace frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from kaggriculture_jax.constants import (
    ANIMALS,
    CROPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    PRODUCTS,
    SHED_ITEMS,
    SHOP_NAMES,
    TileKind,
)
from kaggriculture_jax.types import State


def _bool_flag(flags: int, bit: int) -> bool:
    return bool(int(flags) & bit)


def assert_state_matches_frame(state: State, frame: dict[str, Any]) -> None:
    prefix = f"frame={frame['frame']} step={frame['step']}"
    assert int(state.step) == frame["step"], prefix
    assert int(state.step) // 24 == frame["day"], prefix
    assert int(state.step) % 24 == frame["hour"], prefix
    expected_done = frame["status"] == ["DONE", "DONE"]
    assert bool(state.done) == expected_done, prefix
    np.testing.assert_array_equal(
        state.reward, np.asarray(frame["reward"], dtype=np.int32), err_msg=prefix
    )

    for player, farm in enumerate(frame["farms"]):
        pfx = f"{prefix} player={player}"
        assert int(state.money[player]) == int(farm["money"]), (
            f"{pfx} money actual={int(state.money[player])} "
            f"expected={int(farm['money'])}"
        )
        assert int(state.hires_today[player]) == farm["hires_today"], pfx
        assert int(state.unlocked_count[player]) == len(farm["unlocked_quadrants"]), pfx
        active = np.flatnonzero(state.unit_active[player])
        assert len(active) == 1 + len(farm["hands"]), pfx
        np.testing.assert_array_equal(state.unit_pos[player, 0], farm["farmer"], err_msg=pfx)
        if farm["hands"]:
            np.testing.assert_array_equal(
                state.unit_pos[player, 1 : 1 + len(farm["hands"])],
                farm["hands"],
                err_msg=pfx,
            )

        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                tpfx = f"{pfx} tile=({x},{y})"
                kind = int(state.tile_kind[player, y, x])
                if tile is None:
                    assert kind == TileKind.EMPTY, tpfx
                    continue
                if tile == "LOCKED":
                    assert kind == TileKind.LOCKED, tpfx
                    continue
                official_kind = tile["kind"]
                expected_kind = {
                    "WEED": TileKind.WEED,
                    "PLANT": TileKind.PLANT,
                    "COOP": TileKind.COOP,
                    "PASTURE": TileKind.PASTURE,
                }[official_kind]
                assert kind == expected_kind, tpfx
                if official_kind == "PLANT":
                    assert int(state.tile_crop[player, y, x]) == CROPS.index(tile["crop"]), tpfx
                    assert int(state.tile_origin_day[player, y, x]) == tile["planted_day"], tpfx
                    assert int(state.tile_neglect[player, y, x]) == tile["consecutive_unwatered"], tpfx
                    assert int(state.tile_yield[player, y, x]) == tile["yield_units"], tpfx
                    assert int(state.tile_max_lifespan[player, y, x]) == tile["max_lifespan_step"], tpfx
                    assert int(state.tile_fertilized_until[player, y, x]) == tile["fertilized_until_day"], tpfx
                    assert _bool_flag(
                        state.tile_flags[player, y, x], FLAG_WATERED
                    ) == tile["watered_today"], tpfx
                if "animal" in tile:
                    assert int(state.tile_animal[player, y, x]) == ANIMALS.index(tile["animal"]), tpfx
                    assert int(state.tile_origin_day[player, y, x]) == tile["placed_day"], tpfx
                    assert int(state.tile_yield[player, y, x]) == tile["yield_units"], tpfx
                    assert int(state.tile_neglect[player, y, x]) == tile["consecutive_unfed"], tpfx
                    assert int(state.tile_pending_care[player, y, x]) == tile.get(
                        "pending_care_bonus", 0
                    ), tpfx
                    flags = state.tile_flags[player, y, x]
                    assert _bool_flag(flags, FLAG_FED) == tile["fed_today"], tpfx
                    assert _bool_flag(flags, FLAG_CARED) == tile["cared_today"], tpfx
                    assert _bool_flag(
                        flags, FLAG_FERTILIZER_AVAILABLE
                    ) == tile["fertilizer_available"], tpfx
                else:
                    assert int(state.tile_animal[player, y, x]) == -1, tpfx

        private = frame["private"][player]
        np.testing.assert_array_equal(
            state.shed[player],
            [private["shed"][item] for item in SHED_ITEMS],
            err_msg=pfx,
        )
        np.testing.assert_array_equal(
            state.seeds[player],
            [private["seeds"][crop] for crop in CROPS],
            err_msg=pfx,
        )
        assert len(private["inventories"]) == len(active), pfx
        for unit, inventory in enumerate(private["inventories"]):
            np.testing.assert_array_equal(
                state.unit_inventory[player, unit],
                [inventory.get(item, 0) for item in SHED_ITEMS],
                err_msg=f"{pfx} unit={unit}",
            )

    np.testing.assert_array_equal(
        state.market_inventory,
        [frame["market"]["inventory"][item] for item in PRODUCTS],
        err_msg=prefix,
    )
    np.testing.assert_array_equal(
        state.market_price,
        [frame["market"]["prices"][item] for item in PRODUCTS],
        err_msg=prefix,
    )
    expected_shops = [SHOP_NAMES.index(name) for name in frame["town"]["unlocked_shops"]]
    assert int(state.town_count) == len(expected_shops), prefix
    np.testing.assert_array_equal(
        state.town_shops[: len(expected_shops)], expected_shops, err_msg=prefix
    )
    assert np.all(np.asarray(state.hand_cap_hits) == 0), prefix
    assert int(state.market_loop_cap_hits) == 0, prefix
    assert int(state.price_lut_oob) == 0, prefix
