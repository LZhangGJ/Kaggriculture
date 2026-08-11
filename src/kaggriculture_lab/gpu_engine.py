"""Pure-tensor Kaggriculture transition engine for massive CUDA self-play.

This is a fixed-shape rewrite of the official 1.32.6 Python interpreter.  It keeps
thousands of games in structure-of-arrays tensors and advances the whole batch on
one device.  The public dictionary engine remains the submission oracle.

The high-throughput RL profile is the official 10x10 / 24 turns per day / 720 turn
game with at most 16 simultaneous hands and bounded market quantities.  Invalid
actions are silent no-ops, matching the official interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the GPU extra with `pip install -e '.[gpu]'`") from exc

from .triton_ops import (
    COMMON_UNIT_OPS,
    INVENTORY_UNIT_OPS,
    TRITON_AVAILABLE,
    run_common_interactions,
    run_decay_plants,
    run_dynamic_market,
    run_end_of_day,
    run_inventory_interactions,
    run_move_units,
    run_town_consume,
)


# Board codes.
EMPTY = 0
LOCKED = 1
WEED = 2
PLANT_BASE = 3  # + crop index (0..4)
COOP = 8
PASTURE = 9
ANIMAL_BASE = 10  # + animal index (0..2)

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = PRODUCTS + ANIMALS
SHOP_NAMES = tuple(sorted(("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")))

CROP_INDEX = {name: index for index, name in enumerate(CROPS)}
ANIMAL_INDEX = {name: index for index, name in enumerate(ANIMALS)}
PRODUCT_INDEX = {name: index for index, name in enumerate(PRODUCTS)}
ITEM_INDEX = {name: index for index, name in enumerate(ITEMS)}

# Unit operation codes.
U_PASS, U_NORTH, U_SOUTH, U_EAST, U_WEST = range(5)
U_DROP, U_DIG, U_WATER, U_HARVEST, U_FERTILIZE = range(5, 10)
U_BUILD_COOP, U_BUILD_PASTURE, U_FEED, U_COLLECT_FERTILIZER, U_CARE = range(10, 15)
U_PLANT, U_PICKUP, U_PLACE = range(15, 18)

UNIT_OP_INDEX = {
    "PASS": U_PASS,
    "NORTH": U_NORTH,
    "SOUTH": U_SOUTH,
    "EAST": U_EAST,
    "WEST": U_WEST,
    "DROP": U_DROP,
    "DIG": U_DIG,
    "WATER": U_WATER,
    "HARVEST": U_HARVEST,
    "FERTILIZE": U_FERTILIZE,
    "BUILD_COOP": U_BUILD_COOP,
    "BUILD_PASTURE": U_BUILD_PASTURE,
    "FEED": U_FEED,
    "COLLECT_FERTILIZER": U_COLLECT_FERTILIZER,
    "CARE": U_CARE,
    "PLANT": U_PLANT,
    "PICKUP": U_PICKUP,
    "PLACE": U_PLACE,
}

# Market operation codes.
M_NONE, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL = range(7)
MARKET_OP_INDEX = {
    "HIRE": M_HIRE,
    "BUY_LAND": M_BUY_LAND,
    "BUY_SEED": M_BUY_SEED,
    "BUY_PRODUCT": M_BUY_PRODUCT,
    "BUY_ANIMAL": M_BUY_ANIMAL,
    "SELL": M_SELL,
}


@dataclass(frozen=True)
class GpuEngineConfig:
    board_size: int = 10
    episode_steps: int = 720
    turns_per_day: int = 24
    starting_money: int = 3000
    max_hands: int = 16
    max_market_orders: int = 10
    max_market_quantity: int = 100
    shed_capacity: int = 100
    weed_spawn_chance: float = 0.005
    town_shop_unlock_interval: int = 3
    town_shop_sell_interval: int = 4
    town_center_sell_interval: int = 24
    farm_hand_cost_mult: int = 1
    use_triton: bool = True

    def __post_init__(self) -> None:
        if self.board_size != 10:
            raise ValueError("The CUDA engine currently specializes the official 10x10 board")
        if self.max_hands <= 0:
            raise ValueError("max_hands must be positive")
        if self.max_market_orders <= 0 or self.max_market_quantity <= 0:
            raise ValueError("market bounds must be positive")


@dataclass
class TensorActions:
    unit_ops: torch.Tensor  # [B, 2, U]
    unit_args: torch.Tensor  # [B, 2, U]
    unit_quantities: torch.Tensor  # [B, 2, U]
    market_ops: torch.Tensor  # [B, 2, M]
    market_args: torch.Tensor  # [B, 2, M]
    market_quantities: torch.Tensor  # [B, 2, M]


@dataclass
class TensorState:
    money: torch.Tensor
    tile_type: torch.Tensor
    planted_day: torch.Tensor
    placed_day: torch.Tensor
    watered: torch.Tensor
    fed: torch.Tensor
    cared: torch.Tensor
    consecutive: torch.Tensor
    yield_units: torch.Tensor
    max_lifespan_step: torch.Tensor
    fertilized_until_day: torch.Tensor
    fertilizer_available: torch.Tensor
    pending_care_bonus: torch.Tensor
    positions: torch.Tensor
    unit_active: torch.Tensor
    hands_count: torch.Tensor
    hires_today: torch.Tensor
    unlocked_count: torch.Tensor
    shed: torch.Tensor
    seeds: torch.Tensor
    unit_inventory: torch.Tensor
    market_inventory: torch.Tensor
    market_prices: torch.Tensor
    shop_counts: torch.Tensor


@dataclass(frozen=True)
class TensorStep:
    rewards: torch.Tensor
    done: bool
    step: int


def _constant_tensor(values: Sequence[Any], *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


class CudaKaggricultureEnv:
    """Batched fixed-shape Kaggriculture environment living on one device."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cuda",
        config: GpuEngineConfig | None = None,
        seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = config or GpuEngineConfig()
        self.players = 2
        self.max_units = self.config.max_hands + 1
        self.use_triton = self.config.use_triton and self.device.type == "cuda" and TRITON_AVAILABLE
        self.step_index = 0
        self.done = False
        self._setup_constants()
        self.reset(seeds)

    def _setup_constants(self) -> None:
        device = self.device
        self.crop_seed_cost = _constant_tensor((10, 20, 50, 100, 80), device=device)
        self.crop_first = _constant_tensor((2, 2, 8, 10, 10), device=device, dtype=torch.int64)
        self.crop_max_day = _constant_tensor((4, 3, 8, 10, 12), device=device, dtype=torch.int64)
        self.crop_interval = _constant_tensor((0, 0, 1, 2, 0), device=device, dtype=torch.int64)
        self.crop_max_yield = _constant_tensor((6, 4, 4, 4, 6), device=device, dtype=torch.int64)
        self.crop_ongoing = _constant_tensor((False, False, True, True, False), device=device, dtype=torch.bool)
        self.animal_cost = _constant_tensor((300, 400, 500), device=device)
        self.animal_structure = _constant_tensor((COOP, PASTURE, PASTURE), device=device, dtype=torch.int64)
        self.animal_first = _constant_tensor((4, 8, 6), device=device, dtype=torch.int64)
        self.animal_interval = _constant_tensor((1, 2, 3), device=device, dtype=torch.int64)
        self.animal_max_held = _constant_tensor((4, 6, 6), device=device, dtype=torch.int64)
        self.animal_product = _constant_tensor((5, 6, 7), device=device, dtype=torch.int64)

        self.market_base = _constant_tensor((25, 35, 60, 120, 250, 50, 160, 200, 100), device=device)
        self.market_i0 = _constant_tensor((10000,) * 9, device=device)
        self.market_t = _constant_tensor((400, 450, 200, 100, 300, 332, 122, 105, 200), device=device)
        # 0 linear, 1 square, 2 sqrt, 3 log1p.
        self.market_below_func = _constant_tensor((2, 3, 0, 2, 3, 0, 2, 3, 0), device=device, dtype=torch.int64)
        self.market_below_target = _constant_tensor((0.80, 0.20, 0.40, 0.70, 0.20, 0.40, 0.60, 0.20, 0.40), device=device)
        self.market_above_func = _constant_tensor((3, 2, 2, 0, 1, 3, 0, 1, 0), device=device, dtype=torch.int64)
        self.market_above_target = _constant_tensor((0.20, 0.70, 0.60, 1.60, 3.60, 0.20, 1.60, 3.20, 0.40), device=device)

        demand = torch.zeros((len(SHOP_NAMES), len(PRODUCTS)), device=device, dtype=torch.int64)
        definitions = {
            "BAKERY": ("EGG", "WHEAT"),
            "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
            "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
            "YARN_STORE": ("WOOL",),
            "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
            "PET_CAFE": ("CARROT",),
            "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
            "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
        }
        for shop_index, shop in enumerate(SHOP_NAMES):
            multiplier = 2 if len(definitions[shop]) == 1 else 1
            for product in definitions[shop]:
                demand[shop_index, PRODUCT_INDEX[product]] = multiplier
        self.shop_demand = demand
        fib = [1, 1]
        for _ in range(self.config.max_hands + 2):
            fib.append(fib[-1] + fib[-2])
        self.hire_cost = _constant_tensor(fib[: self.config.max_hands + 1], device=device) * self.config.farm_hand_cost_mult
        self.land_prices = _constant_tensor((1000, 2000, 4000, 10**9), device=device)
        self._batch = torch.arange(self.num_envs, device=device, dtype=torch.long)[:, None].expand(-1, 2)
        self._player = torch.arange(2, device=device, dtype=torch.long)[None, :].expand(self.num_envs, -1)

    def reset(self, seeds: Sequence[int] | torch.Tensor | None = None) -> TensorState:
        cfg = self.config
        device = self.device
        b, p, h, w, u = self.num_envs, 2, cfg.board_size, cfg.board_size, self.max_units
        if seeds is None:
            seeds = torch.arange(b, device=device, dtype=torch.int64)
        self.episode_seeds = torch.as_tensor(seeds, device=device, dtype=torch.int64).reshape(b)
        self.step_index = 0
        self.done = False

        tile_type = torch.full((b, p, h, w), LOCKED, device=device, dtype=torch.int8)
        tile_type[:, :, : h // 2, : w // 2] = EMPTY
        positions = torch.zeros((b, p, u, 2), device=device, dtype=torch.int8)
        positions[:, :, 0, 0] = w // 2 - 1
        positions[:, :, 0, 1] = h // 2 - 1
        unit_active = torch.zeros((b, p, u), device=device, dtype=torch.bool)
        unit_active[:, :, 0] = True

        board_i16 = lambda fill=0: torch.full((b, p, h, w), fill, device=device, dtype=torch.int16)
        self.state = TensorState(
            money=torch.full((b, p), float(cfg.starting_money), device=device),
            tile_type=tile_type,
            planted_day=board_i16(-1),
            placed_day=board_i16(-1),
            watered=torch.zeros((b, p, h, w), device=device, dtype=torch.bool),
            fed=torch.zeros((b, p, h, w), device=device, dtype=torch.bool),
            cared=torch.zeros((b, p, h, w), device=device, dtype=torch.bool),
            consecutive=torch.zeros((b, p, h, w), device=device, dtype=torch.int8),
            yield_units=board_i16(0),
            max_lifespan_step=board_i16(-1),
            fertilized_until_day=board_i16(-1),
            fertilizer_available=torch.zeros((b, p, h, w), device=device, dtype=torch.bool),
            pending_care_bonus=board_i16(0),
            positions=positions,
            unit_active=unit_active,
            hands_count=torch.zeros((b, p), device=device, dtype=torch.int16),
            hires_today=torch.zeros((b, p), device=device, dtype=torch.int16),
            unlocked_count=torch.ones((b, p), device=device, dtype=torch.int8),
            shed=torch.zeros((b, p, len(ITEMS)), device=device, dtype=torch.int16),
            seeds=torch.zeros((b, p, len(CROPS)), device=device, dtype=torch.int16),
            unit_inventory=torch.zeros((b, p, u, len(ITEMS)), device=device, dtype=torch.int16),
            market_inventory=torch.full((b, len(PRODUCTS)), 10000, device=device, dtype=torch.int32),
            market_prices=self.market_base.to(torch.int32).expand(b, -1).clone(),
            shop_counts=torch.zeros((b, len(SHOP_NAMES)), device=device, dtype=torch.int8),
        )
        return self.state

    def empty_actions(self) -> TensorActions:
        cfg = self.config
        shape_units = (self.num_envs, 2, self.max_units)
        shape_market = (self.num_envs, 2, cfg.max_market_orders)
        return TensorActions(
            unit_ops=torch.zeros(shape_units, device=self.device, dtype=torch.int64),
            unit_args=torch.zeros(shape_units, device=self.device, dtype=torch.int64),
            unit_quantities=torch.ones(shape_units, device=self.device, dtype=torch.int64),
            market_ops=torch.zeros(shape_market, device=self.device, dtype=torch.int64),
            market_args=torch.zeros(shape_market, device=self.device, dtype=torch.int64),
            market_quantities=torch.zeros(shape_market, device=self.device, dtype=torch.int64),
        )

    def _shape(self, code: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp_min(x, 0.0)
        return torch.where(
            code == 0,
            x,
            torch.where(code == 1, x * x, torch.where(code == 2, torch.sqrt(x), torch.log1p(x))),
        )

    def _prices_for(self, item: torch.Tensor, inventory: torch.Tensor) -> torch.Tensor:
        item = item.clamp(0, len(PRODUCTS) - 1)
        base = self.market_base[item]
        i0 = self.market_i0[item]
        target = torch.where(inventory < i0, self.market_below_target[item], self.market_above_target[item])
        func = torch.where(inventory < i0, self.market_below_func[item], self.market_above_func[item])
        x = torch.abs(inventory.float() - i0)
        amp = target * base / self._shape(func, self.market_t[item])
        price = torch.where(inventory < i0, base + amp * self._shape(func, x), base - amp * self._shape(func, x))
        return torch.round(price).clamp_min(1.0)

    def _refresh_prices(self) -> None:
        inventory = self.state.market_inventory.float()
        item = torch.arange(len(PRODUCTS), device=self.device, dtype=torch.long)[None, :].expand(self.num_envs, -1)
        self.state.market_prices.copy_(self._prices_for(item, inventory).to(torch.int32))

    def _clear_tile_fields(self, mask: torch.Tensor) -> None:
        state = self.state
        for name in ("planted_day", "placed_day", "max_lifespan_step", "fertilized_until_day"):
            getattr(state, name)[mask] = -1
        for name in ("watered", "fed", "cared", "fertilizer_available"):
            getattr(state, name)[mask] = False
        for name in ("consecutive", "yield_units", "pending_care_bonus"):
            getattr(state, name)[mask] = 0

    def _board_indices(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (((self._batch * 2 + self._player) * 100) + y * 10 + x).reshape(-1)

    def _gather_board(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1).gather(0, indices).reshape(self.num_envs, 2)

    def _scatter_board(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor | int | float | bool,
        mask: torch.Tensor,
    ) -> None:
        flat = tensor.reshape(-1)
        old = flat.gather(0, indices)
        value_tensor = torch.as_tensor(values, device=self.device, dtype=tensor.dtype).expand_as(mask)
        flat.scatter_(0, indices, torch.where(mask, value_tensor, old.reshape_as(mask)).reshape(-1))

    def _move_units_dense(self, actions: TensorActions, unit_limit: int) -> None:
        """Advance every active moving unit in one fixed-shape CUDA operation."""
        s = self.state
        if self.use_triton:
            run_move_units(s, actions, unit_limit)
            return
        op = actions.unit_ops[:, :, :unit_limit]
        active = s.unit_active[:, :, :unit_limit]
        x = s.positions[:, :, :unit_limit, 0].long()
        y = s.positions[:, :, :unit_limit, 1].long()
        dx = (op == U_EAST).long() - (op == U_WEST).long()
        dy = (op == U_SOUTH).long() - (op == U_NORTH).long()
        moving = active & (op >= U_NORTH) & (op <= U_WEST)
        nx, ny = x + dx, y + dy
        valid = moving & (nx >= 0) & (nx < 10) & (ny >= 0) & (ny < 10)
        s.positions[:, :, :unit_limit, 0].copy_(torch.where(valid, nx, x).to(torch.int8))
        s.positions[:, :, :unit_limit, 1].copy_(torch.where(valid, ny, y).to(torch.int8))

    def _step_unit_dense(
        self,
        actions: TensorActions,
        unit: int,
        day: int,
        plant_allowed: torch.Tensor,
        active_override: torch.Tensor | None = None,
    ) -> None:
        """Static-shape unit transition with no data-dependent host branches."""
        s = self.state
        cfg = self.config
        op = actions.unit_ops[:, :, unit]
        arg = actions.unit_args[:, :, unit]
        quantity = actions.unit_quantities[:, :, unit].clamp_min(1)
        active = s.unit_active[:, :, unit]
        if active_override is not None:
            active = active & active_override
        x = s.positions[:, :, unit, 0].long()
        y = s.positions[:, :, unit, 1].long()

        board_index = self._board_indices(x, y)
        tile = self._gather_board(s.tile_type, board_index).long()
        half = 5
        shed_adjacent = ((x == half - 1) | (x == half)) & ((y == half - 1) | (y == half))
        item = arg.clamp(0, len(ITEMS) - 1)
        inv = s.unit_inventory[:, :, unit]

        pickup = active & (op == U_PICKUP) & shed_adjacent & (arg >= 0) & (arg < len(ITEMS))
        available = s.shed.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        take = torch.where(pickup, torch.minimum(quantity, available), 0)
        s.shed.scatter_add_(2, item.unsqueeze(-1), (-take).to(s.shed.dtype).unsqueeze(-1))
        inv.scatter_add_(2, item.unsqueeze(-1), take.to(inv.dtype).unsqueeze(-1))

        drop = active & (op == U_DROP) & shed_adjacent
        for item_index in range(len(ITEMS)):
            room = (cfg.shed_capacity - s.shed.sum(dim=2).long()).clamp_min(0)
            amount = torch.where(drop, torch.minimum(inv[:, :, item_index].long(), room), 0)
            s.shed[:, :, item_index].add_(amount.to(s.shed.dtype))
            inv[:, :, item_index].sub_(amount.to(inv.dtype))

        is_animal_arg = (arg >= len(PRODUCTS)) & (arg < len(ITEMS))
        animal_index = (arg - len(PRODUCTS)).clamp(0, len(ANIMALS) - 1)
        structure = self.animal_structure[animal_index]
        animal_structure_match = active & (op == U_PLACE) & is_animal_arg & (tile == structure)
        carried = inv.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        place_animal = animal_structure_match & (carried > 0)
        inv.scatter_add_(2, item.unsqueeze(-1), (-place_animal.long()).to(inv.dtype).unsqueeze(-1))
        self._scatter_board(s.tile_type, board_index, ANIMAL_BASE + animal_index, place_animal)
        self._scatter_board(s.placed_day, board_index, day, place_animal)
        for name in ("yield_units", "consecutive", "pending_care_bonus"):
            self._scatter_board(getattr(s, name), board_index, 0, place_animal)
        for name in ("fed", "cared", "fertilizer_available"):
            self._scatter_board(getattr(s, name), board_index, False, place_animal)

        place_shed = active & (op == U_PLACE) & shed_adjacent & ~animal_structure_match & (arg >= 0) & (arg < len(ITEMS))
        carried = inv.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        room = (cfg.shed_capacity - s.shed.sum(dim=2).long()).clamp_min(0)
        place_amount = torch.where(place_shed, torch.minimum(quantity, torch.minimum(carried, room)), 0)
        inv.scatter_add_(2, item.unsqueeze(-1), (-place_amount).to(inv.dtype).unsqueeze(-1))
        s.shed.scatter_add_(2, item.unsqueeze(-1), place_amount.to(s.shed.dtype).unsqueeze(-1))

        owned = active & (tile != LOCKED)
        crop_arg = arg.clamp(0, len(CROPS) - 1)
        seed_available = s.seeds.gather(2, crop_arg.unsqueeze(-1)).squeeze(-1) > 0
        plant = owned & (op == U_PLANT) & plant_allowed[:, :, unit] & (tile == EMPTY) & seed_available & (arg >= 0) & (arg < len(CROPS))
        s.seeds.scatter_add_(2, crop_arg.unsqueeze(-1), (-plant.long()).to(s.seeds.dtype).unsqueeze(-1))
        self._scatter_board(s.tile_type, board_index, PLANT_BASE + crop_arg, plant)
        self._scatter_board(s.planted_day, board_index, day, plant)
        self._scatter_board(s.watered, board_index, False, plant)
        self._scatter_board(s.consecutive, board_index, 1, plant)
        self._scatter_board(s.yield_units, board_index, torch.where(self.crop_ongoing[crop_arg], 0, 1), plant)
        max_life = torch.where(
            self.crop_ongoing[crop_arg],
            torch.full_like(crop_arg, -1),
            (day + self.crop_max_day[crop_arg] + 1) * cfg.turns_per_day,
        )
        self._scatter_board(s.max_lifespan_step, board_index, max_life, plant)
        self._scatter_board(s.fertilized_until_day, board_index, -1, plant)

        plant_tile = (tile >= PLANT_BASE) & (tile < PLANT_BASE + len(CROPS))
        crop = (tile - PLANT_BASE).clamp(0, len(CROPS) - 1)
        board_watered = self._gather_board(s.watered, board_index)
        water = owned & (op == U_WATER) & plant_tile & ~board_watered
        self._scatter_board(s.watered, board_index, True, water)
        planted_day = self._gather_board(s.planted_day, board_index).long()
        current_yield = self._gather_board(s.yield_units, board_index).long()
        fert_until = self._gather_board(s.fertilized_until_day, board_index).long()
        age = day - planted_day
        window_start = (self.crop_max_day[crop] + 1) // 2
        bonus_window = ~self.crop_ongoing[crop] & (age >= window_start) & (age <= self.crop_max_day[crop])
        bonus = torch.where(fert_until >= day, 2, 1)
        watered_yield = torch.minimum(current_yield + bonus, self.crop_max_yield[crop])
        self._scatter_board(s.yield_units, board_index, watered_yield, water & bonus_window)

        mature = age >= self.crop_first[crop]
        harvest_plant = owned & (op == U_HARVEST) & plant_tile & (current_yield > 0) & mature
        plant_amount = torch.where(harvest_plant, current_yield, 0)
        inv.scatter_add_(2, crop.unsqueeze(-1), plant_amount.to(inv.dtype).unsqueeze(-1))
        self._scatter_board(s.yield_units, board_index, 0, harvest_plant)
        clear_plant = harvest_plant & ~self.crop_ongoing[crop]
        self._scatter_board(s.tile_type, board_index, EMPTY, clear_plant)
        for name in ("planted_day", "placed_day", "max_lifespan_step", "fertilized_until_day"):
            self._scatter_board(getattr(s, name), board_index, -1, clear_plant)
        for name in ("watered", "fed", "cared", "fertilizer_available"):
            self._scatter_board(getattr(s, name), board_index, False, clear_plant)
        for name in ("consecutive", "yield_units", "pending_care_bonus"):
            self._scatter_board(getattr(s, name), board_index, 0, clear_plant)

        animal_tile = (tile >= ANIMAL_BASE) & (tile < ANIMAL_BASE + len(ANIMALS))
        animal = (tile - ANIMAL_BASE).clamp(0, len(ANIMALS) - 1)
        harvest_animal = owned & (op == U_HARVEST) & animal_tile & (current_yield > 0)
        animal_amount = torch.where(harvest_animal, current_yield, 0)
        inv.scatter_add_(2, self.animal_product[animal].unsqueeze(-1), animal_amount.to(inv.dtype).unsqueeze(-1))
        self._scatter_board(s.yield_units, board_index, 0, harvest_animal)

        fertilizer = inv[:, :, PRODUCT_INDEX["FERTILIZER"]] > 0
        fertilize = owned & (op == U_FERTILIZE) & plant_tile & fertilizer
        inv[:, :, PRODUCT_INDEX["FERTILIZER"]].sub_(fertilize.to(inv.dtype))
        self._scatter_board(s.fertilized_until_day, board_index, torch.maximum(fert_until, torch.full_like(fert_until, day + 2)), fertilize)

        dig = owned & (op == U_DIG) & (tile != EMPTY) & ~animal_tile
        self._scatter_board(s.tile_type, board_index, EMPTY, dig)
        for name in ("planted_day", "placed_day", "max_lifespan_step", "fertilized_until_day"):
            self._scatter_board(getattr(s, name), board_index, -1, dig)
        for name in ("watered", "fed", "cared", "fertilizer_available"):
            self._scatter_board(getattr(s, name), board_index, False, dig)
        for name in ("consecutive", "yield_units", "pending_care_bonus"):
            self._scatter_board(getattr(s, name), board_index, 0, dig)

        self._scatter_board(s.tile_type, board_index, COOP, owned & (op == U_BUILD_COOP) & (tile == EMPTY))
        self._scatter_board(s.tile_type, board_index, PASTURE, owned & (op == U_BUILD_PASTURE) & (tile == EMPTY))

        board_fed = self._gather_board(s.fed, board_index)
        feed = owned & (op == U_FEED) & animal_tile & ~board_fed & (inv[:, :, PRODUCT_INDEX["WHEAT"]] > 0)
        inv[:, :, PRODUCT_INDEX["WHEAT"]].sub_(feed.to(inv.dtype))
        self._scatter_board(s.fed, board_index, True, feed)
        available_fert = self._gather_board(s.fertilizer_available, board_index)
        collect = owned & (op == U_COLLECT_FERTILIZER) & animal_tile & available_fert
        self._scatter_board(s.fertilizer_available, board_index, False, collect)
        inv[:, :, PRODUCT_INDEX["FERTILIZER"]].add_(collect.to(inv.dtype))
        board_cared = self._gather_board(s.cared, board_index)
        self._scatter_board(s.cared, board_index, True, owned & (op == U_CARE) & animal_tile & ~board_cared)

    def _step_unit(self, actions: TensorActions, unit: int, day: int, plant_allowed: torch.Tensor) -> None:
        s = self.state
        cfg = self.config
        op = actions.unit_ops[:, :, unit]
        arg = actions.unit_args[:, :, unit]
        quantity = actions.unit_quantities[:, :, unit].clamp_min(1)
        active = s.unit_active[:, :, unit]
        x = s.positions[:, :, unit, 0].long()
        y = s.positions[:, :, unit, 1].long()

        dx = (op == U_EAST).long() - (op == U_WEST).long()
        dy = (op == U_SOUTH).long() - (op == U_NORTH).long()
        moving = active & ((op >= U_NORTH) & (op <= U_WEST))
        nx, ny = x + dx, y + dy
        valid_move = moving & (nx >= 0) & (nx < cfg.board_size) & (ny >= 0) & (ny < cfg.board_size)
        s.positions[:, :, unit, 0] = torch.where(valid_move, nx, x).to(torch.int8)
        s.positions[:, :, unit, 1] = torch.where(valid_move, ny, y).to(torch.int8)
        x = s.positions[:, :, unit, 0].long()
        y = s.positions[:, :, unit, 1].long()
        tile = s.tile_type[self._batch, self._player, y, x].long()

        half = cfg.board_size // 2
        shed_adjacent = ((x == half - 1) | (x == half)) & ((y == half - 1) | (y == half))
        item = arg.clamp(0, len(ITEMS) - 1)
        inv = s.unit_inventory[:, :, unit]

        pickup = active & (op == U_PICKUP) & shed_adjacent & (arg >= 0) & (arg < len(ITEMS))
        available = s.shed.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        take = torch.minimum(quantity, available)
        take = torch.where(pickup, take, torch.zeros_like(take))
        s.shed.scatter_add_(2, item.unsqueeze(-1), (-take).to(s.shed.dtype).unsqueeze(-1))
        inv.scatter_add_(2, item.unsqueeze(-1), take.to(inv.dtype).unsqueeze(-1))

        drop = active & (op == U_DROP) & shed_adjacent
        for item_index in range(len(ITEMS)):
            room = cfg.shed_capacity - s.shed.sum(dim=2).long()
            amount = torch.minimum(inv[:, :, item_index].long(), room.clamp_min(0))
            amount = torch.where(drop, amount, torch.zeros_like(amount))
            s.shed[:, :, item_index] += amount.to(s.shed.dtype)
            inv[:, :, item_index] -= amount.to(inv.dtype)

        # PLACE first tries animal placement; matching structures return even if
        # the unit does not carry the requested animal.
        is_animal_arg = (arg >= len(PRODUCTS)) & (arg < len(ITEMS))
        animal_index = (arg - len(PRODUCTS)).clamp(0, len(ANIMALS) - 1)
        structure = self.animal_structure[animal_index]
        animal_structure_match = active & (op == U_PLACE) & is_animal_arg & (tile == structure)
        carried = inv.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        place_animal = animal_structure_match & (carried > 0)
        bi, pi = self._batch[place_animal], self._player[place_animal]
        yi, xi = y[place_animal], x[place_animal]
        if bi.numel():
            chosen = animal_index[place_animal]
            s.tile_type[bi, pi, yi, xi] = (ANIMAL_BASE + chosen).to(torch.int8)
            s.placed_day[bi, pi, yi, xi] = day
            for name in ("yield_units", "consecutive", "pending_care_bonus"):
                getattr(s, name)[bi, pi, yi, xi] = 0
            for name in ("fed", "cared", "fertilizer_available"):
                getattr(s, name)[bi, pi, yi, xi] = False
            s.unit_inventory[bi, pi, unit, item[place_animal]] -= 1

        place_shed = active & (op == U_PLACE) & shed_adjacent & ~animal_structure_match & (arg >= 0) & (arg < len(ITEMS))
        carried = inv.gather(2, item.unsqueeze(-1)).squeeze(-1).long()
        room = cfg.shed_capacity - s.shed.sum(dim=2).long()
        place_amount = torch.minimum(quantity, torch.minimum(carried, room.clamp_min(0)))
        place_amount = torch.where(place_shed, place_amount, torch.zeros_like(place_amount))
        inv.scatter_add_(2, item.unsqueeze(-1), (-place_amount).to(inv.dtype).unsqueeze(-1))
        s.shed.scatter_add_(2, item.unsqueeze(-1), place_amount.to(s.shed.dtype).unsqueeze(-1))

        owned = active & (tile != LOCKED)
        crop_arg = arg.clamp(0, len(CROPS) - 1)
        plant = owned & (op == U_PLANT) & plant_allowed[:, :, unit] & (tile == EMPTY) & (arg >= 0) & (arg < len(CROPS))
        seed_available = s.seeds.gather(2, crop_arg.unsqueeze(-1)).squeeze(-1) > 0
        plant &= seed_available
        bi, pi = self._batch[plant], self._player[plant]
        yi, xi = y[plant], x[plant]
        if bi.numel():
            chosen = crop_arg[plant]
            s.seeds[bi, pi, chosen] -= 1
            s.tile_type[bi, pi, yi, xi] = (PLANT_BASE + chosen).to(torch.int8)
            s.planted_day[bi, pi, yi, xi] = day
            s.watered[bi, pi, yi, xi] = False
            s.consecutive[bi, pi, yi, xi] = 1
            s.yield_units[bi, pi, yi, xi] = torch.where(self.crop_ongoing[chosen], 0, 1).to(torch.int16)
            s.max_lifespan_step[bi, pi, yi, xi] = torch.where(
                self.crop_ongoing[chosen],
                torch.full_like(chosen, -1),
                (day + self.crop_max_day[chosen] + 1) * cfg.turns_per_day,
            ).to(torch.int16)
            s.fertilized_until_day[bi, pi, yi, xi] = -1

        plant_tile = (tile >= PLANT_BASE) & (tile < PLANT_BASE + len(CROPS))
        crop = (tile - PLANT_BASE).clamp(0, len(CROPS) - 1)
        water = owned & (op == U_WATER) & plant_tile & ~s.watered[self._batch, self._player, y, x]
        bi, pi = self._batch[water], self._player[water]
        yi, xi = y[water], x[water]
        if bi.numel():
            chosen = crop[water]
            s.watered[bi, pi, yi, xi] = True
            age = day - s.planted_day[bi, pi, yi, xi].long()
            window_start = (self.crop_max_day[chosen] + 1) // 2
            bonus_mask = ~self.crop_ongoing[chosen] & (age >= window_start) & (age <= self.crop_max_day[chosen])
            bonus = torch.where(s.fertilized_until_day[bi, pi, yi, xi] >= day, 2, 1)
            updated = torch.minimum(s.yield_units[bi, pi, yi, xi].long() + bonus, self.crop_max_yield[chosen])
            s.yield_units[bi, pi, yi, xi] = torch.where(bonus_mask, updated, s.yield_units[bi, pi, yi, xi].long()).to(torch.int16)

        current_yield = s.yield_units[self._batch, self._player, y, x].long()
        planted = s.planted_day[self._batch, self._player, y, x].long()
        mature = day - planted >= self.crop_first[crop]
        harvest_plant = owned & (op == U_HARVEST) & plant_tile & (current_yield > 0) & mature
        amount = torch.where(harvest_plant, current_yield, torch.zeros_like(current_yield))
        inv.scatter_add_(2, crop.unsqueeze(-1), amount.to(inv.dtype).unsqueeze(-1))
        bi, pi = self._batch[harvest_plant], self._player[harvest_plant]
        yi, xi = y[harvest_plant], x[harvest_plant]
        if bi.numel():
            chosen = crop[harvest_plant]
            s.yield_units[bi, pi, yi, xi] = 0
            one_time = ~self.crop_ongoing[chosen]
            if one_time.any():
                mask = torch.zeros_like(s.tile_type, dtype=torch.bool)
                mask[bi[one_time], pi[one_time], yi[one_time], xi[one_time]] = True
                s.tile_type[mask] = EMPTY
                self._clear_tile_fields(mask)

        animal_tile = (tile >= ANIMAL_BASE) & (tile < ANIMAL_BASE + len(ANIMALS))
        animal = (tile - ANIMAL_BASE).clamp(0, len(ANIMALS) - 1)
        harvest_animal = owned & (op == U_HARVEST) & animal_tile & (current_yield > 0)
        animal_amount = torch.where(harvest_animal, current_yield, torch.zeros_like(current_yield))
        product = self.animal_product[animal]
        inv.scatter_add_(2, product.unsqueeze(-1), animal_amount.to(inv.dtype).unsqueeze(-1))
        bi, pi = self._batch[harvest_animal], self._player[harvest_animal]
        if bi.numel():
            s.yield_units[bi, pi, y[harvest_animal], x[harvest_animal]] = 0

        fertilizer = inv[:, :, PRODUCT_INDEX["FERTILIZER"]] > 0
        fertilize = owned & (op == U_FERTILIZE) & plant_tile & fertilizer
        bi, pi = self._batch[fertilize], self._player[fertilize]
        if bi.numel():
            inv[:, :, PRODUCT_INDEX["FERTILIZER"]][fertilize] -= 1
            yi, xi = y[fertilize], x[fertilize]
            s.fertilized_until_day[bi, pi, yi, xi] = torch.maximum(
                s.fertilized_until_day[bi, pi, yi, xi], torch.full_like(s.fertilized_until_day[bi, pi, yi, xi], day + 2)
            )

        diggable = (tile != EMPTY) & ~animal_tile
        dig = owned & (op == U_DIG) & diggable
        bi, pi = self._batch[dig], self._player[dig]
        if bi.numel():
            mask = torch.zeros_like(s.tile_type, dtype=torch.bool)
            mask[bi, pi, y[dig], x[dig]] = True
            s.tile_type[mask] = EMPTY
            self._clear_tile_fields(mask)

        build_coop = owned & (op == U_BUILD_COOP) & (tile == EMPTY)
        build_pasture = owned & (op == U_BUILD_PASTURE) & (tile == EMPTY)
        s.tile_type[self._batch[build_coop], self._player[build_coop], y[build_coop], x[build_coop]] = COOP
        s.tile_type[self._batch[build_pasture], self._player[build_pasture], y[build_pasture], x[build_pasture]] = PASTURE

        board_fed = s.fed[self._batch, self._player, y, x]
        feed = owned & (op == U_FEED) & animal_tile & ~board_fed & (inv[:, :, PRODUCT_INDEX["WHEAT"]] > 0)
        if feed.any():
            inv[:, :, PRODUCT_INDEX["WHEAT"]][feed] -= 1
            s.fed[self._batch[feed], self._player[feed], y[feed], x[feed]] = True
        available_fert = s.fertilizer_available[self._batch, self._player, y, x]
        collect = owned & (op == U_COLLECT_FERTILIZER) & animal_tile & available_fert
        if collect.any():
            s.fertilizer_available[self._batch[collect], self._player[collect], y[collect], x[collect]] = False
            inv[:, :, PRODUCT_INDEX["FERTILIZER"]][collect] += 1
        board_cared = s.cared[self._batch, self._player, y, x]
        care = owned & (op == U_CARE) & animal_tile & ~board_cared
        s.cared[self._batch[care], self._player[care], y[care], x[care]] = True

    def _hire(self, mask: torch.Tensor) -> None:
        s = self.state
        cfg = self.config
        hires = s.hires_today.long().clamp(0, cfg.max_hands)
        cost = self.hire_cost[hires]
        valid = mask & (s.hands_count < cfg.max_hands) & (s.money >= cost)
        s.money -= torch.where(valid, cost, torch.zeros_like(cost))
        slots = (s.hands_count.long() + 1).clamp_max(cfg.max_hands)
        candidates = torch.tensor(((4, 4), (5, 4), (4, 5), (5, 5)), device=self.device, dtype=torch.int8)
        occupancy = torch.zeros((self.num_envs, 2, 4), device=self.device, dtype=torch.int16)
        for candidate in range(4):
            same = s.unit_active & (s.positions[..., 0] == candidates[candidate, 0]) & (s.positions[..., 1] == candidates[candidate, 1])
            occupancy[:, :, candidate] = same.sum(dim=2)
        chosen = occupancy.argmin(dim=2)
        slot_index = slots.unsqueeze(-1).unsqueeze(-1)
        position_index = slot_index.expand(-1, -1, 1, 2)
        old_position = s.positions.gather(2, position_index).squeeze(2)
        new_position = torch.where(valid.unsqueeze(-1), candidates[chosen], old_position)
        s.positions.scatter_(2, position_index, new_position.unsqueeze(2))
        active_index = slots.unsqueeze(-1)
        old_active = s.unit_active.gather(2, active_index).squeeze(-1)
        s.unit_active.scatter_(2, active_index, (old_active | valid).unsqueeze(-1))
        inventory_index = slot_index.expand(-1, -1, 1, len(ITEMS))
        old_inventory = s.unit_inventory.gather(2, inventory_index).squeeze(2)
        new_inventory = torch.where(valid.unsqueeze(-1), torch.zeros_like(old_inventory), old_inventory)
        s.unit_inventory.scatter_(2, inventory_index, new_inventory.unsqueeze(2))
        s.hands_count += valid.to(s.hands_count.dtype)
        s.hires_today += valid.to(s.hires_today.dtype)

    def _buy_land(self, mask: torch.Tensor) -> None:
        s = self.state
        extra = (s.unlocked_count.long() - 1).clamp(0, 3)
        cost = self.land_prices[extra]
        valid = mask & (s.unlocked_count < 4) & (s.money >= cost)
        s.money -= torch.where(valid, cost, torch.zeros_like(cost))
        # Unlock order NE, SW, SE.
        y_grid = torch.arange(10, device=self.device)[None, None, :, None]
        x_grid = torch.arange(10, device=self.device)[None, None, None, :]
        for stage, quadrant in enumerate((1, 2, 3)):
            stage_mask = valid & (extra == stage)
            if quadrant == 1:
                cells = (y_grid < 5) & (x_grid >= 5)
            elif quadrant == 2:
                cells = (y_grid >= 5) & (x_grid < 5)
            else:
                cells = (y_grid >= 5) & (x_grid >= 5)
            unlock = stage_mask[:, :, None, None] & cells & (s.tile_type == LOCKED)
            s.tile_type.masked_fill_(unlock, EMPTY)
        s.unlocked_count += valid.to(s.unlocked_count.dtype)

    def _process_market(self, actions: TensorActions, has_orders: bool | None = None) -> None:
        s = self.state
        cfg = self.config
        # Most turns contain no market order.  Avoid the bounded per-unit order
        # interpreter (and its synchronization points) on that hot path.
        if has_orders is None:
            has_orders = bool(actions.market_ops.any().item())
        if not has_orders:
            return
        for order in range(cfg.max_market_orders):
            op = actions.market_ops[:, :, order]
            arg = actions.market_args[:, :, order]
            quantity = actions.market_quantities[:, :, order].clamp(0, cfg.max_market_quantity)
            self._hire(op == M_HIRE)
            self._buy_land(op == M_BUY_LAND)

            # Seeds and animals have fixed prices, unlike products whose quotes
            # change after every unit. Their sequential official loop therefore
            # has an exact closed form and can be settled in one GPU pass.
            crop = arg.clamp(0, len(CROPS) - 1)
            seed_cost = self.crop_seed_cost[crop]
            buy_seed = (op == M_BUY_SEED) & (arg >= 0) & (arg < len(CROPS))
            seed_units = torch.where(buy_seed, torch.minimum(quantity, s.money // seed_cost), 0)
            s.money.sub_(seed_units * seed_cost)
            s.seeds.scatter_add_(2, crop.unsqueeze(-1), seed_units.to(s.seeds.dtype).unsqueeze(-1))

            animal = arg.clamp(0, len(ANIMALS) - 1)
            animal_cost = self.animal_cost[animal]
            buy_animal = (op == M_BUY_ANIMAL) & (arg >= 0) & (arg < len(ANIMALS))
            shed_room = (cfg.shed_capacity - s.shed.sum(dim=2).long()).clamp_min(0)
            animal_units = torch.where(
                buy_animal,
                torch.minimum(quantity, torch.minimum(s.money // animal_cost, shed_room)),
                0,
            )
            s.money.sub_(animal_units * animal_cost)
            animal_item = animal + len(PRODUCTS)
            s.shed.scatter_add_(2, animal_item.unsqueeze(-1), animal_units.to(s.shed.dtype).unsqueeze(-1))

            dynamic_order = (op == M_BUY_PRODUCT) | (op == M_SELL)
            remaining = torch.where(dynamic_order, quantity, torch.zeros_like(quantity))
            dynamic_rounds = int(remaining.max().item())
            if dynamic_rounds == 0:
                continue
            if self.use_triton:
                run_dynamic_market(s, actions, self, order, dynamic_rounds)
                continue
            for _ in range(dynamic_rounds):
                active = remaining > 0
                product_arg = arg.clamp(0, len(PRODUCTS) - 1)
                inventory = s.market_inventory.gather(1, product_arg).reshape(self.num_envs, 2)
                quoted = torch.zeros_like(s.money)
                sell = active & (op == M_SELL) & (arg >= 0) & (arg < len(PRODUCTS))
                buy_product = active & (op == M_BUY_PRODUCT) & ((arg == PRODUCT_INDEX["WHEAT"]) | (arg == PRODUCT_INDEX["FERTILIZER"]))
                buy_seed = active & (op == M_BUY_SEED) & (arg >= 0) & (arg < len(CROPS))
                buy_animal = active & (op == M_BUY_ANIMAL) & (arg >= 0) & (arg < len(ANIMALS))
                quoted = torch.where(sell, self._prices_for(product_arg, inventory), quoted)
                quoted = torch.where(buy_product, self._prices_for(product_arg, inventory - 1), quoted)
                quoted = torch.where(buy_seed, self.crop_seed_cost[arg.clamp(0, len(CROPS) - 1)], quoted)
                quoted = torch.where(buy_animal, self.animal_cost[arg.clamp(0, len(ANIMALS) - 1)], quoted)

                for player in range(2):
                    p_sell = sell[:, player]
                    p_buy_product = buy_product[:, player]
                    p_buy_seed = buy_seed[:, player]
                    p_buy_animal = buy_animal[:, player]
                    p_arg = arg[:, player]
                    p_quote = quoted[:, player]
                    p_product = product_arg[:, player]
                    p_remaining = remaining[:, player]
                    product_index = p_product.unsqueeze(-1)

                    available = s.shed[:, player].gather(1, product_index).squeeze(-1) > 0
                    success = p_sell & available
                    s.shed[:, player].scatter_add_(1, product_index, (-success.long()).to(s.shed.dtype).unsqueeze(-1))
                    s.money[:, player].add_(torch.where(success, p_quote, torch.zeros_like(p_quote)))
                    increase = success & (p_quote > 1)
                    s.market_inventory.scatter_add_(1, product_index, increase.to(s.market_inventory.dtype).unsqueeze(-1))
                    p_remaining = torch.where(p_sell, torch.where(success, p_remaining - 1, torch.zeros_like(p_remaining)), p_remaining)

                    capacity = s.shed[:, player].sum(dim=1) < cfg.shed_capacity
                    success = p_buy_product & capacity & (s.money[:, player] >= p_quote)
                    s.money[:, player].sub_(torch.where(success, p_quote, torch.zeros_like(p_quote)))
                    s.shed[:, player].scatter_add_(1, product_index, success.to(s.shed.dtype).unsqueeze(-1))
                    s.market_inventory.scatter_add_(1, product_index, (-success.long()).to(s.market_inventory.dtype).unsqueeze(-1))
                    p_remaining = torch.where(p_buy_product, torch.where(success, p_remaining - 1, torch.zeros_like(p_remaining)), p_remaining)

                    crop = p_arg.clamp(0, len(CROPS) - 1)
                    success = p_buy_seed & (s.money[:, player] >= p_quote)
                    s.money[:, player].sub_(torch.where(success, p_quote, torch.zeros_like(p_quote)))
                    s.seeds[:, player].scatter_add_(1, crop.unsqueeze(-1), success.to(s.seeds.dtype).unsqueeze(-1))
                    p_remaining = torch.where(p_buy_seed, torch.where(success, p_remaining - 1, torch.zeros_like(p_remaining)), p_remaining)

                    animal = p_arg.clamp(0, len(ANIMALS) - 1)
                    item_index = animal + len(PRODUCTS)
                    capacity = s.shed[:, player].sum(dim=1) < cfg.shed_capacity
                    success = p_buy_animal & capacity & (s.money[:, player] >= p_quote)
                    s.money[:, player].sub_(torch.where(success, p_quote, torch.zeros_like(p_quote)))
                    s.shed[:, player].scatter_add_(1, item_index.unsqueeze(-1), success.to(s.shed.dtype).unsqueeze(-1))
                    p_remaining = torch.where(p_buy_animal, torch.where(success, p_remaining - 1, torch.zeros_like(p_remaining)), p_remaining)
                    malformed = active[:, player] & ~(p_sell | p_buy_product | p_buy_seed | p_buy_animal)
                    remaining[:, player] = torch.where(malformed, torch.zeros_like(p_remaining), p_remaining)

    def _town_consume(self) -> None:
        s = self.state
        cfg = self.config
        if self.use_triton:
            run_town_consume(
                s,
                self,
                shop_active=self.step_index % cfg.town_shop_sell_interval == 0,
                center_active=self.step_index % cfg.town_center_sell_interval == 0,
            )
            return
        if self.step_index % cfg.town_shop_sell_interval == 0:
            demand = (s.shop_counts.long().unsqueeze(-1) * self.shop_demand.unsqueeze(0)).sum(dim=1)
            s.market_inventory -= demand.to(s.market_inventory.dtype)
        if self.step_index % cfg.town_center_sell_interval == 0:
            s.market_inventory -= 1
            s.market_inventory[:, PRODUCT_INDEX["FERTILIZER"]] += 1
        self._refresh_prices()

    def _decay_plants(self) -> None:
        s = self.state
        if self.use_triton:
            run_decay_plants(s, self.step_index)
            return
        plant = (s.tile_type >= PLANT_BASE) & (s.tile_type < PLANT_BASE + len(CROPS))
        decay = plant & (s.max_lifespan_step >= 0) & (self.step_index >= s.max_lifespan_step) & (((self.step_index - s.max_lifespan_step) % 2) == 0)
        s.yield_units[decay] -= 1
        weed = decay & (s.yield_units <= 0)
        s.tile_type[weed] = WEED
        self._clear_tile_fields(weed)

    def _uniform_board(self, day: int) -> torch.Tensor:
        # Stateless per-episode LCG hash.  It preserves the official Bernoulli
        # distribution and determinism, but not Python random.Random bit identity.
        index = torch.arange(2 * 100, device=self.device, dtype=torch.int64).reshape(1, 2, 10, 10)
        seed = self.episode_seeds[:, None, None, None]
        hashed = (seed * 48_271 + index * 69_621 + day * 1_000_003 + 12_345) % 2_147_483_647
        return hashed.float() / 2_147_483_647.0

    def _end_of_day(self, day: int) -> None:
        s = self.state
        cfg = self.config
        if self.use_triton:
            run_end_of_day(s, self, day)
            return
        next_day = day + 1
        plant = (s.tile_type >= PLANT_BASE) & (s.tile_type < PLANT_BASE + len(CROPS))
        was_watered = s.watered.clone()
        s.consecutive[plant & was_watered] = 0
        s.consecutive[plant & ~was_watered] += 1
        s.watered[plant] = False
        dead = plant & (s.consecutive >= 2)
        s.tile_type[dead] = WEED
        self._clear_tile_fields(dead)

        plant = (s.tile_type >= PLANT_BASE) & (s.tile_type < PLANT_BASE + len(CROPS))
        crop = (s.tile_type.long() - PLANT_BASE).clamp(0, len(CROPS) - 1)
        ongoing = plant & self.crop_ongoing[crop]
        days_since_first = next_day - s.planted_day.long() - self.crop_first[crop]
        interval = self.crop_interval[crop].clamp_min(1)
        production_count = torch.div(days_since_first, interval, rounding_mode="floor") + 1
        produce = ongoing & (days_since_first >= 0) & ((days_since_first % interval) == 0) & (production_count <= self.crop_max_yield[crop])
        fertilized = was_watered & (s.fertilized_until_day >= day)
        addition = torch.where(fertilized, 2, 1)
        s.yield_units[produce] = torch.minimum(
            s.yield_units[produce].long() + addition[produce], self.crop_max_yield[crop][produce]
        ).to(torch.int16)
        last = produce & (production_count == self.crop_max_yield[crop])
        s.max_lifespan_step[last] = (next_day + 1) * cfg.turns_per_day

        animal_tile = (s.tile_type >= ANIMAL_BASE) & (s.tile_type < ANIMAL_BASE + len(ANIMALS))
        animal = (s.tile_type.long() - ANIMAL_BASE).clamp(0, len(ANIMALS) - 1)
        s.consecutive[animal_tile & s.fed] = 0
        s.consecutive[animal_tile & ~s.fed] += 1
        escaped = animal_tile & (s.consecutive >= 2)
        s.tile_type[escaped] = self.animal_structure[animal][escaped].to(torch.int8)
        self._clear_tile_fields(escaped)

        animal_tile = (s.tile_type >= ANIMAL_BASE) & (s.tile_type < ANIMAL_BASE + len(ANIMALS))
        animal = (s.tile_type.long() - ANIMAL_BASE).clamp(0, len(ANIMALS) - 1)
        days_since_first = next_day - s.placed_day.long() - self.animal_first[animal]
        interval = self.animal_interval[animal]
        production = animal_tile & (days_since_first >= 0) & ((days_since_first % interval) == 0)
        bonus = torch.where(s.fed, s.pending_care_bonus.long(), torch.zeros_like(s.pending_care_bonus.long()))
        updated = torch.minimum(s.yield_units.long() + 1 + bonus, self.animal_max_held[animal])
        s.yield_units[production] = updated[production].to(torch.int16)
        s.pending_care_bonus[production] = 0
        bank = animal_tile & s.cared & s.fed
        s.pending_care_bonus[bank] += 1
        s.fertilizer_available[animal_tile] = True
        s.fed[animal_tile] = False
        s.cared[animal_tile] = False

        if cfg.weed_spawn_chance > 0:
            spawn = (s.tile_type == EMPTY) & (self._uniform_board(day) < cfg.weed_spawn_chance)
            s.tile_type[spawn] = WEED

        # Fixed item order makes overflow deterministic and GPU-friendly.
        for item in range(len(ITEMS)):
            carried = s.unit_inventory[..., item].sum(dim=2).long()
            room = (cfg.shed_capacity - s.shed.sum(dim=2).long()).clamp_min(0)
            take = torch.minimum(carried, room)
            s.shed[..., item] += take.to(s.shed.dtype)
        s.unit_inventory.zero_()
        s.positions.zero_()
        s.positions[:, :, 0, 0] = 4
        s.positions[:, :, 0, 1] = 4
        s.unit_active.zero_()
        s.unit_active[:, :, 0] = True
        s.hands_count.zero_()
        s.hires_today.zero_()

        if next_day > 0 and next_day % cfg.town_shop_unlock_interval == 0:
            current = s.shop_counts.sum(dim=1)
            can_unlock = current < 8
            hashed = (self.episode_seeds * 48_271 + day * 1_000_003 + 97_531) % 2_147_483_647
            choice = (hashed % len(SHOP_NAMES)).long()
            batch = torch.arange(self.num_envs, device=self.device)
            s.shop_counts[batch[can_unlock], choice[can_unlock]] += 1

    def step(self, actions: TensorActions) -> TensorStep:
        if self.done:
            raise RuntimeError("Episode is done; call reset before stepping again")
        expected_units = (self.num_envs, 2, self.max_units)
        expected_market = (self.num_envs, 2, self.config.max_market_orders)
        if tuple(actions.unit_ops.shape) != expected_units or tuple(actions.market_ops.shape) != expected_market:
            raise ValueError(f"Expected unit {expected_units} and market {expected_market} action tensors")
        day = self.step_index // self.config.turns_per_day

        # One compact device-to-host transfer controls both sparse interaction
        # dispatch and the market fast path. Processing all 17 position slots is
        # cheaper than synchronizing once more to discover the current hand cap.
        unit_limit = self.max_units
        unit_ops = actions.unit_ops[:, :, :unit_limit]
        active = self.state.unit_active[:, :, :unit_limit]
        interaction = (unit_ops >= U_DROP) & active
        if self.use_triton:
            common = torch.zeros_like(interaction)
            for supported_op in COMMON_UNIT_OPS:
                common |= unit_ops == supported_op
            common &= active
            inventory_masks = [((unit_ops == supported_op) & active) for supported_op in INVENTORY_UNIT_OPS]
            inventory = inventory_masks[0] | inventory_masks[1] | inventory_masks[2]
            supported = common | inventory
            supported &= active
            common_by_unit = common.any(dim=(0, 1))
            inventory_by_kind = [mask.any(dim=(0, 1)) for mask in inventory_masks]
            unsupported_by_unit = (interaction & ~supported).any(dim=(0, 1))
            plant_present = ((unit_ops == U_PLANT) & active).any().reshape(1)
            control = torch.cat(
                (
                    common_by_unit,
                    *inventory_by_kind,
                    unsupported_by_unit,
                    actions.market_ops.any().reshape(1),
                    plant_present,
                )
            ).tolist()
            common_units = control[:unit_limit]
            inventory_units = [
                control[(kind + 1) * unit_limit : (kind + 2) * unit_limit] for kind in range(len(INVENTORY_UNIT_OPS))
            ]
            unsupported_units = control[4 * unit_limit : 5 * unit_limit]
            has_market_orders = bool(control[-2])
            has_plant_actions = bool(control[-1])
        else:
            unsupported_by_unit = interaction.any(dim=(0, 1))
            control = torch.cat((unsupported_by_unit, actions.market_ops.any().reshape(1))).tolist()
            supported = torch.zeros_like(interaction)
            common_units = [False] * unit_limit
            inventory_units = [[False] * unit_limit for _ in INVENTORY_UNIT_OPS]
            unsupported_units = control[:unit_limit]
            has_market_orders = bool(control[-1])
            has_plant_actions = False

        self._move_units_dense(actions, unit_limit)
        # Movement is independent across units. Board and inventory interactions
        # remain sequential in official unit order, but only for slots that have
        # at least one interaction anywhere in the batch.
        if any(unsupported_units) or has_plant_actions:
            plant = (unit_ops == U_PLANT) & active
            crop = actions.unit_args[:, :, :unit_limit].clamp(0, len(CROPS) - 1)
            one_hot = torch.nn.functional.one_hot(crop, len(CROPS)).to(torch.int16)
            demand = (one_hot * plant.unsqueeze(-1)).sum(dim=2)
            blocked = demand > self.state.seeds
            plant_allowed = ~blocked.gather(2, crop) & (actions.unit_args[:, :, :unit_limit] >= 0) & (
                actions.unit_args[:, :, :unit_limit] < len(CROPS)
            )
        else:
            plant_allowed = torch.zeros_like(active)

        for unit in range(unit_limit):
            if common_units[unit]:
                run_common_interactions(
                    self.state,
                    actions.unit_ops,
                    actions.unit_args,
                    plant_allowed,
                    unit,
                    day,
                    self.max_units,
                )
            for kind, op_kind in enumerate(INVENTORY_UNIT_OPS):
                if inventory_units[kind][unit]:
                    run_inventory_interactions(
                        self.state,
                        actions.unit_ops,
                        actions.unit_args,
                        actions.unit_quantities,
                        unit,
                        day,
                        self.max_units,
                        self.config.shed_capacity,
                        op_kind,
                    )
            if unsupported_units[unit]:
                active_override = ~supported[:, :, unit] if self.use_triton else None
                self._step_unit_dense(actions, unit, day, plant_allowed, active_override)
        self._process_market(actions, has_market_orders)
        self._town_consume()
        self._decay_plants()
        if (self.step_index + 1) % self.config.turns_per_day == 0:
            self._end_of_day(day)

        terminal = self.step_index >= self.config.episode_steps - 2
        self.step_index += 1
        self.done = terminal
        rewards = self.state.money.clone() if terminal else torch.zeros_like(self.state.money)
        return TensorStep(rewards=rewards, done=terminal, step=self.step_index)

    def clone_state(self) -> TensorState:
        return TensorState(**{field.name: getattr(self.state, field.name).clone() for field in fields(TensorState)})


def encode_action_dicts(
    action_pairs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    device: str | torch.device,
    config: GpuEngineConfig | None = None,
) -> TensorActions:
    """Bridge official dictionary actions into fixed tensor actions for testing."""
    cfg = config or GpuEngineConfig()
    batch_size = len(action_pairs)
    max_units = cfg.max_hands + 1
    unit_ops = torch.zeros((batch_size, 2, max_units), dtype=torch.int64)
    unit_args = torch.zeros_like(unit_ops)
    unit_quantities = torch.ones_like(unit_ops)
    market_ops = torch.zeros((batch_size, 2, cfg.max_market_orders), dtype=torch.int64)
    market_args = torch.zeros_like(market_ops)
    market_quantities = torch.zeros_like(market_ops)

    for batch, pair in enumerate(action_pairs):
        if len(pair) != 2:
            raise ValueError("Each environment needs two player actions")
        for player, action in enumerate(pair):
            raw_units = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]
            for unit, raw in enumerate(raw_units[:max_units]):
                if not isinstance(raw, Sequence) or not raw:
                    continue
                op = str(raw[0])
                unit_ops[batch, player, unit] = UNIT_OP_INDEX.get(op, U_PASS)
                if op == "PLANT" and len(raw) > 1:
                    unit_args[batch, player, unit] = CROP_INDEX.get(str(raw[1]), -1)
                elif op in ("PICKUP", "PLACE") and len(raw) > 1:
                    unit_args[batch, player, unit] = ITEM_INDEX.get(str(raw[1]), -1)
                if len(raw) > 2:
                    try:
                        unit_quantities[batch, player, unit] = int(raw[2])
                    except (TypeError, ValueError):
                        unit_quantities[batch, player, unit] = 1
            for order, raw in enumerate(list(action.get("market", []) or [])[: cfg.max_market_orders]):
                if not isinstance(raw, Sequence) or not raw:
                    continue
                op = str(raw[0])
                market_ops[batch, player, order] = MARKET_OP_INDEX.get(op, M_NONE)
                if op == "BUY_SEED" and len(raw) > 1:
                    market_args[batch, player, order] = CROP_INDEX.get(str(raw[1]), -1)
                elif op == "BUY_ANIMAL" and len(raw) > 1:
                    market_args[batch, player, order] = ANIMAL_INDEX.get(str(raw[1]), -1)
                elif op in ("BUY_PRODUCT", "SELL") and len(raw) > 1:
                    market_args[batch, player, order] = PRODUCT_INDEX.get(str(raw[1]), -1)
                market_quantities[batch, player, order] = 1
                if len(raw) > 2:
                    try:
                        market_quantities[batch, player, order] = int(raw[2])
                    except (TypeError, ValueError):
                        market_quantities[batch, player, order] = 0
    return TensorActions(
        unit_ops.to(device),
        unit_args.to(device),
        unit_quantities.to(device),
        market_ops.to(device),
        market_args.to(device),
        market_quantities.to(device),
    )
