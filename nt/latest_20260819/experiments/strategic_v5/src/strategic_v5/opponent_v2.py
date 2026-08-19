"""Actor-visible opponent features for the Strategic V5 V2 policy.

The module deliberately reads only public opponent fields from ``State``:
money, tiles, unit positions/activity, hires and unlocked land.  Opponent shed,
seeds and carried inventories are never referenced.  Own private state remains
available through the frozen V1 feature prefix.

V2 keeps the exact 96 V1 global fields and 16 V1 candidate fields as prefixes.
This makes checkpoint migration mechanical: old kernels are copied and all
new input rows are zero-initialized.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_ONGOING,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_TILES,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State, StaticTables

from .constants import TaskTypeV1
from .learned_v1 import (
    CANDIDATE_FEATURE_DIM_V1,
    GLOBAL_FEATURE_DIM_V1,
    build_candidate_features_v1,
    build_global_features_v1,
)
from .schema import CandidateV1, EconFeaturesV1, FeasibilityV1


OPPONENT_CURRENT_FEATURE_DIM_V2 = 64
PUBLIC_SNAPSHOT_DIM_V2 = 36
OPPONENT_HISTORY_FEATURE_DIM_V2 = PUBLIC_SNAPSHOT_DIM_V2 * 2
GLOBAL_FEATURE_DIM_V2 = (
    GLOBAL_FEATURE_DIM_V1
    + OPPONENT_CURRENT_FEATURE_DIM_V2
    + OPPONENT_HISTORY_FEATURE_DIM_V2
)
CANDIDATE_INTERACTION_FEATURE_DIM_V2 = 8
CANDIDATE_FEATURE_DIM_V2 = (
    CANDIDATE_FEATURE_DIM_V1 + CANDIDATE_INTERACTION_FEATURE_DIM_V2
)
HISTORY_EMA_DECAY_V2 = 0.75


PRODUCT_NAMES_V2 = (
    "wheat",
    "carrot",
    "tomato",
    "strawberry",
    "melon",
    "egg",
    "milk",
    "wool",
    "fertilizer",
)


OPPONENT_CURRENT_FEATURE_NAMES_V2 = (
    *(f"opponent_visible_supply_now_{name}" for name in PRODUCT_NAMES_V2),
    *(f"opponent_producer_count_{name}" for name in PRODUCT_NAMES_V2),
    *(f"opponent_visible_projection_1d_{name}" for name in PRODUCT_NAMES_V2),
    *(f"opponent_visible_projection_3d_{name}" for name in PRODUCT_NAMES_V2),
    "opponent_crops_need_water",
    "opponent_crops_loss_risk",
    "opponent_crops_decaying",
    "opponent_animals_need_feed",
    "opponent_animals_escape_risk",
    "opponent_animals_at_capacity",
    "opponent_empty_unlocked_tiles",
    "opponent_weeds",
    "opponent_empty_coops",
    "opponent_empty_pastures",
    "opponent_pending_care_total",
    "opponent_minus_own_cash",
    "opponent_minus_own_unlocked_land",
    "opponent_minus_own_active_units",
    "opponent_minus_own_visible_supply_value",
    "opponent_farmer_x",
    "opponent_farmer_y",
    "opponent_mean_unit_center_distance",
    "opponent_min_unit_to_urgent_distance",
    "opponent_mean_urgent_target_distance",
    "opponent_urgent_task_count",
    "opponent_urgent_tasks_per_active_unit",
    "opponent_productive_tiles_per_active_unit",
    "opponent_visible_supply_value_now",
    "opponent_visible_supply_value_1d",
    "opponent_visible_supply_value_3d",
    "opponent_visible_value_at_risk",
    "opponent_product_concentration",
)


PUBLIC_SNAPSHOT_FEATURE_NAMES_V2 = (
    "opponent_money",
    "opponent_unlocked_land",
    "opponent_active_units",
    "opponent_hires_today",
    *(f"opponent_visible_supply_{name}" for name in PRODUCT_NAMES_V2),
    *(f"opponent_crop_count_{name}" for name in PRODUCT_NAMES_V2[:NUM_CROPS]),
    *(f"opponent_animal_count_{name}" for name in ("goose", "cow", "sheep")),
    "opponent_crop_loss_risk",
    "opponent_animal_escape_risk",
    "opponent_crop_decay_risk",
    "opponent_animal_capacity_risk",
    "opponent_weed_count",
    *(f"market_inventory_{name}" for name in PRODUCT_NAMES_V2),
    "town_shop_count",
)


OPPONENT_HISTORY_FEATURE_NAMES_V2 = (
    *(f"recent_delta_{name}" for name in PUBLIC_SNAPSHOT_FEATURE_NAMES_V2),
    *(f"ema_delta_{name}" for name in PUBLIC_SNAPSHOT_FEATURE_NAMES_V2),
)


CANDIDATE_INTERACTION_FEATURE_NAMES_V2 = (
    "opponent_visible_supply_now",
    "opponent_visible_projection_1d",
    "opponent_visible_projection_3d",
    "opponent_same_product_producers",
    "opponent_earliest_visible_supply_steps",
    "candidate_revenue_lead_vs_opponent",
    "opponent_scenario_post_supply_price_ratio",
    "opponent_scenario_profit_loss",
)


if len(OPPONENT_CURRENT_FEATURE_NAMES_V2) != OPPONENT_CURRENT_FEATURE_DIM_V2:
    raise RuntimeError("opponent current feature schema width mismatch")
if len(PUBLIC_SNAPSHOT_FEATURE_NAMES_V2) != PUBLIC_SNAPSHOT_DIM_V2:
    raise RuntimeError("public snapshot schema width mismatch")
if len(OPPONENT_HISTORY_FEATURE_NAMES_V2) != OPPONENT_HISTORY_FEATURE_DIM_V2:
    raise RuntimeError("opponent history feature schema width mismatch")
if len(CANDIDATE_INTERACTION_FEATURE_NAMES_V2) != CANDIDATE_INTERACTION_FEATURE_DIM_V2:
    raise RuntimeError("candidate interaction feature schema width mismatch")


class OpponentHistoryV2(NamedTuple):
    """Short actor-visible history for both viewpoints.

    Axis 1 is the observing player.  Each snapshot contains the other player's
    public farm plus shared market/town state.
    """

    previous_snapshot: jax.Array
    recent_delta: jax.Array
    ema_delta: jax.Array


class VisibleProductProjectionV2(NamedTuple):
    supply_now: jax.Array
    producer_count: jax.Array
    projection_1d: jax.Array
    projection_3d: jax.Array


_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
_CROP_INTERVAL = jnp.asarray(CROP_INTERVAL, dtype=jnp.int16)
_CROP_MAX = jnp.asarray(CROP_MAX_YIELD, dtype=jnp.int16)
_CROP_ONGOING = jnp.asarray(CROP_ONGOING, dtype=jnp.bool_)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_TILE_Y, _TILE_X = jnp.indices((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int16)
_TILE_POS = jnp.stack((_TILE_X.reshape(-1), _TILE_Y.reshape(-1)), axis=-1)


def _flag(flags: jax.Array, value: int) -> jax.Array:
    return (flags & jnp.uint8(value)) != 0


def _counts_by_id(values: jax.Array, valid: jax.Array, count: int) -> jax.Array:
    safe = jnp.clip(values.astype(jnp.int32), 0, count - 1)
    one_hot = jax.nn.one_hot(safe, count, dtype=jnp.float32)
    return jnp.sum(one_hot * valid[..., None], axis=(1, 2))


def _supply_by_id(
    values: jax.Array,
    valid: jax.Array,
    quantities: jax.Array,
    count: int,
) -> jax.Array:
    safe = jnp.clip(values.astype(jnp.int32), 0, count - 1)
    one_hot = jax.nn.one_hot(safe, count, dtype=jnp.float32)
    return jnp.sum(
        one_hot * valid[..., None] * quantities.astype(jnp.float32)[..., None],
        axis=(1, 2),
    )


def _project_supply_for_horizon(
    states: State, owner: int, horizon_days: int
) -> jax.Array:
    """Visible production projection assuming public assets are maintained.

    It is intentionally not presented as known future inventory.  One-time
    crop watering bonuses and hidden feasibility are not invented.  Existing
    visible yield is carried forward; scheduled ongoing/animal production uses
    only official public lifecycle fields.
    """

    day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    crop = states.tile_crop[:, owner]
    animal = states.tile_animal[:, owner]
    crop_valid = (states.tile_kind[:, owner] == TileKind.PLANT) & (crop >= 0)
    animal_valid = animal >= 0
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    safe_animal = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    origin = states.tile_origin_day[:, owner].astype(jnp.int16)
    current_yield = states.tile_yield[:, owner].astype(jnp.int16)

    crop_due_count = jnp.zeros_like(current_yield, dtype=jnp.int16)
    animal_due_count = jnp.zeros_like(current_yield, dtype=jnp.int16)
    for offset in range(1, horizon_days + 1):
        future_day = day[:, None, None] + jnp.int16(offset)
        crop_since_first = future_day - origin - _CROP_FIRST[safe_crop]
        crop_interval = jnp.maximum(_CROP_INTERVAL[safe_crop], 1)
        crop_production_index = crop_since_first // crop_interval + 1
        crop_due = (
            crop_valid
            & _CROP_ONGOING[safe_crop]
            & (crop_since_first >= 0)
            & ((crop_since_first % crop_interval) == 0)
            & (crop_production_index <= _CROP_MAX[safe_crop])
        )
        crop_due_count = crop_due_count + crop_due.astype(jnp.int16)

        animal_since_first = future_day - origin - _ANIMAL_FIRST[safe_animal]
        animal_due = (
            animal_valid
            & (animal_since_first >= 0)
            & ((animal_since_first % _ANIMAL_INTERVAL[safe_animal]) == 0)
        )
        animal_due_count = animal_due_count + animal_due.astype(jnp.int16)

    target_day = day[:, None, None] + jnp.int16(horizon_days)
    crop_mature = (target_day - origin) >= _CROP_FIRST[safe_crop]
    crop_projected = jnp.where(
        _CROP_ONGOING[safe_crop],
        jnp.minimum(_CROP_MAX[safe_crop], current_yield + crop_due_count),
        current_yield,
    )
    crop_supply = _supply_by_id(
        crop,
        crop_valid & crop_mature,
        crop_projected,
        NUM_CROPS,
    )

    pending = states.tile_pending_care[:, owner].astype(jnp.int16)
    animal_projected = jnp.minimum(
        _ANIMAL_MAX[safe_animal],
        current_yield
        + animal_due_count
        + jnp.where(animal_due_count > 0, pending, 0),
    )
    animal_by_type = _supply_by_id(
        animal, animal_valid, animal_projected, NUM_ANIMALS
    )
    animal_supply = jnp.zeros(
        (states.step.shape[0], NUM_PRODUCTS), dtype=jnp.float32
    ).at[:, _ANIMAL_PRODUCT].set(animal_by_type)

    crop_products = jnp.pad(crop_supply, ((0, 0), (0, NUM_PRODUCTS - NUM_CROPS)))
    animal_count = jnp.sum(animal_valid, axis=(1, 2), dtype=jnp.float32)
    fertilizer_pipeline = animal_count * float(horizon_days)
    return (crop_products + animal_supply).at[:, NUM_PRODUCTS - 1].set(
        fertilizer_pipeline
    )


def visible_product_projection_v2(
    states: State, owner: int
) -> VisibleProductProjectionV2:
    day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    crop = states.tile_crop[:, owner]
    animal = states.tile_animal[:, owner]
    crop_valid = (states.tile_kind[:, owner] == TileKind.PLANT) & (crop >= 0)
    animal_valid = animal >= 0
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    crop_mature = (
        day[:, None, None] - states.tile_origin_day[:, owner].astype(jnp.int16)
    ) >= _CROP_FIRST[safe_crop]
    crop_supply = _supply_by_id(
        crop,
        crop_valid & crop_mature,
        states.tile_yield[:, owner],
        NUM_CROPS,
    )
    animal_by_type = _supply_by_id(
        animal,
        animal_valid,
        states.tile_yield[:, owner],
        NUM_ANIMALS,
    )
    animal_supply = jnp.zeros(
        (states.step.shape[0], NUM_PRODUCTS), dtype=jnp.float32
    ).at[:, _ANIMAL_PRODUCT].set(animal_by_type)
    fertilizer_ready = jnp.sum(
        animal_valid
        & _flag(states.tile_flags[:, owner], FLAG_FERTILIZER_AVAILABLE),
        axis=(1, 2),
        dtype=jnp.float32,
    )
    supply_now = (
        jnp.pad(crop_supply, ((0, 0), (0, NUM_PRODUCTS - NUM_CROPS)))
        + animal_supply
    ).at[:, NUM_PRODUCTS - 1].set(fertilizer_ready)

    crop_count = _counts_by_id(crop, crop_valid, NUM_CROPS)
    animal_count = _counts_by_id(animal, animal_valid, NUM_ANIMALS)
    producer = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.float32)
    producer = producer.at[:, :NUM_CROPS].set(crop_count)
    producer = producer.at[:, _ANIMAL_PRODUCT].set(animal_count)
    producer = producer.at[:, NUM_PRODUCTS - 1].set(
        jnp.sum(animal_count, axis=-1)
    )
    return VisibleProductProjectionV2(
        supply_now=supply_now,
        producer_count=producer,
        projection_1d=_project_supply_for_horizon(states, owner, 1),
        projection_3d=_project_supply_for_horizon(states, owner, 3),
    )


def _public_risk_masks(states: State, owner: int) -> tuple[jax.Array, ...]:
    kind = states.tile_kind[:, owner]
    flags = states.tile_flags[:, owner]
    plant = kind == TileKind.PLANT
    animal = states.tile_animal[:, owner] >= 0
    watered = _flag(flags, FLAG_WATERED)
    fed = _flag(flags, FLAG_FED)
    crop_need_water = plant & (~watered)
    crop_loss = crop_need_water & (states.tile_neglect[:, owner] >= 1)
    crop_decay = (
        plant
        & (states.tile_max_lifespan[:, owner] >= 0)
        & (states.step[:, None, None] >= states.tile_max_lifespan[:, owner])
        & (states.tile_yield[:, owner] > 0)
    )
    animal_need_feed = animal & (~fed)
    animal_escape = animal_need_feed & (states.tile_neglect[:, owner] >= 1)
    safe_animal = jnp.clip(
        states.tile_animal[:, owner].astype(jnp.int32), 0, NUM_ANIMALS - 1
    )
    animal_capacity = animal & (
        states.tile_yield[:, owner] >= _ANIMAL_MAX[safe_animal]
    )
    return (
        crop_need_water,
        crop_loss,
        crop_decay,
        animal_need_feed,
        animal_escape,
        animal_capacity,
    )


def _count(mask: jax.Array) -> jax.Array:
    return jnp.sum(mask, axis=(1, 2), dtype=jnp.float32)


def _spatial_workload_features(
    states: State, owner: int, urgent: jax.Array
) -> tuple[jax.Array, ...]:
    active = states.unit_active[:, owner]
    positions = states.unit_pos[:, owner].astype(jnp.int16)
    active_count = jnp.maximum(jnp.sum(active, axis=-1, dtype=jnp.float32), 1.0)

    center_distance = jnp.sum(
        jnp.abs(positions[:, :, None, :] - _SHED_ACCESS[None, None, :, :]),
        axis=-1,
    )
    nearest_center = jnp.min(center_distance, axis=-1).astype(jnp.float32)
    mean_center = jnp.sum(jnp.where(active, nearest_center, 0.0), axis=-1) / active_count

    target_mask = urgent.reshape((urgent.shape[0], NUM_TILES))
    distance = jnp.sum(
        jnp.abs(positions[:, :, None, :] - _TILE_POS[None, None, :, :]),
        axis=-1,
    ).astype(jnp.float32)
    pair_valid = active[:, :, None] & target_mask[:, None, :]
    safe_distance = jnp.where(pair_valid, distance, 10_000.0)
    has_urgent = jnp.any(target_mask, axis=-1)
    min_any = jnp.where(has_urgent, jnp.min(safe_distance, axis=(1, 2)), 0.0)
    nearest_unit_per_target = jnp.min(
        jnp.where(active[:, :, None], distance, 10_000.0), axis=1
    )
    urgent_count = jnp.sum(target_mask, axis=-1, dtype=jnp.float32)
    mean_target = jnp.where(
        has_urgent,
        jnp.sum(jnp.where(target_mask, nearest_unit_per_target, 0.0), axis=-1)
        / jnp.maximum(urgent_count, 1.0),
        0.0,
    )
    return mean_center, min_any, mean_target, urgent_count, active_count


def build_opponent_current_features_v2(states: State, player: int) -> jax.Array:
    opponent = 1 - player
    opp = visible_product_projection_v2(states, opponent)
    own = visible_product_projection_v2(states, player)
    (
        crop_need_water,
        crop_loss,
        crop_decay,
        animal_need_feed,
        animal_escape,
        animal_capacity,
    ) = _public_risk_masks(states, opponent)
    kind = states.tile_kind[:, opponent]
    animal_present = states.tile_animal[:, opponent] >= 0
    empty = kind == TileKind.EMPTY
    weeds = kind == TileKind.WEED
    empty_coops = (kind == TileKind.COOP) & (~animal_present)
    empty_pastures = (kind == TileKind.PASTURE) & (~animal_present)
    urgent = crop_loss | crop_decay | animal_escape | animal_capacity
    mean_center, min_urgent, mean_urgent, urgent_count, active_count = (
        _spatial_workload_features(states, opponent, urgent)
    )
    productive_count = _count((kind == TileKind.PLANT) | animal_present)
    price = states.market_price.astype(jnp.float32)
    opp_value_now = jnp.sum(opp.supply_now * price, axis=-1)
    own_value_now = jnp.sum(own.supply_now * price, axis=-1)
    opp_value_1d = jnp.sum(opp.projection_1d * price, axis=-1)
    opp_value_3d = jnp.sum(opp.projection_3d * price, axis=-1)
    at_risk_supply = jnp.sum(
        jnp.where(
            crop_decay | animal_capacity,
            states.tile_yield[:, opponent].astype(jnp.float32),
            0.0,
        ),
        axis=(1, 2),
    )
    mean_price = jnp.maximum(jnp.mean(price, axis=-1), 1.0)
    supply_total = jnp.sum(opp.supply_now, axis=-1)
    concentration = jnp.sum(
        jnp.square(opp.supply_now / jnp.maximum(supply_total[:, None], 1.0)),
        axis=-1,
    )
    farmer = states.unit_pos[:, opponent, 0].astype(jnp.float32)

    pieces = (
        opp.supply_now / 100.0,
        opp.producer_count / 100.0,
        opp.projection_1d / 100.0,
        opp.projection_3d / 100.0,
        (_count(crop_need_water) / NUM_TILES)[:, None],
        (_count(crop_loss) / NUM_TILES)[:, None],
        (_count(crop_decay) / NUM_TILES)[:, None],
        (_count(animal_need_feed) / NUM_TILES)[:, None],
        (_count(animal_escape) / NUM_TILES)[:, None],
        (_count(animal_capacity) / NUM_TILES)[:, None],
        (_count(empty) / NUM_TILES)[:, None],
        (_count(weeds) / NUM_TILES)[:, None],
        (_count(empty_coops) / NUM_TILES)[:, None],
        (_count(empty_pastures) / NUM_TILES)[:, None],
        (
            jnp.sum(
                states.tile_pending_care[:, opponent].astype(jnp.float32),
                axis=(1, 2),
            )
            / 100.0
        )[:, None],
        (
            (states.money[:, opponent] - states.money[:, player]).astype(jnp.float32)
            / 100_000.0
        )[:, None],
        (
            (states.unlocked_count[:, opponent] - states.unlocked_count[:, player]).astype(
                jnp.float32
            )
            / 4.0
        )[:, None],
        (
            (
                jnp.sum(states.unit_active[:, opponent], axis=-1)
                - jnp.sum(states.unit_active[:, player], axis=-1)
            ).astype(jnp.float32)
            / MAX_UNITS
        )[:, None],
        ((opp_value_now - own_value_now) / 100_000.0)[:, None],
        (farmer[:, 0] / float(BOARD_SIZE - 1))[:, None],
        (farmer[:, 1] / float(BOARD_SIZE - 1))[:, None],
        (mean_center / (2.0 * (BOARD_SIZE - 1)))[:, None],
        (min_urgent / (2.0 * (BOARD_SIZE - 1)))[:, None],
        (mean_urgent / (2.0 * (BOARD_SIZE - 1)))[:, None],
        (urgent_count / NUM_TILES)[:, None],
        (urgent_count / active_count / 10.0)[:, None],
        (productive_count / active_count / 10.0)[:, None],
        (opp_value_now / 100_000.0)[:, None],
        (opp_value_1d / 100_000.0)[:, None],
        (opp_value_3d / 100_000.0)[:, None],
        (at_risk_supply * mean_price / 100_000.0)[:, None],
        concentration[:, None],
    )
    result = jnp.concatenate(pieces, axis=-1)
    if result.shape[-1] != OPPONENT_CURRENT_FEATURE_DIM_V2:
        raise ValueError("opponent current feature width changed")
    return result.astype(jnp.float32)


def build_public_snapshot_v2(states: State, player: int) -> jax.Array:
    opponent = 1 - player
    projection = visible_product_projection_v2(states, opponent)
    crop = states.tile_crop[:, opponent]
    animal = states.tile_animal[:, opponent]
    crop_valid = (states.tile_kind[:, opponent] == TileKind.PLANT) & (crop >= 0)
    animal_valid = animal >= 0
    crop_counts = _counts_by_id(crop, crop_valid, NUM_CROPS)
    animal_counts = _counts_by_id(animal, animal_valid, NUM_ANIMALS)
    _, crop_loss, crop_decay, _, animal_escape, animal_capacity = _public_risk_masks(
        states, opponent
    )
    snapshot = jnp.concatenate(
        (
            (states.money[:, opponent].astype(jnp.float32) / 100_000.0)[:, None],
            (states.unlocked_count[:, opponent].astype(jnp.float32) / 4.0)[:, None],
            (
                jnp.sum(states.unit_active[:, opponent], axis=-1).astype(jnp.float32)
                / MAX_UNITS
            )[:, None],
            (states.hires_today[:, opponent].astype(jnp.float32) / MAX_UNITS)[:, None],
            projection.supply_now / 100.0,
            crop_counts / NUM_TILES,
            animal_counts / NUM_TILES,
            (_count(crop_loss) / NUM_TILES)[:, None],
            (_count(animal_escape) / NUM_TILES)[:, None],
            (_count(crop_decay) / NUM_TILES)[:, None],
            (_count(animal_capacity) / NUM_TILES)[:, None],
            (_count(states.tile_kind[:, opponent] == TileKind.WEED) / NUM_TILES)[:, None],
            (states.market_inventory.astype(jnp.float32) - 10_000.0) / 2_000.0,
            (states.town_count.astype(jnp.float32) / MAX_SHOPS)[:, None],
        ),
        axis=-1,
    )
    if snapshot.shape[-1] != PUBLIC_SNAPSHOT_DIM_V2:
        raise ValueError("public snapshot width changed")
    return snapshot.astype(jnp.float32)


def initialize_opponent_history_v2(states: State) -> OpponentHistoryV2:
    snapshot = jnp.stack(
        (build_public_snapshot_v2(states, 0), build_public_snapshot_v2(states, 1)),
        axis=1,
    )
    zeros = jnp.zeros_like(snapshot)
    return OpponentHistoryV2(snapshot, zeros, zeros)


def update_opponent_history_v2(
    history: OpponentHistoryV2,
    next_states: State,
    *,
    ema_decay: float = HISTORY_EMA_DECAY_V2,
) -> OpponentHistoryV2:
    snapshot = jnp.stack(
        (
            build_public_snapshot_v2(next_states, 0),
            build_public_snapshot_v2(next_states, 1),
        ),
        axis=1,
    )
    delta = snapshot - history.previous_snapshot
    ema = ema_decay * history.ema_delta + (1.0 - ema_decay) * delta
    return OpponentHistoryV2(
        previous_snapshot=snapshot.astype(jnp.float32),
        recent_delta=delta.astype(jnp.float32),
        ema_delta=ema.astype(jnp.float32),
    )


def build_opponent_history_features_v2(
    history: OpponentHistoryV2 | None,
    *,
    batch_size: int,
    player: int,
) -> jax.Array:
    if history is None:
        return jnp.zeros(
            (batch_size, OPPONENT_HISTORY_FEATURE_DIM_V2), dtype=jnp.float32
        )
    return jnp.concatenate(
        (history.recent_delta[:, player], history.ema_delta[:, player]), axis=-1
    ).astype(jnp.float32)


def build_global_features_v2(
    states: State,
    player: int,
    history: OpponentHistoryV2 | None = None,
    *,
    include_opponent: bool = True,
    include_history: bool = False,
) -> jax.Array:
    base = build_global_features_v1(states, player)
    current = (
        build_opponent_current_features_v2(states, player)
        if include_opponent
        else jnp.zeros(
            (states.step.shape[0], OPPONENT_CURRENT_FEATURE_DIM_V2),
            dtype=jnp.float32,
        )
    )
    historical = (
        build_opponent_history_features_v2(
            history, batch_size=states.step.shape[0], player=player
        )
        if include_history
        else jnp.zeros(
            (states.step.shape[0], OPPONENT_HISTORY_FEATURE_DIM_V2),
            dtype=jnp.float32,
        )
    )
    return jnp.concatenate((base, current, historical), axis=-1).astype(jnp.float32)


def candidate_product_id_v2(candidates: CandidateV1) -> tuple[jax.Array, jax.Array]:
    task = candidates.task_type
    item = candidates.item_id.astype(jnp.int32)
    safe_animal_from_inventory = jnp.clip(item - NUM_PRODUCTS, 0, NUM_ANIMALS - 1)
    safe_animal_direct = jnp.clip(item, 0, NUM_ANIMALS - 1)
    is_animal_inventory = (task == TaskTypeV1.ANIMAL_PURCHASE) | (
        task == TaskTypeV1.ANIMAL_PLACE
    )
    is_animal_direct = (task == TaskTypeV1.ANIMAL_CARE) | (
        task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
    )
    product = jnp.where(
        is_animal_inventory,
        _ANIMAL_PRODUCT[safe_animal_from_inventory],
        jnp.where(is_animal_direct, _ANIMAL_PRODUCT[safe_animal_direct], item),
    )
    product = jnp.where(task == TaskTypeV1.ANIMAL_FEED, 0, product)
    product = jnp.where(
        task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER, NUM_PRODUCTS - 1, product
    )
    relevant = (
        (task == TaskTypeV1.CROP_PRODUCTION)
        | (task == TaskTypeV1.WATER_CROP)
        | (task == TaskTypeV1.ANIMAL_PURCHASE)
        | (task == TaskTypeV1.ANIMAL_PLACE)
        | (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.ANIMAL_CARE)
        | (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        | (task == TaskTypeV1.BUY_PRODUCT)
        | (task == TaskTypeV1.APPLY_FERTILIZER)
        | (task == TaskTypeV1.SELL_INVENTORY)
        | (task == TaskTypeV1.TERMINAL_LIQUIDATION)
    ) & (product >= 0) & (product < NUM_PRODUCTS)
    return jnp.clip(product, 0, NUM_PRODUCTS - 1), relevant


def build_candidate_interaction_features_v2(
    states: State,
    candidates: CandidateV1,
    econ: EconFeaturesV1,
    player: int,
    tables: StaticTables,
) -> jax.Array:
    opponent = 1 - player
    projection = visible_product_projection_v2(states, opponent)
    product, relevant = candidate_product_id_v2(candidates)
    now = jnp.take_along_axis(projection.supply_now, product, axis=1)
    day1 = jnp.take_along_axis(projection.projection_1d, product, axis=1)
    day3 = jnp.take_along_axis(projection.projection_3d, product, axis=1)
    producers = jnp.take_along_axis(projection.producer_count, product, axis=1)
    remaining_to_day_end = (
        TURNS_PER_DAY - (states.step % TURNS_PER_DAY)
    ).astype(jnp.float32)[:, None]
    earliest = jnp.where(
        now > 0,
        0.0,
        jnp.where(
            day1 > 0,
            remaining_to_day_end,
            jnp.where(day3 > 0, remaining_to_day_end + 2 * TURNS_PER_DAY, EPISODE_STEPS),
        ),
    )
    revenue_lead = earliest - econ.steps_to_first_revenue.astype(jnp.float32)
    price = jnp.take_along_axis(
        states.market_price.astype(jnp.float32), product, axis=1
    )
    inventory = jnp.take_along_axis(
        states.market_inventory.astype(jnp.int32), product, axis=1
    )
    supply_at_revenue = jnp.where(
        econ.steps_to_first_revenue <= 1,
        now,
        jnp.where(econ.steps_to_first_revenue <= TURNS_PER_DAY, day1, day3),
    )
    # Official sales at price $1 do not increase market inventory.  Otherwise
    # this public-pressure scenario assumes the opponent's visible projected
    # supply reaches the shared market before our first revenue.
    scenario_inventory = jnp.where(
        price > 1.0,
        inventory + jnp.rint(supply_at_revenue).astype(jnp.int32),
        inventory,
    )
    scenario_lut_index = jnp.clip(
        scenario_inventory - MARKET_MIN_INVENTORY,
        0,
        MARKET_LUT_SIZE - 1,
    )
    scenario_price = tables.market_price[product, scenario_lut_index].astype(
        jnp.float32
    )
    price_ratio = scenario_price / jnp.maximum(price, 1.0)
    current_profit = jnp.maximum(econ.expected_bank_delta_current_price, 0.0)
    scenario_profit_loss = jnp.maximum(
        current_profit - current_profit * price_ratio, 0.0
    )
    mask = relevant.astype(jnp.float32)
    result = jnp.stack(
        (
            now / 100.0,
            day1 / 100.0,
            day3 / 100.0,
            producers / 100.0,
            earliest / EPISODE_STEPS,
            revenue_lead / EPISODE_STEPS,
            price_ratio,
            scenario_profit_loss / 10_000.0,
        ),
        axis=-1,
    )
    return (result * mask[..., None]).astype(jnp.float16)


def build_candidate_features_v2(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    player: int,
    tables: StaticTables,
    *,
    include_opponent: bool = True,
) -> jax.Array:
    base = build_candidate_features_v1(
        states, candidates, feasibility, econ, player
    )
    interaction = (
        build_candidate_interaction_features_v2(
            states, candidates, econ, player, tables
        )
        if include_opponent
        else jnp.zeros(
            (
                states.step.shape[0],
                candidates.task_type.shape[1],
                CANDIDATE_INTERACTION_FEATURE_DIM_V2,
            ),
            dtype=jnp.float16,
        )
    )
    return jnp.concatenate((base, interaction), axis=-1).astype(jnp.float16)
