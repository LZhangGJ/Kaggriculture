"""M3A/M3B BUY_ONLY animal project controller.

The module owns animal commitments, feed planning, coarse layout and market
admission.  Official primitive legality and effect checking remain in the
Strategic V5 Full-core compiler.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    HIRE_COST,
    LAND_PRICES,
    MARKET_BASE_PRICES,
    MARKET_INITIAL_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.state import load_tables
from kaggriculture_jax.types import State, StaticTables
from strategic_v5.constants import (
    FailureCodeV1,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
)
from strategic_v5.e4_executor import (
    E4PlayerActionV1,
    FullCoreEffectDiagnosticsV1,
    compile_full_core_player_action_v1,
    update_full_core_controller_from_effects_v1,
)
from strategic_v5.e5_econ import exact_market_quote_v1

from .constants import (
    ProjectStatusV2,
    ProjectTypeV2,
)
from .m3_constants import (
    M3AnimalCarePolicyV2,
    M3AnimalFertilizerPolicyV2,
    M3AnimalLayoutPolicyV2,
    M3FeedSourcePolicyV2,
)
from .lifecycle import empty_market_tasks_v2, empty_unit_tasks_v2, reconcile_project_controller_v2
from .m3_schema import AnimalCommitmentLedgerV2, M3AnimalGenomeV2
from .schema import ProjectControllerStateV2


_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_LAND_COST = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_HIRE_COST_PREFIX = jnp.concatenate(
    (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(_HIRE_COST, dtype=jnp.int32))
)
_BASE_PRICE = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)
_INVALID_SCORE = jnp.int32(1_000_000_000)
_DEFAULT_TABLES = load_tables()

# Rows follow the frozen sorted SHOP_NAMES contract; columns are PRODUCTS.
_SHOP_DEMAND = jnp.asarray(
    (
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 0, 1, 0, 0, 0),
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
    ),
    dtype=jnp.int16,
)


def m3_phase_v2(states: State, genome: M3AnimalGenomeV2) -> jax.Array:
    slots = jnp.arange(genome.phase_start_step.shape[1], dtype=jnp.int8)[None]
    active = slots < genome.phase_count[:, None]
    crossed = active & (states.step[:, None] >= genome.phase_start_step)
    return jnp.maximum(jnp.sum(crossed, axis=-1) - 1, 0).astype(jnp.int8)


def _phase_value(values: jax.Array, phase: jax.Array) -> jax.Array:
    batch = jnp.arange(phase.shape[0], dtype=jnp.int32)
    return values[batch, phase.astype(jnp.int32)]


def _maintenance_cost_by_species(care_policy: jax.Array) -> jax.Array:
    """Conservative daily route-work units for one animal of each species."""

    return jnp.where(
        care_policy == M3AnimalCarePolicyV2.OFF,
        14,
        jnp.where(
            care_policy == M3AnimalCarePolicyV2.FIRST_CYCLE_ONLY,
            18,
            22,
        ),
    ).astype(jnp.int16)


def m3_phase_targets_v2(
    states: State, genome: M3AnimalGenomeV2
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    phase = m3_phase_v2(states, genome)
    raw_target = _phase_value(genome.animal_target, phase).astype(jnp.int16)
    hand_target = _phase_value(genome.hand_target, phase).astype(jnp.int16)
    requested_species = raw_target > 0
    species_count = jnp.maximum(
        jnp.sum(requested_species, axis=-1, dtype=jnp.int16), 1
    )
    # Dynamic productive cap: convert the available worker/day budget into a
    # conservative sustainable animal count.  CARE-heavy projects reserve more
    # route capacity for feed, care, harvest and depot return.  Equal per-
    # species shares prevent cow/sheep from being starved by allocation order.
    action_budget = jnp.floor(
        (hand_target.astype(jnp.float32) + 1.0)
        * TURNS_PER_DAY
        * genome.maintenance_utilization_cap
    ).astype(jnp.int16)
    species_budget = action_budget[:, None] // species_count[:, None]
    maintenance_cost = _maintenance_cost_by_species(genome.care_policy)
    dynamic_cap = species_budget // maintenance_cost
    target = jnp.where(
        genome.enforce_productive_cap[:, None],
        jnp.minimum(raw_target, dynamic_cap),
        raw_target,
    ).astype(jnp.int16)
    return (
        phase,
        target,
        _phase_value(genome.land_target, phase).astype(jnp.int8),
        hand_target.astype(jnp.int8),
    )


def _active_animal_counts(states: State, player: int) -> jax.Array:
    animal = states.tile_animal[:, player].reshape(states.step.shape[0], -1)
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    return jnp.sum(
        jax.nn.one_hot(safe, NUM_ANIMALS, dtype=jnp.int16)
        * (animal >= 0)[..., None],
        axis=1,
        dtype=jnp.int16,
    )


def _production_today_by_tile(states: State, player: int) -> jax.Array:
    animal = states.tile_animal[:, player]
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)[:, None, None]
    days_since_first = (
        current_day
        + 1
        - states.tile_origin_day[:, player].astype(jnp.int16)
        - _ANIMAL_FIRST[safe]
    )
    return (
        (animal >= 0)
        & (days_since_first >= 0)
        & (jnp.mod(days_since_first, _ANIMAL_INTERVAL[safe]) == 0)
    )


def _care_plan_by_tile(
    states: State, genome: M3AnimalGenomeV2, player: int
) -> jax.Array:
    animal = states.tile_animal[:, player]
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None, None]
    policy = genome.care_policy[batch, safe]
    first_target = genome.first_cycle_care_bonus_target[batch, safe]
    steady_target = genome.steady_cycle_care_bonus_target[batch, safe]
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)[:, None, None]
    age = current_day - states.tile_origin_day[:, player].astype(jnp.int16)
    before_first_production = age < (_ANIMAL_FIRST[safe] - 1)
    target = jnp.where(before_first_production, first_target, steady_target)
    policy_active = (policy == M3AnimalCarePolicyV2.CAPACITY_AWARE_EVERY_CYCLE) | (
        (policy == M3AnimalCarePolicyV2.FIRST_CYCLE_ONLY) & before_first_production
    )
    pending = states.tile_pending_care[:, player].astype(jnp.int16)
    held = states.tile_yield[:, player].astype(jnp.int16)
    capacity_ok = (held + 1 + pending + 1 <= _ANIMAL_MAX[safe]) | (
        held >= genome.animal_harvest_trigger_units[batch, safe]
    )
    flags = states.tile_flags[:, player]
    return (
        (animal >= 0)
        & policy_active
        & (pending < target)
        & capacity_ok
        & ((flags & jnp.uint8(FLAG_CARED)) == 0)
    )


def m3_feed_obligation_masks_v2(
    states: State, genome: M3AnimalGenomeV2, player: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-tile survival, production-bonus and CARE feed obligations.

    M3 originally derived these masks only inside its species ledger.  The
    M3.6 joint scheduler also needs the tile-level workload before it assigns
    units between crops and animals; otherwise ordinary CARE feed is invisible
    until it becomes a next-day escape risk.
    """

    animal = states.tile_animal[:, player]
    flags = states.tile_flags[:, player]
    unfed = (flags & jnp.uint8(FLAG_FED)) == 0
    survival = (animal >= 0) & unfed & (states.tile_neglect[:, player] == 1)
    production = _production_today_by_tile(states, player)
    bonus = (
        (animal >= 0)
        & unfed
        & production
        & (states.tile_pending_care[:, player] > 0)
    )
    care = _care_plan_by_tile(states, genome, player) & unfed
    return survival, bonus, care


def m3_care_obligation_mask_v2(
    states: State, genome: M3AnimalGenomeV2, player: int
) -> jax.Array:
    """Return animals whose current-day CARE investment is still pending."""

    return _care_plan_by_tile(states, genome, player)


def derive_m3_commitment_ledger_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
) -> AnimalCommitmentLedgerV2:
    """Derive physical inventory and exact current-day feed obligations."""

    active = _active_animal_counts(states, player)
    in_shed = states.shed[
        :, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].astype(jnp.int16)
    carried = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    market = controller.market_tasks
    pending_mask = (
        (market.status == TaskStatusV1.ACTIVE)
        & (market.task_type == TaskTypeV1.ANIMAL_PURCHASE)
    )
    market_animal = jnp.clip(
        market.item_id.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
    )
    pending = jnp.sum(
        jax.nn.one_hot(market_animal, NUM_ANIMALS, dtype=jnp.int16)
        * pending_mask[..., None]
        * market.quantity[..., None].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    committed = active + in_shed + carried + pending

    kind = states.tile_kind[:, player].reshape(states.step.shape[0], -1)
    animal_flat = states.tile_animal[:, player].reshape(states.step.shape[0], -1)
    empty_structure = jnp.stack(
        tuple(
            jnp.sum(
                (kind == _ANIMAL_STRUCTURE[animal]) & (animal_flat < 0),
                axis=-1,
                dtype=jnp.int16,
            )
            for animal in range(NUM_ANIMALS)
        ),
        axis=-1,
    )

    animal = states.tile_animal[:, player]
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    survival, bonus, care = m3_feed_obligation_masks_v2(states, genome, player)
    required = survival | bonus | care

    def by_species(mask):
        return jnp.sum(
            jax.nn.one_hot(safe, NUM_ANIMALS, dtype=jnp.int16)
            * mask[..., None],
            axis=(1, 2),
            dtype=jnp.int16,
        )

    horizon = genome.feed_stock_horizon_days.astype(jnp.int16)[:, None]
    policy = genome.care_policy
    daily = jnp.where(policy == M3AnimalCarePolicyV2.OFF, (horizon + 1) // 2, horizon)
    planned = active * daily + jnp.minimum(in_shed + carried + pending, 1)
    return AnimalCommitmentLedgerV2(
        active=active,
        in_shed=in_shed,
        carried=carried,
        pending_purchase=pending,
        committed=committed,
        empty_structures=empty_structure,
        survival_feed_due_today=by_species(survival),
        bonus_feed_due_today=by_species(bonus),
        care_feed_due_today=by_species(care),
        feed_required_today=by_species(required),
        planned_feed_horizon=planned.astype(jnp.int16),
    )


def ensure_m3_projects_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    phase, target, land_target, hand_target = m3_phase_targets_v2(states, genome)
    ledger = derive_m3_commitment_ledger_v2(states, controller, genome, player)
    projects = controller.projects
    animal_base = NUM_CROPS
    for animal in range(NUM_ANIMALS):
        slot = animal_base + animal
        active = ledger.active[:, animal]
        committed = ledger.committed[:, animal]
        desired = target[:, animal]
        status = jnp.where(
            (desired > 0) | (committed > 0),
            ProjectStatusV2.ACTIVE,
            ProjectStatusV2.PLANNED,
        ).astype(jnp.int8)
        projects = projects._replace(
            project_id=projects.project_id.at[:, slot].set(jnp.int16(slot)),
            project_type=projects.project_type.at[:, slot].set(
                jnp.int8(ProjectTypeV2.ANIMAL_LOT)
            ),
            item_id=projects.item_id.at[:, slot].set(jnp.int8(animal)),
            status=projects.status.at[:, slot].set(status),
            phase=projects.phase.at[:, slot].set(phase),
            target_count=projects.target_count.at[:, slot].set(desired),
            committed_count=projects.committed_count.at[:, slot].set(committed),
            active_count=projects.active_count.at[:, slot].set(active),
            start_step=projects.start_step.at[:, slot].set(
                jnp.where(
                    (projects.start_step[:, slot] < 0) & (desired > 0),
                    states.step,
                    projects.start_step[:, slot],
                )
            ),
            stop_step=projects.stop_step.at[:, slot].set(
                genome.animal_investment_stop_step[:, animal]
            ),
            latest_bank_step=projects.latest_bank_step.at[:, slot].set(
                genome.liquidation_start_step
            ),
            cash_budget=projects.cash_budget.at[:, slot].set(
                genome.animal_project_cash_cap[:, animal]
            ),
            layout_policy_id=projects.layout_policy_id.at[:, slot].set(
                genome.animal_layout_policy[:, animal]
            ),
        )

    for slot, project_type, desired, current in (
        (
            animal_base + NUM_ANIMALS,
            ProjectTypeV2.LAND_EXPANSION,
            land_target,
            states.unlocked_count[:, player],
        ),
        (
            animal_base + NUM_ANIMALS + 1,
            ProjectTypeV2.WORKFORCE,
            hand_target,
            states.hires_today[:, player],
        ),
    ):
        projects = projects._replace(
            project_id=projects.project_id.at[:, slot].set(jnp.int16(slot)),
            project_type=projects.project_type.at[:, slot].set(jnp.int8(project_type)),
            item_id=projects.item_id.at[:, slot].set(jnp.int8(-1)),
            status=projects.status.at[:, slot].set(jnp.int8(ProjectStatusV2.ACTIVE)),
            phase=projects.phase.at[:, slot].set(phase),
            target_count=projects.target_count.at[:, slot].set(desired.astype(jnp.int16)),
            active_count=projects.active_count.at[:, slot].set(current.astype(jnp.int16)),
        )

    tile_animal = states.tile_animal[:, player].astype(jnp.int16)
    kind = states.tile_kind[:, player]
    physical_project = jnp.where(tile_animal >= 0, animal_base + tile_animal, -1)
    keep_reservation = (
        ((kind == TileKind.COOP) | (kind == TileKind.PASTURE) | (kind == TileKind.EMPTY))
        & (controller.tile_project_id >= animal_base)
        & (controller.tile_project_id < animal_base + NUM_ANIMALS)
    )
    tile_project = jnp.where(
        tile_animal >= 0,
        physical_project,
        jnp.where(keep_reservation, controller.tile_project_id, -1),
    ).astype(jnp.int16)
    return controller._replace(
        projects=projects,
        tile_project_id=tile_project,
        route_phase=phase,
        liquidation_mode=states.step >= genome.liquidation_start_step,
    )


def clear_invalidated_m3_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    tasks = controller.unit_tasks
    active = tasks.status == TaskStatusV1.ACTIVE
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    kind = states.tile_kind[batch, player, y, x]
    animal = states.tile_animal[batch, player, y, x]
    task = tasks.task_type
    animal_service = (
        (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.ANIMAL_CARE)
        | (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    place = task == TaskTypeV1.ANIMAL_PLACE
    desired_animal = jnp.clip(tasks.item_id.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1)
    place_invalid = place & (
        (animal >= 0) | (kind != _ANIMAL_STRUCTURE[desired_animal])
    )
    build = task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
    desired_structure = _ANIMAL_STRUCTURE[jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_ANIMALS - 1)]
    build_completed_stale = build & (kind == desired_structure) & (animal < 0)
    # A weed or another project may claim a previously reserved empty tile
    # before the builder arrives.  The official BUILD then becomes a legal
    # silent no-op.  Retire the stale route before compilation so the planner
    # can select another empty tile instead of misclassifying that no-op as an
    # unexplained effect failure.
    build_target_invalid = build & (
        (kind != TileKind.EMPTY) & (kind != desired_structure)
    )
    returning_animal_product = (
        (
            (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
            | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        )
        & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
    )
    unit = jnp.arange(tasks.task_type.shape[1], dtype=jnp.int32)[None]
    inventory_empty = jnp.sum(
        states.unit_inventory[batch, player, unit].astype(jnp.int32),
        axis=-1,
        dtype=jnp.int32,
    ) == 0
    # The official day-end can auto-deposit a returning unit.  E3 deliberately
    # keeps the route task until a DROP is observed, so clear that now-empty
    # return leg before it emits a spurious DROP/effect mismatch next morning.
    stale_auto_deposit = returning_animal_product & inventory_empty
    # Fertilizer is a soft by-product.  Once liquidation starts, do not finish
    # or create fertilizer routes that cannot improve the terminal bank.
    stale_liquidation_fertilizer = (
        (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        & (states.step[:, None] >= genome.liquidation_start_step[:, None])
    )
    invalid = active & (
        (animal_service & (animal < 0))
        | place_invalid
        | build_completed_stale
        | build_target_invalid
        | stale_auto_deposit
        | stale_liquidation_fertilizer
    )
    empty = empty_unit_tasks_v2(states.step.shape[0])
    unit_tasks = jax.tree.map(
        lambda old, replacement: jnp.where(invalid, replacement, old),
        tasks,
        empty,
    )
    return controller._replace(unit_tasks=unit_tasks)


def _nearest_shed_distance(position: jax.Array) -> jax.Array:
    return jnp.min(
        jnp.abs(position[:, 0, None].astype(jnp.int16) - _SHED_ACCESS[None, :, 0])
        + jnp.abs(position[:, 1, None].astype(jnp.int16) - _SHED_ACCESS[None, :, 1]),
        axis=-1,
    ).astype(jnp.int16)


def _shed_to_tile_distance() -> jax.Array:
    return jnp.min(
        jnp.abs(_TILE_X[:, None] - _SHED_ACCESS[None, :, 0])
        + jnp.abs(_TILE_Y[:, None] - _SHED_ACCESS[None, :, 1]),
        axis=-1,
    ).astype(jnp.int16)


_SHED_TO_TILE = _shed_to_tile_distance()


def _via_one_shed_access_distance(position: jax.Array) -> jax.Array:
    """Return pickup-plus-travel distance to every tile via one shed access.

    A pickup does not teleport a unit between the four shed access cells.  The
    old expression added the independently nearest access-to-unit distance and
    nearest access-to-tile distance, which could combine two different access
    cells and underestimate a route by up to two turns.  Minimise the complete
    route over the same access cell instead.  The returned cost includes the
    one PICKUP operation but excludes the final operation at the target.
    """

    to_access = (
        jnp.abs(
            position[:, 0, None].astype(jnp.int16) - _SHED_ACCESS[None, :, 0]
        )
        + jnp.abs(
            position[:, 1, None].astype(jnp.int16) - _SHED_ACCESS[None, :, 1]
        )
    ).astype(jnp.int16)
    access_to_tile = (
        jnp.abs(_SHED_ACCESS[:, 0, None] - _TILE_X[None])
        + jnp.abs(_SHED_ACCESS[:, 1, None] - _TILE_Y[None])
    ).astype(jnp.int16)
    return jnp.min(
        to_access[:, :, None] + jnp.int16(1) + access_to_tile[None],
        axis=1,
    ).astype(jnp.int16)


def _layout_cost(animal: jax.Array, policy: jax.Array, active_count: jax.Array) -> jax.Array:
    shed = _SHED_TO_TILE[None]
    qx = jnp.where(animal == 0, 3, jnp.where(animal == 1, 7, 2))[:, None]
    qy = jnp.where(animal == 0, 3, jnp.where(animal == 1, 2, 7))[:, None]
    quadrant = jnp.abs(_TILE_X[None] - qx) + jnp.abs(_TILE_Y[None] - qy)
    serpentine = _TILE_Y * BOARD_SIZE + jnp.where((_TILE_Y % 2) == 0, _TILE_X, 9 - _TILE_X)
    rank = (animal.astype(jnp.int32) * 29 + active_count.astype(jnp.int32)) % 100
    strip = jnp.abs(serpentine[None] - rank[:, None])
    # Cluster expansion starts from a stable species-specific anchor.  M4 will
    # replace this coarse proxy with joint layout optimization.
    ax = jnp.where(animal == 0, 4, jnp.where(animal == 1, 6, 3))[:, None]
    ay = jnp.where(animal == 0, 4, jnp.where(animal == 1, 3, 6))[:, None]
    cluster = jnp.abs(_TILE_X[None] - ax) + jnp.abs(_TILE_Y[None] - ay)
    return jnp.where(
        policy[:, None] == M3AnimalLayoutPolicyV2.CENTER_COMPACT,
        shed,
        jnp.where(
            policy[:, None] == M3AnimalLayoutPolicyV2.CLUSTER_EXPANSION,
            cluster,
            jnp.where(
                policy[:, None] == M3AnimalLayoutPolicyV2.QUADRANT_ZONED,
                quadrant,
                strip,
            ),
        ),
    ).astype(jnp.int32)


def _set_unit_task(
    tasks,
    unit: int,
    assign: jax.Array,
    *,
    task_type: jax.Array,
    target_id: jax.Array,
    item_id: jax.Array,
    phase: jax.Array,
    states: State,
    distance: jax.Array,
    deadline: jax.Array,
    quantity: jax.Array | None = None,
):
    target_x = _TILE_X[jnp.clip(target_id, 0, BOARD_SIZE * BOARD_SIZE - 1)]
    target_y = _TILE_Y[jnp.clip(target_id, 0, BOARD_SIZE * BOARD_SIZE - 1)]

    def set_field(field, value):
        return field.at[:, unit].set(jnp.where(assign, value, field[:, unit]))

    task_quantity = (
        jnp.ones_like(states.step, dtype=jnp.int16)
        if quantity is None
        else jnp.asarray(quantity, dtype=jnp.int16)
    )
    return tasks._replace(
        task_type=set_field(tasks.task_type, task_type.astype(jnp.int8)),
        target_id=set_field(tasks.target_id, target_id.astype(jnp.int16)),
        target_x=set_field(tasks.target_x, target_x.astype(jnp.int8)),
        target_y=set_field(tasks.target_y, target_y.astype(jnp.int8)),
        item_id=set_field(tasks.item_id, item_id.astype(jnp.int8)),
        quantity=set_field(tasks.quantity, task_quantity),
        phase=set_field(tasks.phase, phase.astype(jnp.int8)),
        start_step=set_field(tasks.start_step, states.step.astype(jnp.int16)),
        last_progress_step=set_field(tasks.last_progress_step, states.step.astype(jnp.int16)),
        expected_finish_step=set_field(
            tasks.expected_finish_step, (states.step + distance + 1).astype(jnp.int16)
        ),
        deadline_step=set_field(tasks.deadline_step, deadline.astype(jnp.int16)),
        status=set_field(
            tasks.status,
            jnp.full_like(states.step, TaskStatusV1.ACTIVE, dtype=jnp.int8),
        ),
        failure_code=set_field(
            tasks.failure_code,
            jnp.full_like(states.step, FailureCodeV1.NONE, dtype=jnp.int8),
        ),
    )


def materialize_m3_unit_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
    eligible_units: jax.Array | None = None,
    unlock_commitment_mask: jax.Array | None = None,
    externally_reserved_tiles: jax.Array | None = None,
) -> ProjectControllerStateV2:
    """Greedily assign sticky animal service, placement and building tasks.

    ``unlock_commitment_mask`` is the M3.7 value/deadline gate for animals that
    have already consumed cash and shed capacity.  It is deliberately optional
    so frozen M3/M3.6 callers retain byte-for-byte scheduling semantics.  When
    enabled, survival FEED and capacity-loss collection remain absolute; only
    deferrable service may yield to a viable PICKUP/PLACE/BUILD pipeline.
    """

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    if eligible_units is None:
        eligible_units = jnp.ones((batch_size, MAX_UNITS), dtype=jnp.bool_)
    _, target, _, _ = m3_phase_targets_v2(states, genome)
    ledger = derive_m3_commitment_ledger_v2(states, controller, genome, player)
    tasks = controller.unit_tasks
    active_task = tasks.status == TaskStatusV1.ACTIVE
    reserved_tile = (
        jnp.zeros((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_)
        if externally_reserved_tiles is None
        else jnp.asarray(externally_reserved_tiles, dtype=jnp.bool_)
    )
    active_target = jnp.clip(tasks.target_id.astype(jnp.int32), 0, BOARD_SIZE * BOARD_SIZE - 1)
    # A collection task no longer owns the animal tile after the operation has
    # succeeded and its unit is returning to the shed.  Keeping the reservation
    # through MOVE_TO_DEPOT blocked another unit from feeding/caring for the
    # same animal until it was too late in the day.
    target_reserved = active_task & (
        (tasks.phase != TaskPhaseV1.MOVE_TO_DEPOT)
        & (tasks.phase != TaskPhaseV1.DEPOSIT)
    )
    reserved_tile = reserved_tile.at[batch[:, None], active_target].max(
        target_reserved
    )

    active_feed = active_task & (tasks.task_type == TaskTypeV1.ANIMAL_FEED)
    unit_wheat = states.unit_inventory[:, player, :, 0].astype(jnp.int16)
    # Wheat reserved by an active one-animal feed route cannot fund another
    # worker's pickup.
    active_feed_pickup_reserve = jnp.sum(
        jnp.where(
            active_feed & (unit_wheat <= 0),
            jnp.maximum(tasks.quantity, 1),
            0,
        ),
        axis=-1,
        dtype=jnp.int16,
    )
    shed_wheat_available = jnp.maximum(
        states.shed[:, player, 0].astype(jnp.int16)
        - active_feed_pickup_reserve,
        0,
    )
    active_place = active_task & (tasks.task_type == TaskTypeV1.ANIMAL_PLACE)
    active_place_animal = jnp.clip(tasks.item_id.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1)
    place_reserved = jnp.sum(
        jax.nn.one_hot(active_place_animal, NUM_ANIMALS, dtype=jnp.int16)
        * active_place[..., None],
        axis=1,
        dtype=jnp.int16,
    )
    animal_available = jnp.maximum(ledger.in_shed + ledger.carried - place_reserved, 0)

    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    tile_animal = states.tile_animal[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    neglect = states.tile_neglect[:, player].reshape(batch_size, -1)
    held = states.tile_yield[:, player].reshape(batch_size, -1).astype(jnp.int16)
    pending_care = states.tile_pending_care[:, player].reshape(batch_size, -1).astype(jnp.int16)
    safe_animal = jnp.clip(tile_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    production = _production_today_by_tile(states, player).reshape(batch_size, -1)
    care_plan = _care_plan_by_tile(states, genome, player).reshape(batch_size, -1)
    care_plan = care_plan & (
        states.step[:, None] < genome.liquidation_start_step[:, None]
    )
    fed = (flags & jnp.uint8(FLAG_FED)) != 0
    fertilizer_ready = (flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    day_end = ((current_day + 1) * TURNS_PER_DAY - 1).astype(jnp.int16)
    survival = (tile_animal >= 0) & (~fed) & (neglect == 1)
    bonus_feed = (
        (tile_animal >= 0) & (~fed) & production & (pending_care > 0)
    )
    care_feed = (tile_animal >= 0) & (~fed) & care_plan
    feed_required = survival | bonus_feed | care_feed
    # Step 695 is the final executable end-of-day refresh.  States 696+ have
    # no later neglect/escape transition, so feeding there cannot preserve an
    # animal and only steals terminal collection capacity.
    feed_required = feed_required & (
        states.step[:, None]
        <= jnp.int16(EPISODE_STEPS - TURNS_PER_DAY - 1)
    )
    # Anonymous second-stop reservations are not tied to a tile, but they must
    # still reduce the amount later workers may pick up.  Without this budget,
    # four due feeds produced pickup quantities 2+2+2+1: each worker counted
    # the same future targets again even though only four FEED operations
    # existed.  Wheat already carried and active pickup batches cover part of
    # the live demand before any new task is assigned.
    carried_wheat_units = jnp.sum(unit_wheat, axis=-1, dtype=jnp.int16)
    feed_pickup_demand = jnp.maximum(
        jnp.sum(feed_required, axis=-1, dtype=jnp.int16)
        - carried_wheat_units
        - active_feed_pickup_reserve,
        0,
    ).astype(jnp.int16)
    project_id = controller.tile_project_id.reshape(batch_size, -1)
    active_build = active_task & (tasks.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
    active_build_animal = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    build_reserved = jnp.sum(
        jax.nn.one_hot(active_build_animal, NUM_ANIMALS, dtype=jnp.int16)
        * active_build[..., None],
        axis=1,
        dtype=jnp.int16,
    )
    # Structure capacity is physical, not project-owned.  In particular cows
    # and sheep share one pasture pool; a pasture first created while serving a
    # sheep project remains a legal cow placement tile (and vice versa).
    # Project tags still record the current assignment, but must not partition
    # official capacity or force redundant structures.
    coop_empty = jnp.sum(
        (kind == TileKind.COOP) & (tile_animal < 0),
        axis=-1,
        dtype=jnp.int16,
    )
    pasture_empty = jnp.sum(
        (kind == TileKind.PASTURE) & (tile_animal < 0),
        axis=-1,
        dtype=jnp.int16,
    )
    animal_gap = jnp.maximum(target - ledger.active, 0).astype(jnp.int16)
    coop_need = jnp.maximum(
        animal_gap[:, 0] - coop_empty - build_reserved[:, 0], 0
    ).astype(jnp.int16)
    pasture_need = jnp.maximum(
        jnp.sum(animal_gap[:, 1:], axis=-1, dtype=jnp.int16)
        - pasture_empty
        - jnp.sum(build_reserved[:, 1:], axis=-1, dtype=jnp.int16),
        0,
    ).astype(jnp.int16)
    # Tag a new shared pasture with the species whose already purchased stock
    # and target debt put more value at risk.  This is only deterministic task
    # attribution; either species may use the resulting physical pasture.
    pasture_priority = (
        (ledger.in_shed[:, 1:] + ledger.carried[:, 1:]).astype(jnp.int32)
        * 1_000
        + animal_gap[:, 1:].astype(jnp.int32) * 10
        + jnp.asarray((1, 0), dtype=jnp.int32)[None]
    )
    pasture_species = jnp.argmax(pasture_priority, axis=-1).astype(jnp.int32) + 1
    need_build = jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int16)
    need_build = need_build.at[:, 0].set(coop_need)
    need_build = need_build.at[batch, pasture_species].set(pasture_need)
    # Cows and sheep share the *kind* of structure, not one tile's occupancy:
    # every active animal still consumes one physical pasture tile.  Preserve
    # both species' admitted placement budgets when forming the shared build
    # pool; using max(cow, sheep) stranded half of a 2-cow/2-sheep opening and
    # could make a late-placed sheep escape before its first service cycle.
    coop_wave_room = jnp.maximum(
        genome.animal_place_wave_size[:, 0].astype(jnp.int16)
        - coop_empty
        - build_reserved[:, 0],
        0,
    )
    pasture_wave = jnp.sum(
        genome.animal_place_wave_size[:, 1:].astype(jnp.int16),
        axis=-1,
        dtype=jnp.int16,
    )
    pasture_wave_room = jnp.maximum(
        pasture_wave
        - pasture_empty
        - jnp.sum(build_reserved[:, 1:], axis=-1, dtype=jnp.int16),
        0,
    )
    need_build = need_build.at[:, 0].set(
        jnp.minimum(need_build[:, 0], coop_wave_room)
    )
    need_build = need_build.at[batch, pasture_species].set(
        jnp.minimum(need_build[batch, pasture_species], pasture_wave_room)
    )

    for unit in range(MAX_UNITS):
        unit_free = (
            states.unit_active[:, player, unit]
            & eligible_units[:, unit]
            & (~active_task[:, unit])
        )
        position = states.unit_pos[:, player, unit].astype(jnp.int16)
        travel = (
            jnp.abs(_TILE_X[None] - position[:, 0, None])
            + jnp.abs(_TILE_Y[None] - position[:, 1, None])
        ).astype(jnp.int16)

        unit_has_wheat = states.unit_inventory[:, player, unit, 0] > 0
        feed_distance = jnp.where(
            unit_has_wheat[:, None],
            travel,
            _via_one_shed_access_distance(position),
        ).astype(jnp.int16)
        remaining_today = (day_end - states.step).astype(jnp.int16)
        can_finish_feed = feed_distance + 1 <= remaining_today[:, None]
        feed_resource_available = unit_has_wheat | (
            (shed_wheat_available > 0) & (feed_pickup_demand > 0)
        )
        feed_mask = (
            feed_required
            & can_finish_feed
            & (~reserved_tile)
            & feed_resource_available[:, None]
        )
        expected_incoming = jnp.where(
            production,
            1 + jnp.where(fed | feed_required, pending_care, 0),
            0,
        ).astype(jnp.int16)
        capacity_loss = jnp.maximum(
            held + expected_incoming - _ANIMAL_MAX[safe_animal], 0
        )
        threshold = genome.animal_harvest_trigger_units[batch[:, None], safe_animal]
        terminal_harvest = (
            states.step[:, None] >= genome.liquidation_start_step[:, None]
        ) & (held > 0)
        harvest_mask = (
            (tile_animal >= 0)
            & (held > 0)
            & ((capacity_loss > 0) | (held >= threshold) | terminal_harvest)
            & (~reserved_tile)
        )
        bank_route_steps = travel + 1 + _SHED_TO_TILE[None] + 1
        harvest_deadline = jnp.where(
            (capacity_loss > 0), day_end[:, None] - 1, EPISODE_STEPS - 2
        )
        harvest_mask = harvest_mask & (
            travel + 1 <= (harvest_deadline - states.step[:, None])
        )
        # Harvest is only a complete task after the product is deposited.
        # Hands disappear at day end, so every assigned collection must reserve
        # the entire outbound-operation-return-deposit route within the day.
        harvest_mask = harvest_mask & (
            bank_route_steps <= remaining_today[:, None]
        )
        harvest_mask = harvest_mask & (
            states.step[:, None] + bank_route_steps <= EPISODE_STEPS - 2
        )

        care_mask = (
            (tile_animal >= 0)
            & care_plan
            & fed
            & (~reserved_tile)
            & (travel + 1 <= remaining_today[:, None])
        )
        fert_policy = genome.animal_fertilizer_policy[:, None]
        visiting = (
            survival
            | bonus_feed
            | care_feed
            | harvest_mask
            | care_plan
            | (travel == 0)
        )
        fertilizer_mask = (
            (tile_animal >= 0)
            & fertilizer_ready
            & (fert_policy != M3AnimalFertilizerPolicyV2.IGNORE)
            & (
                (
                    (fert_policy == M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL)
                    | (fert_policy == M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS)
                )
                | visiting
            )
            & (~reserved_tile)
            & (states.step[:, None] < genome.liquidation_start_step[:, None])
            & (
                travel + 1 + _SHED_TO_TILE[None] + 1
                <= remaining_today[:, None]
            )
        )

        # Value-aware priority: survival is absolute; capacity and CARE bonus
        # risks use current market value rather than a permanently fixed order.
        product = _ANIMAL_PRODUCT[safe_animal]
        product_value = states.market_price[batch[:, None], product].astype(jnp.int32)
        service_score = jnp.full((batch_size, BOARD_SIZE * BOARD_SIZE), _INVALID_SCORE, dtype=jnp.int32)
        feed_risk = jnp.where(survival, 100_000, pending_care * product_value)
        harvest_risk = capacity_loss.astype(jnp.int32) * product_value
        feed_priority = jnp.where(survival, 0, 2_000 - jnp.minimum(feed_risk, 1_500))
        harvest_priority = jnp.where(capacity_loss > 0, 1_500 - jnp.minimum(harvest_risk, 1_000), 4_500)
        service_score = jnp.where(
            feed_mask,
            feed_priority + feed_distance.astype(jnp.int32) * 10 + _TILE_ID[None],
            service_score,
        )
        service_score = jnp.where(
            harvest_mask,
            jnp.minimum(
                service_score,
                harvest_priority + travel.astype(jnp.int32) * 10 + _TILE_ID[None],
            ),
            service_score,
        )
        service_score = jnp.where(
            care_mask,
            jnp.minimum(service_score, 5_000 + travel.astype(jnp.int32) * 10 + _TILE_ID[None]),
            service_score,
        )
        service_score = jnp.where(
            fertilizer_mask,
            jnp.minimum(service_score, 6_000 + travel.astype(jnp.int32) * 10 + _TILE_ID[None]),
            service_score,
        )
        service_target = jnp.argmin(service_score, axis=-1)
        service_valid = jnp.min(service_score, axis=-1) < _INVALID_SCORE
        ssafe = jnp.clip(service_target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        service_is_feed = feed_mask[batch, ssafe] & (
            service_score[batch, ssafe]
            == feed_priority[batch, ssafe]
            + feed_distance[batch, ssafe].astype(jnp.int32) * 10
            + ssafe
        )
        service_is_harvest = (~service_is_feed) & harvest_mask[batch, ssafe] & (
            service_score[batch, ssafe]
            == harvest_priority[batch, ssafe]
            + travel[batch, ssafe].astype(jnp.int32) * 10
            + ssafe
        )
        service_is_care = (
            (~service_is_feed)
            & (~service_is_harvest)
            & care_mask[batch, ssafe]
            & (
                service_score[batch, ssafe]
                == 5_000 + travel[batch, ssafe].astype(jnp.int32) * 10 + ssafe
            )
        )
        service_task = jnp.where(
            service_is_feed,
            TaskTypeV1.ANIMAL_FEED,
            jnp.where(
                service_is_harvest,
                TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
                jnp.where(
                    service_is_care,
                    TaskTypeV1.ANIMAL_CARE,
                    TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                ),
            ),
        ).astype(jnp.int8)
        service_animal = safe_animal[batch, ssafe]
        service_item = jnp.where(
            service_is_feed,
            0,
            jnp.where(
                service_is_harvest,
                _ANIMAL_PRODUCT[service_animal],
                jnp.where(service_is_care, service_animal, 8),
            ),
        ).astype(jnp.int8)
        service_distance = jnp.where(
            service_is_feed, feed_distance[batch, ssafe], travel[batch, ssafe]
        ).astype(jnp.int16)
        service_deadline = jnp.where(
            service_is_feed | (service_is_harvest & (capacity_loss[batch, ssafe] > 0)),
            day_end,
            EPISODE_STEPS - 2,
        ).astype(jnp.int16)
        service_hard = (
            (service_is_feed & survival[batch, ssafe])
            | (
                service_is_harvest
                & (capacity_loss[batch, ssafe] > 0)
            )
        )

        carried_by_species = states.unit_inventory[
            :, player, unit, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int16)
        has_carried_animal = jnp.any(carried_by_species > 0, axis=-1)
        carried_species = jnp.argmax(carried_by_species, axis=-1).astype(jnp.int32)
        stock_valid = animal_available > 0
        first_stock = jnp.argmax(stock_valid, axis=-1).astype(jnp.int32)
        place_species = jnp.where(has_carried_animal, carried_species, first_stock)
        has_stock = has_carried_animal | jnp.any(stock_valid, axis=-1)
        matching_structure = (
            (kind == _ANIMAL_STRUCTURE[place_species][:, None])
            & (tile_animal < 0)
            & (~reserved_tile)
        )
        place_score = jnp.where(
            matching_structure,
            travel.astype(jnp.int32) * 10 + _TILE_ID[None],
            _INVALID_SCORE,
        )
        place_target = jnp.argmin(place_score, axis=-1)
        place_valid = has_stock & (jnp.min(place_score, axis=-1) < _INVALID_SCORE)
        place_distance = jnp.take_along_axis(travel, place_target[:, None], axis=-1)[:, 0]
        place_route_distance = jnp.where(
            has_carried_animal,
            place_distance,
            jnp.take_along_axis(
                _via_one_shed_access_distance(position),
                place_target[:, None],
                axis=-1,
            )[:, 0],
        ).astype(jnp.int16)
        # Every unit is returned to the farm entrance and drops its inventory
        # at day end (including the persistent farmer).  Starting a route that
        # cannot reach PLACE before that reset makes no durable progress: the
        # animal is returned to the shed and the same route is materialized
        # again on a later day.  Completion on the final action of the day is
        # legal, hence ``distance <= day_end - step`` rather than a stricter
        # off-by-one check.
        place_valid = place_valid & (place_route_distance <= remaining_today)

        need_species_mask = need_build > 0
        build_species = jnp.argmax(need_species_mask, axis=-1).astype(jnp.int32)
        has_build_need = jnp.any(need_species_mask, axis=-1)
        empty_tile = (kind == TileKind.EMPTY) & (~reserved_tile)
        build_layout = _layout_cost(
            build_species,
            genome.animal_layout_policy[batch, build_species],
            ledger.active[batch, build_species],
        )
        build_score = jnp.where(
            empty_tile,
            build_layout * 100 + travel.astype(jnp.int32) * 10 + _TILE_ID[None],
            _INVALID_SCORE,
        )
        build_target = jnp.argmin(build_score, axis=-1)
        build_valid = has_build_need & (jnp.min(build_score, axis=-1) < _INVALID_SCORE)
        build_distance = jnp.take_along_axis(travel, build_target[:, None], axis=-1)[:, 0]
        # BUILD also has no persistent movement progress across a day reset.
        # Only admit a new build route when its terminal operation can execute
        # by the last action of the current day.
        build_valid = build_valid & (build_distance <= remaining_today)

        product_inventory = jnp.sum(
            states.unit_inventory[:, player, unit, :NUM_PRODUCTS].astype(jnp.int32),
            axis=-1,
            dtype=jnp.int32,
        )
        fertilizer_inventory = (
            states.unit_inventory[:, player, unit, NUM_PRODUCTS - 1] > 0
        )
        # M3.6 batches fertilizer: after collecting every still-feasible tile,
        # the carrying worker must bank the accumulated units even before
        # liquidation.  Other products retain the original liquidation rule.
        deposit_valid = (
            ((states.step >= genome.liquidation_start_step) & (product_inventory > 0))
            | fertilizer_inventory
        )
        deposit_distance = _nearest_shed_distance(position)

        if unlock_commitment_mask is not None:
            unlock_mask = jnp.asarray(unlock_commitment_mask, dtype=jnp.bool_)
            unlock_place = place_valid & unlock_mask[batch, place_species]
            unlock_build = build_valid & unlock_mask[batch, build_species]
            # Mask only deferrable service when this lane can immediately
            # release viable locked capital.  The frozen selection code below
            # then naturally emits PLACE/BUILD, while survival/capacity tasks,
            # deposits and every legality check keep their original ordering.
            service_valid = service_valid & (
                service_hard | (~(unlock_place | unlock_build))
            )

        # Frozen M3/M3.6 ordering; M3.7 affects it solely through the narrow
        # value-gated service mask above.
        choose_service = unit_free & service_valid
        choose_deposit = unit_free & (~choose_service) & deposit_valid
        choose_place = unit_free & (~choose_service) & (~choose_deposit) & place_valid
        choose_build = (
            unit_free
            & (~choose_service)
            & (~choose_place)
            & (~choose_deposit)
            & build_valid
            & (states.step < genome.liquidation_start_step)
        )
        assign = choose_service | choose_place | choose_deposit | choose_build
        chosen_target = jnp.where(
            choose_service,
            service_target,
            jnp.where(choose_place, place_target, jnp.where(choose_deposit, 44, build_target)),
        )
        chosen_task = jnp.where(
            choose_service,
            service_task,
            jnp.where(
                choose_place,
                TaskTypeV1.ANIMAL_PLACE,
                jnp.where(
                    choose_deposit,
                    TaskTypeV1.SHED_DEPOSIT,
                    TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
                ),
            ),
        ).astype(jnp.int8)
        chosen_species = jnp.where(
            choose_service, service_animal, jnp.where(choose_place, place_species, build_species)
        ).astype(jnp.int32)
        chosen_item = jnp.where(
            choose_service,
            service_item,
            jnp.where(
                choose_place,
                NUM_PRODUCTS + place_species,
                jnp.where(choose_deposit, -1, build_species),
            ),
        ).astype(jnp.int8)
        chosen_distance = jnp.where(
            choose_service,
            service_distance,
            jnp.where(
                choose_place,
                place_route_distance,
                jnp.where(choose_deposit, deposit_distance, build_distance),
            ),
        ).astype(jnp.int16)
        chosen_deadline = jnp.where(
            choose_service, service_deadline, EPISODE_STEPS - 2
        ).astype(jnp.int16)
        # A one-target fallback task picks exactly one unit.  Multi-unit pickup
        # is owned by an explicit FEED_TOUR whose future targets are reserved;
        # increasing this quantity without such a route previously stranded
        # wheat and duplicated anonymous demand reservations.
        chosen_quantity = jnp.ones_like(states.step, dtype=jnp.int16)
        chosen_phase = jnp.where(
            choose_place,
            jnp.where(
                has_carried_animal,
                TaskPhaseV1.MOVE_TO_TARGET,
                TaskPhaseV1.MOVE_TO_SHED,
            ),
            TaskPhaseV1.MOVE_TO_TARGET,
        ).astype(jnp.int8)
        tasks = _set_unit_task(
            tasks,
            unit,
            assign,
            task_type=chosen_task,
            target_id=chosen_target,
            item_id=chosen_item,
            phase=chosen_phase,
            states=states,
            distance=chosen_distance,
            deadline=chosen_deadline,
            quantity=chosen_quantity,
        )

        safe_target = jnp.clip(chosen_target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        reserved_tile = reserved_tile.at[batch, safe_target].max(assign)
        shed_wheat_available = jnp.maximum(
            shed_wheat_available
            - jnp.where(
                choose_service & service_is_feed & (~unit_has_wheat),
                chosen_quantity,
                0,
            ).astype(jnp.int16),
            0,
        )
        feed_pickup_demand = jnp.maximum(
            feed_pickup_demand
            - jnp.where(
                choose_service & service_is_feed & (~unit_has_wheat),
                chosen_quantity,
                0,
            ).astype(jnp.int16),
            0,
        )
        animal_available = animal_available.at[batch, jnp.clip(place_species, 0, 2)].add(
            -choose_place.astype(jnp.int16)
        )
        need_build = need_build.at[batch, jnp.clip(build_species, 0, 2)].add(
            -choose_build.astype(jnp.int16)
        )
        assigned_project = (NUM_CROPS + chosen_species).astype(jnp.int16)
        tile_project_flat = controller.tile_project_id.reshape(batch_size, -1)
        tile_project_flat = tile_project_flat.at[batch, safe_target].set(
            jnp.where(
                choose_place | choose_build,
                assigned_project,
                tile_project_flat[batch, safe_target],
            )
        )
        controller = controller._replace(
            unit_tasks=tasks,
            tile_project_id=tile_project_flat.reshape(batch_size, BOARD_SIZE, BOARD_SIZE),
        )

    return controller._replace(unit_tasks=tasks)


def _visible_town_demand(states: State) -> jax.Array:
    safe = jnp.clip(states.town_shops.astype(jnp.int32), 0, _SHOP_DEMAND.shape[0] - 1)
    slots = jnp.arange(states.town_shops.shape[1], dtype=jnp.int8)[None]
    active = slots < states.town_count[:, None]
    return jnp.sum(_SHOP_DEMAND[safe] * active[..., None], axis=1, dtype=jnp.int16)


def _hire_prefix_cost(current_hires: jax.Array) -> jax.Array:
    offsets = jnp.arange(10, dtype=jnp.int32)[None]
    indices = jnp.clip(
        current_hires[:, None].astype(jnp.int32) + offsets, 0, len(HIRE_COST) - 1
    )
    return jnp.cumsum(_HIRE_COST[indices], axis=-1, dtype=jnp.int32)


def _allocate_units_for_pressure(available: jax.Array, required: jax.Array) -> jax.Array:
    selected = jnp.zeros_like(available, dtype=jnp.int16)
    remaining = required.astype(jnp.int32)
    for product in range(NUM_PRODUCTS):
        take = jnp.minimum(available[:, product].astype(jnp.int32), remaining)
        selected = selected.at[:, product].set(take.astype(jnp.int16))
        remaining = jnp.maximum(remaining - take, 0)
    return selected


def materialize_m3_market_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
    tables: StaticTables | None = None,
    *,
    reserve_future_hires: bool = True,
    rolling_replan: bool = False,
    market_template: jax.Array | None = None,
) -> ProjectControllerStateV2:
    """Create cash-safe sells, land, animals, wheat and daily hires.

    ``reserve_future_hires`` remains enabled for an open-loop M3 genome whose
    future revenue is unknown.  ``rolling_replan`` is narrower: it is valid
    only for the complete-calendar M3.6/M3.7 controller, which replans at the
    start of a day, after the first realized action and after the fertilizer
    collection window, and also has a per-step emergency feed order.  In that
    mode the market planner protects one realized feed day instead of asking a
    newly purchased animal to prepay its whole first-production runway.  The
    animal itself is safe in the shed until placement, while subsequent plans
    react to actual cash, wheat and placement state.  Frozen/open-loop callers
    retain the conservative pre-revenue and replacement reserves by default.
    """

    tables = _DEFAULT_TABLES if tables is None else tables
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    _, target, desired_land, hand_target = m3_phase_targets_v2(states, genome)
    ledger = derive_m3_commitment_ledger_v2(states, controller, genome, player)
    closing = states.step >= genome.liquidation_start_step
    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int16)
    # The final executable action is step 718; the last end-of-day animal
    # refresh is therefore step 695.  Do not reserve unsellable wheat for a
    # day-end that will never occur.
    remaining_day_ends = jnp.maximum(
        (jnp.int16(EPISODE_STEPS - TURNS_PER_DAY - 1) - states.step) // TURNS_PER_DAY + 1,
        0,
    ).astype(jnp.int16)
    configured_horizon = jnp.minimum(
        genome.feed_stock_horizon_days.astype(jnp.int16), remaining_day_ends
    )
    # GROW_ONLY has no market fallback, so its physical reserve must cover the
    # whole remaining executable feed horizon.  BUY_ONLY/HYBRID can retain the
    # configurable rolling horizon and replenish later.
    effective_horizon = jnp.where(
        genome.feed_source_policy == M3FeedSourcePolicyV2.GROW_ONLY,
        remaining_day_ends,
        configured_horizon,
    )[:, None]
    # Gold Replay 94051618 demonstrates a rolling one-day feed policy: on day
    # four it buys a third cow with four wheat already supporting four active
    # animals, then replenishes from realized production/market cash on the
    # next plan.  A complete-calendar controller can safely express that
    # policy because an unplaced purchase does not escape and every realized
    # step is reconciled.  GROW_ONLY remains fully physically reserved.
    effective_horizon = jnp.where(
        rolling_replan
        & (genome.feed_source_policy != M3FeedSourcePolicyV2.GROW_ONLY)[:, None],
        jnp.minimum(effective_horizon, jnp.int16(1)),
        effective_horizon,
    )
    feed_days = jnp.where(
        genome.care_policy == M3AnimalCarePolicyV2.OFF,
        (effective_horizon + 1) // 2,
        effective_horizon,
    )
    feed_target_by_species = ledger.active * feed_days + jnp.where(
        effective_horizon > 0,
        jnp.minimum(ledger.in_shed + ledger.carried + ledger.pending_purchase, 1),
        0,
    )
    feed_target = jnp.sum(feed_target_by_species, axis=-1, dtype=jnp.int16)
    wheat_held = (
        states.shed[:, player, 0].astype(jnp.int16)
        + jnp.sum(states.unit_inventory[:, player, :, 0].astype(jnp.int16), axis=-1)
    )
    sellable = shed.at[:, 0].set(jnp.maximum(shed[:, 0] - feed_target, 0))

    fertilizer_policy = genome.animal_fertilizer_policy
    sellable = sellable.at[:, 8].set(
        jnp.where(
            closing,
            shed[:, 8],
            jnp.where(
                (
                    (fertilizer_policy == M3AnimalFertilizerPolicyV2.COLLECT_WHEN_VISITING_AND_SELL)
                    | (fertilizer_policy == M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL)
                ),
                sellable[:, 8],
                0,
            ),
        )
    )
    due = (
        jnp.mod(states.step[:, None] - genome.sell_phase, jnp.maximum(genome.sell_interval, 1))
        == 0
    )
    town_demand = _visible_town_demand(states)
    floor = genome.sell_price_floor_ratio + 0.05 * (town_demand > 0) + 0.05 * (
        states.market_inventory > MARKET_INITIAL_INVENTORY
    )
    price_ok = states.market_price.astype(jnp.float32) >= _BASE_PRICE[None] * floor
    normal_sell = jnp.ceil(sellable.astype(jnp.float32) * genome.sell_fraction).astype(jnp.int16)
    normal_sell = jnp.where(due & price_ok, normal_sell, 0)
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    pressure = jnp.maximum(shed_used - genome.shed_pressure_trigger.astype(jnp.int32) + 1, 0)
    pressure_sell = _allocate_units_for_pressure(sellable, pressure)
    sell_quantity = jnp.maximum(normal_sell, pressure_sell)
    sell_quantity = jnp.where(closing[:, None], sellable, sell_quantity)
    # Market execution precedes the end-of-day animal refresh.  Selling on the
    # refresh turn can be masked by fertilizer/product inventory generated in
    # the same transition, making an actually executed sale indistinguishable
    # from an effect failure.  There is always a following executable step;
    # the true terminal action (718) is not an end-of-day turn.
    day_end_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
    sell_quantity = jnp.where(day_end_turn[:, None], 0, sell_quantity)

    product_ids = jnp.broadcast_to(
        jnp.arange(NUM_PRODUCTS, dtype=jnp.int8)[None], sell_quantity.shape
    )
    sell_value = jnp.sum(
        exact_market_quote_v1(
            states, tables, product_ids, sell_quantity, buy=False, player=player
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    gross_available = jnp.maximum(
        states.money[:, player] + sell_value, 0
    ).astype(jnp.int32)

    # Feed is an existing irreversible commitment, not optional investment.
    # Reserve both cash and shed capacity for it before land, new animals and
    # hires.  SELL executes before BUY_PRODUCT in the official market phase.
    wheat_deficit = jnp.maximum(feed_target - wheat_held, 0).astype(jnp.int16)
    initial_shed_room = jnp.maximum(SHED_CAPACITY - shed_used, 0).astype(jnp.int16)
    can_buy_feed = genome.feed_source_policy != M3FeedSourcePolicyV2.GROW_ONLY
    wheat_plan = jnp.where(
        can_buy_feed,
        jnp.minimum(wheat_deficit, initial_shed_room),
        0,
    ).astype(jnp.int16)
    # A bulk BUY_PRODUCT order must degrade gracefully when the farm cannot
    # afford the whole horizon at once.  The earlier all-or-nothing check could
    # skip every wheat purchase and let otherwise healthy animals escape.  Use
    # exact official quotes for every feasible quantity and buy the largest
    # affordable prefix instead.
    wheat_quantities = jnp.broadcast_to(
        jnp.arange(SHED_CAPACITY + 1, dtype=jnp.int16)[None],
        (batch_size, SHED_CAPACITY + 1),
    )
    wheat_quotes = exact_market_quote_v1(
        states,
        tables,
        jnp.zeros_like(wheat_quantities, dtype=jnp.int8),
        wheat_quantities,
        buy=True,
        player=player,
        # We need the cost of each requested quantity.  With resource
        # enforcement the quote helper returns the cost of the affordable
        # partial fill, which made an unaffordable request look affordable.
        enforce_resources=False,
    ).astype(jnp.int32)
    wheat_feasible = (
        (wheat_quantities <= wheat_plan[:, None])
        & (wheat_quotes <= gross_available[:, None])
    )
    wheat_buy = jnp.max(
        jnp.where(wheat_feasible, wheat_quantities, 0), axis=-1
    ).astype(jnp.int16)
    wheat_quote = jnp.take_along_axis(
        wheat_quotes, wheat_buy[:, None].astype(jnp.int32), axis=-1
    )[:, 0]
    # Protect the complete pre-revenue feed runway of every admitted animal.
    # A one-cycle reserve is insufficient for cows/sheep: the farm can spend
    # its remaining cash on hires several days before their first product and
    # then starve.  Physical wheat already held (plus today's planned buy)
    # offsets this cash reserve.
    map_animal = states.tile_animal[:, player]
    safe_map_animal = jnp.clip(map_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)[:, None, None]
    first_yield_day = (
        states.tile_origin_day[:, player].astype(jnp.int16)
        + _ANIMAL_FIRST[safe_map_animal]
    )
    interval = _ANIMAL_INTERVAL[safe_map_animal]
    elapsed_after_first = current_day - first_yield_day
    next_production_day = jnp.where(
        elapsed_after_first <= 0,
        first_yield_day,
        first_yield_day
        + ((elapsed_after_first + interval - 1) // interval) * interval,
    )
    harvest_trigger = genome.animal_harvest_trigger_units[
        batch[:, None, None], safe_map_animal
    ].astype(jnp.int16)
    units_needed = jnp.maximum(
        harvest_trigger - states.tile_yield[:, player].astype(jnp.int16), 0
    )
    first_bank_day = jnp.where(
        units_needed > 0,
        next_production_day + jnp.maximum(units_needed - 1, 0) * interval,
        current_day,
    )
    days_to_first_bank = jnp.maximum(first_bank_day - current_day, 0)
    care_enabled = (
        genome.care_policy[batch[:, None, None], safe_map_animal]
        != M3AnimalCarePolicyV2.OFF
    )
    active_runway = jnp.where(
        map_animal >= 0,
        jnp.where(
            care_enabled,
            days_to_first_bank,
            (days_to_first_bank + 1) // 2,
        ),
        0,
    )
    pending_bank_days = (
        _ANIMAL_FIRST[None]
        + (genome.animal_harvest_trigger_units.astype(jnp.int16) - 1)
        * _ANIMAL_INTERVAL[None]
    )
    pending_first_feeds = jnp.where(
        genome.care_policy != M3AnimalCarePolicyV2.OFF,
        pending_bank_days,
        (pending_bank_days + 1) // 2,
    ).astype(jnp.int16)
    pending_runway = jnp.sum(
        (ledger.in_shed + ledger.carried + ledger.pending_purchase)
        * pending_first_feeds,
        axis=-1,
        dtype=jnp.int16,
    )
    pre_revenue_feed_units = jnp.minimum(
        jnp.sum(active_runway, axis=(1, 2), dtype=jnp.int16) + pending_runway,
        jnp.int16(SHED_CAPACITY),
    )
    runway_deficit = jnp.clip(
        pre_revenue_feed_units - wheat_held - wheat_buy, 0, SHED_CAPACITY
    ).astype(jnp.int16)
    runway_quote = jnp.take_along_axis(
        wheat_quotes, runway_deficit[:, None].astype(jnp.int32), axis=-1
    )[:, 0]

    # Every admitted animal needs one rolling replacement cycle in cash,
    # including pre-revenue animals.  Unit FEED executes before the market
    # phase; without this extra cycle the planner sees the pre-action wheat,
    # spends the cash on hires, and only notices the depleted stock one step
    # too late.
    admitted_count = jnp.sum(
        ledger.active + ledger.in_shed + ledger.carried + ledger.pending_purchase,
        axis=-1,
        dtype=jnp.int16,
    )
    replacement_feed_qty = jnp.clip(admitted_count, 0, SHED_CAPACITY).astype(jnp.int16)
    replacement_feed_quote = jnp.take_along_axis(
        wheat_quotes, replacement_feed_qty[:, None].astype(jnp.int32), axis=-1
    )[:, 0]
    protect_feed_cash = jnp.where(
        can_buy_feed & (remaining_day_ends > 1) & (~jnp.asarray(rolling_replan)),
        runway_quote + replacement_feed_quote,
        0,
    )

    # An animal is not a viable commitment if the farm can buy wheat but
    # cannot afford the hands needed to deliver it.  Reserve today's workforce
    # before optional investment, and retain a cash runway until the current
    # commitments can first be harvested and banked.  Current hires may spend
    # the extra one-cycle replacement reserve when the physical feed horizon is
    # already stocked, but they never consume the exact pre-revenue deficit.
    current_hires = states.hires_today[:, player].astype(jnp.int16)
    hand_cap = jnp.clip(hand_target.astype(jnp.int32), 0, len(HIRE_COST))
    route_capacity_per_unit = jnp.maximum(
        jnp.floor(
            TURNS_PER_DAY * genome.maintenance_utilization_cap
        ).astype(jnp.int32),
        1,
    )
    maintenance_cost = _maintenance_cost_by_species(
        genome.care_policy
    ).astype(jnp.int32)
    committed_count = (
        ledger.active + ledger.in_shed + ledger.carried + ledger.pending_purchase
    ).astype(jnp.int32)
    projected_count = jnp.maximum(target.astype(jnp.int32), committed_count)

    def required_hands(counts: jax.Array) -> jax.Array:
        daily_work = jnp.sum(
            counts * maintenance_cost, axis=-1, dtype=jnp.int32
        )
        total_units = (
            daily_work + route_capacity_per_unit - 1
        ) // route_capacity_per_unit
        return jnp.clip(total_units - 1, 0, hand_cap)

    # The genome field is a ceiling, not an instruction to hire an expensive
    # idle workforce.  Today's orders follow already committed obligations;
    # admission reserves the workforce implied by the full effective target.
    hire_goal = required_hands(committed_count)
    projected_hire_goal = required_hands(projected_count)
    desired_hires = jnp.clip(
        hire_goal.astype(jnp.int16) - current_hires, 0, 10
    )
    hire_prefix = _hire_prefix_cost(current_hires)
    hire_slot = jnp.arange(10, dtype=jnp.int16)[None]
    last_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
    terminal_animal_collection = closing & jnp.any(
        (states.tile_animal[:, player] >= 0)
        & (states.tile_yield[:, player] > 0),
        axis=(1, 2),
    )
    hire_funds = jnp.maximum(
        gross_available
        - jnp.where(wheat_buy > 0, wheat_quote, 0),
        0,
    ).astype(jnp.int32)
    hire_affordable = (
        ((remaining_day_ends > 0) | terminal_animal_collection)[:, None]
        & (~last_turn)[:, None]
        & (hire_slot < desired_hires[:, None])
        & (hire_prefix <= hire_funds[:, None])
    )
    today_hire_quote = jnp.max(
        jnp.where(hire_affordable, hire_prefix, 0), axis=-1
    ).astype(jnp.int32)

    active_workforce_days = jnp.max(
        jnp.where(map_animal >= 0, days_to_first_bank, 0), axis=(1, 2)
    ).astype(jnp.int16)
    unplaced_count = ledger.in_shed + ledger.carried + ledger.pending_purchase
    pending_workforce_days = jnp.max(
        jnp.where(unplaced_count > 0, pending_bank_days, 0), axis=-1
    ).astype(jnp.int16)
    target_gap = target > ledger.committed
    gap_workforce_days = jnp.max(
        jnp.where(target_gap & (~closing[:, None]), pending_bank_days, 0), axis=-1
    ).astype(jnp.int16)
    workforce_runway_days = jnp.minimum(
        jnp.maximum(
            active_workforce_days,
            jnp.maximum(pending_workforce_days, gap_workforce_days),
        ),
        remaining_day_ends,
    )
    future_hire_days = jnp.maximum(workforce_runway_days - 1, 0).astype(jnp.int32)
    daily_hire_quote = _HIRE_COST_PREFIX[projected_hire_goal]
    future_hire_reserve = jnp.where(
        reserve_future_hires,
        daily_hire_quote * future_hire_days,
        0,
    ).astype(jnp.int32)
    available = jnp.maximum(
        gross_available
        - jnp.where(wheat_buy > 0, wheat_quote, 0)
        - protect_feed_cash
        - today_hire_quote
        - future_hire_reserve
        - genome.cash_floor,
        0,
    ).astype(jnp.int32)
    shed_room_for_animals = jnp.maximum(
        initial_shed_room - wheat_buy, 0
    ).astype(jnp.int16)

    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    land_needed = unlocked < desired_land.astype(jnp.int32)
    land_cost = _LAND_COST[jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)]
    buy_land = (~closing) & land_needed & (available >= land_cost)
    available = available - jnp.where(buy_land, land_cost, 0)

    # Allocate current empty structures once.  Cow and sheep share the same
    # pasture pool and are processed deterministically without double booking.
    unplaced_by_species = (
        ledger.in_shed + ledger.carried + ledger.pending_purchase
    ).astype(jnp.int16)
    coop_free = jnp.maximum(
        ledger.empty_structures[:, 0].astype(jnp.int16)
        - unplaced_by_species[:, 0],
        0,
    )
    # Cow and sheep consume one shared pasture pool.  Subtract every already
    # committed unplaced pasture animal once before allocating either species;
    # subtracting per species inside the loop double-counted the same free tile.
    pasture_free = jnp.maximum(
        ledger.empty_structures[:, 1].astype(jnp.int16)
        - jnp.sum(unplaced_by_species[:, 1:], axis=-1, dtype=jnp.int16),
        0,
    )
    grow_feed_room = jnp.maximum(wheat_held - feed_target, 0).astype(jnp.int16)
    animal_buy = jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int16)
    for animal in range(NUM_ANIMALS):
        gap = jnp.maximum(target[:, animal] - ledger.committed[:, animal], 0)
        free = coop_free if animal == 0 else pasture_free
        wave = genome.animal_place_wave_size[:, animal].astype(jnp.int16)
        cap_count = genome.animal_project_cash_cap[:, animal] // _ANIMAL_COST[animal]
        cap_gap = jnp.maximum(cap_count.astype(jnp.int16) - ledger.committed[:, animal], 0)
        bankable = (
            states.step
            + _ANIMAL_FIRST[animal] * TURNS_PER_DAY
            + jnp.int16(12)
            < genome.liquidation_start_step
        )
        within_window = states.step < genome.animal_investment_stop_step[:, animal]
        desired = jnp.minimum(jnp.minimum(gap, free), jnp.minimum(wave, cap_gap))
        desired = jnp.minimum(desired, shed_room_for_animals)
        candidate_quantity = jnp.broadcast_to(
            jnp.arange(9, dtype=jnp.int16)[None], (batch_size, 9)
        )
        admission_feed_units = jnp.clip(
            candidate_quantity * pending_first_feeds[:, animal, None],
            0,
            SHED_CAPACITY,
        ).astype(jnp.int16)
        admission_feed_quote = jnp.take_along_axis(
            wheat_quotes, admission_feed_units.astype(jnp.int32), axis=-1
        )
        admission_total = (
            candidate_quantity.astype(jnp.int32) * _ANIMAL_COST[animal]
            + jnp.where(
                can_buy_feed[:, None] & (~jnp.asarray(rolling_replan)),
                admission_feed_quote,
                0,
            )
        )
        # GROW_ONLY admits a new irreversible animal commitment only when
        # already harvested physical wheat covers its complete pre-revenue
        # feed runway.  Future crop yield is intentionally not counted here;
        # a delayed harvest must not turn into an animal escape.
        grow_feed_ok = admission_feed_units <= grow_feed_room[:, None]
        admission_ok = (
            (candidate_quantity <= desired[:, None])
            & (admission_total <= available[:, None])
            & (can_buy_feed[:, None] | grow_feed_ok)
        )
        quantity = jnp.max(
            jnp.where(admission_ok, candidate_quantity, 0), axis=-1
        ).astype(jnp.int16)
        quantity = jnp.where((~closing) & bankable & within_window, quantity, 0)
        animal_buy = animal_buy.at[:, animal].set(quantity)
        selected_total = jnp.take_along_axis(
            admission_total, quantity[:, None].astype(jnp.int32), axis=-1
        )[:, 0]
        selected_feed_units = jnp.take_along_axis(
            admission_feed_units, quantity[:, None].astype(jnp.int32), axis=-1
        )[:, 0]
        available = available - jnp.where(quantity > 0, selected_total, 0)
        grow_feed_room = jnp.maximum(
            grow_feed_room
            - jnp.where(can_buy_feed, 0, selected_feed_units).astype(jnp.int16),
            0,
        )
        shed_room_for_animals = jnp.maximum(shed_room_for_animals - quantity, 0)
        if animal == 0:
            coop_free = jnp.maximum(coop_free - quantity, 0)
        else:
            pasture_free = jnp.maximum(pasture_free - quantity, 0)

    hire_present = (
        # Liquidation stops new investment, but not the workers needed to keep
        # existing animals alive and bank their final products.  Hires may
        # continue through the last real day-end (step 695), and on the final
        # action day when products generated at that refresh still need to be
        # collected, deposited and sold.
        ((remaining_day_ends > 0) | terminal_animal_collection)[:, None]
        & (~last_turn)[:, None]
        & hire_affordable
        & (hire_prefix <= today_hire_quote[:, None])
    )

    candidate_count = NUM_PRODUCTS + 1 + NUM_ANIMALS + 1 + 10
    present = jnp.zeros((batch_size, candidate_count), dtype=jnp.bool_)
    task_type = jnp.zeros((batch_size, candidate_count), dtype=jnp.int8)
    item = jnp.full((batch_size, candidate_count), -1, dtype=jnp.int8)
    quantity = jnp.zeros((batch_size, candidate_count), dtype=jnp.int16)
    present = present.at[:, :NUM_PRODUCTS].set(sell_quantity > 0)
    task_type = task_type.at[:, :NUM_PRODUCTS].set(
        jnp.where(
            closing[:, None],
            TaskTypeV1.TERMINAL_LIQUIDATION,
            TaskTypeV1.SELL_INVENTORY,
        ).astype(jnp.int8)
    )
    item = item.at[:, :NUM_PRODUCTS].set(product_ids)
    quantity = quantity.at[:, :NUM_PRODUCTS].set(sell_quantity)
    land_slot = NUM_PRODUCTS
    present = present.at[:, land_slot].set(buy_land)
    task_type = task_type.at[:, land_slot].set(jnp.int8(TaskTypeV1.BUY_LAND))
    quantity = quantity.at[:, land_slot].set(jnp.int16(1))
    animal_start = land_slot + 1
    present = present.at[:, animal_start : animal_start + NUM_ANIMALS].set(animal_buy > 0)
    task_type = task_type.at[:, animal_start : animal_start + NUM_ANIMALS].set(
        jnp.int8(TaskTypeV1.ANIMAL_PURCHASE)
    )
    item = item.at[:, animal_start : animal_start + NUM_ANIMALS].set(
        NUM_PRODUCTS
        + jnp.broadcast_to(jnp.arange(NUM_ANIMALS, dtype=jnp.int8)[None], animal_buy.shape)
    )
    quantity = quantity.at[:, animal_start : animal_start + NUM_ANIMALS].set(animal_buy)
    wheat_slot = animal_start + NUM_ANIMALS
    present = present.at[:, wheat_slot].set(wheat_buy > 0)
    task_type = task_type.at[:, wheat_slot].set(jnp.int8(TaskTypeV1.BUY_PRODUCT))
    item = item.at[:, wheat_slot].set(jnp.int8(0))
    quantity = quantity.at[:, wheat_slot].set(wheat_buy)
    hire_start = wheat_slot + 1
    present = present.at[:, hire_start : hire_start + 10].set(hire_present)
    task_type = task_type.at[:, hire_start : hire_start + 10].set(jnp.int8(TaskTypeV1.HIRE_WORKER))
    quantity = quantity.at[:, hire_start : hire_start + 10].set(jnp.int16(1))

    # M3 historically packed product sells before the joint M3.5 planner saw
    # them.  Nine sells could therefore consume nine of ten slots and erase a
    # sheep/cow candidate even when the day calendar explicitly prioritized
    # investment.  Apply the same coarse 0/1/2 transaction template before the
    # sub-plan's own ten-slot boundary; M3-only callers retain the legacy order.
    if market_template is not None:
        template = jnp.asarray(market_template, dtype=jnp.int8)[:, None]
        candidate_ordinal = jnp.arange(candidate_count, dtype=jnp.int32)[None]
        candidate_sell = (task_type == TaskTypeV1.SELL_INVENTORY) | (
            task_type == TaskTypeV1.TERMINAL_LIQUIDATION
        )
        candidate_input = task_type == TaskTypeV1.BUY_PRODUCT
        candidate_hire = task_type == TaskTypeV1.HIRE_WORKER
        candidate_animal = task_type == TaskTypeV1.ANIMAL_PURCHASE
        candidate_land = task_type == TaskTypeV1.BUY_LAND
        hire_first = jnp.where(
            candidate_hire,
            0,
            jnp.where(
                candidate_animal,
                1,
                jnp.where(
                    candidate_input,
                    2,
                    jnp.where(candidate_land, 3, jnp.where(candidate_sell, 4, 5)),
                ),
            ),
        )
        sell_first = jnp.where(
            candidate_sell,
            0,
            jnp.where(
                candidate_hire,
                1,
                jnp.where(
                    candidate_animal,
                    2,
                    jnp.where(candidate_input, 3, jnp.where(candidate_land, 4, 5)),
                ),
            ),
        )
        input_first = jnp.where(
            candidate_input,
            0,
            jnp.where(
                candidate_hire,
                1,
                jnp.where(
                    candidate_animal,
                    2,
                    jnp.where(candidate_sell, 3, jnp.where(candidate_land, 4, 5)),
                ),
            ),
        )
        prepack_priority = jnp.where(
            template == 1,
            sell_first,
            jnp.where(template == 2, input_first, hire_first),
        ).astype(jnp.int32)
        prepack_order = jnp.argsort(
            jnp.where(
                present,
                prepack_priority * 100 + candidate_ordinal,
                100_000 + candidate_ordinal,
            ),
            axis=-1,
        )
        present = jnp.take_along_axis(present, prepack_order, axis=-1)
        task_type = jnp.take_along_axis(task_type, prepack_order, axis=-1)
        item = jnp.take_along_axis(item, prepack_order, axis=-1)
        quantity = jnp.take_along_axis(quantity, prepack_order, axis=-1)

    def pack_lane(p, t, i, q):
        indices = jnp.nonzero(p, size=MAX_MARKET_ORDERS, fill_value=-1)[0]
        valid = indices >= 0
        safe = jnp.clip(indices, 0, candidate_count - 1)
        return (
            valid,
            jnp.where(valid, t[safe], 0).astype(jnp.int8),
            jnp.where(valid, i[safe], -1).astype(jnp.int8),
            jnp.where(valid, q[safe], 0).astype(jnp.int16),
        )

    packed_valid, packed_type, packed_item, packed_quantity = jax.vmap(pack_lane)(
        present, task_type, item, quantity
    )
    desired = empty_market_tasks_v2(batch_size)._replace(
        task_type=packed_type,
        item_id=packed_item,
        quantity=packed_quantity,
        start_step=jnp.where(packed_valid, states.step[:, None], -1).astype(jnp.int16),
        deadline_step=jnp.where(
            packed_valid, jnp.minimum(states.step[:, None] + 1, EPISODE_STEPS - 2), -1
        ).astype(jnp.int16),
        status=jnp.where(
            packed_valid, TaskStatusV1.ACTIVE, TaskStatusV1.EMPTY
        ).astype(jnp.int8),
        failure_code=jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int8),
    )
    old_active = controller.market_tasks.status == TaskStatusV1.ACTIVE
    market = jax.tree.map(
        lambda old, new: jnp.where(old_active, old, new),
        controller.market_tasks,
        desired,
    )
    return controller._replace(market_tasks=market)


def m3_policy_step_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M3AnimalGenomeV2,
    player: int,
    tables: StaticTables | None = None,
) -> tuple[E4PlayerActionV1, ProjectControllerStateV2]:
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_m3_tasks_v2(states, controller, genome, player)
    controller = ensure_m3_projects_v2(states, controller, genome, player)
    controller = materialize_m3_unit_tasks_v2(states, controller, genome, player)
    controller = materialize_m3_market_tasks_v2(states, controller, genome, player, tables)
    return compile_full_core_player_action_v1(states, controller, player), controller


def update_m3_controller_from_effects_v2(
    states: State,
    next_states: State,
    controller: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    genome: M3AnimalGenomeV2,
    player: int,
) -> tuple[ProjectControllerStateV2, FullCoreEffectDiagnosticsV1]:
    controller, diagnostics = update_full_core_controller_from_effects_v1(
        states, next_states, controller, action, player
    )
    controller = ensure_m3_projects_v2(next_states, controller, genome, player)
    controller = controller._replace(
        unexplained_effect_failures=(
            controller.unexplained_effect_failures
            + diagnostics.effect_mismatch_count
            + diagnostics.owner_inactive_count
            + diagnostics.deadline_missed_count
            + diagnostics.resource_unavailable_count
        ).astype(jnp.int32)
    )
    return controller, diagnostics


def m3_player_action_dict_v2(action: E4PlayerActionV1) -> dict:
    return {
        "unit_op": action.unit_op,
        "unit_item": action.unit_item,
        "unit_amount": action.unit_amount,
        "unit_count": action.unit_count,
        "market_op": action.market_op,
        "market_item": action.market_item,
        "market_amount": action.market_amount,
        "market_count": action.market_count,
    }


__all__ = [
    "clear_invalidated_m3_tasks_v2",
    "derive_m3_commitment_ledger_v2",
    "ensure_m3_projects_v2",
    "m3_care_obligation_mask_v2",
    "m3_feed_obligation_masks_v2",
    "m3_phase_targets_v2",
    "m3_phase_v2",
    "m3_player_action_dict_v2",
    "m3_policy_step_v2",
    "materialize_m3_market_tasks_v2",
    "materialize_m3_unit_tasks_v2",
    "update_m3_controller_from_effects_v2",
]
