"""Fixed-shape JAX schema declarations frozen by the M0 contract.

These declarations contain no lifecycle or execution logic. Constructors and
episode reset semantics are deliberately deferred to M1.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from kaggriculture_jax.types import State
from strategic_v5.schema import MarketTaskStateV1, UnitTaskStateV1


Array = jax.Array


class RouteGenomeV2(NamedTuple):
    candidate_id: Array
    family_id: Array
    domain_mode: Array
    phase_count: Array
    phase_start_step: Array
    crop_target: Array
    animal_target: Array
    land_target: Array
    hand_target: Array
    crop_lot_size: Array
    animal_lot_size: Array
    max_open_projects: Array
    cash_floor: Array
    project_cash_cap: Array
    maintenance_utilization_cap: Array
    feed_reserve_days: Array
    crop_harvest_age: Array
    crop_fertilizer_policy: Array
    animal_care_policy: Array
    animal_harvest_threshold: Array
    sell_interval: Array
    sell_phase: Array
    sell_price_floor: Array
    sell_cash_trigger: Array
    shed_pressure_trigger: Array
    investment_stop_step: Array
    liquidation_start_step: Array
    crop_layout_policy: Array
    animal_layout_policy: Array
    land_order_policy: Array
    layout_weights: Array
    scheduler_weights: Array
    visible_market_trigger: Array
    visible_demand_trigger: Array


class ProjectStateV2(NamedTuple):
    project_id: Array
    project_type: Array
    item_id: Array
    status: Array
    phase: Array
    target_count: Array
    committed_count: Array
    active_count: Array
    completed_count: Array
    start_step: Array
    last_progress_step: Array
    stop_step: Array
    latest_bank_step: Array
    cash_budget: Array
    cash_spent: Array
    expected_bank_value: Array
    layout_policy_id: Array
    priority_class: Array
    failure_code: Array


class ObligationV2(NamedTuple):
    obligation_type: Array
    project_id: Array
    target_id: Array
    target_x: Array
    target_y: Array
    item_id: Array
    quantity: Array
    priority_level: Array
    earliest_step: Array
    deadline_step: Array
    expected_finish_step: Array
    slack: Array
    required_unit_item: Array
    required_unit_quantity: Array
    required_shed_item: Array
    required_shed_quantity: Array
    cash_required: Array
    market_slots_required: Array
    expected_bank_delta: Array
    value_at_risk: Array
    future_maintenance_actions: Array
    route_insertion_cost: Array
    present: Array
    hard_mask: Array
    failure_code: Array


class UnitPlanStateV2(NamedTuple):
    home_region: Array
    primary_project_id: Array
    active_obligation_id: Array
    route_obligation_ids: Array
    route_length: Array
    route_cursor: Array
    commitment_until_step: Array
    last_switch_step: Array
    switches_today: Array
    last_target_x: Array
    last_target_y: Array


class RouteCardStateV3(NamedTuple):
    """Bounded executor route owned by one unit.

    A route card records future targets and their remaining same-tile action
    bundle before the worker leaves.  This is intentionally separate from the
    legacy one-target ``UnitTaskStateV1``: the latter remains the parity-tested
    atomic executor input, while this record supplies one atomic stop at a
    time.
    """

    card_type: Array
    status: Array
    target_ids: Array
    target_items: Array
    action_masks: Array
    route_length: Array
    route_cursor: Array
    pickup_item: Array
    pickup_quantity: Array
    return_mode: Array
    start_step: Array
    deadline_step: Array
    last_progress_step: Array
    failure_code: Array


class ProjectControllerStateV2(NamedTuple):
    schema_version: Array
    projects: ProjectStateV2
    unit_tasks: UnitTaskStateV1
    unit_plans: UnitPlanStateV2
    route_cards: RouteCardStateV3
    market_tasks: MarketTaskStateV1
    tile_project_id: Array
    route_phase: Array
    liquidation_mode: Array
    last_money: Array
    last_shed: Array
    last_seeds: Array
    last_tile_kind: Array
    last_tile_yield: Array
    project_cap_hits: Array
    obligation_cap_hits: Array
    unexplained_effect_failures: Array


class CommitmentEnvelopeV2(NamedTuple):
    min_actions_by_day: Array
    feed_required_by_day: Array
    expected_shed_in_by_day: Array
    expected_cash_out_by_day: Array
    expected_cash_in_by_day: Array
    occupied_tiles_by_day: Array


class StepDiagnosticsV2(NamedTuple):
    project_cap_hits: Array
    obligation_cap_hits: Array
    unexplained_effect_failures: Array
    plant_without_same_day_water: Array
    unplanned_animal_escape: Array
    terminal_sellable_shed_value: Array
    avoidable_liquidation_loss: Array


class ControllerReconcileDiagnosticsV2(NamedTuple):
    snapshot_was_valid: Array
    money_delta: Array
    shed_delta: Array
    seed_delta: Array
    tile_kind_changed_count: Array
    tile_yield_changed_count: Array
    cleared_project_count: Array
    cleared_unit_task_count: Array
    cleared_market_task_count: Array
    cleared_unit_plan_count: Array


class ControllerScanCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2


class ControllerCarryDiagnosticsV2(NamedTuple):
    episode_reset: Array
    cross_episode_contamination: Array
    reconcile: ControllerReconcileDiagnosticsV2


class CropProjectConfigV2(NamedTuple):
    crop_id: Array
    target_tiles: Array
    seed_batch_size: Array
    cash_floor: Array
    investment_stop_step: Array
    liquidation_start_step: Array
    harvest_age_days: Array


class M25CropExpansionConfigV2(NamedTuple):
    """Fixed-shape staged crop-business genome used by the M2.5 controller."""

    primary_crop_id: Array
    support_crop_id: Array
    phase_start_step: Array
    primary_target: Array
    support_target: Array
    hand_target: Array
    land_target: Array
    land_start_step: Array
    primary_seed_batch: Array
    support_seed_batch: Array
    hire_batch_max: Array
    parallel_plant_lanes: Array
    harvest_dispatch_lanes: Array
    deposit_lanes: Array
    cash_floor: Array
    sell_interval: Array
    sell_phase: Array
    investment_stop_step: Array
    liquidation_start_step: Array


class M26CropGenomeV2(NamedTuple):
    """Minimal non-redundant crop genome frozen for M2.6.

    M2.5 remains as a historical calibration baseline.  All new crop search
    code must use this five-crop tensor contract and must not introduce
    primary/support aliases.
    """

    candidate_id: Array
    phase_count: Array
    phase_start_step: Array
    crop_target: Array
    land_target: Array
    land_start_step: Array
    hand_target: Array
    crop_last_plant_step: Array
    seed_buy_batch: Array
    plant_wave_size: Array
    harvest_min_age_days: Array
    harvest_trigger_units: Array
    fertilizer_policy: Array
    crop_layout_policy: Array
    crop_cash_cap: Array
    weed_recovery_policy: Array
    crop_abandon_policy: Array
    maintenance_utilization_cap: Array
    deposit_min_value: Array
    sell_interval: Array
    sell_phase: Array
    sell_price_floor_ratio: Array
    sell_fraction: Array
    shed_pressure_trigger: Array
    cash_floor: Array
    liquidation_start_step: Array


class M26CoverageMetricsV2(NamedTuple):
    phase_step_count: Array
    crop_target_step_count: Array
    crop_plant_actions: Array
    crop_harvest_actions: Array
    layout_plant_actions: Array
    fertilizer_actions: Array
    weed_recovery_actions: Array
    abandon_actions: Array
    sell_orders_by_product: Array
    seed_orders_by_crop: Array
    land_orders: Array
    hire_orders: Array
    expansion_events: Array
    shrink_events: Array
    stop_events: Array
    restart_events: Array


class M26RolloutCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2
    metrics: M25RolloutMetricsV2
    coverage: M26CoverageMetricsV2


class M26RolloutSummaryV2(NamedTuple):
    final_bank: Array
    done: Array
    terminal_sellable_shed_value: Array
    terminal_unit_inventory_value: Array
    terminal_harvestable_map_value: Array
    avoidable_liquidation_loss: Array
    plant_without_same_day_water: Array
    unexplained_failure_count: Array
    planted_tiles_by_crop: Array
    max_hires_observed: Array
    final_unlocked_count: Array
    plant_success_count: Array
    harvest_success_count: Array
    sold_product_units: Array
    coverage: M26CoverageMetricsV2


class M25RolloutMetricsV2(NamedTuple):
    invalid_raw_action_count: Array
    unexpected_pass_count: Array
    effect_mismatch_count: Array
    owner_inactive_count: Array
    deadline_missed_count: Array
    resource_unavailable_count: Array
    unit_compiler_overlap_count: Array
    market_compiler_overlap_count: Array
    plant_actions: Array
    water_actions: Array
    harvest_actions: Array
    deposit_actions: Array
    seed_buy_orders: Array
    hire_orders: Array
    land_orders: Array
    sell_orders: Array
    plant_success_count: Array
    water_success_count: Array
    harvest_success_count: Array
    deposit_success_count: Array
    seed_purchase_success_count: Array
    hire_success_count: Array
    land_purchase_success_count: Array
    sold_product_units: Array
    max_hires_observed: Array
    plant_without_same_day_water: Array
    project_cap_hits: Array
    obligation_cap_hits: Array


class M25RolloutCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2
    metrics: M25RolloutMetricsV2


class M25RolloutSummaryV2(NamedTuple):
    final_bank: Array
    done: Array
    terminal_sellable_shed_value: Array
    terminal_unit_inventory_value: Array
    terminal_harvestable_map_value: Array
    avoidable_liquidation_loss: Array
    plant_without_same_day_water: Array
    unexplained_failure_count: Array
    primary_planted_tiles: Array
    support_planted_tiles: Array
    max_hires_observed: Array
    final_unlocked_count: Array
    plant_success_count: Array
    harvest_success_count: Array
    sold_product_units: Array


class CropRolloutMetricsV2(NamedTuple):
    invalid_raw_action_count: Array
    unexpected_pass_count: Array
    effect_mismatch_count: Array
    owner_inactive_count: Array
    deadline_missed_count: Array
    resource_unavailable_count: Array
    plant_actions: Array
    water_actions: Array
    harvest_actions: Array
    deposit_actions: Array
    seed_buy_orders: Array
    sell_orders: Array
    plant_success_count: Array
    water_success_count: Array
    harvest_success_count: Array
    deposit_success_count: Array
    seed_purchase_success_count: Array
    sold_product_units: Array
    plant_without_same_day_water: Array
    project_cap_hits: Array
    obligation_cap_hits: Array


class CropRolloutCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2
    metrics: CropRolloutMetricsV2


class CropRolloutSummaryV2(NamedTuple):
    final_bank: Array
    done: Array
    terminal_sellable_shed_units: Array
    terminal_sellable_shed_value: Array
    terminal_unit_inventory_units: Array
    terminal_unit_inventory_value: Array
    terminal_harvestable_map_units: Array
    terminal_harvestable_map_value: Array
    avoidable_liquidation_loss: Array
    plant_without_same_day_water: Array
    unexplained_failure_count: Array
    plant_success_count: Array
    water_success_count: Array
    harvest_success_count: Array
    deposit_success_count: Array
    sold_product_units: Array


SCHEMA_FIELDS_V2 = {
    "RouteGenomeV2": RouteGenomeV2._fields,
    "ProjectStateV2": ProjectStateV2._fields,
    "ObligationV2": ObligationV2._fields,
    "UnitPlanStateV2": UnitPlanStateV2._fields,
    "RouteCardStateV3": RouteCardStateV3._fields,
    "ProjectControllerStateV2": ProjectControllerStateV2._fields,
    "CommitmentEnvelopeV2": CommitmentEnvelopeV2._fields,
    "StepDiagnosticsV2": StepDiagnosticsV2._fields,
    "ControllerReconcileDiagnosticsV2": ControllerReconcileDiagnosticsV2._fields,
    "ControllerScanCarryV2": ControllerScanCarryV2._fields,
    "ControllerCarryDiagnosticsV2": ControllerCarryDiagnosticsV2._fields,
    "CropProjectConfigV2": CropProjectConfigV2._fields,
    "M25CropExpansionConfigV2": M25CropExpansionConfigV2._fields,
    "M25RolloutMetricsV2": M25RolloutMetricsV2._fields,
    "M25RolloutCarryV2": M25RolloutCarryV2._fields,
    "M25RolloutSummaryV2": M25RolloutSummaryV2._fields,
    "M26CropGenomeV2": M26CropGenomeV2._fields,
    "M26CoverageMetricsV2": M26CoverageMetricsV2._fields,
    "M26RolloutCarryV2": M26RolloutCarryV2._fields,
    "M26RolloutSummaryV2": M26RolloutSummaryV2._fields,
    "CropRolloutMetricsV2": CropRolloutMetricsV2._fields,
    "CropRolloutCarryV2": CropRolloutCarryV2._fields,
    "CropRolloutSummaryV2": CropRolloutSummaryV2._fields,
}


__all__ = [
    "CommitmentEnvelopeV2",
    "ControllerCarryDiagnosticsV2",
    "ControllerReconcileDiagnosticsV2",
    "ControllerScanCarryV2",
    "CropProjectConfigV2",
    "CropRolloutCarryV2",
    "CropRolloutMetricsV2",
    "CropRolloutSummaryV2",
    "M25CropExpansionConfigV2",
    "M25RolloutCarryV2",
    "M25RolloutMetricsV2",
    "M25RolloutSummaryV2",
    "M26CropGenomeV2",
    "M26CoverageMetricsV2",
    "M26RolloutCarryV2",
    "M26RolloutSummaryV2",
    "ObligationV2",
    "ProjectControllerStateV2",
    "ProjectStateV2",
    "RouteGenomeV2",
    "RouteCardStateV3",
    "SCHEMA_FIELDS_V2",
    "StepDiagnosticsV2",
    "UnitPlanStateV2",
]
