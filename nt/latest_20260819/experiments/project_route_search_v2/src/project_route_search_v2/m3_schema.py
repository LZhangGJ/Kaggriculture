"""Fixed-shape JAX records introduced by M3.

M3 records are intentionally separate from the byte-frozen M0 schema module.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from kaggriculture_jax.types import State

from .schema import ProjectControllerStateV2


Array = jax.Array


class M3AnimalGenomeV2(NamedTuple):
    candidate_id: Array
    phase_count: Array
    phase_start_step: Array
    animal_target: Array
    land_target: Array
    hand_target: Array
    animal_investment_stop_step: Array
    animal_layout_policy: Array
    animal_place_wave_size: Array
    animal_project_cash_cap: Array
    feed_source_policy: Array
    feed_stock_horizon_days: Array
    care_policy: Array
    first_cycle_care_bonus_target: Array
    steady_cycle_care_bonus_target: Array
    animal_harvest_trigger_units: Array
    animal_fertilizer_policy: Array
    maintenance_utilization_cap: Array
    enforce_productive_cap: Array
    deposit_min_value: Array
    sell_interval: Array
    sell_phase: Array
    sell_price_floor_ratio: Array
    sell_fraction: Array
    shed_pressure_trigger: Array
    cash_floor: Array
    liquidation_start_step: Array


class AnimalCommitmentLedgerV2(NamedTuple):
    active: Array
    in_shed: Array
    carried: Array
    pending_purchase: Array
    committed: Array
    empty_structures: Array
    survival_feed_due_today: Array
    bonus_feed_due_today: Array
    care_feed_due_today: Array
    feed_required_today: Array
    planned_feed_horizon: Array


class M3CoverageMetricsV2(NamedTuple):
    phase_step_count: Array
    animal_target_step_count: Array
    build_actions: Array
    purchase_orders: Array
    place_actions: Array
    feed_actions: Array
    care_actions: Array
    harvest_actions: Array
    fertilizer_actions: Array
    sell_orders_by_product: Array
    land_orders: Array
    hire_orders: Array


class M3RolloutMetricsV2(NamedTuple):
    invalid_raw_action_count: Array
    unexpected_pass_count: Array
    effect_mismatch_count: Array
    owner_inactive_count: Array
    deadline_missed_count: Array
    resource_unavailable_count: Array
    unit_compiler_overlap_count: Array
    market_compiler_overlap_count: Array
    build_success_count: Array
    animal_purchase_success_count: Array
    place_success_count: Array
    feed_success_count: Array
    care_success_count: Array
    harvest_success_count: Array
    collect_fertilizer_success_count: Array
    deposit_success_count: Array
    wheat_purchase_success_count: Array
    sold_product_units: Array
    unplanned_animal_escape: Array
    animal_capacity_loss: Array
    feed_hard_deadline_miss: Array
    care_bonus_forfeited_unexplained: Array
    care_bonus_capacity_clipped_unexplained: Array
    animal_bought_without_place_plan: Array
    cow_sheep_pasture_conflict: Array
    duplicate_structure_reservation: Array
    max_hires_observed: Array


class M3RolloutCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2
    metrics: M3RolloutMetricsV2
    coverage: M3CoverageMetricsV2


class M3RolloutSummaryV2(NamedTuple):
    final_bank: Array
    done: Array
    terminal_sellable_shed_value: Array
    terminal_unit_inventory_value: Array
    terminal_animal_product_value: Array
    avoidable_liquidation_loss: Array
    animals_stranded_in_shed_at_terminal: Array
    animals_stranded_in_unit_inventory_at_terminal: Array
    active_animals_by_species: Array
    unexplained_failure_count: Array
    unplanned_animal_escape: Array
    animal_capacity_loss: Array
    feed_hard_deadline_miss: Array
    care_bonus_forfeited_unexplained: Array
    care_bonus_capacity_clipped_unexplained: Array
    animal_bought_without_place_plan: Array
    cow_sheep_pasture_conflict: Array
    coverage: M3CoverageMetricsV2


__all__ = [
    "AnimalCommitmentLedgerV2",
    "M3AnimalGenomeV2",
    "M3CoverageMetricsV2",
    "M3RolloutCarryV2",
    "M3RolloutMetricsV2",
    "M3RolloutSummaryV2",
]
