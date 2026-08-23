"""Fixed-shape JAX schemas shared by all V5 strategic modules."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class UnitTaskStateV1(NamedTuple):
    task_type: Array
    owner_unit: Array
    target_id: Array
    target_x: Array
    target_y: Array
    item_id: Array
    quantity: Array
    phase: Array
    start_step: Array
    last_progress_step: Array
    expected_finish_step: Array
    deadline_step: Array
    status: Array
    failure_code: Array


class MarketTaskStateV1(NamedTuple):
    task_type: Array
    item_id: Array
    quantity: Array
    start_step: Array
    deadline_step: Array
    status: Array
    failure_code: Array


class ControllerStateV1(NamedTuple):
    schema_version: Array
    unit_tasks: UnitTaskStateV1
    market_tasks: MarketTaskStateV1


class CandidateV1(NamedTuple):
    task_type: Array
    owner_unit: Array
    target_id: Array
    target_x: Array
    target_y: Array
    item_id: Array
    quantity: Array
    source: Array
    mandatory: Array
    present: Array
    hard_mask: Array
    source_slot: Array
    replay_priority: Array


class FeasibilityV1(NamedTuple):
    legal_now: Array
    unit_required: Array
    plot_required: Array
    path_steps: Array
    operation_steps: Array
    expected_finish_step: Array
    deadline_step: Array
    bankable_before_terminal: Array
    cash_required: Array
    seed_item: Array
    seed_required: Array
    shed_item: Array
    shed_item_required: Array
    unit_item: Array
    unit_item_required: Array
    shed_reserved_in: Array
    market_slots_required: Array
    land_purchases_required: Array


class EconFeaturesV1(NamedTuple):
    """V5 section 7.1 fields, all shaped ``[batch, candidate]``."""

    expected_bank_delta_current_price: Array
    cash_required: Array
    cash_flow_before_revenue: Array
    minimum_cash_during_task: Array
    steps_to_first_revenue: Array
    total_required_actions: Array
    maintenance_actions: Array
    plot_occupancy_duration: Array
    storage_required: Array
    cycles_before_terminal: Array
    bankable_before_terminal: Array
    simple_cash_buffer_after_commit: Array
    action_opportunity_cost: Array
    transport_actions: Array
    input_opportunity_cost: Array
    expected_overflow_loss: Array
    expected_decay_or_capacity_loss: Array
    known_market_impact: Array
    known_town_demand_before_sale: Array
    stochastic_event_risk: Array
    terminal_salvage_value: Array
    exact_field_mask: Array
    expected_field_mask: Array
    scenario_field_mask: Array


class LedgerV1(NamedTuple):
    unit_free: Array
    plot_reserved: Array
    cash_available: Array
    cash_reserved: Array
    seeds_available: Array
    seeds_reserved: Array
    shed_available: Array
    shed_reserved_out: Array
    shed_reserved_in: Array
    unit_inventory_available: Array
    unit_inventory_reserved: Array
    market_slots_used: Array
    market_product_buy_reserved: Array
    market_product_sell_reserved: Array
    land_purchases_reserved: Array
    maintenance_reserved: Array


class CompileDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: Array
    unexpected_pass_count: Array
    land_orders: Array
    fertilizer_buy_orders: Array
    fertilizer_pickup_actions: Array
    fertilizer_apply_actions: Array


class EffectDiagnosticsV1(NamedTuple):
    effect_mismatch_count: Array
    owner_inactive_count: Array
    deadline_missed_count: Array
    resource_unavailable_count: Array
    land_purchase_success_count: Array
    fertilizer_buy_success_count: Array
    fertilizer_pickup_success_count: Array
    fertilizer_apply_success_count: Array
