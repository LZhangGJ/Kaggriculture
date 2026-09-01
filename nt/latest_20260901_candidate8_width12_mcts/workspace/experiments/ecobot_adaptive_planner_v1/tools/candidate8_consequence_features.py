#!/usr/bin/env python3
"""Compile deployable Candidate8 consequence features from public inputs.

This module deliberately accepts only the 230-dimensional Candidate8 public
feature matrix and its names.  It never receives future seeds, rollout labels,
terminal rewards, opponent identities, or hidden inventories.  Every output is
therefore reproducible online from the same public state and candidate delta.

The compiler is an *obligation forecast*, not a second simulator.  It exposes
the interactions that a shallow tree otherwise has to rediscover from sparse
data: capital commitment, cash reserve pressure, labour/land load, first-output
timing, short-horizon production potential, terminal feasibility and visible
market crowding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_PROJECTS = 8
N_PRODUCTS = 9
TOTAL_DAYS = 30
STEPS_PER_DAY = 24

# Frozen official-rule constants mirrored by native_adaptive.cpp.  They are
# mechanics, not learned route dates or Replay-specific values.
CROP_FIRST = np.asarray([2, 2, 8, 10, 10], dtype=np.float32)
CROP_MAX_DAY = np.asarray([4, 3, 8, 10, 12], dtype=np.float32)
CROP_INTERVAL = np.asarray([0, 0, 1, 2, 0], dtype=np.float32)
CROP_MAX_YIELD = np.asarray([6, 4, 4, 4, 6], dtype=np.float32)
CROP_ONGOING = np.asarray([0, 0, 1, 1, 0], dtype=np.float32)
ANIMAL_FIRST = np.asarray([4, 8, 6], dtype=np.float32)
ANIMAL_INTERVAL = np.asarray([1, 2, 3], dtype=np.float32)
ANIMAL_MAX_HELD = np.asarray([4, 6, 6], dtype=np.float32)
PROJECT_FIRST = np.concatenate([CROP_FIRST, ANIMAL_FIRST])
PROJECT_PRODUCT = np.arange(N_PROJECTS, dtype=np.int64)


@dataclass(frozen=True)
class ConsequenceContract:
    version: str = "candidate8-consequence-public-v1"
    horizons_steps: tuple[int, int] = (24, 48)
    source_feature_count: int = 230
    future_information_used: bool = False


CONTRACT = ConsequenceContract()


def _indices(names: list[str], pattern: str, count: int) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    missing = [pattern.format(i) for i in range(count)
               if pattern.format(i) not in lookup]
    if missing:
        raise ValueError(f"missing consequence inputs: {missing}")
    return np.asarray([lookup[pattern.format(i)] for i in range(count)])


def _column(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise ValueError(f"missing consequence input: {name}") from error


def _scaled(raw: np.ndarray, names: list[str], indices: np.ndarray) -> np.ndarray:
    values = np.asarray(raw[:, indices], dtype=np.float32).copy()
    for local, source in enumerate(indices):
        if "x100" in names[int(source)]:
            values[:, local] /= 100.0
    return values


def _production_potential(
    positive_units: np.ndarray,
    effective_lag_days: np.ndarray,
    horizon_days: float,
) -> np.ndarray:
    """Rules-derived potential from newly committed units by one horizon.

    This is intentionally conservative and does not claim that the scheduler
    will realise every unit.  The downstream model sees feasibility/load
    features alongside it.  For one-shot crops the first mature output is used;
    ongoing crops and animals use official production intervals and holding
    caps.
    """

    rows = positive_units.shape[0]
    output = np.zeros((rows, N_PROJECTS), dtype=np.float32)
    ready = horizon_days >= effective_lag_days
    for project in range(N_PROJECTS):
        if project < 5:
            if CROP_ONGOING[project] > 0:
                cycles = 1.0 + np.floor(
                    np.maximum(0.0, horizon_days - effective_lag_days[:, project])
                    / max(1.0, float(CROP_INTERVAL[project]))
                )
                units_per_asset = np.minimum(CROP_MAX_YIELD[project], cycles)
            else:
                # A non-ongoing crop matures into one harvestable lot.  Yield
                # growth after first maturity is not assumed in this short
                # forecast, avoiding optimistic water/harvest assumptions.
                units_per_asset = np.ones(rows, dtype=np.float32)
        else:
            animal = project - 5
            cycles = 1.0 + np.floor(
                np.maximum(0.0, horizon_days - effective_lag_days[:, project])
                / max(1.0, float(ANIMAL_INTERVAL[animal]))
            )
            units_per_asset = np.minimum(ANIMAL_MAX_HELD[animal], cycles)
        output[:, project] = (
            positive_units[:, project]
            * np.where(ready[:, project], units_per_asset, 0.0)
        )
    return output


def compile_consequence_features(
    raw_features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return candidate-conditioned consequence features and stable names."""

    raw = np.asarray(raw_features)
    if raw.ndim != 2:
        raise ValueError("raw_features must be a 2D matrix")
    if raw.shape[1] != len(feature_names):
        raise ValueError("raw feature width/name contract mismatch")
    if len(feature_names) != CONTRACT.source_feature_count:
        raise ValueError(
            f"expected {CONTRACT.source_feature_count} public features, "
            f"got {len(feature_names)}"
        )

    delta = _scaled(raw, feature_names, _indices(feature_names, "target_delta_{}", 8))
    target = _scaled(raw, feature_names, _indices(feature_names, "context_target_{}", 8))
    floor = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_irreversible_floor_{}", 8),
    )
    cap = _scaled(raw, feature_names, _indices(feature_names, "context_cap_{}", 8))
    marginal = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_marginal_value_x100_{}", 8),
    )
    purchase = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_purchase_cost_x100_{}", 8),
    )
    unit_load = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_daily_action_load_x100_{}", 8),
    )
    first_lag = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_first_cash_lag_days_{}", 8),
    )
    opponent_count = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_opponent_project_count_{}", 8),
    )
    opponent_sale_lead = _scaled(
        raw, feature_names,
        _indices(
            feature_names, "context_opponent_estimated_sale_lead_steps_{}", 8
        ),
    )
    price = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_market_price_{}", N_PRODUCTS),
    )
    demand = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_demand_within_day_{}", N_PRODUCTS),
    )
    opponent_h2 = _scaled(
        raw, feature_names,
        _indices(
            feature_names, "context_opponent_committed_supply_x100_h2_item{}",
            N_PRODUCTS,
        ),
    )
    inventory = _scaled(
        raw, feature_names,
        _indices(feature_names, "context_sellable_inventory_{}", N_PRODUCTS),
    )

    def scalar(name: str) -> np.ndarray:
        return _scaled(
            raw, feature_names, np.asarray([_column(feature_names, name)])
        )[:, 0]

    day = scalar("context_day")
    liquid_cash = scalar("context_liquid_cash_x100")
    protected_cash = scalar("context_protected_cash_x100")
    financeable_inventory = scalar("context_financeable_inventory_value_x100")
    quadrants = scalar("context_unlocked_quadrants")
    max_quadrants = scalar("context_maximum_quadrants")
    hands = scalar("context_hands")
    max_hands = scalar("context_maximum_hands")
    next_hand_cost = scalar("context_next_hand_cost_x100")
    next_land_cost = scalar("context_next_quadrant_cost_x100")
    tiles_per_quadrant = scalar("context_tiles_per_quadrant")
    productive_tiles = scalar("context_productive_tiles")
    current_load = scalar("context_current_daily_action_load_x100")
    hard_load = scalar("context_hard_deadline_load_x100")
    travel_load = scalar("context_estimated_travel_load_x100")
    delayed_loss = scalar("context_delayed_loss_x100")
    market_slots = scalar("context_market_slots_available")
    recovery_issues = scalar("context_recovery_issues")
    hand_delta = scalar("hand_delta")
    quadrant_delta = scalar("quadrant_delta")
    delay_days = np.maximum(0.0, scalar("effective_delay_days"))
    estimated_value = scalar("estimated_value_x100")
    estimated_cost = scalar("estimated_cash_cost_x100")
    estimated_load = scalar("estimated_daily_action_load_x100")

    proposed = np.clip(target + delta, floor, cap)
    realised_delta = proposed - target
    positive = np.maximum(realised_delta, 0.0)
    released = np.maximum(-realised_delta, 0.0)
    reversible_before = np.maximum(target - floor, 0.0)
    reversible_after = np.maximum(proposed - floor, 0.0)
    capital_by_project = positive * purchase
    load_delta_by_project = realised_delta * unit_load
    project_value_delta = realised_delta * marginal
    effective_lag = first_lag + delay_days[:, None]
    terminal_slack = (TOTAL_DAYS - day)[:, None] - effective_lag
    output24 = _production_potential(positive, effective_lag, 1.0)
    output48 = _production_potential(positive, effective_lag, 2.0)
    project_prices = price[:, PROJECT_PRODUCT]
    revenue24 = output24 * project_prices
    revenue48 = output48 * project_prices
    own_sale_lead = effective_lag * STEPS_PER_DAY + 2.0
    sale_lead_advantage = opponent_sale_lead - own_sale_lead
    visible_crowding = proposed * opponent_count
    competition_h2 = output48 + opponent_h2[:, PROJECT_PRODUCT]
    demand_h2 = 2.0 * demand[:, PROJECT_PRODUCT]
    demand_surplus_h2 = demand_h2 - competition_h2

    worker_add = np.maximum(hand_delta, 0.0)
    land_add = np.maximum(quadrant_delta, 0.0)
    worker_cost = worker_add * next_hand_cost
    land_cost = land_add * next_land_cost
    project_cost = capital_by_project.sum(axis=1)
    total_commitment = project_cost + worker_cost + land_cost
    cash_after_commitment = liquid_cash - total_commitment
    protected_cash_gap = cash_after_commitment - protected_cash
    financeable_cash_gap = (
        liquid_cash + financeable_inventory - protected_cash - total_commitment
    )

    projected_load = current_load + load_delta_by_project.sum(axis=1)
    projected_workers = np.minimum(max_hands + 1.0, hands + 1.0 + worker_add)
    worker_capacity = np.maximum(1.0, projected_workers * STEPS_PER_DAY)
    load_utilisation = projected_load / worker_capacity
    hard_utilisation = (hard_load + np.maximum(0.0, load_delta_by_project.sum(axis=1))) / worker_capacity
    logistics_utilisation = (
        projected_load + hard_load + travel_load
    ) / worker_capacity
    load_overflow = np.maximum(0.0, projected_load + hard_load - worker_capacity)

    projected_quadrants = np.minimum(max_quadrants, quadrants + land_add)
    land_capacity = np.maximum(1.0, projected_quadrants * tiles_per_quadrant)
    projected_tiles = productive_tiles + positive.sum(axis=1)
    land_slack = land_capacity - projected_tiles
    land_utilisation = projected_tiles / land_capacity
    land_overflow = np.maximum(0.0, -land_slack)

    positive_units = positive.sum(axis=1)
    weighted_lag = (
        (positive * effective_lag).sum(axis=1)
        / np.maximum(1.0, positive_units)
    )
    earliest_lag = np.where(
        positive_units > 0,
        np.min(np.where(positive > 0, effective_lag, 999.0), axis=1),
        0.0,
    )
    terminal_infeasible_units = (
        positive * (terminal_slack <= 0.0)
    ).sum(axis=1)
    inventory_value = (inventory * price).sum(axis=1)
    market_slot_pressure = (
        np.count_nonzero(inventory > 0, axis=1) / np.maximum(1.0, market_slots)
    )
    recovery_pressure = (
        hard_load + travel_load + delayed_loss / 100.0
        + (recovery_issues > 0).astype(np.float32)
    ) / worker_capacity

    blocks: list[tuple[str, np.ndarray]] = []

    def add_vector(prefix: str, values: np.ndarray) -> None:
        for project in range(values.shape[1]):
            blocks.append((f"{prefix}_{project}", values[:, project]))

    def add_scalar(name: str, values: np.ndarray) -> None:
        blocks.append((name, values))

    add_vector("consequence_proposed_target", proposed)
    add_vector("consequence_reversible_before", reversible_before)
    add_vector("consequence_reversible_after", reversible_after)
    add_vector("consequence_positive_units", positive)
    add_vector("consequence_released_units", released)
    add_vector("consequence_capital_commitment", capital_by_project)
    add_vector("consequence_daily_load_delta", load_delta_by_project)
    add_vector("consequence_project_value_delta", project_value_delta)
    add_vector("consequence_effective_lag_days", effective_lag)
    add_vector("consequence_terminal_slack_days", terminal_slack)
    add_vector("consequence_output_potential_24", output24)
    add_vector("consequence_output_potential_48", output48)
    add_vector("consequence_revenue_potential_24", revenue24)
    add_vector("consequence_revenue_potential_48", revenue48)
    add_vector("consequence_sale_lead_advantage_steps", sale_lead_advantage)
    add_vector("consequence_visible_crowding", visible_crowding)
    add_vector("consequence_h2_demand_surplus", demand_surplus_h2)

    add_scalar("consequence_positive_units_total", positive_units)
    add_scalar("consequence_released_units_total", released.sum(axis=1))
    add_scalar("consequence_project_commitment_cost", project_cost)
    add_scalar("consequence_worker_commitment_cost", worker_cost)
    add_scalar("consequence_land_commitment_cost", land_cost)
    add_scalar("consequence_total_commitment_cost", total_commitment)
    add_scalar("consequence_cash_after_commitment", cash_after_commitment)
    add_scalar("consequence_protected_cash_gap", protected_cash_gap)
    add_scalar("consequence_financeable_cash_gap", financeable_cash_gap)
    add_scalar("consequence_projected_daily_load", projected_load)
    add_scalar("consequence_worker_capacity", worker_capacity)
    add_scalar("consequence_load_utilisation", load_utilisation)
    add_scalar("consequence_hard_utilisation", hard_utilisation)
    add_scalar("consequence_logistics_utilisation", logistics_utilisation)
    add_scalar("consequence_load_overflow", load_overflow)
    add_scalar("consequence_projected_tiles", projected_tiles)
    add_scalar("consequence_land_capacity", land_capacity)
    add_scalar("consequence_land_slack", land_slack)
    add_scalar("consequence_land_utilisation", land_utilisation)
    add_scalar("consequence_land_overflow", land_overflow)
    add_scalar("consequence_weighted_first_output_lag_days", weighted_lag)
    add_scalar("consequence_earliest_output_lag_days", earliest_lag)
    add_scalar("consequence_terminal_infeasible_units", terminal_infeasible_units)
    add_scalar("consequence_output_potential_24_total", output24.sum(axis=1))
    add_scalar("consequence_output_potential_48_total", output48.sum(axis=1))
    add_scalar("consequence_revenue_potential_24_total", revenue24.sum(axis=1))
    add_scalar("consequence_revenue_potential_48_total", revenue48.sum(axis=1))
    add_scalar("consequence_inventory_value", inventory_value)
    add_scalar("consequence_market_slot_pressure", market_slot_pressure)
    add_scalar("consequence_recovery_pressure", recovery_pressure)
    add_scalar("consequence_estimated_value", estimated_value)
    add_scalar("consequence_estimated_cost", estimated_cost)
    add_scalar("consequence_estimated_daily_load", estimated_load)

    names = [name for name, _ in blocks]
    matrix = np.column_stack([values for _, values in blocks]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("consequence compiler produced non-finite values")
    return matrix, names

