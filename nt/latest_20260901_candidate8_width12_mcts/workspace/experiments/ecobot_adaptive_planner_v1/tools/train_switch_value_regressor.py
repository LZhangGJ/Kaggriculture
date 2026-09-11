#!/usr/bin/env python3
"""Train a noise-aware SWITCH value model and audit on independent futures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np


PROJECT_COST = np.asarray([10, 20, 50, 100, 80, 300, 400, 500])
PROJECT_LEAD = np.asarray([2, 2, 8, 10, 10, 4, 8, 6])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    return {key: np.asarray(raw[key]) for key in raw.files}


def feature_index(names: np.ndarray) -> dict[str, int]:
    return {str(name): index for index, name in enumerate(names)}


def engineered_features(
    features: np.ndarray, names: np.ndarray
) -> tuple[np.ndarray, list[str], list[int]]:
    x = np.asarray(features, dtype=np.float64)
    at = feature_index(names)

    def col(name: str) -> np.ndarray:
        return x[:, at[name]]

    source = col("source_project").astype(np.int64)
    destination = col("destination_project").astype(np.int64)
    valid_source = (source >= 0) & (source < 8)
    valid_destination = (destination >= 0) & (destination < 8)

    def project_lookup(prefix: str, project: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=np.float64)
        project_names = [
            "wheat", "carrot", "tomato", "strawberry", "melon",
            "geese", "cows", "sheep",
        ]
        for project_id, project_name in enumerate(project_names):
            selected = project == project_id
            out[selected] = col(f"{prefix}_{project_name}")[selected]
        return out

    def product_lookup(prefix: str, project: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=np.float64)
        for product in range(8):
            selected = project == product
            out[selected] = col(f"{prefix}_{product}")[selected]
        return out

    source_cost = np.where(valid_source, PROJECT_COST[np.clip(source, 0, 7)], 0)
    destination_cost = np.where(
        valid_destination, PROJECT_COST[np.clip(destination, 0, 7)], 0
    )
    source_lead = np.where(valid_source, PROJECT_LEAD[np.clip(source, 0, 7)], 0)
    destination_lead = np.where(
        valid_destination, PROJECT_LEAD[np.clip(destination, 0, 7)], 0
    )
    source_price = product_lookup("market_price", source)
    destination_price = product_lookup("market_price", destination)
    source_inventory = product_lookup("market_inventory", source)
    destination_inventory = product_lookup("market_inventory", destination)
    source_opponent_supply = product_lookup(
        "opponent_visible_supply_x100", source
    )
    destination_opponent_supply = product_lookup(
        "opponent_visible_supply_x100", destination
    )
    source_baseline = project_lookup("baseline", source)
    destination_baseline = project_lookup("baseline", destination)
    source_candidate = project_lookup("candidate", source)
    destination_candidate = project_lookup("candidate", destination)
    source_field = project_lookup("field", source)
    destination_field = project_lookup("field", destination)
    source_yield = product_lookup("self_project_yield", source)
    destination_yield = product_lookup("self_project_yield", destination)
    source_age = product_lookup("self_project_age_sum", source)
    destination_age = product_lookup("self_project_age_sum", destination)
    source_stress = product_lookup("self_project_stress", source)
    destination_stress = product_lookup("self_project_stress", destination)
    source_distance = product_lookup("self_project_shed_distance", source)
    destination_distance = product_lookup("self_project_shed_distance", destination)
    source_self_lag2 = product_lookup("self_committed_supply_lag2_x100", source)
    destination_self_lag2 = product_lookup(
        "self_committed_supply_lag2_x100", destination
    )
    source_self_lag8 = product_lookup("self_committed_supply_lag8_x100", source)
    destination_self_lag8 = product_lookup(
        "self_committed_supply_lag8_x100", destination
    )
    source_opponent_lag2 = product_lookup(
        "opponent_committed_supply_lag2_x100", source
    )
    destination_opponent_lag2 = product_lookup(
        "opponent_committed_supply_lag2_x100", destination
    )
    source_opponent_lag8 = product_lookup(
        "opponent_committed_supply_lag8_x100", source
    )
    destination_opponent_lag8 = product_lookup(
        "opponent_committed_supply_lag8_x100", destination
    )
    source_count_trend = product_lookup(
        "opponent_project_count_trend_x100", source
    )
    destination_count_trend = product_lookup(
        "opponent_project_count_trend_x100", destination
    )
    source_yield_trend = product_lookup(
        "opponent_project_yield_trend_x100", source
    )
    destination_yield_trend = product_lookup(
        "opponent_project_yield_trend_x100", destination
    )
    source_market_drift = product_lookup("market_daily_drift_x100", source)
    destination_market_drift = product_lookup(
        "market_daily_drift_x100", destination
    )
    remaining_days = np.maximum(0.0, 30.0 - col("day"))
    source_window = np.maximum(0.0, remaining_days - source_lead)
    destination_window = np.maximum(0.0, remaining_days - destination_lead)
    source_gap = source_baseline - source_field
    destination_gap = destination_candidate - destination_field
    source_pressure = (
        source_inventory * 100.0 + source_self_lag8 + source_opponent_lag8
    )
    destination_pressure = (
        destination_inventory * 100.0
        + destination_self_lag8 + destination_opponent_lag8
    )
    incremental_workload = (
        col("candidate_workload_x100") - col("baseline_workload_x100")
    )
    incremental_hands = (
        col("candidate_estimated_hands") - col("baseline_estimated_hands")
    )
    effective_hands = np.maximum(1.0, col("hands"))

    edit_family = np.where(
        valid_source & valid_destination, source * 8 + destination, 64
    ).astype(np.float64)
    derived_names = [
        "analytic_gain", "estimated_hand_delta", "quadrant_delta",
        "unmet_crop_delta", "unmet_animal_delta", "target_count_delta",
        "workload_delta_x100", "source_price", "destination_price",
        "source_inventory", "destination_inventory", "source_cost",
        "destination_cost", "source_lead", "destination_lead",
        "source_opponent_supply_x100", "destination_opponent_supply_x100",
        "source_baseline_target", "destination_baseline_target",
        "source_candidate_target", "destination_candidate_target",
        "gross_price_spread", "capital_delta", "lead_delta", "edit_family",
        "remaining_days", "source_production_window",
        "destination_production_window", "source_field_count",
        "destination_field_count", "source_target_gap",
        "destination_target_gap", "source_yield_held",
        "destination_yield_held", "source_age_sum", "destination_age_sum",
        "source_stress", "destination_stress", "source_shed_distance",
        "destination_shed_distance", "source_self_supply_lag2_x100",
        "destination_self_supply_lag2_x100", "source_self_supply_lag8_x100",
        "destination_self_supply_lag8_x100",
        "source_opponent_supply_lag2_x100",
        "destination_opponent_supply_lag2_x100",
        "source_opponent_supply_lag8_x100",
        "destination_opponent_supply_lag8_x100",
        "source_opponent_count_trend_x100",
        "destination_opponent_count_trend_x100",
        "source_opponent_yield_trend_x100",
        "destination_opponent_yield_trend_x100",
        "source_market_drift_x100", "destination_market_drift_x100",
        "source_total_supply_pressure_x100",
        "destination_total_supply_pressure_x100",
        "supply_pressure_delta_x100", "incremental_workload_per_hand_x100",
        "incremental_hands", "destination_price_window_value",
        "source_price_window_value", "destination_capital_pressure",
        "cash_to_incremental_capital_ratio",
    ]
    derived = np.column_stack([
        col("candidate_analytic_score") - col("baseline_analytic_score"),
        col("candidate_estimated_hands") - col("baseline_estimated_hands"),
        col("candidate_quadrants") - col("baseline_quadrants"),
        col("candidate_unmet_crops") - col("baseline_unmet_crops"),
        col("candidate_unmet_animals") - col("baseline_unmet_animals"),
        col("candidate_total_targets") - col("baseline_total_targets"),
        col("candidate_workload_x100") - col("baseline_workload_x100"),
        source_price,
        destination_price,
        source_inventory,
        destination_inventory,
        source_cost,
        destination_cost,
        source_lead,
        destination_lead,
        source_opponent_supply,
        destination_opponent_supply,
        source_baseline,
        destination_baseline,
        source_candidate,
        destination_candidate,
        destination_price - source_price,
        col("add_count") * destination_cost - col("remove_count") * source_cost,
        destination_lead - source_lead,
        edit_family,
        remaining_days,
        source_window,
        destination_window,
        source_field,
        destination_field,
        source_gap,
        destination_gap,
        source_yield,
        destination_yield,
        source_age,
        destination_age,
        source_stress,
        destination_stress,
        source_distance,
        destination_distance,
        source_self_lag2,
        destination_self_lag2,
        source_self_lag8,
        destination_self_lag8,
        source_opponent_lag2,
        destination_opponent_lag2,
        source_opponent_lag8,
        destination_opponent_lag8,
        source_count_trend,
        destination_count_trend,
        source_yield_trend,
        destination_yield_trend,
        source_market_drift,
        destination_market_drift,
        source_pressure,
        destination_pressure,
        destination_pressure - source_pressure,
        incremental_workload / effective_hands,
        incremental_hands,
        col("add_count") * destination_price * destination_window,
        col("remove_count") * source_price * source_window,
        col("add_count") * destination_cost
        + np.maximum(0.0, incremental_hands) * 100.0,
        col("liquid_capital") / np.maximum(
            1.0,
            col("add_count") * destination_cost
            + np.maximum(0.0, incremental_hands) * 100.0,
        ),
    ])
    expanded_names = [str(name) for name in names] + derived_names
    categorical = [
        at["source_project"],
        at["destination_project"],
        at["shop_mask"],
        at["candidate_seat"],
        len(names) + derived_names.index("edit_family"),
    ]
    return np.concatenate([x, derived], axis=1), expanded_names, categorical


def generalized_engineered_features(
    features: np.ndarray,
    names: np.ndarray,
    mode: str = "legacy",
) -> tuple[np.ndarray, list[str], list[int]]:
    """Optionally replace the opaque shop-mask category with reusable bits."""
    expanded, expanded_names, categorical = engineered_features(features, names)
    if mode == "legacy":
        return expanded, expanded_names, categorical
    if mode not in {
        "shop_bits",
        "shop_bits_scale",
        "shop_bits_scale_catfix",
        "shop_bits_scale_marginal",
        "shop_bits_scale_forecast",
        "shop_bits_scale_forecast_catfix",
        "shop_bits_scale_forecast_marginal_catfix",
    }:
        raise ValueError(f"unsupported feature mode: {mode}")
    at = feature_index(names)
    shop_index = at["shop_mask"]
    shop_mask = np.asarray(features[:, shop_index], dtype=np.int64)
    robust = np.asarray(expanded, dtype=np.float64).copy()
    if mode in {
        "shop_bits_scale_catfix",
        "shop_bits_scale_forecast_catfix",
        "shop_bits_scale_forecast_marginal_catfix",
    }:
        # LightGBM reserves negative categorical values for missing data.
        # KEEP legitimately represents source/destination as -1, so this
        # opt-in encoding maps {-1, 0..7} to {0..8}.  Keep it a separate mode:
        # previously frozen models intentionally learned the legacy missing-
        # value representation and must remain reproducible.
        robust[:, at["source_project"]] += 1.0
        robust[:, at["destination_project"]] += 1.0
    robust[:, shop_index] = 0.0
    shop_bits = np.column_stack([
        ((shop_mask >> item) & 1).astype(np.float64) for item in range(9)
    ])
    robust = np.concatenate([robust, shop_bits], axis=1)
    robust_names = expanded_names + [f"shop_available_{item}" for item in range(9)]
    robust_categorical = [index for index in categorical if index != shop_index]
    if mode == "shop_bits":
        return robust, robust_names, robust_categorical

    raw = np.asarray(features, dtype=np.float64)

    def raw_col(name: str) -> np.ndarray:
        return raw[:, at[name]]

    def optional_raw_col(name: str) -> np.ndarray:
        # Corpora generated before the W3 value decomposition remain readable
        # for rollback audits.  New 310-field corpora provide the real values.
        if name not in at:
            return np.zeros(len(raw), dtype=np.float64)
        return raw[:, at[name]]

    source = raw_col("source_project").astype(np.int64)
    destination = raw_col("destination_project").astype(np.int64)
    add_count = raw_col("add_count")
    remove_count = raw_col("remove_count")
    remaining_days = np.maximum(0.0, 30.0 - raw_col("day"))
    first = np.asarray([2, 2, 8, 10, 10, 4, 8, 6], dtype=np.float64)
    interval = np.asarray([2, 2, 1, 2, 10, 1, 2, 3], dtype=np.float64)
    cost = PROJECT_COST.astype(np.float64)

    def project_value(values: np.ndarray, project: np.ndarray) -> np.ndarray:
        out = np.zeros(len(raw), dtype=np.float64)
        valid = (project >= 0) & (project < len(values))
        out[valid] = values[project[valid]]
        return out

    destination_first = project_value(first, destination)
    destination_interval = np.maximum(1.0, project_value(interval, destination))
    source_first = project_value(first, source)
    source_interval = np.maximum(1.0, project_value(interval, source))
    destination_cycles = np.where(
        remaining_days >= destination_first,
        1.0 + np.floor(
            np.maximum(0.0, remaining_days - destination_first)
            / destination_interval
        ),
        0.0,
    )
    source_cycles = np.where(
        remaining_days >= source_first,
        1.0 + np.floor(
            np.maximum(0.0, remaining_days - source_first) / source_interval
        ),
        0.0,
    )

    destination_price = np.zeros(len(raw), dtype=np.float64)
    source_price = np.zeros(len(raw), dtype=np.float64)
    destination_inventory = np.zeros(len(raw), dtype=np.float64)
    destination_pressure = np.zeros(len(raw), dtype=np.float64)
    for project_id in range(8):
        destination_selected = destination == project_id
        source_selected = source == project_id
        destination_price[destination_selected] = raw_col(
            f"market_price_{project_id}"
        )[destination_selected]
        source_price[source_selected] = raw_col(
            f"market_price_{project_id}"
        )[source_selected]
        destination_inventory[destination_selected] = raw_col(
            f"market_inventory_{project_id}"
        )[destination_selected]
        destination_pressure[destination_selected] = (
            raw_col(f"self_committed_supply_lag8_x100_{project_id}")
            + raw_col(f"opponent_committed_supply_lag8_x100_{project_id}")
        )[destination_selected] / 100.0

    destination_supply = add_count * destination_cycles
    source_supply = remove_count * source_cycles
    destination_gross_value = destination_supply * destination_price
    source_gross_value = source_supply * source_price
    spatial_capacity = np.where(
        destination < 5,
        raw_col("empty_tiles"),
        np.where(
            destination == 5,
            raw_col("empty_coops"),
            raw_col("empty_pastures"),
        ),
    )
    current_units = np.maximum(1.0, raw_col("hands") + 1.0)
    candidate_units = np.maximum(1.0, raw_col("candidate_estimated_hands") + 1.0)
    destination_target = np.zeros(len(raw), dtype=np.float64)
    destination_field = np.zeros(len(raw), dtype=np.float64)
    project_names = [
        "wheat", "carrot", "tomato", "strawberry", "melon",
        "geese", "cows", "sheep",
    ]
    for project_id, project_name in enumerate(project_names):
        selected = destination == project_id
        destination_target[selected] = raw_col(
            f"candidate_{project_name}"
        )[selected]
        destination_field[selected] = raw_col(
            f"field_{project_name}"
        )[selected]
    target_gap = np.maximum(0.0, destination_target - destination_field)
    incremental_hands = np.maximum(
        0.0,
        raw_col("candidate_estimated_hands")
        - raw_col("baseline_estimated_hands"),
    )
    incremental_capital = add_count * project_value(cost, destination) + 100.0 * incremental_hands
    scale_names = [
        "add_count_squared", "remove_count_squared",
        "destination_project_cycles", "source_project_cycles",
        "destination_incremental_supply", "source_removed_supply",
        "destination_incremental_gross_value", "source_removed_gross_value",
        "destination_incremental_market_share",
        "destination_spatial_headroom", "destination_capacity_ratio",
        "destination_target_per_current_unit",
        "candidate_workload_per_current_unit_x100",
        "candidate_workload_per_candidate_unit_x100",
        "destination_target_gap_per_remaining_day",
        "incremental_capital_headroom",
        "gross_value_per_candidate_workload",
    ]
    scale = np.column_stack([
        np.square(add_count),
        np.square(remove_count),
        destination_cycles,
        source_cycles,
        destination_supply,
        source_supply,
        destination_gross_value,
        source_gross_value,
        destination_supply / np.maximum(
            1.0, destination_inventory + destination_pressure
        ),
        spatial_capacity - add_count,
        add_count / np.maximum(1.0, spatial_capacity),
        destination_target / current_units,
        raw_col("candidate_workload_x100") / current_units,
        raw_col("candidate_workload_x100") / candidate_units,
        target_gap / np.maximum(1.0, remaining_days),
        raw_col("liquid_capital") - incremental_capital,
        destination_gross_value
        / np.maximum(1.0, raw_col("candidate_workload_x100") / 100.0),
    ])
    breakdown_components = (
        "crop_gross", "animal_gross", "fertilizer_gross",
        "seed_cost", "feed_cost", "animal_purchase_cost",
        "action_cost", "move_cost", "hire_cost", "land_cost",
        "lockup_cost", "crop_units", "animal_product_units",
        "fertilizer_used", "fertilizer_sellable", "setup_turns",
    )
    breakdown_delta = np.column_stack([
        optional_raw_col(f"candidate_{component}")
        - optional_raw_col(f"baseline_{component}")
        for component in breakdown_components
    ])
    baseline_gross = sum(
        optional_raw_col(f"baseline_{component}")
        for component in ("crop_gross", "animal_gross", "fertilizer_gross")
    )
    candidate_gross = sum(
        optional_raw_col(f"candidate_{component}")
        for component in ("crop_gross", "animal_gross", "fertilizer_gross")
    )
    cost_components = (
        "seed_cost", "feed_cost", "animal_purchase_cost", "action_cost",
        "move_cost", "hire_cost", "land_cost", "lockup_cost",
    )
    baseline_cost = sum(
        optional_raw_col(f"baseline_{component}")
        for component in cost_components
    )
    candidate_cost = sum(
        optional_raw_col(f"candidate_{component}")
        for component in cost_components
    )
    baseline_output = (
        optional_raw_col("baseline_crop_units")
        + optional_raw_col("baseline_animal_product_units")
        + optional_raw_col("baseline_fertilizer_sellable")
    )
    candidate_output = (
        optional_raw_col("candidate_crop_units")
        + optional_raw_col("candidate_animal_product_units")
        + optional_raw_col("candidate_fertilizer_sellable")
    )
    breakdown_summary = np.column_stack([
        candidate_gross - baseline_gross,
        candidate_cost - baseline_cost,
        candidate_output - baseline_output,
        (candidate_gross - candidate_cost)
        / np.maximum(1.0, candidate_output),
    ])
    breakdown_names = [
        f"value_delta_{component}" for component in breakdown_components
    ] + [
        "value_delta_total_gross", "value_delta_total_cost",
        "value_delta_projected_sellable_units",
        "candidate_net_value_per_projected_unit",
    ]
    base_matrix = np.concatenate(
        [robust, scale, breakdown_delta, breakdown_summary], axis=1
    )
    base_names = robust_names + scale_names + breakdown_names
    if mode in {
        "shop_bits_scale_forecast",
        "shop_bits_scale_forecast_catfix",
        "shop_bits_scale_forecast_marginal_catfix",
    }:
        # Convert the already-public opponent history into an explicit
        # short-horizon capacity forecast.  The raw features contain current
        # project counts/yields and their observed daily trends, but asking a
        # shallow tree to discover every project-selection x trend x horizon
        # interaction generalises poorly to unseen route families.  These
        # features use no route identity and inspect no future action tape.
        opponent_count = np.column_stack([
            raw_col(f"opponent_field_{project_name}")
            for project_name in project_names
        ])
        opponent_yield = np.column_stack([
            raw_col(f"opponent_{product_name}_yield")
            for product_name in (
                "wheat", "carrot", "tomato", "strawberry", "melon",
                "egg", "milk", "wool",
            )
        ])
        opponent_count_trend = np.column_stack([
            raw_col(f"opponent_project_count_trend_x100_{project_id}")
            for project_id in range(8)
        ]) / 100.0
        opponent_yield_trend = np.column_stack([
            raw_col(f"opponent_project_yield_trend_x100_{project_id}")
            for project_id in range(8)
        ]) / 100.0
        market_price = np.column_stack([
            raw_col(f"market_price_{project_id}") for project_id in range(8)
        ])
        market_drift = np.column_stack([
            raw_col(f"market_daily_drift_x100_{project_id}")
            for project_id in range(8)
        ]) / 100.0

        def select_project(matrix: np.ndarray, project: np.ndarray) -> np.ndarray:
            out = np.zeros(len(raw), dtype=np.float64)
            valid = (project >= 0) & (project < matrix.shape[1])
            rows = np.flatnonzero(valid)
            out[rows] = matrix[rows, project[rows]]
            return out

        forecast_columns: list[np.ndarray] = []
        forecast_names: list[str] = []
        for role, project in (("source", source), ("destination", destination)):
            for lag in (2, 4, 8):
                effective_lag = np.minimum(float(lag), remaining_days)
                count_forecast = np.maximum(
                    0.0, opponent_count + opponent_count_trend * effective_lag[:, None]
                )
                yield_forecast = np.maximum(
                    0.0, opponent_yield + opponent_yield_trend * effective_lag[:, None]
                )
                price_forecast = np.maximum(
                    1.0, market_price + market_drift * effective_lag[:, None]
                )
                expansion = np.maximum(0.0, count_forecast - opponent_count)
                maturity = np.maximum(
                    0.0,
                    (effective_lag[:, None] - first[None, :])
                    / np.maximum(1.0, interval[None, :]),
                )
                expansion_supply = expansion * maturity
                self_committed = np.column_stack([
                    raw_col(f"self_committed_supply_lag{lag}_x100_{item}")
                    for item in range(8)
                ]) / 100.0
                opponent_committed = np.column_stack([
                    raw_col(f"opponent_committed_supply_lag{lag}_x100_{item}")
                    for item in range(8)
                ]) / 100.0
                total_pressure = self_committed + opponent_committed + expansion_supply
                for suffix, matrix in (
                    ("opponent_count", count_forecast),
                    ("opponent_yield", yield_forecast),
                    ("market_price", price_forecast),
                    ("opponent_expansion_supply", expansion_supply),
                    ("total_supply_pressure", total_pressure),
                ):
                    forecast_columns.append(select_project(matrix, project))
                    forecast_names.append(f"{role}_{suffix}_forecast_lag{lag}")
        destination_pressure_lag8 = forecast_columns[-1]
        forecast_columns.extend([
            destination_supply / np.maximum(1.0, destination_pressure_lag8),
            np.maximum(
                0.0,
                raw_col("opponent_cash")
                + raw_col("opponent_cash_trend")
                * np.minimum(8.0, remaining_days),
            ),
            np.maximum(
                0.0,
                raw_col("opponent_hands")
                + raw_col("opponent_hands_trend_x100")
                * np.minimum(8.0, remaining_days) / 100.0,
            ),
            np.clip(
                raw_col("opponent_quadrants")
                + raw_col("opponent_quadrants_trend_x100")
                * np.minimum(8.0, remaining_days) / 100.0,
                1.0,
                4.0,
            ),
        ])
        forecast_names.extend([
            "destination_incremental_supply_to_forecast_pressure_lag8",
            "opponent_cash_forecast_lag8",
            "opponent_hands_forecast_lag8",
            "opponent_quadrants_forecast_lag8",
        ])
        forecast = np.column_stack(forecast_columns)
        forecast_matrix = np.concatenate([base_matrix, forecast], axis=1)
        forecast_feature_names = base_names + forecast_names
        if mode != "shop_bits_scale_forecast_marginal_catfix":
            return forecast_matrix, forecast_feature_names, robust_categorical
        # The forecast and marginal views are complementary: the former says
        # how crowded the future market may be, while the latter says whether
        # the last added/removed unit pays for its own capital and workload.
        # Keep both in one candidate scorer so same-industry scale edits do not
        # discard the public-state market forecast.
        base_matrix = forecast_matrix
        base_names = forecast_feature_names

    if mode not in {
        "shop_bits_scale_marginal",
        "shop_bits_scale_forecast_marginal_catfix",
    }:
        return base_matrix, base_names, robust_categorical

    # A SWITCH within the same industry is mainly a scale decision.  Aggregate
    # candidate value alone does not tell the ranker whether the *last* added
    # animal/crop is worth its capital, workload and setup turns.  These ratios
    # expose that marginal comparison without adding route identity, author
    # names, fixed dates, or simulator outcomes to online inference.
    baseline_net = baseline_gross - baseline_cost
    candidate_net = candidate_gross - candidate_cost
    delta_gross = candidate_gross - baseline_gross
    delta_cost = candidate_cost - baseline_cost
    delta_net = candidate_net - baseline_net
    delta_output = candidate_output - baseline_output
    changed_units = np.maximum(1.0, add_count + remove_count)
    signed_target_change = add_count - remove_count
    baseline_workload = np.maximum(
        1.0, raw_col("baseline_workload_x100") / 100.0
    )
    candidate_workload = np.maximum(
        1.0, raw_col("candidate_workload_x100") / 100.0
    )
    delta_workload = (
        raw_col("candidate_workload_x100")
        - raw_col("baseline_workload_x100")
    ) / 100.0
    baseline_setup_turns = np.maximum(
        1.0, optional_raw_col("baseline_setup_turns")
    )
    candidate_setup_turns = np.maximum(
        1.0, optional_raw_col("candidate_setup_turns")
    )
    same_project = (
        (source >= 0) & (source == destination)
    ).astype(np.float64)
    positive_incremental_cost = np.maximum(1.0, np.maximum(0.0, delta_cost))
    marginal = np.column_stack([
        baseline_gross,
        candidate_gross,
        baseline_cost,
        candidate_cost,
        baseline_net,
        candidate_net,
        delta_net,
        same_project,
        signed_target_change,
        delta_gross / changed_units,
        delta_cost / changed_units,
        delta_net / changed_units,
        delta_output / changed_units,
        baseline_net / baseline_workload,
        candidate_net / candidate_workload,
        delta_net / np.maximum(1.0, np.abs(delta_workload)),
        baseline_net / baseline_setup_turns,
        candidate_net / candidate_setup_turns,
        delta_net / positive_incremental_cost,
        same_project * delta_net / changed_units,
    ])
    marginal_names = [
        "baseline_total_gross",
        "candidate_total_gross",
        "baseline_total_cost",
        "candidate_total_cost",
        "baseline_total_net",
        "candidate_total_net",
        "value_delta_total_net",
        "same_project_scale_edit",
        "signed_target_change",
        "value_delta_gross_per_changed_unit",
        "value_delta_cost_per_changed_unit",
        "value_delta_net_per_changed_unit",
        "value_delta_output_per_changed_unit",
        "baseline_net_per_workload",
        "candidate_net_per_workload",
        "value_delta_net_per_abs_workload_delta",
        "baseline_net_per_setup_turn",
        "candidate_net_per_setup_turn",
        "incremental_net_return_on_positive_cost",
        "same_project_net_delta_per_changed_unit",
    ]
    return (
        np.concatenate([base_matrix, marginal], axis=1),
        base_names + marginal_names,
        robust_categorical,
    )


def training_weight(data: dict[str, np.ndarray]) -> np.ndarray:
    delta = np.asarray(data["expected_delta"], dtype=np.float64)
    std = np.asarray(data["future_std"], dtype=np.float64)
    samples = np.asarray(
        data.get(
            "future_sample_count",
            np.full(len(delta), data["future_delta_samples"].shape[1]),
        ),
        dtype=np.float64,
    )
    standard_error = std / np.sqrt(np.maximum(1.0, samples))
    confidence = 1.0 / (1.0 + np.square(standard_error / 2500.0))
    decision_importance = 1.0 + np.minimum(np.abs(delta) / 3000.0, 3.0)
    return np.clip(confidence * decision_importance, 0.05, 4.0)


def metrics(
    prediction: np.ndarray,
    actual: np.ndarray,
    seed: np.ndarray,
    seat: np.ndarray,
    candidate_rank: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    group = seed.astype(np.int64) * 2 + seat.astype(np.int64)
    pair_total = 0
    pair_correct = 0
    pair_gap_total = 0
    pair_gap_correct = 0
    top1 = 0
    top3 = 0
    chosen: list[float] = []
    switch_count = 0
    groups = np.unique(group)
    for key in groups:
        indices = np.flatnonzero(group == key)
        group_prediction = np.asarray(prediction[indices], dtype=np.float64).copy()
        group_actual = np.asarray(actual[indices], dtype=np.float64)
        group_rank = np.asarray(candidate_rank[indices], dtype=np.int64)
        keep_rows = np.flatnonzero(group_rank < 0)
        if keep_rows.size != 1:
            raise ValueError(
                f"group {key} must contain exactly one KEEP row, got {keep_rows.size}"
            )
        # Runtime assigns KEEP the calibrated threshold.  Do the same for both
        # pairwise and selection metrics instead of duplicating a second KEEP
        # arm (the dataset already contains candidate_rank == -1).
        group_prediction[keep_rows[0]] = threshold
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                difference = group_actual[left] - group_actual[right]
                if difference == 0:
                    continue
                correct = np.sign(difference) == np.sign(
                    group_prediction[left] - group_prediction[right]
                )
                pair_total += 1
                pair_correct += int(correct)
                if abs(difference) >= 2000.0:
                    pair_gap_total += 1
                    pair_gap_correct += int(correct)
        actual_best = int(np.argmax(group_actual))
        predicted_order = np.argsort(-group_prediction, kind="stable")
        top1 += actual_best == int(predicted_order[0])
        top3 += actual_best in predicted_order[:3]
        chosen_value = float(group_actual[int(predicted_order[0])])
        chosen.append(chosen_value)
        switch_count += int(group_rank[int(predicted_order[0])] >= 0)
    selected = np.asarray(chosen, dtype=np.float64)
    return {
        "groups": int(len(groups)),
        "threshold": float(threshold),
        "pairwise_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "pairwise_comparisons": pair_total,
        "pairwise_gap2000_accuracy": (
            pair_gap_correct / pair_gap_total if pair_gap_total else 0.0
        ),
        "pairwise_gap2000_comparisons": pair_gap_total,
        "top1_oracle_recall_with_keep": top1 / len(groups) if len(groups) else 0.0,
        "top3_oracle_recall_with_keep": top3 / len(groups) if len(groups) else 0.0,
        "mean_realized_delta": float(selected.mean()) if selected.size else 0.0,
        "median_realized_delta": (
            float(np.median(selected)) if selected.size else 0.0
        ),
        "p10_realized_delta": (
            float(np.quantile(selected, 0.10)) if selected.size else 0.0
        ),
        "positive_choice_rate": (
            float(np.mean(selected > 0)) if selected.size else 0.0
        ),
        "negative_choice_rate": (
            float(np.mean(selected < 0)) if selected.size else 0.0
        ),
        "switch_rate": switch_count / len(groups) if len(groups) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--test",
        type=Path,
        help=(
            "Optional untouched test corpus. Model variant and KEEP threshold "
            "are selected only on --validation, then evaluated once here."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    train = load(args.train)
    validation = load(args.validation)
    if not np.array_equal(train["feature_names"], validation["feature_names"]):
        raise ValueError("train/validation feature schema mismatch")
    test = load(args.test) if args.test is not None else None
    if test is not None and not np.array_equal(
        train["feature_names"], test["feature_names"]
    ):
        raise ValueError("train/test feature schema mismatch")
    train_x, expanded_names, categorical = engineered_features(
        train["features"], train["feature_names"]
    )
    validation_x, _, _ = engineered_features(
        validation["features"], validation["feature_names"]
    )
    test_x = None
    if test is not None:
        test_x, _, _ = engineered_features(
            test["features"], test["feature_names"]
        )
    train_y = np.clip(
        np.asarray(train["expected_delta"], dtype=np.float64), -30000.0, 30000.0
    )
    validation_y = np.asarray(validation["expected_delta"], dtype=np.float64)
    test_y = (
        np.asarray(test["expected_delta"], dtype=np.float64)
        if test is not None else None
    )
    weights = training_weight(train)

    variants = [
        {
            "objective": "regression_l1", "num_leaves": 15,
            "max_depth": 6, "min_child_samples": 15,
            "reg_lambda": 5.0, "reg_alpha": 1.0,
        },
        {
            "objective": "regression_l1", "num_leaves": 31,
            "max_depth": 7, "min_child_samples": 20,
            "reg_lambda": 10.0, "reg_alpha": 2.0,
        },
        {
            "objective": "regression", "num_leaves": 15,
            "max_depth": 6, "min_child_samples": 20,
            "reg_lambda": 10.0, "reg_alpha": 2.0,
        },
        {
            "objective": "regression", "num_leaves": 31,
            "max_depth": 8, "min_child_samples": 20,
            "reg_lambda": 15.0, "reg_alpha": 3.0,
        },
        {
            "objective": "regression", "num_leaves": 63,
            "max_depth": 9, "min_child_samples": 12,
            "reg_lambda": 5.0, "reg_alpha": 1.0,
        },
        {
            "objective": "huber", "num_leaves": 31,
            "max_depth": 8, "min_child_samples": 20,
            "reg_lambda": 5.0, "reg_alpha": 1.0,
        },
    ]
    thresholds = [0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0]
    audits: list[dict[str, object]] = []
    models: list[lgb.LGBMRegressor] = []
    best_key: tuple[float, float, float] | None = None
    best_model = -1
    best_threshold = 0.0
    for index, variant in enumerate(variants):
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.025,
            verbosity=-1,
            n_jobs=16,
            random_state=700 + index,
            **variant,
        )
        model.fit(
            train_x,
            train_y,
            sample_weight=weights,
            categorical_feature=categorical,
        )
        prediction = model.predict(validation_x)
        threshold_audits = [
            metrics(
                prediction,
                validation_y,
                validation["prefix_seed"],
                validation["seat"],
                validation["candidate_rank"],
                threshold,
            )
            for threshold in thresholds
        ]
        local_best = max(
            threshold_audits,
            key=lambda item: (
                float(item["mean_realized_delta"]),
                float(item["p10_realized_delta"]),
                -float(item["negative_choice_rate"]),
            ),
        )
        key = (
            float(local_best["mean_realized_delta"]),
            float(local_best["p10_realized_delta"]),
            -float(local_best["negative_choice_rate"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = index
            best_threshold = float(local_best["threshold"])
        audits.append({
            "variant_index": index,
            "parameters": variant,
            "threshold_audits": threshold_audits,
            "selected_threshold": local_best,
        })
        models.append(model)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    models[best_model].booster_.save_model(str(args.model_output))
    importance = models[best_model].feature_importances_
    order = np.argsort(-importance)[:40]
    payload = {
        "schema": "kaggriculture.switch-value-regressor-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": {
            "path": str(args.train),
            "sha256": sha256(args.train),
            "rows": int(len(train_y)),
            "future_samples": int(train["future_delta_samples"].shape[1]),
        },
        "validation": {
            "path": str(args.validation),
            "sha256": sha256(args.validation),
            "rows": int(len(validation_y)),
            "future_samples": int(validation["future_delta_samples"].shape[1]),
        },
        "feature_dim": int(train_x.shape[1]),
        "raw_feature_dim": int(train["features"].shape[1]),
        "variants": audits,
        "selected_variant_index": best_model,
        "selected_threshold": best_threshold,
        "selected_validation_metrics": audits[best_model]["selected_threshold"],
        "model": {
            "path": str(args.model_output),
            "sha256": sha256(args.model_output),
        },
        "feature_importance": [
            {"feature": expanded_names[i], "importance": float(importance[i])}
            for i in order
        ],
    }
    if test is not None and test_x is not None and test_y is not None:
        test_prediction = models[best_model].predict(test_x)
        payload["test"] = {
            "path": str(args.test),
            "sha256": sha256(args.test),
            "rows": int(len(test_y)),
            "future_samples": int(test["future_delta_samples"].shape[1]),
            "used_for_model_or_threshold_selection": False,
        }
        payload["selected_test_metrics"] = metrics(
            test_prediction,
            test_y,
            test["prefix_seed"],
            test["seat"],
            test["candidate_rank"],
            best_threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_variant_index": best_model,
        "selected_threshold": best_threshold,
        "metrics": payload["selected_validation_metrics"],
        "test_metrics": payload.get("selected_test_metrics"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
