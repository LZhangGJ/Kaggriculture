from __future__ import annotations

import jax
import jax.numpy as jnp

from project_route_search_v2.constants import (
    MAX_MARKET_ORDERS,
    MAX_OBLIGATIONS_V2,
    MAX_PHASES_V2,
    MAX_PROJECTS_V2,
    MAX_ROUTE_STOPS,
    MAX_ROUTE_CARD_STOPS_V3,
    MAX_UNITS,
    OFFICIAL_PACKAGE_VERSION,
    STATIC_SPECIES_CAP,
)
from project_route_search_v2.schema import (
    CommitmentEnvelopeV2,
    ObligationV2,
    ProjectControllerStateV2,
    ProjectStateV2,
    RouteCardStateV3,
    RouteGenomeV2,
    M26CropGenomeV2,
    SCHEMA_FIELDS_V2,
    StepDiagnosticsV2,
    UnitPlanStateV2,
)


def test_frozen_caps_match_v11_contract() -> None:
    assert OFFICIAL_PACKAGE_VERSION == "1.32.7"
    assert MAX_PROJECTS_V2 == 24
    assert MAX_PHASES_V2 == 6
    assert MAX_OBLIGATIONS_V2 == 128
    assert MAX_ROUTE_STOPS == 8
    assert MAX_ROUTE_CARD_STOPS_V3 == 12
    assert MAX_UNITS == 33
    assert MAX_MARKET_ORDERS == 10
    assert STATIC_SPECIES_CAP == 100


def test_schema_field_sets_contain_contract_critical_fields() -> None:
    assert SCHEMA_FIELDS_V2["RouteGenomeV2"][:3] == (
        "candidate_id",
        "family_id",
        "domain_mode",
    )
    assert "crop_target" in RouteGenomeV2._fields
    assert "animal_target" in RouteGenomeV2._fields
    assert "visible_market_trigger" in RouteGenomeV2._fields
    assert "latest_bank_step" in ProjectStateV2._fields
    assert "deadline_step" in ObligationV2._fields
    assert "route_insertion_cost" in ObligationV2._fields
    assert "route_obligation_ids" in UnitPlanStateV2._fields
    assert "unexplained_effect_failures" in ProjectControllerStateV2._fields
    assert "route_cards" in ProjectControllerStateV2._fields
    assert RouteCardStateV3._fields[:5] == (
        "card_type",
        "status",
        "target_ids",
        "target_items",
        "action_masks",
    )
    assert CommitmentEnvelopeV2._fields == (
        "min_actions_by_day",
        "feed_required_by_day",
        "expected_shed_in_by_day",
        "expected_cash_out_by_day",
        "expected_cash_in_by_day",
        "occupied_tiles_by_day",
    )
    assert "terminal_sellable_shed_value" in StepDiagnosticsV2._fields
    assert M26CropGenomeV2._fields == (
        "candidate_id",
        "phase_count",
        "phase_start_step",
        "crop_target",
        "land_target",
        "land_start_step",
        "hand_target",
        "crop_last_plant_step",
        "seed_buy_batch",
        "plant_wave_size",
        "harvest_min_age_days",
        "harvest_trigger_units",
        "fertilizer_policy",
        "crop_layout_policy",
        "crop_cash_cap",
        "weed_recovery_policy",
        "crop_abandon_policy",
        "maintenance_utilization_cap",
        "deposit_min_value",
        "sell_interval",
        "sell_phase",
        "sell_price_floor_ratio",
        "sell_fraction",
        "shed_pressure_trigger",
        "cash_floor",
        "liquidation_start_step",
    )


def test_schema_named_tuples_are_jax_pytrees() -> None:
    values = [jnp.asarray(index, dtype=jnp.int32) for index in range(len(RouteGenomeV2._fields))]
    genome = RouteGenomeV2(*values)
    leaves, tree = jax.tree.flatten(genome)
    assert len(leaves) == len(RouteGenomeV2._fields)
    assert isinstance(jax.tree.unflatten(tree, leaves), RouteGenomeV2)
