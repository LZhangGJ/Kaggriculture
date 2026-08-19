"""Fixed-shape records for the M3.5 crop/animal closed loop."""

from __future__ import annotations

from typing import NamedTuple

import jax

from kaggriculture_jax.types import State

from .m3_schema import M3AnimalGenomeV2, M3CoverageMetricsV2, M3RolloutMetricsV2
from .schema import (
    M25RolloutMetricsV2,
    M26CoverageMetricsV2,
    M26CropGenomeV2,
    ProjectControllerStateV2,
)


Array = jax.Array


class M35FarmGenomeV2(NamedTuple):
    """One synchronized business plan with crop and animal sub-plans."""

    candidate_id: Array
    crop: M26CropGenomeV2
    animal: M3AnimalGenomeV2
    crop_unit_share: Array


class M35FlowMetricsV2(NamedTuple):
    """Observable cross-project resource flow and conflict counters."""

    wheat_harvest_actions: Array
    wheat_market_buy_units: Array
    animal_feed_actions: Array
    animal_fertilizer_collect_actions: Array
    crop_fertilizer_apply_actions: Array
    fertilizer_market_buy_units: Array
    joint_resource_conflict_count: Array


class M35RolloutCarryV2(NamedTuple):
    environment_state: State
    controller: ProjectControllerStateV2
    crop_metrics: M25RolloutMetricsV2
    animal_metrics: M3RolloutMetricsV2
    crop_coverage: M26CoverageMetricsV2
    animal_coverage: M3CoverageMetricsV2
    flow: M35FlowMetricsV2


class M35RolloutSummaryV2(NamedTuple):
    final_bank: Array
    done: Array
    terminal_sellable_shed_value: Array
    terminal_unit_inventory_value: Array
    terminal_harvestable_crop_value: Array
    terminal_animal_product_value: Array
    avoidable_liquidation_loss: Array
    plant_without_same_day_water: Array
    unexplained_failure_count: Array
    unplanned_animal_escape: Array
    animal_capacity_loss: Array
    feed_hard_deadline_miss: Array
    care_bonus_forfeited_unexplained: Array
    care_bonus_capacity_clipped_unexplained: Array
    animal_bought_without_place_plan: Array
    cow_sheep_pasture_conflict: Array
    animals_stranded_in_shed_at_terminal: Array
    animals_stranded_in_unit_inventory_at_terminal: Array
    planted_tiles_by_crop: Array
    active_animals_by_species: Array
    flow: M35FlowMetricsV2
    crop_coverage: M26CoverageMetricsV2
    animal_coverage: M3CoverageMetricsV2


__all__ = [
    "M35FarmGenomeV2",
    "M35FlowMetricsV2",
    "M35RolloutCarryV2",
    "M35RolloutSummaryV2",
]

