"""Compile and execute one full M3.6C split-JIT season with progress receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
for source in (
    PROJECT / "src",
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source))

import jax  # noqa: E402

CACHE = ROOT / ".jax_cache"
CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from kaggriculture_jax.constants import FLAG_FED, NUM_ANIMALS, NUM_PRODUCTS  # noqa: E402
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2  # noqa: E402
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
)
from project_route_search_v2.m36_rollout import summarize_m36c_rollout_v3  # noqa: E402
from project_route_search_v2.m36_split_rollout import (  # noqa: E402
    initialize_m36c_split_v3,
    make_m36c_light_chunk_v3,
    m36c_daily_plan_carry_v3,
)


REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)
EVENT_BANK = PROJECT / "artifacts" / "events" / "e0_seed_panels_v1.npz"
OUTPUT = PROJECT / "receipts" / "m36c_split_semantic_v1.json"
JOURNAL = PROJECT / "receipts" / "m36c_split_semantic_v1.progress.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-bank", type=Path, default=EVENT_BANK)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def log(event: str, **values) -> None:
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with JOURNAL.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def compile_kernel(name, function, arguments):
    log("compile_started", kernel=name)
    started = time.perf_counter()
    executable = jax.jit(function).lower(*arguments).compile()
    seconds = time.perf_counter() - started
    log("compile_completed", kernel=name, seconds=seconds)
    return executable, seconds


def scalar_summary(summary) -> dict:
    return {
        "final_bank": int(summary.farm.final_bank[0]),
        "done": bool(summary.farm.done[0]),
        "hard_error_count": int(summary.hard_error_count[0]),
        "transaction_hard_error_count": int(
            summary.transaction_hard_error_count[0]
        ),
        "illegal_action_count": int(summary.illegal_action_count[0]),
        "unexplained_effect_error_count": int(
            summary.unexplained_effect_error_count[0]
        ),
        "unplanned_deadline_miss_count": int(
            summary.unplanned_deadline_miss_count[0]
        ),
        "duplicate_resource_reservation_count": int(
            summary.duplicate_resource_reservation_count[0]
        ),
        "planned_release_count": int(summary.planned_release_count[0]),
        "unplanned_animal_escape": int(summary.unplanned_animal_escape[0]),
        "plant_without_same_day_water": int(
            summary.farm.plant_without_same_day_water[0]
        ),
        "fertilizer_market_slot_overflow_count": int(
            summary.fertilizer_market_slot_overflow_count[0]
        ),
        "terminal_shed_value": int(summary.farm.terminal_sellable_shed_value[0]),
        "terminal_unit_inventory_value": int(
            summary.farm.terminal_unit_inventory_value[0]
        ),
        "terminal_harvestable_crop_value": int(
            summary.farm.terminal_harvestable_crop_value[0]
        ),
        "terminal_animal_product_value": int(
            summary.farm.terminal_animal_product_value[0]
        ),
        "scheduler_crop_selection_count": int(
            summary.scheduler_crop_selection_count[0]
        ),
        "scheduler_animal_selection_count": int(
            summary.scheduler_animal_selection_count[0]
        ),
        "scheduler_deadline_preemption_count": int(
            summary.scheduler_deadline_preemption_count[0]
        ),
        "scheduler_local_swap_count": int(summary.scheduler_local_swap_count[0]),
        "max_planned_release_tiles": int(summary.max_planned_release_tiles[0]),
    }


def timeline_row(carry) -> dict:
    """Small host-side day boundary snapshot for hard-error attribution."""

    state = jax.device_get(carry.farm.environment_state)
    aggregate = jax.device_get(carry.aggregates)
    animals = np.asarray(state.tile_animal[0, 0]).reshape(-1)
    flags = np.asarray(state.tile_flags[0, 0]).reshape(-1)
    neglect = np.asarray(state.tile_neglect[0, 0]).reshape(-1)
    placed = [int(np.sum(animals == species)) for species in range(NUM_ANIMALS)]
    shed = np.asarray(state.shed[0, 0])
    carried = np.asarray(state.unit_inventory[0, 0]).sum(axis=0)
    return {
        "step": int(state.step[0]),
        "money": int(state.money[0, 0]),
        "hands": int(state.hires_today[0, 0]),
        "placed_animals": placed,
        "shed_animals": shed[NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
        .astype(int)
        .tolist(),
        "carried_animals": carried[NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
        .astype(int)
        .tolist(),
        "wheat_stock": int(shed[0] + carried[0]),
        "unfed_animals": int(
            np.sum((animals >= 0) & ((flags & int(FLAG_FED)) == 0))
        ),
        "neglected_animals": int(np.sum((animals >= 0) & (neglect >= 1))),
        "deadline_miss_cumulative": int(
            aggregate.unplanned_deadline_miss_count[0]
        ),
        "unplanned_escape_cumulative": int(
            aggregate.unplanned_animal_escape_count[0]
        ),
        "planned_release_cumulative": int(
            aggregate.planned_release_escape_count[0]
        ),
        "crop_selection_cumulative": int(
            aggregate.scheduler_crop_selection_count[0]
        ),
        "animal_selection_cumulative": int(
            aggregate.scheduler_animal_selection_count[0]
        ),
        "deadline_preemption_cumulative": int(
            aggregate.scheduler_deadline_preemption_count[0]
        ),
        "crop_effect_mismatch_cumulative": int(
            carry.farm.crop_metrics.effect_mismatch_count[0]
        ),
        "animal_effect_mismatch_cumulative": int(
            carry.farm.animal_metrics.effect_mismatch_count[0]
        ),
        "animal_resource_unavailable_cumulative": int(
            carry.farm.animal_metrics.resource_unavailable_count[0]
        ),
        "plant_without_same_day_water_cumulative": int(
            carry.farm.crop_metrics.plant_without_same_day_water[0]
        ),
    }


def main() -> None:
    args = parse_args()
    JOURNAL.unlink(missing_ok=True)
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    calendar, diagnostic = compile_gold_replay_calendar_v3(
        replay, player=1, candidate_id=3_600
    )
    source_seeds, bank = load_event_bank(args.event_bank)
    if args.event_index < 0 or args.event_index >= len(source_seeds):
        raise ValueError("event-index is outside the event bank")
    selected = slice(args.event_index, args.event_index + 1)
    seeds = jnp.asarray((int(source_seeds[args.event_index]),), dtype=jnp.int32)
    events = Events(bank.weed_spawn[selected], bank.shop_choice[selected])
    tables = load_tables()
    template = default_m35_farm_genome_v2(1)
    opening_args = (seeds, calendar, events, tables, template)
    opening, opening_compile = compile_kernel(
        "opening", initialize_m36c_split_v3, opening_args
    )
    started = time.perf_counter()
    carry = opening(*opening_args)
    jax.block_until_ready(carry)
    opening_execute = time.perf_counter() - started
    log("opening_executed", seconds=opening_execute, step=int(carry.farm.environment_state.step[0]))
    timeline = [timeline_row(carry)]

    chunk_fn = make_m36c_light_chunk_v3(chunk_steps=24)
    chunk_args = (carry, calendar, events, tables, template, jnp.int32(23))
    chunk, chunk_compile = compile_kernel("light_chunk_24_masked", chunk_fn, chunk_args)
    daily_args = (carry, calendar, tables, template)
    daily, daily_compile = compile_kernel(
        "daily_plan", m36c_daily_plan_carry_v3, daily_args
    )

    started = time.perf_counter()
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(23))
    jax.block_until_ready(carry)
    timeline.append(timeline_row(carry))
    for _day in range(1, 29):
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(23))
        jax.block_until_ready(carry)
        timeline.append(timeline_row(carry))
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(22))
    jax.block_until_ready(carry)
    timeline.append(timeline_row(carry))
    warm_seconds = time.perf_counter() - started
    summary = jax.device_get(summarize_m36c_rollout_v3(carry))
    values = scalar_summary(summary)
    values["final_step"] = int(carry.farm.environment_state.step[0])
    log("season_completed", seconds=warm_seconds, **values)
    checks = {
        "final_step_719": values["final_step"] == 719,
        "done": values["done"],
        "transaction_hard_zero": values["transaction_hard_error_count"] == 0,
        "illegal_action_zero": values["illegal_action_count"] == 0,
        "unexplained_effect_zero": values["unexplained_effect_error_count"] == 0,
        "unplanned_deadline_zero": values["unplanned_deadline_miss_count"] == 0,
        "duplicate_reservation_zero": values[
            "duplicate_resource_reservation_count"
        ]
        == 0,
        "unplanned_escape_zero": values["unplanned_animal_escape"] == 0,
        "fertilizer_slot_overflow_zero": values[
            "fertilizer_market_slot_overflow_count"
        ]
        == 0,
        "plant_without_same_day_water_zero": values[
            "plant_without_same_day_water"
        ]
        == 0,
        "hard_error_zero": values["hard_error_count"] == 0,
    }
    receipt = {
        "schema": "kaggriculture.m36c_split_semantic.v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "calendar_source": diagnostic.to_dict(),
        "event_bank": str(args.event_bank.resolve()),
        "event_index": args.event_index,
        "event_seed": int(source_seeds[args.event_index]),
        "compile_seconds": {
            "opening": opening_compile,
            "light_chunk_24_masked": chunk_compile,
            "daily_plan": daily_compile,
        },
        "opening_execute_seconds": opening_execute,
        "season_warm_seconds": warm_seconds,
        "effective_transitions_per_second": 719 / warm_seconds,
        "summary": values,
        "day_boundary_timeline": timeline,
        "checks": checks,
        "boundary": "ONE_CALENDAR_ONE_EVENT_SEED_SEMANTIC_DIAGNOSTIC_NOT_FINAL_M36C_ACCEPTANCE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
