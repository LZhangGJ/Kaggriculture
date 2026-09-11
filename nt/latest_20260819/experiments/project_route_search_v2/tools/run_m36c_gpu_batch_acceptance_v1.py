"""M3.6C batched GPU semantic and throughput acceptance.

The runner broadcasts one high-level RouteCalendar over independent official
event streams.  It does not store or play Replay raw actions.
"""

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
from project_route_search_v2.m38_route_cards import RouteCardModeV4  # noqa: E402


REPLAY = (
    ROOT
    / "replay"
    / "gold_latest_2026-08-19_154928"
    / "latest_per_gold"
    / "episode-94424738-replay.json"
)
DEFAULT_EVENT_BANK = (
    ROOT
    / "experiments"
    / "strategic_v4"
    / "artifacts"
    / "shared_event_banks"
    / "training_events_64k_v1"
    / "events_00_seeds_1000000_1002047.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--event-bank", type=Path, default=DEFAULT_EVENT_BANK)
    parser.add_argument("--timed-runs", type=int, default=2)
    parser.add_argument("--enable-m37-unlock", action="store_true")
    parser.add_argument("--enable-route-cards", action="store_true")
    parser.add_argument(
        "--route-card-mode",
        choices=[mode.name.lower() for mode in RouteCardModeV4],
        default=RouteCardModeV4.LEGACY_M38.name.lower(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m36c_gpu_batch2048_acceptance_v1.json",
    )
    return parser.parse_args()


def log(event: str, **values) -> None:
    print(
        json.dumps(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **values,
            }
        ),
        flush=True,
    )


def compile_kernel(name, function, arguments):
    log("compile_started", kernel=name)
    started = time.perf_counter()
    executable = jax.jit(function).lower(*arguments).compile()
    seconds = time.perf_counter() - started
    log("compile_completed", kernel=name, seconds=seconds)
    return executable, seconds


def broadcast_tree(value, batch_size: int):
    return jax.tree.map(
        lambda array: jnp.broadcast_to(
            array, (batch_size, *array.shape[1:])
        ),
        value,
    )


def run_season(opening, chunk, daily, arguments, *, enable_m37_unlock: bool):
    seeds, calendar, events, tables, template = arguments
    carry = opening(seeds, calendar, events, tables, template)
    # Opening consumes step 0.  M3.7 additionally replans at hour 20 to unlock
    # committed capital; keep that schedule in throughput measurements so the
    # benchmark times the same controller semantics as the fixed-seed trace.
    if enable_m37_unlock:
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(19))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(4))
        for _day in range(1, 29):
            carry = daily(carry, calendar, tables, template)
            carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
            carry = daily(carry, calendar, tables, template)
            carry = chunk(carry, calendar, events, tables, template, jnp.int32(19))
            carry = daily(carry, calendar, tables, template)
            carry = chunk(carry, calendar, events, tables, template, jnp.int32(4))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(19))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(3))
        return carry

    # Frozen M3.6 schedule: day-boundary plan plus one hour-1 retry for daily
    # bundles longer than the ten official market slots.
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(23))
    for _day in range(1, 29):
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
        carry = daily(carry, calendar, tables, template)
        carry = chunk(carry, calendar, events, tables, template, jnp.int32(23))
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(1))
    carry = daily(carry, calendar, tables, template)
    carry = chunk(carry, calendar, events, tables, template, jnp.int32(22))
    return carry


def integer_stats(values) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "sum": int(array.sum()),
    }


def error_stats(values) -> dict[str, int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "sum": int(array.sum()),
        "max": int(array.max()),
        "affected_episode_count": int(np.count_nonzero(array)),
    }


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    if args.timed_runs <= 0:
        raise ValueError("timed-runs must be positive")

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    one_calendar, diagnostic = compile_gold_replay_calendar_v3(
        replay, player=args.player, candidate_id=3_600
    )
    calendar = broadcast_tree(one_calendar, args.batch)
    source_seeds, source_events = load_event_bank(args.event_bank)
    if len(source_seeds) < args.batch:
        raise ValueError(
            f"event bank has {len(source_seeds)} rows, needs {args.batch}"
        )
    seeds = jnp.asarray(source_seeds[: args.batch], dtype=jnp.int32)
    events = Events(
        source_events.weed_spawn[: args.batch],
        source_events.shop_choice[: args.batch],
    )
    tables = load_tables()
    template = default_m35_farm_genome_v2(args.batch)
    arguments = (seeds, calendar, events, tables, template)

    opening_fn = lambda s, c, e, t, g: initialize_m36c_split_v3(
        s, c, e, t, g, player=args.player
    )
    opening, opening_compile = compile_kernel(
        "opening", opening_fn, arguments
    )
    sample_carry = opening(*arguments)
    jax.block_until_ready(sample_carry)
    chunk_fn = make_m36c_light_chunk_v3(
        chunk_steps=24,
        player=args.player,
        enable_unlock_committed_capital=args.enable_m37_unlock,
        enable_route_cards=args.enable_route_cards,
        route_card_mode=int(RouteCardModeV4[args.route_card_mode.upper()]),
    )
    chunk_args = (
        sample_carry,
        calendar,
        events,
        tables,
        template,
        jnp.int32(24),
    )
    chunk, chunk_compile = compile_kernel(
        "light_chunk_24_masked", chunk_fn, chunk_args
    )
    daily_fn = lambda carry, c, t, g: m36c_daily_plan_carry_v3(
        carry,
        c,
        t,
        g,
        player=args.player,
        enable_unlock_committed_capital=args.enable_m37_unlock,
    )
    daily_args = (sample_carry, calendar, tables, template)
    daily, daily_compile = compile_kernel(
        "daily_plan", daily_fn, daily_args
    )

    timings = []
    final_carry = None
    for run in range(args.timed_runs):
        started = time.perf_counter()
        final_carry = run_season(
            opening,
            chunk,
            daily,
            arguments,
            enable_m37_unlock=args.enable_m37_unlock,
        )
        jax.block_until_ready(final_carry)
        seconds = time.perf_counter() - started
        timings.append(seconds)
        log(
            "timed_run_completed",
            run=run,
            seconds=seconds,
            transitions_per_second=args.batch * 719 / seconds,
        )

    assert final_carry is not None
    summary = jax.device_get(summarize_m36c_rollout_v3(final_carry))
    final_steps = np.asarray(final_carry.farm.environment_state.step)
    done = np.asarray(summary.farm.done)
    hard = np.asarray(summary.hard_error_count)
    component_errors = {
        "transaction_hard_error_count": error_stats(
            summary.transaction_hard_error_count
        ),
        "illegal_action_count": error_stats(summary.illegal_action_count),
        "unexplained_effect_error_count": error_stats(
            summary.unexplained_effect_error_count
        ),
        "unplanned_deadline_miss_count": error_stats(
            summary.unplanned_deadline_miss_count
        ),
        "duplicate_resource_reservation_count": error_stats(
            summary.duplicate_resource_reservation_count
        ),
        "unplanned_animal_escape": error_stats(summary.unplanned_animal_escape),
        "plant_without_same_day_water": error_stats(
            summary.farm.plant_without_same_day_water
        ),
        "fertilizer_market_slot_overflow_count": error_stats(
            summary.fertilizer_market_slot_overflow_count
        ),
    }
    affected_indices = np.flatnonzero(hard)
    affected_examples = []
    for index in affected_indices[:64]:
        affected_examples.append(
            {
                "batch_index": int(index),
                "event_seed": int(source_seeds[index]),
                "hard_error_count": int(hard[index]),
                "unexplained_effect_error_count": int(
                    summary.unexplained_effect_error_count[index]
                ),
                "unplanned_deadline_miss_count": int(
                    summary.unplanned_deadline_miss_count[index]
                ),
                "unplanned_animal_escape": int(
                    summary.unplanned_animal_escape[index]
                ),
                "crop_effect_mismatch_count": int(
                    final_carry.farm.crop_metrics.effect_mismatch_count[index]
                ),
                "crop_resource_unavailable_count": int(
                    final_carry.farm.crop_metrics.resource_unavailable_count[index]
                ),
                "animal_effect_mismatch_count": int(
                    final_carry.farm.animal_metrics.effect_mismatch_count[index]
                ),
                "animal_resource_unavailable_count": int(
                    final_carry.farm.animal_metrics.resource_unavailable_count[index]
                ),
                "animal_deadline_missed_count": int(
                    final_carry.farm.animal_metrics.deadline_missed_count[index]
                ),
                "final_bank": int(summary.farm.final_bank[index]),
                "active_animals_by_species": np.asarray(
                    summary.farm.active_animals_by_species[index], dtype=np.int64
                ).tolist(),
            }
        )
    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "all_final_step_719": bool(np.all(final_steps == 719)),
        "all_done": bool(np.all(done)),
        "hard_error_zero_all_episodes": bool(np.all(hard == 0)),
        "all_component_error_sums_zero": all(
            value["sum"] == 0 for value in component_errors.values()
        ),
    }
    best_seconds = min(timings)
    panel_size = min(args.batch, 64)
    fixed_seed_panel = [
        {
            "batch_index": int(index),
            "event_seed": int(source_seeds[index]),
            "final_bank": int(summary.farm.final_bank[index]),
            "hard_error_count": int(hard[index]),
        }
        for index in range(panel_size)
    ]
    receipt = {
        "schema": "kaggriculture.m36c_gpu_batch_acceptance.v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "batch_size": args.batch,
        "replay": str(args.replay.resolve()),
        "player": args.player,
        "m37_unlock_committed_capital": bool(args.enable_m37_unlock),
        "route_cards_enabled": bool(args.enable_route_cards),
        "route_card_mode": args.route_card_mode,
        "calendar_source": diagnostic.to_dict(),
        "raw_action_payload_stored": False,
        "event_bank": str(args.event_bank.resolve()),
        "event_seed_first": int(source_seeds[0]),
        "event_seed_last": int(source_seeds[args.batch - 1]),
        "compile_seconds": {
            "opening": opening_compile,
            "light_chunk_24_masked": chunk_compile,
            "daily_plan": daily_compile,
        },
        "timed_run_seconds": timings,
        "best_full_season_transitions_per_second": args.batch * 719 / best_seconds,
        "hard_error": error_stats(hard),
        "component_errors": component_errors,
        "affected_examples": affected_examples,
        "final_bank": integer_stats(summary.farm.final_bank),
        "fixed_seed_panel_first_64": fixed_seed_panel,
        "soft_terminal_diagnostics": {
            "terminal_shed_value": integer_stats(
                summary.farm.terminal_sellable_shed_value
            ),
            "terminal_unit_inventory_value": integer_stats(
                summary.farm.terminal_unit_inventory_value
            ),
            "terminal_harvestable_crop_value": integer_stats(
                summary.farm.terminal_harvestable_crop_value
            ),
            "terminal_animal_product_value": integer_stats(
                summary.farm.terminal_animal_product_value
            ),
            "planned_release_count": integer_stats(summary.planned_release_count),
        },
        "scheduler": {
            "crop_selection_count": integer_stats(
                summary.scheduler_crop_selection_count
            ),
            "animal_selection_count": integer_stats(
                summary.scheduler_animal_selection_count
            ),
            "deadline_preemption_count": integer_stats(
                summary.scheduler_deadline_preemption_count
            ),
            "local_swap_count": integer_stats(summary.scheduler_local_swap_count),
        },
        "checks": checks,
        "boundary": "M36C_GPU_BATCH_ACCEPTANCE_NULL_OPPONENT_NOT_OFFICIAL_PARITY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
