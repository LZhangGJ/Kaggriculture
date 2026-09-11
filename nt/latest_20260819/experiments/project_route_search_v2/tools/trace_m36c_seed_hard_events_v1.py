"""Locate and explain the exact M3.6C step that increments a hard counter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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

from kaggriculture_jax.constants import TURNS_PER_DAY  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m35_controller import (  # noqa: E402
    m35_player_action_dict_v2,
    update_m35_controller_from_effects_v2,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2  # noqa: E402
from project_route_search_v2.m36_controller import m36c_light_policy_step_v3  # noqa: E402
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
)
from project_route_search_v2.m36_split_rollout import (  # noqa: E402
    M36_MARKET_PLAN_HOURS_V3,
    initialize_m36c_split_v3,
    make_m36c_light_chunk_v3,
    m36c_daily_plan_carry_v3,
)
from project_route_search_v2.null_opponent import combine_with_null_opponent  # noqa: E402


REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
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
    parser.add_argument("--event-bank", type=Path, default=DEFAULT_EVENT_BANK)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline-start", type=int)
    parser.add_argument("--timeline-end", type=int)
    parser.add_argument(
        "--compact-inspect",
        action="store_true",
        help="Compile only the fields needed to explain hard-counter changes.",
    )
    return parser.parse_args()


def host(value):
    value = jax.device_get(value)
    if hasattr(value, "_fields"):
        return {name: host(getattr(value, name)) for name in value._fields}
    if isinstance(value, dict):
        return {name: host(item) for name, item in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape[0] == 1:
        array = array[0]
    return array.tolist()


def compact_tasks(tasks) -> list[dict]:
    values = host(tasks)
    result = []
    for unit, status in enumerate(values["status"]):
        if status != 1:
            continue
        result.append(
            {
                "unit": unit,
                "task_type": values["task_type"][unit],
                "target": [values["target_x"][unit], values["target_y"][unit]],
                "item_id": values["item_id"][unit],
                "quantity": values["quantity"][unit],
                "phase": values["phase"][unit],
                "deadline_step": values["deadline_step"][unit],
            }
        )
    return result


def compact_market(tasks) -> list[dict]:
    values = host(tasks)
    result = []
    for slot, status in enumerate(values["status"]):
        if status != 1:
            continue
        result.append(
            {
                "slot": slot,
                "task_type": values["task_type"][slot],
                "item_id": values["item_id"][slot],
                "quantity": values["quantity"][slot],
                "deadline_step": values["deadline_step"][slot],
            }
        )
    return result


def compact_animal_tiles(state) -> list[dict]:
    values = host(state)
    result = []
    animal = np.asarray(values["tile_animal"])[0]
    flags = np.asarray(values["tile_flags"])[0]
    neglect = np.asarray(values["tile_neglect"])[0]
    held = np.asarray(values["tile_yield"])[0]
    pending = np.asarray(values["tile_pending_care"])[0]
    for y, x in np.argwhere(animal >= 0):
        result.append(
            {
                "tile": [int(x), int(y)],
                "species": int(animal[y, x]),
                "flags": int(flags[y, x]),
                "neglect": int(neglect[y, x]),
                "held": int(held[y, x]),
                "pending_care": int(pending[y, x]),
            }
        )
    return result


def compact_timeline_step(step, carry, action, controller, scheduler) -> dict:
    state = host(carry.farm.environment_state)
    return {
        "step": step,
        "money": state["money"][0],
        "shed_wheat": state["shed"][0][0],
        "unit_active": state["unit_active"][0],
        "unit_pos": state["unit_pos"][0],
        "unit_wheat": [row[0] for row in state["unit_inventory"][0]],
        "animal_tiles": compact_animal_tiles(carry.farm.environment_state),
        "action": host(m35_player_action_dict_v2(action)),
        "unit_tasks": compact_tasks(controller.unit_tasks),
        "scheduler": host(scheduler),
    }


def inspect_step(carry, calendar, events, tables, template):
    states = carry.farm.environment_state
    action, controller, genome, scheduler = m36c_light_policy_step_v3(
        states, carry.farm.controller, calendar, template, 0
    )
    joint = combine_with_null_opponent(
        m35_player_action_dict_v2(action), player_seat=0
    )
    next_states = batched_step_sync(states, joint, events, tables)
    _, effects = update_m35_controller_from_effects_v2(
        states, next_states, controller, action, genome, 0
    )
    return action, controller, scheduler, next_states, effects


def inspect_step_compact(carry, calendar, events, tables, template):
    """Replay one step while returning only hard-event diagnostic fields.

    The full inspector returns complete State/Controller pytrees and can take
    several minutes for XLA to optimise.  This projection keeps strict effect
    recomputation but lets dead-code elimination remove unrelated farm fields.
    """

    states = carry.farm.environment_state
    action, controller, genome, _ = m36c_light_policy_step_v3(
        states, carry.farm.controller, calendar, template, 0
    )
    joint = combine_with_null_opponent(
        m35_player_action_dict_v2(action), player_seat=0
    )
    next_states = batched_step_sync(states, joint, events, tables)
    _, effects = update_m35_controller_from_effects_v2(
        states, next_states, controller, action, genome, 0
    )
    return {
        "next_step": next_states.step,
        "money_before": states.money[:, 0],
        "money_after": next_states.money[:, 0],
        "shed_wheat_before": states.shed[:, 0, 0],
        "shed_wheat_after": next_states.shed[:, 0, 0],
        "shed_before": states.shed[:, 0],
        "shed_after": next_states.shed[:, 0],
        "unit_pos": states.unit_pos[:, 0],
        "unit_wheat_before": states.unit_inventory[:, 0, :, 0],
        "unit_wheat_after": next_states.unit_inventory[:, 0, :, 0],
        "unit_inventory_before": states.unit_inventory[:, 0],
        "unit_inventory_after": next_states.unit_inventory[:, 0],
        "action_unit_op": action.unit_op,
        "action_unit_item": action.unit_item,
        "action_unit_amount": action.unit_amount,
        "action_market_count": action.market_count,
        "action_market_op": action.market_op,
        "action_market_item": action.market_item,
        "action_market_amount": action.market_amount,
        "task_type": controller.unit_tasks.task_type,
        "task_item": controller.unit_tasks.item_id,
        "task_quantity": controller.unit_tasks.quantity,
        "task_phase": controller.unit_tasks.phase,
        "task_target_x": controller.unit_tasks.target_x,
        "task_target_y": controller.unit_tasks.target_y,
        "task_deadline": controller.unit_tasks.deadline_step,
        "task_status": controller.unit_tasks.status,
        "market_task_type": controller.market_tasks.task_type,
        "market_task_item": controller.market_tasks.item_id,
        "market_task_quantity": controller.market_tasks.quantity,
        "market_task_status": controller.market_tasks.status,
        "tile_animal_before": states.tile_animal[:, 0],
        "tile_animal_after": next_states.tile_animal[:, 0],
        "tile_flags_before": states.tile_flags[:, 0],
        "tile_flags_after": next_states.tile_flags[:, 0],
        "tile_neglect_before": states.tile_neglect[:, 0],
        "effect_mismatch_count": effects.effect_mismatch_count,
        "e3_effect_mismatch_count": effects.e3.effect_mismatch_count,
        "crop_effect_mismatch_count": effects.crop_inventory.effect_mismatch_count,
        "e3_deadline_missed_count": effects.e3.deadline_missed_count,
    }


def counter_tuple(carry) -> tuple[int, ...]:
    aggregate = jax.device_get(carry.aggregates)
    return (
        int(aggregate.unexplained_effect_error_count[0]),
        int(aggregate.unplanned_deadline_miss_count[0]),
        int(aggregate.unplanned_animal_escape_count[0]),
        int(carry.farm.crop_metrics.plant_without_same_day_water[0]),
    )


def main() -> None:
    args = parse_args()
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    calendar, diagnostic = compile_gold_replay_calendar_v3(
        replay, player=1, candidate_id=3_600
    )
    source_seeds, source_events = load_event_bank(args.event_bank)
    if args.event_index < 0 or args.event_index >= len(source_seeds):
        raise ValueError("event-index is outside the event bank")
    selected = slice(args.event_index, args.event_index + 1)
    seeds = jnp.asarray((int(source_seeds[args.event_index]),), dtype=jnp.int32)
    events = Events(
        source_events.weed_spawn[selected], source_events.shop_choice[selected]
    )
    tables = load_tables()
    template = default_m35_farm_genome_v2(1)
    opening = jax.jit(initialize_m36c_split_v3)
    daily = jax.jit(m36c_daily_plan_carry_v3)
    one_step = jax.jit(make_m36c_light_chunk_v3(chunk_steps=24))
    inspect = jax.jit(
        inspect_step_compact if args.compact_inspect else inspect_step
    )

    carry = opening(seeds, calendar, events, tables, template)
    jax.block_until_ready(carry)
    prior = counter_tuple(carry)
    events_found = []
    timeline = []
    while int(carry.farm.environment_state.step[0]) < 719:
        step = int(carry.farm.environment_state.step[0])
        if step >= 1 and step % TURNS_PER_DAY in M36_MARKET_PLAN_HOURS_V3:
            carry = daily(carry, calendar, tables, template)
        before = carry
        inspected = None
        if (
            args.timeline_start is not None
            and args.timeline_end is not None
            and args.timeline_start <= step <= args.timeline_end
        ):
            inspected = inspect(before, calendar, events, tables, template)
            jax.block_until_ready(inspected)
            if args.compact_inspect:
                timeline.append({"step": step, "compact": host(inspected)})
            else:
                action, controller, scheduler, _, _ = inspected
                timeline.append(
                    compact_timeline_step(
                        step, before, action, controller, scheduler
                    )
                )
        carry = one_step(
            carry, calendar, events, tables, template, jnp.int32(1)
        )
        jax.block_until_ready(carry)
        current = counter_tuple(carry)
        if current != prior:
            if inspected is None:
                inspected = inspect(before, calendar, events, tables, template)
            if args.compact_inspect:
                jax.block_until_ready(inspected)
                events_found.append(
                    {
                        "pre_step": step,
                        "counter_before": list(prior),
                        "counter_after": list(current),
                        "compact": host(inspected),
                    }
                )
                prior = current
                continue
            action, controller, scheduler, next_states, effects = inspected
            jax.block_until_ready((action, controller, next_states, effects))
            pre_state = host(before.farm.environment_state)
            post_state = host(next_states)
            action_dict = host(m35_player_action_dict_v2(action))
            events_found.append(
                {
                    "pre_step": step,
                    "post_step": int(np.asarray(next_states.step)[0]),
                    "counter_before": list(prior),
                    "counter_after": list(current),
                    "money_before": pre_state["money"][0],
                    "money_after": post_state["money"][0],
                    "shed_before": pre_state["shed"][0],
                    "shed_after": post_state["shed"][0],
                    "seeds_before": pre_state["seeds"][0],
                    "seeds_after": post_state["seeds"][0],
                    "market_price_before": pre_state["market_price"],
                    "market_inventory_before": pre_state["market_inventory"],
                    "action": action_dict,
                    "unit_tasks": compact_tasks(controller.unit_tasks),
                    "market_tasks": compact_market(controller.market_tasks),
                    "scheduler": host(scheduler),
                    "effects": host(effects),
                }
            )
        prior = current

    receipt = {
        "schema": "kaggriculture.m36c_seed_hard_trace.v1",
        "event_bank": str(args.event_bank.resolve()),
        "event_index": args.event_index,
        "event_seed": int(source_seeds[args.event_index]),
        "calendar_source": diagnostic.to_dict(),
        "final_counters": list(prior),
        "events": events_found,
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
