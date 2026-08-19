"""Generate the exact split-kernel M3.6 controller trace for official replay.

The trace contains the controller's own actions and every post-action JAX
state.  It never stores or plays the gold player's raw Replay actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
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

from kaggriculture_jax.constants import (  # noqa: E402
    NUM_DAYS,
    SHOP_NAMES,
    TURNS_PER_DAY,
)
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events, State  # noqa: E402
from project_route_search_v2.constants import ProjectStatusV2  # noqa: E402
from project_route_search_v2.m35_controller import (  # noqa: E402
    m35_player_action_dict_v2,
    update_m35_controller_from_effects_v2,
)
from project_route_search_v2.m35_genome import (  # noqa: E402
    default_m35_farm_genome_v2,
)
from project_route_search_v2.m36_controller import (  # noqa: E402
    m36c_light_policy_step_v3,
)
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
)
from project_route_search_v2.m36_rollout import (  # noqa: E402
    summarize_m36c_rollout_v3,
)
from project_route_search_v2.m36_split_rollout import (  # noqa: E402
    M36_MARKET_PLAN_HOURS_V3,
    initialize_m36c_split_v3,
    make_m36c_light_chunk_v3,
    m36c_daily_plan_carry_v3,
)
from project_route_search_v2.m36_transaction import (  # noqa: E402
    m36_player_action_dict_v3,
    m36a_policy_step_v3,
)
from project_route_search_v2.m38_route_cards import RouteCardModeV4  # noqa: E402
from project_route_search_v2.null_opponent import (  # noqa: E402
    combine_with_null_opponent,
)
from strategic_v5.constants import TaskStatusV1  # noqa: E402


REPLAY = (
    ROOT
    / "replay"
    / "gold_latest_2026-08-19_154928"
    / "latest_per_gold"
    / "episode-94424738-replay.json"
)
MILESTONE_STEPS = (1, 6 * 24, 11 * 24, 12 * 24, 20 * 24, 25 * 24, 29 * 24, 719)
EVENT_DRAWS = 200
WEED_CHANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--include-internal-lifecycle",
        action="store_true",
        help="Store controller/task/project/scheduler snapshots for one-seed diagnostics.",
    )
    parser.add_argument(
        "--enable-m37-unlock",
        action="store_true",
        help="Enable the value-gated M3.7 committed-capital scheduler arm.",
    )
    parser.add_argument(
        "--enable-route-cards",
        action="store_true",
        help="Enable bounded FEED/HARVEST route cards while retaining primitive execution.",
    )
    parser.add_argument(
        "--route-card-mode",
        choices=[mode.name.lower() for mode in RouteCardModeV4],
        default=RouteCardModeV4.LEGACY_M38.name.lower(),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=PROJECT / "artifacts" / "traces" / "m36d_controller_trace_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m36d_controller_trace_generation_v1.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_events(seed: int) -> Events:
    weed = np.zeros((1, NUM_DAYS, EVENT_DRAWS), dtype=np.bool_)
    choice = np.zeros((1, NUM_DAYS, EVENT_DRAWS + 1), dtype=np.int8)
    for day in range(NUM_DAYS):
        rng = random.Random((seed * 1_000_003) ^ day)
        for consumed in range(EVENT_DRAWS + 1):
            clone = random.Random()
            clone.setstate(rng.getstate())
            choice[0, day, consumed] = SHOP_NAMES.index(clone.choice(SHOP_NAMES))
            if consumed < EVENT_DRAWS:
                weed[0, day, consumed] = rng.random() < WEED_CHANCE
    return Events(jnp.asarray(weed), jnp.asarray(choice))


def player_action(data: dict[str, jax.Array]) -> Action:
    return Action(*(data[name] for name in Action._fields))


def inspect_step(
    carry,
    calendar,
    events,
    tables,
    template,
    *,
    player: int,
    enable_m37_unlock: bool = False,
    enable_route_cards: bool = False,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
):
    states = carry.farm.environment_state
    action, controller, genome, scheduler = m36c_light_policy_step_v3(
        states,
        carry.farm.controller,
        calendar,
        template,
        player,
        enable_unlock_committed_capital=enable_m37_unlock,
        enable_route_cards=enable_route_cards,
        route_card_mode=route_card_mode,
    )
    joint = combine_with_null_opponent(
        m35_player_action_dict_v2(action), player_seat=player
    )
    next_states = batched_step_sync(states, joint, events, tables)
    _, effects = update_m35_controller_from_effects_v2(
        states, next_states, controller, action, genome, player
    )
    return action, controller, scheduler, next_states, effects


def tree_equal(left, right) -> bool:
    left = jax.device_get(left)
    right = jax.device_get(right)
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def backlog(controller, state: State, calendar, player: int) -> dict[str, object]:
    controller, state, calendar = jax.device_get((controller, state, calendar))
    step = int(state.step[0])
    day = min(step // TURNS_PER_DAY, NUM_DAYS - 1)
    unit_status = np.asarray(controller.unit_tasks.status[0])
    market_status = np.asarray(controller.market_tasks.status[0])
    projects = np.asarray(controller.projects.status[0])
    active_units = unit_status == int(TaskStatusV1.ACTIVE)
    active_market = market_status == int(TaskStatusV1.ACTIVE)
    active_projects = (projects == int(ProjectStatusV2.PLANNED)) | (
        projects == int(ProjectStatusV2.ACTIVE)
    )
    route_status = np.asarray(controller.route_cards.status[0])
    crop = np.asarray(state.tile_crop[0, player])
    animal = np.asarray(state.tile_animal[0, player])
    crop_actual = np.asarray([(crop == item).sum() for item in range(5)], dtype=np.int32)
    animal_actual = np.asarray(
        [(animal == item).sum() for item in range(3)], dtype=np.int32
    )
    crop_target = np.asarray(calendar.crop_target_by_day[0, day], dtype=np.int32)
    animal_target = np.asarray(
        calendar.animal_service_target_by_day[0, day], dtype=np.int32
    )
    task_types = np.asarray(controller.unit_tasks.task_type[0], dtype=np.int32)
    return {
        "active_projects": int(active_projects.sum()),
        "active_unit_tasks": int(active_units.sum()),
        "active_market_tasks": int(active_market.sum()),
        "active_route_cards": int(
            ((route_status >= 1) & (route_status <= 3)).sum()
        ),
        "unit_task_type_counts": {
            str(task): int((active_units & (task_types == task)).sum())
            for task in np.unique(task_types[active_units]).tolist()
        },
        "unmet_crop_target": np.maximum(crop_target - crop_actual, 0).tolist(),
        "unmet_animal_service_target": np.maximum(
            animal_target - animal_actual, 0
        ).tolist(),
        "project_cap_hits": int(controller.project_cap_hits[0]),
        "obligation_cap_hits": int(controller.obligation_cap_hits[0]),
        "unexplained_effect_failures": int(
            controller.unexplained_effect_failures[0]
        ),
    }


def append_action(storage: dict[str, list[np.ndarray]], action: Action) -> None:
    action = jax.device_get(action)
    for name, value in zip(Action._fields, action, strict=True):
        storage[name].append(np.asarray(value))


def append_state(storage: dict[str, list[np.ndarray]], state: State) -> None:
    state = jax.device_get(state)
    for name, value in zip(State._fields, state, strict=True):
        storage[name].append(np.asarray(value))


def append_internal_controller(
    storage: dict[str, list[np.ndarray]], controller
) -> None:
    """Append only controller fields needed to locate a lifecycle blocker.

    This host-side trace is intentionally excluded from the batched rollout
    carry.  It is a single-seed diagnostic and therefore cannot reduce GPU
    search throughput.
    """

    controller = jax.device_get(controller)
    for prefix, record in (
        ("project", controller.projects),
        ("unit_task", controller.unit_tasks),
        ("route_card", controller.route_cards),
        ("market_task", controller.market_tasks),
    ):
        for name, value in zip(record._fields, record, strict=True):
            storage.setdefault(f"{prefix}_{name}", []).append(np.asarray(value))
    for name in (
        "project_cap_hits",
        "obligation_cap_hits",
        "unexplained_effect_failures",
    ):
        storage.setdefault(f"controller_{name}", []).append(
            np.asarray(getattr(controller, name))
        )


def append_internal_record(
    storage: dict[str, list[np.ndarray]], prefix: str, record
) -> None:
    record = jax.device_get(record)
    if hasattr(record, "_fields"):
        for name in record._fields:
            append_internal_record(
                storage, f"{prefix}_{name}", getattr(record, name)
            )
        return
    storage.setdefault(prefix, []).append(np.asarray(record))


def append_internal_scheduler(
    storage: dict[str, list[np.ndarray]], scheduler, *, batch_size: int
) -> None:
    if scheduler is None:
        # Opening action is generated by M3.6A and has no unified scheduler.
        # A zero row keeps every internal array aligned to the 719 actions.
        for name in (
            "sticky_task_count",
            "crop_candidate_count",
            "animal_candidate_count",
            "hard_candidate_count",
            "selected_crop_count",
            "selected_animal_count",
            "deadline_preemption_count",
            "duplicate_reservation_prevented",
            "local_swap_count",
            "planned_release_tile_count",
        ):
            storage.setdefault(f"scheduler_{name}", []).append(
                np.zeros((batch_size,), dtype=np.int32)
            )
        return
    scheduler = jax.device_get(scheduler)
    for name, value in zip(scheduler._fields, scheduler, strict=True):
        storage.setdefault(f"scheduler_{name}", []).append(np.asarray(value))


def main() -> None:
    args = parse_args()
    if args.player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay.get("module_version") != "1.32.7":
        raise ValueError("M3.6D requires an official 1.32.7 Replay")
    seed = int(replay["info"]["seed"])
    calendar, calendar_diagnostic = compile_gold_replay_calendar_v3(
        replay, player=args.player, candidate_id=3_600
    )
    seeds = jnp.asarray((seed,), dtype=jnp.int32)
    events = build_events(seed)
    tables = load_tables()
    template = default_m35_farm_genome_v2(1)

    opening_fn = jax.jit(
        lambda s, c, e, t, g: initialize_m36c_split_v3(
            s, c, e, t, g, player=args.player
        )
    )
    daily_fn = jax.jit(
        lambda carry, c, t, g: m36c_daily_plan_carry_v3(
            carry,
            c,
            t,
            g,
            player=args.player,
            enable_unlock_committed_capital=args.enable_m37_unlock,
        )
    )
    one_step_fn = jax.jit(
        make_m36c_light_chunk_v3(
            chunk_steps=24,
            player=args.player,
            enable_unlock_committed_capital=args.enable_m37_unlock,
            enable_route_cards=args.enable_route_cards,
            route_card_mode=int(RouteCardModeV4[args.route_card_mode.upper()]),
        )
    )
    inspect_fn = jax.jit(
        lambda carry, c, e, t, g: inspect_step(
            carry,
            c,
            e,
            t,
            g,
            player=args.player,
            enable_m37_unlock=args.enable_m37_unlock,
            enable_route_cards=args.enable_route_cards,
            route_card_mode=int(RouteCardModeV4[args.route_card_mode.upper()]),
        )
    )

    started = time.perf_counter()
    reset_states = jax.vmap(reset)(seeds)
    opening, _ = m36a_policy_step_v3(reset_states, calendar, args.player)
    opening_action = player_action(m36_player_action_dict_v3(opening))
    opening_joint = combine_with_null_opponent(
        opening_action._asdict(), player_seat=args.player
    )
    direct_opening_state = batched_step_sync(
        reset_states, opening_joint, events, tables
    )
    carry = opening_fn(seeds, calendar, events, tables, template)
    jax.block_until_ready(carry)
    if not tree_equal(direct_opening_state, carry.farm.environment_state):
        raise AssertionError("opening trace state differs from split initializer")

    action_storage = {name: [] for name in Action._fields}
    state_storage = {name: [] for name in State._fields}
    internal_storage: dict[str, list[np.ndarray]] = {}
    effect_storage: dict[str, list[np.ndarray]] = {}
    effect_pre_steps: list[int] = []
    append_action(action_storage, opening_action)
    append_state(state_storage, carry.farm.environment_state)
    if args.include_internal_lifecycle:
        append_internal_controller(internal_storage, carry.farm.controller)
        append_internal_scheduler(internal_storage, None, batch_size=1)
    milestone_backlog = {
        "1": backlog(
            carry.farm.controller,
            carry.farm.environment_state,
            calendar,
            args.player,
        )
    }
    internal_state_mismatch_count = 0

    while int(carry.farm.environment_state.step[0]) < 719:
        step = int(carry.farm.environment_state.step[0])
        market_plan_hours = (
            (0, 1, 20) if args.enable_m37_unlock else M36_MARKET_PLAN_HOURS_V3
        )
        if step >= 1 and step % TURNS_PER_DAY in market_plan_hours:
            carry = daily_fn(carry, calendar, tables, template)
        inspected = inspect_fn(carry, calendar, events, tables, template)
        action, planned_controller, scheduler, inspected_next_state, effects = inspected
        carry = one_step_fn(
            carry, calendar, events, tables, template, jnp.int32(1)
        )
        jax.block_until_ready((inspected, carry))
        if not tree_equal(inspected_next_state, carry.farm.environment_state):
            internal_state_mismatch_count += 1
        append_action(action_storage, player_action(m35_player_action_dict_v2(action)))
        append_state(state_storage, carry.farm.environment_state)
        if args.include_internal_lifecycle:
            append_internal_controller(internal_storage, planned_controller)
            append_internal_scheduler(internal_storage, scheduler, batch_size=1)
            append_internal_record(effect_storage, "effect", effects)
            effect_pre_steps.append(step)
        state_step = int(carry.farm.environment_state.step[0])
        if state_step in MILESTONE_STEPS:
            milestone_backlog[str(state_step)] = backlog(
                carry.farm.controller,
                carry.farm.environment_state,
                calendar,
                args.player,
            )

    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m36c_rollout_v3(carry, player=args.player))
    payload: dict[str, np.ndarray] = {
        "seed": np.asarray((seed,), dtype=np.int64),
        "player": np.asarray((args.player,), dtype=np.int8),
        "milestone_steps": np.asarray(MILESTONE_STEPS, dtype=np.int16),
        "milestone_backlog_json": np.asarray(
            (json.dumps(milestone_backlog, sort_keys=True),), dtype="U8192"
        ),
    }
    for name, values in action_storage.items():
        payload[f"action_{name}"] = np.stack(values, axis=0)
    for name, values in state_storage.items():
        payload[f"state_{name}"] = np.stack(values, axis=0)
    for name, values in internal_storage.items():
        payload[f"internal_{name}"] = np.stack(values, axis=0)
    if effect_storage:
        payload["internal_effect_pre_step"] = np.asarray(
            effect_pre_steps, dtype=np.int16
        )
        for name, values in effect_storage.items():
            payload[f"internal_{name}"] = np.stack(values, axis=0)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace, **payload)

    receipt = {
        "receipt_id": "M36D_CONTROLLER_TRACE_GENERATION_V1",
        "status": "PASS"
        if internal_state_mismatch_count == 0
        and int(summary.hard_error_count[0]) == 0
        else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "official_version": replay["module_version"],
        "episode_id": int(replay["info"]["EpisodeId"]),
        "seed": seed,
        "player": args.player,
        "opponent": "NullOpponent",
        "calendar_source": calendar_diagnostic.to_dict(),
        "raw_replay_action_payload_stored": False,
        "internal_lifecycle_trace": bool(args.include_internal_lifecycle),
        "m37_unlock_committed_capital": bool(args.enable_m37_unlock),
        "route_cards_enabled": bool(args.enable_route_cards),
        "route_card_mode": args.route_card_mode,
        "steps": 719,
        "internal_state_mismatch_count": internal_state_mismatch_count,
        "hard_error_count": int(summary.hard_error_count[0]),
        "hard_error_breakdown": {
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
            "unplanned_animal_escape": int(
                summary.unplanned_animal_escape[0]
            ),
            "fertilizer_market_slot_overflow_count": int(
                summary.fertilizer_market_slot_overflow_count[0]
            ),
            "plant_without_same_day_water": int(
                summary.farm.plant_without_same_day_water[0]
            ),
        },
        "final_bank": int(summary.farm.final_bank[0]),
        "milestone_backlog": milestone_backlog,
        "compile_and_execute_seconds": elapsed,
        "trace": args.trace.resolve().relative_to(ROOT.resolve()).as_posix(),
        "trace_sha256": sha256(args.trace),
        "trace_bytes": args.trace.stat().st_size,
        "boundary": "M36D_JAX_CONTROLLER_TRACE_NULL_OPPONENT_AWAITING_OFFICIAL_1327_REPLAY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
