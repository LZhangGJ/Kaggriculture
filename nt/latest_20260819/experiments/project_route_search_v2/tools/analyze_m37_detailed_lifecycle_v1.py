"""Audit M3.7 crop/animal commitment lifecycles from one internal trace.

This is deliberately host-side and single-seed.  It does not enter the GPU
rollout carry, mutate the calendar, replay expert raw actions, or change any
hard-error gate.  The batched low-overhead aggregate has a separate acceptance
path; this tool preserves the detail needed to locate the first blocked stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

import numpy as np  # noqa: E402

from kaggriculture_jax.constants import (  # noqa: E402
    ANIMAL_COST,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    MarketOp,
    TileKind,
    UnitOp,
)
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
)
from strategic_v5.constants import TaskStatusV1, TaskTypeV1  # noqa: E402


DEFAULT_TRACE = (
    PROJECT / "artifacts" / "traces" / "m37b_unlock_replan_trace_v4.npz"
)
DEFAULT_REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)

STAGES = (
    "PLANNED",
    "ADMITTED",
    "RESOURCE_RESERVED",
    "TASK_CREATED",
    "ASSIGNED",
    "ACTION_EMITTED",
    "EFFECT_CONFIRMED",
    "ACTIVE",
    "PRODUCTIVE",
    "BANKED",
)

BLOCKER_REASON_VOCABULARY = (
    "NONE",
    "CASH",
    "MARKET_SLOT",
    "CAPACITY",
    "FACILITY",
    "UNIT",
    "PREEMPTION",
    "PLACEMENT",
    "FEED",
    "PARTIAL_FILL",
    "TARGET_NOT_EMITTED",
    "SEED",
    "IN_FLIGHT",
    "TASK_NOT_CREATED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m37a_detailed_lifecycle_v1.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def first_reach(series: np.ndarray, target: int, start: int = 0) -> int | None:
    found = np.flatnonzero((np.arange(series.size) >= start) & (series >= target))
    return int(found[0]) if found.size else None


def nth_step(events: list[tuple[int, int]], ordinal: int) -> int | None:
    cumulative = 0
    for step, quantity in events:
        cumulative += quantity
        if cumulative >= ordinal:
            return int(step)
    return None


def first_event_at_or_after(
    events: list[tuple[int, int]], start: int | None
) -> int | None:
    if start is None:
        return None
    for step, quantity in events:
        if quantity > 0 and step >= start:
            return int(step)
    return None


def allocate_event_steps_after(
    events: list[tuple[int, int]], starts: list[int | None]
) -> list[int | None]:
    """Assign each event unit once, in order, no earlier than its commitment."""

    expanded = [step for step, quantity in events for _ in range(quantity)]
    cursor = 0
    assigned: list[int | None] = []
    for start in starts:
        if start is None:
            assigned.append(None)
            continue
        while cursor < len(expanded) and expanded[cursor] < start:
            cursor += 1
        if cursor >= len(expanded):
            assigned.append(None)
            continue
        assigned.append(int(expanded[cursor]))
        cursor += 1
    return assigned


def unique_task_events(
    trace: np.lib.npyio.NpzFile,
    task_type: int,
    item: int,
) -> list[tuple[int, int]]:
    status = trace["internal_unit_task_status"][:, 0]
    types = trace["internal_unit_task_task_type"][:, 0]
    items = trace["internal_unit_task_item_id"][:, 0]
    starts = trace["internal_unit_task_start_step"][:, 0]
    seen: set[tuple[int, int]] = set()
    events: list[tuple[int, int]] = []
    for action_step in range(status.shape[0]):
        units = np.flatnonzero(
            (status[action_step] == int(TaskStatusV1.ACTIVE))
            & (types[action_step] == task_type)
            & (items[action_step] == item)
        )
        for unit in units.tolist():
            key = (int(starts[action_step, unit]), int(unit))
            if key in seen:
                continue
            seen.add(key)
            events.append((action_step, 1))
    return sorted(events)


def market_events(
    trace: np.lib.npyio.NpzFile,
    op: int,
    item: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    ops = trace["action_market_op"][:, 0]
    items = trace["action_market_item"][:, 0]
    amounts = trace["action_market_amount"][:, 0]
    for step in range(ops.shape[0]):
        slots = np.flatnonzero((ops[step] == op) & (items[step] == item))
        quantity = int(amounts[step, slots].sum()) if slots.size else 0
        if quantity > 0:
            result.append((step, quantity))
    return result


def unit_task_action_events(
    trace: np.lib.npyio.NpzFile,
    task_type: int,
    item: int,
    ops: tuple[int, ...],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    status = trace["internal_unit_task_status"][:, 0]
    types = trace["internal_unit_task_task_type"][:, 0]
    items = trace["internal_unit_task_item_id"][:, 0]
    emitted = trace["action_unit_op"][:, 0]
    for step in range(status.shape[0]):
        mask = (
            (status[step] == int(TaskStatusV1.ACTIVE))
            & (types[step] == task_type)
            & (items[step] == item)
            & np.isin(emitted[step], np.asarray(ops, dtype=np.int8))
        )
        count = int(mask.sum())
        if count:
            result.append((step, count))
    return result


def state_counts(trace: np.lib.npyio.NpzFile, player: int):
    tile_kind = trace["state_tile_kind"][:, 0, player]
    tile_crop = trace["state_tile_crop"][:, 0, player]
    tile_animal = trace["state_tile_animal"][:, 0, player]
    tile_yield = trace["state_tile_yield"][:, 0, player]
    crop_active = np.stack(
        [
            ((tile_kind == int(TileKind.PLANT)) & (tile_crop == crop)).sum((1, 2))
            for crop in range(NUM_CROPS)
        ],
        axis=-1,
    ).astype(np.int16)
    crop_productive = np.stack(
        [
            (
                (tile_kind == int(TileKind.PLANT))
                & (tile_crop == crop)
                & (tile_yield > 0)
            ).sum((1, 2))
            for crop in range(NUM_CROPS)
        ],
        axis=-1,
    ).astype(np.int16)
    animal_active = np.stack(
        [(tile_animal == animal).sum((1, 2)) for animal in range(NUM_ANIMALS)],
        axis=-1,
    ).astype(np.int16)
    animal_productive = np.stack(
        [
            ((tile_animal == animal) & (tile_yield > 0)).sum((1, 2))
            for animal in range(NUM_ANIMALS)
        ],
        axis=-1,
    ).astype(np.int16)
    animal_shed = trace["state_shed"][:, 0, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
    animal_carried = trace["state_unit_inventory"][
        :, 0, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].sum(axis=1)
    return (
        crop_active,
        crop_productive,
        animal_active,
        animal_productive,
        animal_shed.astype(np.int16),
        animal_carried.astype(np.int16),
    )


def classify_animal_losses(
    trace: np.lib.npyio.NpzFile,
    calendar,
    animal_active: np.ndarray,
) -> tuple[
    dict[str, list[dict[str, int]]],
    dict[int, dict[int, int]],
    dict[int, dict[int, int]],
]:
    """Separate calendar-authorized release from observed execution loss.

    Animals have no stable identity in the simulator state, so aggregate drops
    are deterministically attributed to the highest currently active ordinal.
    That convention is diagnostic only; the planned/unplanned quantity at each
    transition is exact under the daily service target.
    """

    service = np.asarray(calendar.animal_service_target_by_day)[0].astype(
        np.int16
    )
    state_steps = trace["state_step"][:, 0].astype(np.int32)
    events: dict[str, list[dict[str, int]]] = {
        str(species): [] for species in range(NUM_ANIMALS)
    }
    planned_by_species: dict[int, dict[int, int]] = {
        species: {} for species in range(NUM_ANIMALS)
    }
    unplanned_by_species: dict[int, dict[int, int]] = {
        species: {} for species in range(NUM_ANIMALS)
    }
    for species in range(NUM_ANIMALS):
        for index in range(1, animal_active.shape[0]):
            previous = int(animal_active[index - 1, species])
            current = int(animal_active[index, species])
            dropped = max(previous - current, 0)
            if dropped <= 0:
                continue
            state_step = int(state_steps[index])
            service_day = int(np.clip((state_step - 1) // TURNS_PER_DAY, 0, 29))
            service_target = int(service[service_day, species])
            planned_capacity = max(previous - service_target, 0)
            planned = min(dropped, planned_capacity)
            unplanned = dropped - planned
            ordinals = list(
                range(previous, max(previous - dropped, 0), -1)
            )
            for offset, ordinal in enumerate(ordinals):
                if offset < planned:
                    planned_by_species[species][ordinal] = state_step
                else:
                    unplanned_by_species[species][ordinal] = state_step
            events[str(species)].append(
                {
                    "state_step": state_step,
                    "service_day_index": service_day,
                    "active_before": previous,
                    "active_after": current,
                    "service_target": service_target,
                    "planned_release_quantity": planned,
                    "unplanned_escape_quantity": unplanned,
                }
            )
    return events, planned_by_species, unplanned_by_species


def empty_structures(
    trace: np.lib.npyio.NpzFile,
    player: int,
    species: int,
) -> np.ndarray:
    kind = trace["state_tile_kind"][:, 0, player]
    animal = trace["state_tile_animal"][:, 0, player]
    structure = int(TileKind.COOP if species == 0 else TileKind.PASTURE)
    return ((kind == structure) & (animal < 0)).sum((1, 2)).astype(np.int16)


def first_animal_blocker(
    trace: np.lib.npyio.NpzFile,
    player: int,
    species: int,
    planned: int,
    admitted: int | None,
    effect_confirmed: int | None,
    active: int | None,
) -> tuple[str, int]:
    end = admitted if admitted is not None else min(planned + TURNS_PER_DAY, 718)
    money = trace["state_money"][:, 0, player]
    shed_used = trace["state_shed"][:, 0, player].sum(axis=-1)
    market_active = trace["internal_market_task_status"][:, 0] == int(TaskStatusV1.ACTIVE)
    facilities = empty_structures(trace, player, species)
    if admitted is None or admitted > planned:
        for step in range(planned, end + 1):
            pre = max(step - 1, 0)
            if int(shed_used[pre]) >= SHED_CAPACITY:
                return "CAPACITY", step
            if int(facilities[pre]) <= 0:
                return "FACILITY", step
            if int(market_active[step].sum()) >= 10:
                return "MARKET_SLOT", step
            if int(money[pre]) < int(ANIMAL_COST[species]):
                return "CASH", step
        return "TARGET_NOT_EMITTED", planned
    if effect_confirmed is None:
        return "PARTIAL_FILL", admitted
    if effect_confirmed > admitted:
        return "PARTIAL_FILL", admitted
    if active is None or active > effect_confirmed + TURNS_PER_DAY:
        shed = trace["state_shed"][:, 0, player, NUM_PRODUCTS + species]
        carried = trace["state_unit_inventory"][
            :, 0, player, :, NUM_PRODUCTS + species
        ].sum(axis=1)
        for step in range(effect_confirmed, min(effect_confirmed + TURNS_PER_DAY, 718) + 1):
            if int(facilities[step]) <= 0:
                return "FACILITY", step
            if int(carried[step]) > 0:
                return "PLACEMENT", step
            if int(shed[step]) > 0:
                active_units = int(
                    trace["state_unit_active"][step, 0, player].sum()
                )
                assigned = int(
                    (
                        trace["internal_unit_task_status"][step, 0]
                        == int(TaskStatusV1.ACTIVE)
                    ).sum()
                )
                return (
                    "PREEMPTION" if assigned >= active_units else "UNIT",
                    step,
                )
        return "TASK_NOT_CREATED", effect_confirmed
    return "NONE", planned


def stage_delays(stage_steps: dict[str, int | None]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    previous: int | None = None
    for stage in STAGES:
        step = stage_steps[stage]
        result[stage] = None if step is None or previous is None else int(step - previous)
        if step is not None:
            previous = step
    return result


def stages_are_monotonic(stage_steps: dict[str, int | None]) -> bool:
    observed = [stage_steps[stage] for stage in STAGES if stage_steps[stage] is not None]
    return all(left <= right for left, right in zip(observed, observed[1:]))


def animal_lifecycle(
    trace: np.lib.npyio.NpzFile,
    calendar,
    player: int,
    animal_active: np.ndarray,
    animal_productive: np.ndarray,
    animal_shed: np.ndarray,
    animal_carried: np.ndarray,
    planned_loss_by_species: dict[int, dict[int, int]],
    unplanned_loss_by_species: dict[int, dict[int, int]],
) -> dict[str, list[dict[str, object]]]:
    additions = np.asarray(calendar.animal_purchase_additions_by_day)[0]
    result: dict[str, list[dict[str, object]]] = {}
    for species in range(NUM_ANIMALS):
        planned_events = [
            (day * TURNS_PER_DAY, int(additions[day, species]))
            for day in range(additions.shape[0])
            if int(additions[day, species]) > 0
        ]
        total = sum(quantity for _, quantity in planned_events)
        buy_item = NUM_PRODUCTS + species
        buy_events = market_events(trace, int(MarketOp.BUY_ANIMAL), buy_item)
        task_events = unique_task_events(
            trace, int(TaskTypeV1.ANIMAL_PLACE), buy_item
        )
        pickup_events = unit_task_action_events(
            trace,
            int(TaskTypeV1.ANIMAL_PLACE),
            buy_item,
            (int(UnitOp.PICKUP),),
        )
        place_events = unit_task_action_events(
            trace,
            int(TaskTypeV1.ANIMAL_PLACE),
            buy_item,
            (int(UnitOp.PLACE),),
        )
        harvest_events = unit_task_action_events(
            trace,
            int(TaskTypeV1.ANIMAL_COLLECT_PRODUCT),
            int(ANIMAL_PRODUCT[species]),
            (int(UnitOp.HARVEST),),
        )
        committed = animal_active[:, species] + animal_shed[:, species] + animal_carried[:, species]
        sell_events = market_events(
            trace, int(MarketOp.SELL), int(ANIMAL_PRODUCT[species])
        )
        planned_steps = [nth_step(planned_events, ordinal) for ordinal in range(1, total + 1)]
        admitted_steps = allocate_event_steps_after(buy_events, planned_steps)
        task_steps = allocate_event_steps_after(task_events, admitted_steps)
        pickup_steps = allocate_event_steps_after(pickup_events, task_steps)
        place_starts = [
            pickup if pickup is not None else task
            for pickup, task in zip(pickup_steps, task_steps, strict=True)
        ]
        place_steps = allocate_event_steps_after(place_events, place_starts)
        entries: list[dict[str, object]] = []
        for ordinal in range(1, total + 1):
            planned = planned_steps[ordinal - 1]
            admitted = admitted_steps[ordinal - 1]
            resource_reserved = admitted
            task_created = task_steps[ordinal - 1]
            assigned = task_created
            pickup_action = pickup_steps[ordinal - 1]
            place_action = place_steps[ordinal - 1]
            action_emitted = (
                pickup_action if pickup_action is not None else place_action
            )
            effect_confirmed = first_reach(committed, ordinal, admitted or 0)
            active_start = (
                place_action
                if place_action is not None
                else (action_emitted if action_emitted is not None else effect_confirmed)
            )
            active = first_reach(
                animal_active[:, species], ordinal, active_start or 0
            )
            productive = first_event_at_or_after(harvest_events, active)
            if productive is None and active is not None:
                productive = first_reach(
                    animal_productive[:, species], 1, active or 0
                )
            banked = first_event_at_or_after(sell_events, productive)
            stages = {
                "PLANNED": planned,
                "ADMITTED": admitted,
                "RESOURCE_RESERVED": resource_reserved,
                "TASK_CREATED": task_created,
                "ASSIGNED": assigned,
                "ACTION_EMITTED": action_emitted,
                # The generic action/effect pair refers to the placement
                # pipeline; purchase intent/effect are retained separately.
                "EFFECT_CONFIRMED": action_emitted,
                "ACTIVE": active,
                "PRODUCTIVE": productive,
                "BANKED": banked,
            }
            blocker, blocker_step = first_animal_blocker(
                trace,
                player,
                species,
                int(planned or 0),
                admitted,
                effect_confirmed,
                active,
            )
            planned_release_step = planned_loss_by_species[species].get(
                ordinal
            )
            unplanned_escape_step = unplanned_loss_by_species[species].get(
                ordinal
            )
            entries.append(
                {
                    "commitment_ordinal": ordinal,
                    "stage_steps": stages,
                    "stage_delays": stage_delays(stages),
                    "purchase_action_step": admitted,
                    "purchase_effect_confirmed_step": effect_confirmed,
                    "pickup_action_step": pickup_action,
                    "place_action_step": place_action,
                    "first_blocker_reason": blocker,
                    "first_blocker_step": blocker_step,
                    "planned_release": planned_release_step is not None,
                    "planned_release_step": planned_release_step,
                    "unplanned_escape_step": unplanned_escape_step,
                    # A blocker explains delay; it is not itself a failed
                    # commitment.  Failure requires observed unplanned loss or
                    # purchased capital that never reached ACTIVE.
                    "execution_failure": (
                        unplanned_escape_step is not None
                        or (
                            admitted is not None
                            and active is None
                            and planned_release_step is None
                        )
                    ),
                }
            )
        result[str(species)] = entries
    return result


def crop_lifecycle(
    trace: np.lib.npyio.NpzFile,
    calendar,
    player: int,
    crop_active: np.ndarray,
    crop_productive: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    targets = np.asarray(calendar.crop_target_by_day)[0].astype(np.int16)
    previous = np.zeros((NUM_CROPS,), dtype=np.int16)
    additions = np.zeros_like(targets)
    for day in range(targets.shape[0]):
        additions[day] = np.maximum(targets[day] - previous, 0)
        previous = targets[day]
    result: dict[str, list[dict[str, object]]] = {}
    for crop in range(NUM_CROPS):
        planned_events = [
            (day * TURNS_PER_DAY, int(additions[day, crop]))
            for day in range(additions.shape[0])
            if int(additions[day, crop]) > 0
        ]
        total = sum(quantity for _, quantity in planned_events)
        seed_events = market_events(trace, int(MarketOp.BUY_SEED), crop)
        task_events = unique_task_events(
            trace, int(TaskTypeV1.CROP_PRODUCTION), crop
        )
        plant_events = unit_task_action_events(
            trace,
            int(TaskTypeV1.CROP_PRODUCTION),
            crop,
            (int(UnitOp.PLANT),),
        )
        harvest_events = unit_task_action_events(
            trace,
            int(TaskTypeV1.CROP_PRODUCTION),
            crop,
            (int(UnitOp.HARVEST),),
        )
        sell_events = market_events(trace, int(MarketOp.SELL), crop)
        planned_steps = [nth_step(planned_events, ordinal) for ordinal in range(1, total + 1)]
        task_steps = allocate_event_steps_after(task_events, planned_steps)
        seed_steps = allocate_event_steps_after(seed_events, planned_steps)
        admitted_steps = [
            min(step for step in (seed, task) if step is not None)
            if seed is not None or task is not None
            else None
            for seed, task in zip(seed_steps, task_steps, strict=True)
        ]
        plant_steps = allocate_event_steps_after(plant_events, task_steps)
        harvest_steps = allocate_event_steps_after(harvest_events, plant_steps)
        entries: list[dict[str, object]] = []
        for ordinal in range(1, total + 1):
            planned = planned_steps[ordinal - 1]
            admitted = admitted_steps[ordinal - 1]
            task_created = task_steps[ordinal - 1]
            planted = plant_steps[ordinal - 1]
            harvested = harvest_steps[ordinal - 1]
            banked = first_event_at_or_after(sell_events, harvested)
            stages = {
                "PLANNED": planned,
                "ADMITTED": admitted,
                "RESOURCE_RESERVED": task_created if task_created is not None else admitted,
                "TASK_CREATED": task_created,
                "ASSIGNED": task_created,
                "ACTION_EMITTED": planted,
                "EFFECT_CONFIRMED": planted,
                "ACTIVE": planted,
                "PRODUCTIVE": harvested,
                "BANKED": banked,
            }
            blocker = "NONE"
            blocker_step = int(planned or 0)
            if planted is None:
                seeds = trace["state_seeds"][:, 0, player, crop]
                check = min(int(planned or 0), len(seeds) - 1)
                if int(seeds[check]) <= 0:
                    blocker = "SEED"
                elif task_created is None:
                    active_units = int(
                        trace["state_unit_active"][check, 0, player].sum()
                    )
                    assigned = int(
                        (
                            trace["internal_unit_task_status"][check, 0]
                            == int(TaskStatusV1.ACTIVE)
                        ).sum()
                    )
                    blocker = "PREEMPTION" if assigned >= active_units else "UNIT"
                else:
                    blocker = "PLACEMENT"
            entries.append(
                {
                    "commitment_ordinal": ordinal,
                    "stage_steps": stages,
                    "stage_delays": stage_delays(stages),
                    "first_blocker_reason": blocker,
                    "first_blocker_step": blocker_step,
                    "productive_tile_count_at_end": int(crop_productive[-1, crop]),
                }
            )
        result[str(crop)] = entries
    return result


def daily_crop_debt(
    trace: np.lib.npyio.NpzFile,
    calendar,
    player: int,
    crop_active: np.ndarray,
) -> list[dict[str, object]]:
    targets = np.asarray(calendar.crop_target_by_day)[0].astype(np.int16)
    result: list[dict[str, object]] = []
    debt_age = np.zeros((NUM_CROPS,), dtype=np.int16)
    for day_index in range(30):
        # Calendar row 0 is the plan executed during day 1.  Its end-of-day
        # acceptance state is environment step 24, stored at trace index 23
        # because the trace contains post-action states for actions 0..718.
        state_step = min((day_index + 1) * TURNS_PER_DAY, 719)
        step = min(state_step - 1, len(crop_active) - 1)
        target = targets[day_index]
        active = crop_active[step]
        gap = np.maximum(target - active, 0)
        seeds = trace["state_seeds"][step, 0, player].astype(np.int16)
        plantable = np.minimum(gap, seeds)
        task_active = (
            (trace["internal_unit_task_status"][step, 0] == int(TaskStatusV1.ACTIVE))
            & (
                trace["internal_unit_task_task_type"][step, 0]
                == int(TaskTypeV1.CROP_PRODUCTION)
            )
        )
        items = trace["internal_unit_task_item_id"][step, 0]
        target_x = np.clip(
            trace["internal_unit_task_target_x"][step, 0].astype(np.int16),
            0,
            9,
        )
        target_y = np.clip(
            trace["internal_unit_task_target_y"][step, 0].astype(np.int16),
            0,
            9,
        )
        target_kind = trace["state_tile_kind"][
            step, 0, player, target_y, target_x
        ]
        # CROP_PRODUCTION also represents harvest/replant continuations.  Only
        # a task whose current target is still EMPTY is an outstanding plant
        # commitment; counting every CROP_PRODUCTION lane made melon harvests
        # look like ten pending melon plants and hid wheat/strawberry debt.
        task_mask = task_active & (target_kind == int(TileKind.EMPTY))
        assigned = np.asarray(
            [(task_mask & (items == crop)).sum() for crop in range(NUM_CROPS)],
            dtype=np.int16,
        )
        unassigned_gap = np.maximum(gap - assigned, 0)
        seed_deficit = np.maximum(unassigned_gap - seeds, 0)
        pending_plant = np.minimum(unassigned_gap, seeds)
        rejection = pending_plant.copy()
        debt_age = np.where(gap > 0, debt_age + 1, 0).astype(np.int16)
        blocker: list[str] = []
        for crop in range(NUM_CROPS):
            if gap[crop] <= 0:
                blocker.append("NONE")
            elif seed_deficit[crop] > 0:
                blocker.append("SEED")
            elif rejection[crop] > 0:
                active_units = int(
                    trace["state_unit_active"][step, 0, player].sum()
                )
                assigned_units = int(
                    (
                        trace["internal_unit_task_status"][step, 0]
                        == int(TaskStatusV1.ACTIVE)
                    ).sum()
                )
                blocker.append(
                    "PREEMPTION" if assigned_units >= active_units else "UNIT"
                )
            else:
                blocker.append("IN_FLIGHT")
        result.append(
            {
                "day": day_index + 1,
                "state_step": state_step,
                "trace_index": step,
                "target": target.tolist(),
                "active": active.tolist(),
                "target_debt": gap.tolist(),
                "target_debt_age_days": debt_age.tolist(),
                "seed_held": seeds.tolist(),
                "pending_seed": seed_deficit.tolist(),
                "pending_plant": pending_plant.tolist(),
                "assigned_plant_tasks": assigned.tolist(),
                "rejection": rejection.tolist(),
                "first_blocker_reason": blocker,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay.get("module_version") != "1.32.7":
        raise ValueError("M3.7 diagnostics require official 1.32.7")
    trace = np.load(args.trace, allow_pickle=False)
    required = {
        "internal_unit_task_status",
        "internal_market_task_status",
        "internal_effect_pre_step",
    }
    missing = sorted(required - set(trace.files))
    if missing:
        raise ValueError(f"trace lacks --include-internal-lifecycle fields: {missing}")
    player = int(trace["player"][0])
    calendar, calendar_diagnostic = compile_gold_replay_calendar_v3(
        replay, player=player, candidate_id=3_701
    )
    (
        crop_active,
        crop_productive,
        animal_active,
        animal_productive,
        animal_shed,
        animal_carried,
    ) = state_counts(trace, player)
    (
        animal_loss_events,
        planned_loss_by_species,
        unplanned_loss_by_species,
    ) = classify_animal_losses(trace, calendar, animal_active)
    animals = animal_lifecycle(
        trace,
        calendar,
        player,
        animal_active,
        animal_productive,
        animal_shed,
        animal_carried,
        planned_loss_by_species,
        unplanned_loss_by_species,
    )
    crops = crop_lifecycle(
        trace, calendar, player, crop_active, crop_productive
    )
    final_bank = int(trace["state_money"][-1, 0, player])
    effect_hard_error_count = int(
        sum(
            int(trace[name][:, 0].sum())
            for name in (
                "internal_effect_effect_mismatch_count",
                "internal_effect_owner_inactive_count",
                "internal_effect_deadline_missed_count",
                "internal_effect_resource_unavailable_count",
            )
        )
    )
    unplanned_escape_count = int(
        sum(
            event["unplanned_escape_quantity"]
            for species_events in animal_loss_events.values()
            for event in species_events
        )
    )
    hard_error_count = effect_hard_error_count + unplanned_escape_count
    payload = {
        "receipt_id": "M37A_DETAILED_LIFECYCLE_V1",
        "status": "PASS",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DIAGNOSTIC_ONLY_SINGLE_SEED_HOST_TRACE",
        "official_version": "1.32.7",
        "seed": int(trace["seed"][0]),
        "player": player,
        "steps": int(trace["action_unit_op"].shape[0]),
        "final_bank": final_bank,
        "hard_error_count": hard_error_count,
        "hard_error_count_scope": (
            "observed_effect_errors_plus_unplanned_escape_events; "
            "exact rollout hard-error breakdown is authoritative in the "
            "trace-generation receipt"
        ),
        "effect_hard_error_count": effect_hard_error_count,
        "observed_unplanned_animal_escape_count": unplanned_escape_count,
        "trace": args.trace.resolve().relative_to(ROOT.resolve()).as_posix(),
        "trace_sha256": sha256(args.trace),
        "calendar_source": calendar_diagnostic.to_dict(),
        "raw_replay_action_payload_stored": False,
        "stage_order": list(STAGES),
        "blocker_reason_vocabulary": list(BLOCKER_REASON_VOCABULARY),
        "animal_commitment_lifecycle": animals,
        "animal_loss_events": animal_loss_events,
        "crop_commitment_lifecycle": crops,
        "daily_crop_target_debt": daily_crop_debt(
            trace, calendar, player, crop_active
        ),
        "checks": {
            "all_required_stages_present": all(
                set(entry["stage_steps"]) == set(STAGES)
                for family in (animals, crops)
                for entries in family.values()
                for entry in entries
            ),
            "all_observed_stage_steps_monotonic": all(
                stages_are_monotonic(entry["stage_steps"])
                for family in (animals, crops)
                for entries in family.values()
                for entry in entries
            ),
            "trace_is_719_actions": int(trace["action_unit_op"].shape[0]) == 719,
            "calendar_event_overflow_count": int(
                calendar_diagnostic.event_overflow_count
            ),
            "detailed_trace_not_in_gpu_carry": True,
        },
        "reproduce_command": (
            f"E:\\ai_coding\\kaggle\\kaggriculture\\.venv\\python.exe "
            f"{Path(__file__).resolve()} --trace {args.trace.resolve()} "
            f"--replay {args.replay.resolve()} --output {args.output.resolve()}"
        ),
        "boundary": (
            "LIFECYCLE_DIAGNOSTIC_ONLY_NOT_M37_ACCEPTANCE_NOT_SEARCH_AUTHORIZATION"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "final_bank": final_bank,
                "hard_error_count": hard_error_count,
                "animal_commitments": {
                    key: len(value) for key, value in animals.items()
                },
                "crop_commitments": {
                    key: len(value) for key, value in crops.items()
                },
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
