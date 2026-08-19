"""Build a host-side M3.7 lifecycle baseline from an accepted M3.6 trace.

This is diagnostic-only: it does not change the simulator, controller carry,
calendar schema, or action trace.  It converts physical states and emitted
actions into daily commitment evidence so the first blocked expansion stage is
known before scheduler behavior is changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    ANIMAL_STRUCTURE,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_DAYS,
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


DEFAULT_TRACE = (
    PROJECT / "artifacts" / "traces" / "m36d_controller_trace_v8.npz"
)
DEFAULT_REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m37a_lifecycle_baseline_v1.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def counts(values: np.ndarray, size: int) -> np.ndarray:
    return np.asarray([(values == item).sum() for item in range(size)], dtype=np.int16)


def first_reach(series: np.ndarray, quantity: int) -> int | None:
    found = np.flatnonzero(series >= quantity)
    return int(found[0]) if found.size else None


def animal_blocker(
    *,
    desired: int,
    active: int,
    in_shed: int,
    carried: int,
    empty_structure: int,
    animal_cost: int,
    cash: int,
    shed_used: int,
    active_units: int,
    emitted_buy: bool,
    emitted_build: bool,
    emitted_pickup: bool,
    emitted_place: bool,
) -> str:
    owned = active + in_shed + carried
    if active >= desired:
        return "NONE"
    if carried > 0:
        if empty_structure <= 0:
            return "NO_STRUCTURE_PLAN" if not emitted_build else "STRUCTURE_NOT_FINISHED"
        return "PLACE_ACTION_EMITTED" if emitted_place else "TASK_PREEMPTED_PLACE"
    if in_shed > 0:
        if empty_structure <= 0:
            return "NO_STRUCTURE_PLAN" if not emitted_build else "STRUCTURE_NOT_FINISHED"
        if emitted_pickup:
            return "PICKUP_ACTION_EMITTED"
        return "NO_AVAILABLE_UNIT" if active_units <= 1 else "TASK_PREEMPTED_PICKUP"
    if owned < desired:
        if emitted_buy:
            return "BUY_ACTION_EMITTED"
        if shed_used >= SHED_CAPACITY:
            return "NO_SHED_CAPACITY"
        if cash < animal_cost:
            return "NO_CASH"
        if empty_structure <= 0 and not emitted_build:
            return "NO_STRUCTURE_PLAN"
        return "ADMISSION_BLOCKED"
    return "UNKNOWN"


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    trace = np.load(args.trace, allow_pickle=False)
    player = int(np.asarray(trace["player"])[0])
    calendar, diagnostic = compile_gold_replay_calendar_v3(
        replay, player=player, candidate_id=3_700
    )
    crop_target_by_day = np.asarray(calendar.crop_target_by_day)[0]
    animal_additions = np.asarray(calendar.animal_purchase_additions_by_day)[0]
    animal_target_by_day = np.cumsum(animal_additions, axis=0)
    animal_service_by_day = np.asarray(calendar.animal_service_target_by_day)[0]

    steps = np.asarray(trace["state_step"])[:, 0].astype(np.int32)
    money = np.asarray(trace["state_money"])[:, 0, player].astype(np.int32)
    kind = np.asarray(trace["state_tile_kind"])[:, 0, player]
    crop = np.asarray(trace["state_tile_crop"])[:, 0, player]
    animal = np.asarray(trace["state_tile_animal"])[:, 0, player]
    tile_yield = np.asarray(trace["state_tile_yield"])[:, 0, player]
    shed = np.asarray(trace["state_shed"])[:, 0, player]
    unit_inventory = np.asarray(trace["state_unit_inventory"])[:, 0, player]
    unit_active = np.asarray(trace["state_unit_active"])[:, 0, player]
    unit_op = np.asarray(trace["action_unit_op"])[:, 0]
    unit_item = np.asarray(trace["action_unit_item"])[:, 0]
    market_op = np.asarray(trace["action_market_op"])[:, 0]
    market_item = np.asarray(trace["action_market_item"])[:, 0]
    market_amount = np.asarray(trace["action_market_amount"])[:, 0]

    n = len(steps)
    active_animals = np.stack(
        [counts(animal[index], NUM_ANIMALS) for index in range(n)]
    )
    active_crops = np.stack([counts(crop[index], NUM_CROPS) for index in range(n)])
    shed_animals = shed[:, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS].astype(
        np.int16
    )
    carried_animals = unit_inventory[
        :, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].sum(axis=1, dtype=np.int16)
    owned_animals = active_animals + shed_animals + carried_animals
    productive_animals = np.stack(
        [
            np.asarray(
                [((animal[index] == species) & (tile_yield[index] > 0)).sum()
                 for species in range(NUM_ANIMALS)],
                dtype=np.int16,
            )
            for index in range(n)
        ]
    )
    empty_structures = np.stack(
        [
            np.asarray(
                [
                    ((kind[index] == ANIMAL_STRUCTURE[species]) & (animal[index] < 0)).sum()
                    for species in range(NUM_ANIMALS)
                ],
                dtype=np.int16,
            )
            for index in range(n)
        ]
    )
    # Cow and sheep share one pasture pool.  Report the same physical pool for
    # both lanes; the detailed blocker remains species-specific.
    empty_structures[:, 2] = empty_structures[:, 1]

    requested_purchase = np.zeros((n, NUM_ANIMALS), dtype=np.int32)
    sell_product = np.zeros((n, NUM_PRODUCTS), dtype=np.int32)
    for index in range(n):
        for slot in range(market_op.shape[1]):
            op = int(market_op[index, slot])
            item = int(market_item[index, slot])
            quantity = int(market_amount[index, slot])
            if op == int(MarketOp.BUY_ANIMAL) and NUM_PRODUCTS <= item < NUM_PRODUCTS + NUM_ANIMALS:
                requested_purchase[index, item - NUM_PRODUCTS] += quantity
            if op == int(MarketOp.SELL) and 0 <= item < NUM_PRODUCTS:
                sell_product[index, item] += quantity
    cumulative_requested = np.cumsum(requested_purchase, axis=0)
    cumulative_sell = np.cumsum(sell_product, axis=0)

    lifecycle: dict[str, list[dict[str, object]]] = {}
    for species in range(NUM_ANIMALS):
        entries = []
        planned_steps: list[int] = []
        for day, addition in enumerate(animal_additions[:, species].tolist()):
            planned_steps.extend([day * TURNS_PER_DAY] * int(addition))
        for ordinal, planned_step in enumerate(planned_steps, start=1):
            bank_product = int(ANIMAL_PRODUCT[species])
            entries.append(
                {
                    "commitment_ordinal": ordinal,
                    "planned_step": planned_step,
                    "buy_action_step": first_reach(cumulative_requested[:, species], ordinal),
                    "effect_confirmed_owned_step": first_reach(owned_animals[:, species], ordinal),
                    "active_step": first_reach(active_animals[:, species], ordinal),
                    "productive_step": first_reach(productive_animals[:, species], ordinal),
                    "project_product_sell_step": first_reach(cumulative_sell[:, bank_product], 1),
                }
            )
        lifecycle[str(species)] = entries

    blocker_counts: Counter[str] = Counter()
    first_blocker_step: dict[str, int] = {}
    idle_cash_events: list[dict[str, object]] = []
    daily = []
    milestone_days = {5, 10, 11}
    for day in range(NUM_DAYS):
        index = min((day + 1) * TURNS_PER_DAY - 1, n - 1)
        start = day * TURNS_PER_DAY
        stop = min((day + 1) * TURNS_PER_DAY, n)
        desired = animal_target_by_day[day]
        service = animal_service_by_day[day]
        emitted_buy = np.zeros(NUM_ANIMALS, dtype=np.bool_)
        emitted_build = np.zeros(NUM_ANIMALS, dtype=np.bool_)
        emitted_pickup = np.zeros(NUM_ANIMALS, dtype=np.bool_)
        emitted_place = np.zeros(NUM_ANIMALS, dtype=np.bool_)
        for species in range(NUM_ANIMALS):
            emitted_buy[species] = requested_purchase[start:stop, species].sum() > 0
            structure_op = UnitOp.BUILD_COOP if species == 0 else UnitOp.BUILD_PASTURE
            emitted_build[species] = np.any(unit_op[start:stop] == int(structure_op))
            animal_item = NUM_PRODUCTS + species
            emitted_pickup[species] = np.any(
                (unit_op[start:stop] == int(UnitOp.PICKUP))
                & (unit_item[start:stop] == animal_item)
            )
            emitted_place[species] = np.any(
                (unit_op[start:stop] == int(UnitOp.PLACE))
                & (unit_item[start:stop] == animal_item)
            )
        lane_blockers = []
        for species in range(NUM_ANIMALS):
            blocker = animal_blocker(
                desired=int(desired[species]),
                active=int(active_animals[index, species]),
                in_shed=int(shed_animals[index, species]),
                carried=int(carried_animals[index, species]),
                empty_structure=int(empty_structures[index, species]),
                animal_cost=int(ANIMAL_COST[species]),
                cash=int(money[index]),
                shed_used=int(shed[index].sum()),
                active_units=int(unit_active[index].sum()),
                emitted_buy=bool(emitted_buy[species]),
                emitted_build=bool(emitted_build[species]),
                emitted_pickup=bool(emitted_pickup[species]),
                emitted_place=bool(emitted_place[species]),
            )
            lane_blockers.append(blocker)
            if active_animals[index, species] < service[species] and blocker != "NONE":
                blocker_counts[blocker] += 1
                first_blocker_step.setdefault(blocker, int(steps[index]))
        crop_gap = np.maximum(crop_target_by_day[day] - active_crops[index], 0)
        investment_action = bool(
            np.any(
                np.isin(
                    market_op[start:stop],
                    [int(MarketOp.BUY_LAND), int(MarketOp.BUY_SEED), int(MarketOp.BUY_ANIMAL)],
                )
            )
            or np.any(
                np.isin(
                    unit_op[start:stop],
                    [
                        int(UnitOp.PLANT),
                        int(UnitOp.BUILD_COOP),
                        int(UnitOp.BUILD_PASTURE),
                        int(UnitOp.PLACE),
                    ],
                )
            )
        )
        backlog = bool(np.any(crop_gap > 0) or np.any(service > active_animals[index]))
        min_missing_animal_cost = min(
            [ANIMAL_COST[s] for s in range(NUM_ANIMALS) if active_animals[index, s] < service[s]]
            or [10**9]
        )
        cash_idle = backlog and not investment_action and int(money[index]) >= min_missing_animal_cost
        if cash_idle:
            idle_cash_events.append(
                {
                    "state_step": int(steps[index]),
                    "day": day + 1,
                    "cash": int(money[index]),
                    "animal_gap": np.maximum(service - active_animals[index], 0).tolist(),
                    "crop_gap": crop_gap.tolist(),
                }
            )
        daily.append(
            {
                "day": day + 1,
                "state_step": int(steps[index]),
                "cash": int(money[index]),
                "animal_purchase_target": desired.tolist(),
                "animal_service_target": service.tolist(),
                "animal_active": active_animals[index].tolist(),
                "animal_in_shed": shed_animals[index].tolist(),
                "animal_carried": carried_animals[index].tolist(),
                "empty_structure": empty_structures[index].tolist(),
                "animal_blocker": lane_blockers,
                "crop_target": crop_target_by_day[day].tolist(),
                "crop_active": active_crops[index].tolist(),
                "crop_gap": crop_gap.tolist(),
                "investment_action_emitted": investment_action,
                "cash_idle_while_admissible_backlog": cash_idle,
                "milestone_gate_day": day in milestone_days,
            }
        )

    milestone = {str(row["day"]): row for row in daily if row["milestone_gate_day"]}
    receipt = {
        "receipt_id": "M37A_LIFECYCLE_BASELINE_V1",
        "status": "PASS",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DIAGNOSTIC_ONLY_NO_POLICY_CHANGE",
        "official_version": replay.get("module_version"),
        "episode_id": int(replay["info"]["EpisodeId"]),
        "seed": int(np.asarray(trace["seed"])[0]),
        "player": player,
        "trace": args.trace.resolve().relative_to(ROOT.resolve()).as_posix(),
        "trace_sha256": sha256(args.trace),
        "calendar_source": diagnostic.to_dict(),
        "raw_replay_action_payload_stored": False,
        "lifecycle_semantics": {
            "planned": "calendar cumulative purchase ordinal",
            "admitted": "BUY_ANIMAL action emitted; current controller creates and assigns market tasks together",
            "effect_confirmed": "owned active+shed+carried count reaches ordinal",
            "active": "map animal count reaches ordinal",
            "productive": "map animals with held product reach ordinal",
            "banked": "first project product SELL action; project-level, not individual-animal attribution",
            "resource_reserved_task_created_assigned": "requires internal controller trace in M3.7A-v2",
        },
        "animal_commitment_lifecycle": lifecycle,
        "daily": daily,
        "gate_milestones": milestone,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "first_blocker_step": first_blocker_step,
        "cash_idle_while_admissible_backlog_events": idle_cash_events,
        "checks": {
            "trace_steps_719": n == 719,
            "calendar_30_days": len(daily) == 30,
            "lifecycle_has_all_planned_animals": sum(len(v) for v in lifecycle.values())
            == int(animal_additions.sum()),
            "raw_replay_action_playback": False,
        },
        "boundary": "Baseline physical/action audit. Internal RESOURCE_RESERVED/TASK_CREATED/ASSIGNED stages are explicitly not inferred and must be added by M3.7A-v2.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "blocker_counts": receipt["blocker_counts"],
                "cash_idle_event_count": len(idle_cash_events),
                "gate_milestones": milestone,
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
