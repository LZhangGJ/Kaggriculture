#!/usr/bin/env python3
"""Run the paired BLOCK-MVP-ROUTED-INTENT-v2 falsification experiment.

The selector is frozen by forcing one of three replay blocks.  RAW executes the
block tape directly.  ROUTED removes positional production actions from that
tape and replays them as concurrent, state-conditioned ``RouteTask`` objects.
KEEP, RAW, and ROUTED use identical opponent, seed, and seat tuples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_block_mvp_96_216 as v1
from meta_agent.src.route_compiler import (
    ANIMALS,
    CROPS,
    SHED_ACCESS,
    _expected,
    _get,
    _inventory,
    _move_toward,
    _nearest_actor,
    _positions,
    _private_dict,
    _set_unit_order,
    _tile,
)
from meta_agent.src.route_plan import (
    CarrierRoute,
    CompileReport,
    CompiledRoutePlan,
    RouteTask,
    normalized_action,
)


TARGET_IDS = ("RB96_C02", "RB96_C00", "RB96_C01")
KEY_OPERATIONS = {"BUILD_COOP", "BUILD_PASTURE", "PLACE"}
DEFAULT_V1_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\block_mvp_96_216_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\block_mvp_routed_intent_v2"
)
PHASE_SPECS = {
    "dev": {"split": "validation", "opponents": 4, "seeds": 4, "seed_offset": 2000},
    "holdout": {
        "split": "championship", "opponents": 8, "seeds": 16,
        "seed_offset": 3000,
    },
}


@dataclass(frozen=True)
class RoutedTarget:
    block_id: str
    routed_id: str
    masked_tape: tuple[dict[str, Any], ...]
    plan: CompiledRoutePlan
    task_rows: tuple[dict[str, Any], ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.plan.tasks)

    @property
    def key_task_ids(self) -> tuple[str, ...]:
        return tuple(
            task.task_id for task in self.plan.tasks
            if task.operation in KEY_OPERATIONS
        )


class ConcurrentIntentRouter:
    """State-conditioned intent executor with one concurrent task per actor."""

    def __init__(self, tasks: Sequence[RouteTask]) -> None:
        self.tasks = tuple(sorted(
            tasks, key=lambda task: (task.activation_step, task.due_step, task.task_id)
        ))
        self.next_task = 0
        self.pending: list[RouteTask] = []
        self.active: dict[int, dict[str, Any]] = {}
        self.executed_steps: dict[str, int] = {}
        self.already_satisfied_steps: dict[str, int] = {}
        self.failed_task_ids: list[str] = []
        self.blocked_reasons: dict[str, str] = {}
        self.unfinished_task_ids: list[str] = []
        self.task_retries = 0
        self.prerequisite_waits = 0
        self.actor_rebinds = 0
        self.non_pass_base_unit_overrides = 0
        self.finalized = False
        self._issued: dict[int, str] = {}
        self._before_satisfied: dict[str, bool] = {}
        self._reserved_seeds: dict[str, int] = {}
        self._satisfied_keys: set[tuple[str, str | None, tuple[int, int]]] = set()

    @property
    def realized_steps(self) -> dict[str, int]:
        return {**self.already_satisfied_steps, **self.executed_steps}

    def _enqueue(self, step: int) -> None:
        while (
            self.next_task < len(self.tasks)
            and self.tasks[self.next_task].activation_step <= step
        ):
            self.pending.append(self.tasks[self.next_task])
            self.next_task += 1

    def _expire(self, step: int) -> None:
        kept = []
        for task in self.pending:
            if step > task.due_step + task.grace_steps:
                self.failed_task_ids.append(task.task_id)
            else:
                kept.append(task)
        self.pending = kept
        for actor, state in list(self.active.items()):
            task: RouteTask = state["task"]
            if step > task.due_step + task.grace_steps:
                self.failed_task_ids.append(task.task_id)
                del self.active[actor]

    @staticmethod
    def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _estimated_steps(
        self, observation: Any, actor: int, task: RouteTask,
    ) -> int:
        """Estimate setup travel so actors bind just in time, not at activation."""

        positions = _positions(observation)
        position = positions[actor]
        target = task.target_xy
        tile = _tile(observation, target)
        if _expected(tile, task.operation, task.item):
            return 0
        tile_kind = str(_get(tile, "kind", "") or "") if tile is not None else ""
        tile_animal = _get(tile, "animal", None) if tile is not None else None
        if task.operation == "PLACE" and task.item:
            required = "COOP" if task.item == "GOOSE" else "PASTURE"
            carried = _inventory(observation, actor).get(task.item, 0)
            if tile_animal not in (None, "", "NONE") or tile_kind == "ANIMAL":
                return self._distance(position, target)
            if tile_kind == required:
                prepare = 0
            elif tile is None or tile_kind in ("", "EMPTY"):
                prepare = 1
            else:
                prepare = 2
            if carried > 0:
                return self._distance(position, target) + prepare
            access = min(
                SHED_ACCESS,
                key=lambda xy: self._distance(position, xy),
            )
            return (
                self._distance(position, access) + 1
                + self._distance(access, target) + prepare
            )
        clear = int(
            tile is not None and tile_kind not in ("", "EMPTY")
            and tile_animal in (None, "", "NONE")
        )
        return self._distance(position, target) + clear

    def _assign(self, observation: Any, step: int) -> None:
        positions = _positions(observation)
        free = set(range(len(positions))) - set(self.active)
        active_targets = {
            tuple(state["task"].target_xy) for state in self.active.values()
        }
        remaining: list[RouteTask] = []
        for task in sorted(self.pending, key=lambda value: (value.due_step, value.task_id)):
            if not free or tuple(task.target_xy) in active_targets:
                remaining.append(task)
                continue
            choices = sorted(free)
            if task.operation == "PLACE" and task.item:
                carrying = [
                    index for index in choices
                    if _inventory(observation, index).get(task.item, 0) > 0
                ]
                if carrying:
                    choices = carrying
            local = _nearest_actor([positions[index] for index in choices], task.target_xy)
            actor = choices[local]
            estimate = self._estimated_steps(observation, actor, task)
            if step < task.due_step - estimate:
                remaining.append(task)
                continue
            free.remove(actor)
            active_targets.add(tuple(task.target_xy))
            if task.actor_index is not None and actor != task.actor_index:
                self.actor_rebinds += 1
            self.active[actor] = {"task": task, "attempts": 0}
        self.pending = remaining

    def _complete(self, actor: int, step: int, *, executed: bool) -> None:
        state = self.active.pop(actor)
        task: RouteTask = state["task"]
        destination = self.executed_steps if executed else self.already_satisfied_steps
        destination.setdefault(task.task_id, max(step, task.due_step))
        self._satisfied_keys.add((task.operation, task.item, tuple(task.target_xy)))

    def _fail(self, actor: int, reason: str) -> None:
        state = self.active.pop(actor)
        task: RouteTask = state["task"]
        self.failed_task_ids.append(task.task_id)
        self.blocked_reasons[task.task_id] = reason

    def _task_action(
        self, observation: Any, action: dict[str, Any], actor: int, step: int,
    ) -> None:
        state = self.active.get(actor)
        if state is None:
            return
        task: RouteTask = state["task"]
        positions = _positions(observation)
        if actor >= len(positions):
            del self.active[actor]
            self.pending.append(task)
            self.actor_rebinds += 1
            return
        position = positions[actor]
        target = task.target_xy
        tile = _tile(observation, target)
        satisfied = _expected(tile, task.operation, task.item)
        self._before_satisfied[task.task_id] = satisfied
        if step >= task.due_step and satisfied:
            key = (task.operation, task.item, tuple(task.target_xy))
            if key not in self._satisfied_keys:
                # Preserve the native base action; this intent needs no edit.
                self._complete(actor, step, executed=False)
            return
        if _tile(observation, target) == "LOCKED":
            _set_unit_order(action, actor, ["PASS"])
            self.prerequisite_waits += 1
            return
        tile_kind = str(_get(tile, "kind", "") or "") if tile is not None else ""
        tile_animal = _get(tile, "animal", None) if tile is not None else None
        if task.operation == "PLACE" and task.item:
            if tile_animal not in (None, "", "NONE") or tile_kind == "ANIMAL":
                self._fail(actor, "target_has_animal")
                return
            required = "COOP" if task.item == "GOOSE" else "PASTURE"
            carried = _inventory(observation, actor).get(task.item, 0)
            if carried <= 0:
                shed = _private_dict(observation, "shed")
                access = min(
                    SHED_ACCESS,
                    key=lambda xy: abs(position[0] - xy[0]) + abs(position[1] - xy[1]),
                )
                if position != access:
                    _set_unit_order(action, actor, _move_toward(position, access))
                elif shed.get(task.item, 0) > 0:
                    _set_unit_order(action, actor, ["PICKUP", task.item, 1])
                else:
                    _set_unit_order(action, actor, ["PASS"])
                    self.prerequisite_waits += 1
                return
            if tile_kind != required:
                if position != target:
                    _set_unit_order(action, actor, _move_toward(position, target))
                elif step < task.due_step:
                    _set_unit_order(action, actor, ["PASS"])
                elif tile is not None and tile_kind not in ("", "EMPTY"):
                    _set_unit_order(action, actor, ["DIG"])
                else:
                    _set_unit_order(action, actor, [f"BUILD_{required}"])
                if step >= task.due_step and position == target:
                    state["attempts"] += 1
                    self.task_retries += 1
                return
        if position != target:
            _set_unit_order(action, actor, _move_toward(position, target))
            return
        if step < task.due_step:
            _set_unit_order(action, actor, ["PASS"])
            return
        if (
            task.operation in {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
            and tile is not None
            and not satisfied
        ):
            if tile_animal not in (None, "", "NONE") or tile_kind == "ANIMAL":
                self._fail(actor, "target_has_animal")
                return
            _set_unit_order(action, actor, ["DIG"])
            state["attempts"] += 1
            self.task_retries += 1
            return
        if task.operation == "PLANT" and task.item:
            available = _private_dict(observation, "seeds").get(task.item, 0)
            reserved = self._reserved_seeds.get(task.item, 0)
            if available - reserved <= 0:
                _set_unit_order(action, actor, ["PASS"])
                self.prerequisite_waits += 1
                return
            self._reserved_seeds[task.item] = reserved + 1
        order = [task.operation]
        if task.item:
            order.append(task.item)
        _set_unit_order(action, actor, order)
        self._issued[actor] = task.task_id
        state["attempts"] += 1
        if state["attempts"] > 1:
            self.task_retries += 1

    def route(
        self, observation: Any, base_action: Mapping[str, Any], step: int,
    ) -> dict[str, Any]:
        action = normalized_action(base_action)
        base_units = copy.deepcopy([action["farmer"], *action["hands"]])
        base_market = copy.deepcopy(action["market"])
        self._issued = {}
        self._before_satisfied = {}
        self._reserved_seeds = {}
        if step >= v1.STOP:
            self.finalize(step)
            return action
        self._enqueue(step)
        self._expire(step)
        self._assign(observation, step)
        for actor in sorted(tuple(self.active)):
            self._task_action(observation, action, actor, step)
        routed_units = [action["farmer"], *action["hands"]]
        self.non_pass_base_unit_overrides += sum(
            before != after and bool(before) and before[0] != "PASS"
            for before, after in zip(base_units, routed_units)
        )
        if action["market"] != base_market:
            raise AssertionError("ConcurrentIntentRouter must not edit market orders")
        return action

    def observe_after(self, observation: Any, completed_step: int) -> None:
        """Only an issued task op can count as executed intent completion."""

        for actor, task_id in list(self._issued.items()):
            state = self.active.get(actor)
            if state is None or state["task"].task_id != task_id:
                continue
            task: RouteTask = state["task"]
            if (
                not self._before_satisfied.get(task_id, False)
                and _expected(_tile(observation, task.target_xy), task.operation, task.item)
            ):
                self._complete(actor, completed_step, executed=True)

    def finalize(self, step: int = v1.STOP) -> None:
        if self.finalized:
            return
        self._enqueue(step)
        remaining = [state["task"] for state in self.active.values()]
        remaining.extend(self.pending)
        remaining.extend(self.tasks[self.next_task:])
        realized = set(self.realized_steps)
        failed = set(self.failed_task_ids)
        self.unfinished_task_ids = sorted({
            task.task_id for task in remaining
            if task.task_id not in realized and task.task_id not in failed
        })
        self.finalized = True

    def snapshot(self) -> dict[str, Any]:
        due = {task.task_id: task.due_step for task in self.tasks}
        realized = self.realized_steps
        return {
            "executed_steps": dict(self.executed_steps),
            "already_satisfied_steps": dict(self.already_satisfied_steps),
            "delays": {task_id: step - due[task_id] for task_id, step in realized.items()},
            "failed_task_ids": sorted(set(self.failed_task_ids)),
            "blocked_reasons": dict(self.blocked_reasons),
            "unfinished_task_ids": list(self.unfinished_task_ids),
            "task_retries": self.task_retries,
            "prerequisite_waits": self.prerequisite_waits,
            "market_orders_added": 0,
            "actor_rebinds": self.actor_rebinds,
            "non_pass_base_unit_overrides": self.non_pass_base_unit_overrides,
        }


def _is_positional_intent(order: Sequence[Any]) -> bool:
    if not order:
        return False
    operation = str(order[0])
    item = str(order[1]) if len(order) >= 2 else ""
    return bool(
        (operation == "PLANT" and item in CROPS)
        or operation in {"BUILD_COOP", "BUILD_PASTURE"}
        or (operation == "PLACE" and item in ANIMALS)
    )


def _mask_order(action: dict[str, Any], actor: int) -> None:
    if actor == 0:
        action["farmer"] = ["PASS"]
    else:
        action["hands"][actor - 1] = ["PASS"]


def extract_positional_tasks(
    tape: Sequence[Mapping[str, Any]],
    actor_positions: Sequence[Sequence[tuple[int, int] | None]],
    block_id: str,
    *,
    start: int = v1.START,
    stop: int = v1.STOP,
    lookahead: int = 24,
    grace_steps: int = 24,
) -> tuple[list[dict[str, Any]], list[RouteTask], list[dict[str, Any]]]:
    """Mask replay production actions and express them as positional tasks."""

    if len(tape) != v1.HORIZON or len(actor_positions) < stop:
        raise ValueError("routed intent extraction requires a complete 719-step carrier")
    masked = [normalized_action(action) for action in tape]
    tasks: list[RouteTask] = []
    rows: list[dict[str, Any]] = []
    serial = 0
    for step in range(start, stop):
        action = masked[step]
        orders = [action["farmer"], *action["hands"]]
        for actor, order in enumerate(orders):
            if not _is_positional_intent(order):
                continue
            if actor >= len(actor_positions[step]):
                raise ValueError(f"{block_id} step {step} actor {actor} has no source position")
            position = actor_positions[step][actor]
            if position is None:
                raise ValueError(f"{block_id} step {step} actor {actor} position is missing")
            operation = str(order[0])
            item = str(order[1]) if len(order) >= 2 else None
            xy = (int(position[0]), int(position[1]))
            task_id = f"{block_id}:{serial:03d}:{step}:{actor}:{operation}:{item or '-'}"
            task = RouteTask(
                task_id=task_id,
                due_step=step,
                activation_step=max(start, step - max(0, int(lookahead))),
                operation=operation,
                item=item,
                target_xy=xy,
                # Source actor is provenance only; runtime always binds nearest.
                actor_index=None,
                # The replay micro tape expects this actor at the production tile.
                return_xy=xy,
                source_step=step,
                grace_steps=max(0, int(grace_steps)),
            )
            tasks.append(task)
            rows.append({
                **asdict(task),
                "target_xy": list(task.target_xy),
                "return_xy": list(task.return_xy) if task.return_xy else None,
                "source_actor": actor,
                "key_intent": operation in KEY_OPERATIONS,
            })
            _mask_order(action, actor)
            serial += 1
    tasks.sort(key=lambda value: (value.activation_step, value.due_step, value.task_id))
    rows.sort(key=lambda value: (
        int(value["activation_step"]), int(value["due_step"]), str(value["task_id"])
    ))
    for step in range(start, stop):
        orders = [masked[step]["farmer"], *masked[step]["hands"]]
        if any(_is_positional_intent(order) for order in orders):
            raise AssertionError(f"unmasked positional intent at step {step}")
    return masked, tasks, rows


def compose_routed_carrier(
    baseline_tape: Sequence[Mapping[str, Any]],
    target_tape: Sequence[Mapping[str, Any]],
    *,
    start: int = v1.START,
    stop: int = v1.STOP,
) -> list[dict[str, Any]]:
    """Keep B0 unit micro-actions, target static market, and mask B0 macros."""

    if len(baseline_tape) != v1.HORIZON or len(target_tape) != v1.HORIZON:
        raise ValueError("routed carrier requires complete 719-step tapes")
    routed = [normalized_action(action) for action in baseline_tape]
    target = [normalized_action(action) for action in target_tape]
    for step in range(start, stop):
        routed[step]["market"] = copy.deepcopy(target[step]["market"])
        orders = [routed[step]["farmer"], *routed[step]["hands"]]
        for actor, order in enumerate(orders):
            if _is_positional_intent(order):
                _mask_order(routed[step], actor)
    return routed


def build_routed_target(
    baseline: v1.Block,
    block: v1.Block,
    *,
    lookahead: int = 24,
    grace_steps: int = 24,
) -> RoutedTarget:
    target_source = block.source
    target_carrier = CarrierRoute.from_replay(
        Path(str(target_source["replay_path"])), int(target_source["player_index"]),
        horizon=v1.HORIZON,
    )
    _, tasks, rows = extract_positional_tasks(
        block.tape, target_carrier.actor_positions, block.block_id,
        lookahead=lookahead, grace_steps=grace_steps,
    )
    if not tasks:
        raise ValueError(f"{block.block_id} contains no routed positional intents")
    baseline_source = baseline.source
    carrier = CarrierRoute.from_replay(
        Path(str(baseline_source["replay_path"])), int(baseline_source["player_index"]),
        horizon=v1.HORIZON,
    )
    masked = compose_routed_carrier(baseline.tape, block.tape)
    report = CompileReport(block.block_id, f"ROUTED_{block.block_id}")
    report.tasks = len(tasks)
    plan = CompiledRoutePlan(
        carrier=carrier,
        actions=tuple(masked),
        tasks=tuple(tasks),
        report=report,
    )
    return RoutedTarget(
        block_id=block.block_id,
        routed_id=f"ROUTED_{block.block_id}",
        masked_tape=tuple(masked),
        plan=plan,
        task_rows=tuple(rows),
    )


def _v1_inputs(root: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    contract = dict(manifest["generation_contract"])
    values: dict[str, Any] = {
        name: Path(str(manifest["inputs"][name]["path"]))
        for name in (
            "records", "source", "opening_actions", "opening_metadata",
            "base_actions", "base_metadata",
        )
    }
    values.update({
        "output_root": root,
        "opening": str(contract["opening"]),
        "prefilter": int(contract["prefilter"]),
        "carriers": int(contract["carriers"]),
        "random_seed": int(contract["random_seed"]),
    })
    return SimpleNamespace(**values), manifest


def load_formal_targets(
    root: Path,
) -> tuple[v1.Block, dict[str, v1.Block], SimpleNamespace, dict[str, Any]]:
    args, manifest = _v1_inputs(root)
    baseline, carriers, _ = v1.load_blocks(root, args)
    by_id = {block.block_id: block for block in carriers}
    missing = [block_id for block_id in TARGET_IDS if block_id not in by_id]
    if missing:
        raise KeyError(f"formal v1 artifacts do not contain {missing[0]}")
    return baseline, {block_id: by_id[block_id] for block_id in TARGET_IDS}, args, manifest


def _lineage_rows(metadata_path: Path) -> dict[str, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        str(row["family"]): {
            "family": str(row["family"]),
            "lineage": str(row.get("team") or f"family:{row['family']}"),
        }
        for row in metadata["opponent_routes"]
        if bool(row.get("selected", True))
    }


def select_unique_lineage_opponents(
    families: Sequence[str], metadata_path: Path, count: int,
) -> list[dict[str, str]]:
    details = _lineage_rows(metadata_path)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for family in families:
        row = details[str(family)]
        if row["lineage"] in seen:
            continue
        selected.append({"family": row["family"], "lineage": row["lineage"]})
        seen.add(row["lineage"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} unique lineages available; need {count}")
    return selected


def build_panel(
    phase: str,
    splits: Mapping[str, Sequence[str]],
    metadata_path: Path,
    seed_base: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = PHASE_SPECS[phase]
    opponents = select_unique_lineage_opponents(
        splits[str(spec["split"])], metadata_path, int(spec["opponents"])
    )
    seeds = list(range(
        int(seed_base) + int(spec["seed_offset"]),
        int(seed_base) + int(spec["seed_offset"]) + int(spec["seeds"]),
    ))
    states = [
        {"opponent": row["family"], "lineage": row["lineage"], "seed": seed, "seat": seat}
        for row in opponents for seed in seeds for seat in (0, 1)
    ]
    return states, {
        "phase": phase,
        "source_split": spec["split"],
        "opponents": opponents,
        "seeds": seeds,
        "states": len(states),
        "games": len(states) * (1 + 2 * len(TARGET_IDS)),
        "pairing": "same opponent, seed, and seat for KEEP/RAW/ROUTED",
    }


def _native_api() -> tuple[Any, ...]:
    import fast_kaggriculture as native

    required = (
        "Config", "FastEnv", "NativeAgentState", "audit_raw_tapes_detailed",
        "raw_tape_audit_metric_names", "raw_tape_first_failure_names",
    )
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        raise RuntimeError(
            "fast_kaggriculture is missing " + ", ".join(missing)
            + "; rebuild the extension with the action_at/NativeAgentState bindings"
        )
    return tuple(getattr(native, name) for name in required)


def _outcome(own: float, other: float) -> int:
    return 2 if own > other else 1 if own == other else 0


def _state_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["seat"])


def _own_route_index(
    step: int,
    mode: str,
    baseline_index: int,
    raw_index: int | None,
    routed_index: int | None,
    tail_index: int | None = None,
) -> int:
    """Resolve the own route while keeping the optional tail switch isolated."""

    if mode == "KEEP" or step < v1.START:
        return baseline_index
    if step >= v1.STOP:
        return (
            int(tail_index)
            if mode == "ROUTED" and tail_index is not None
            else baseline_index
        )
    if mode == "RAW":
        if raw_index is None:
            raise ValueError("RAW requires a target route")
        return int(raw_index)
    if mode == "ROUTED":
        if routed_index is None:
            raise ValueError("ROUTED requires a masked route")
        return int(routed_index)
    raise ValueError(f"unsupported episode mode: {mode}")


def _has_positional_unit_failure(
    observation: Mapping[str, Any], action: Mapping[str, Any],
) -> bool:
    """Mirror the C++ macro-unit audit enough to locate a failure step."""

    normalized = normalized_action(action)
    orders = [normalized["farmer"], *normalized["hands"]]
    positions = _positions(observation)
    seeds = dict(_private_dict(observation, "seeds"))
    inventories = [dict(_inventory(observation, actor)) for actor in range(len(positions))]
    demand = {
        crop: sum(
            bool(order) and order[0] == "PLANT"
            and len(order) >= 2 and order[1] == crop
            for order in orders
        )
        for crop in CROPS
    }
    blocked_seed = {
        crop: demand[crop] > int(seeds.get(crop, 0)) for crop in CROPS
    }
    tiles: dict[tuple[int, int], Any] = {}
    for actor, order in enumerate(orders):
        if not order:
            continue
        operation = str(order[0])
        item = str(order[1]) if len(order) >= 2 else ""
        if not (
            operation in {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
            or (operation == "PLACE" and item in ANIMALS)
        ):
            continue
        if actor >= len(positions):
            return True
        xy = positions[actor]
        tile = tiles.get(xy, _tile(observation, xy))
        kind = str(_get(tile, "kind", "") or "") if tile is not None else ""
        animal = _get(tile, "animal", None) if tile is not None else None
        if operation == "PLANT":
            if item not in CROPS or blocked_seed[item]:
                return True
            if tile is not None:
                return True
            seeds[item] = int(seeds.get(item, 0)) - 1
            tiles[xy] = {"kind": "PLANT", "crop": item}
            continue
        if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
            if tile is not None:
                return True
            tiles[xy] = {"kind": operation.removeprefix("BUILD_")}
            continue
        required = "COOP" if item == "GOOSE" else "PASTURE"
        if kind != required or animal not in (None, "", "NONE"):
            return True
        if actor >= len(inventories) or int(inventories[actor].get(item, 0)) <= 0:
            return True
        inventories[actor][item] = int(inventories[actor][item]) - 1
        tiles[xy] = {"kind": required, "animal": item}
    return False


def _has_market_failure(
    action: Mapping[str, Any], fills: Sequence[int],
) -> bool:
    """Locate an unfilled audited market request after FastEnv.step()."""

    for index, order in enumerate(normalized_action(action)["market"]):
        if not order:
            continue
        operation = str(order[0])
        quantity = int(order[2]) if len(order) >= 3 else 1
        if operation in {"BUY_SEED", "BUY_ANIMAL"}:
            requested = max(0, quantity)
        elif operation in {"HIRE", "BUY_LAND"}:
            requested = int(quantity > 0)
        else:
            continue
        filled = int(fills[index]) if index < len(fills) else 0
        if filled < requested:
            return True
    return False


def _router_runtime(agent: ConcurrentIntentRouter | None) -> dict[str, Any]:
    return {} if agent is None else agent.snapshot()


def run_episode(
    bundle: Any,
    baseline_index: int,
    opponent_index: int,
    seed: int,
    seat: int,
    mode: str,
    target: RoutedTarget | None,
    raw_index: int | None,
    routed_index: int | None,
    tail_index: int | None = None,
) -> dict[str, Any]:
    (
        Config, FastEnv, NativeAgentState, audit_raw_tapes_detailed,
        raw_tape_audit_metric_names, raw_tape_first_failure_names,
    ) = _native_api()
    if not hasattr(bundle.executor, "action_at"):
        raise RuntimeError("NativeTeammateExecutor.action_at is unavailable; rebuild extension")
    config = Config()
    env = FastEnv(config, int(seed))
    states = [NativeAgentState(), NativeAgentState()]
    router = (
        ConcurrentIntentRouter(target.plan.tasks)
        if mode == "ROUTED" and target else None
    )
    trace: list[list[dict[str, Any]]] = [[], []]
    turns = 0
    first_post216_unit_failure_step = -1
    first_post216_market_failure_step = -1
    while not bool(env.done):
        step = int(env.step_count)
        if router is not None and step == v1.STOP:
            # Step 215 has already been settled; step 216 gets no router action.
            router.finalize(step)
        actions: list[dict[str, Any]] = [{}, {}]
        for player in (0, 1):
            if player != seat:
                route_index = opponent_index
            else:
                route_index = _own_route_index(
                    step, mode, baseline_index, raw_index, routed_index, tail_index
                )
            actions[player] = normalized_action(
                bundle.executor.action_at(env, player, route_index, states[player])
            )
        if router is not None and v1.START <= step < v1.STOP:
            observation = dict(env.observation(seat))
            observation["step"] = step
            actions[seat] = router.route(observation, actions[seat], step)
        trace[0].append(copy.deepcopy(actions[0]))
        trace[1].append(copy.deepcopy(actions[1]))
        if (
            step >= v1.STOP and first_post216_unit_failure_step < 0
            and _has_positional_unit_failure(env.observation(seat), actions[seat])
        ):
            first_post216_unit_failure_step = step
        env.step(actions)
        turns += 1
        if (
            step >= v1.STOP and first_post216_market_failure_step < 0
            and _has_market_failure(actions[seat], env.last_market_fills[seat])
        ):
            first_post216_market_failure_step = step
        if router is not None and v1.START <= step < v1.STOP:
            after = dict(env.observation(seat))
            after["step"] = step + 1
            router.observe_after(after, step)
        if turns > v1.HORIZON:
            raise RuntimeError("FastEnv exceeded the 719-step horizon")
    if router is not None:
        router.finalize(v1.STOP)
    rewards = tuple(float(value) for value in env.rewards)
    metrics, replay_rewards, first = audit_raw_tapes_detailed(
        trace[0], trace[1], int(seed), config
    )
    replay_rewards = tuple(float(value) for value in replay_rewards)
    if not np.allclose(rewards, replay_rewards, rtol=0.0, atol=1e-9):
        raise AssertionError(
            f"trace replay reward mismatch: live={rewards}, replay={replay_rewards}"
        )
    metric_names = list(raw_tape_audit_metric_names())
    first_names = list(raw_tape_first_failure_names())
    metric = {
        str(name): float(np.asarray(metrics)[seat, index])
        for index, name in enumerate(metric_names)
    }
    first_row = np.asarray(first, dtype=np.int32)[seat]
    first_failures = {
        str(name): int(first_row[index])
        for index, name in enumerate(first_names)
    }
    valid_first = [value for value in first_failures.values() if value >= 0]
    unit_failures = metric["unit_attempts"] - metric["unit_valid"]
    market_failures = metric["market_requested"] - metric["market_filled"]
    delays: list[int] = []
    key_realized = 0
    intent_total = key_total = 0
    realized_ids: list[str] = []
    executed_ids: list[str] = []
    already_ids: list[str] = []
    failed_ids: list[str] = []
    unfinished_ids: list[str] = []
    if target is not None and router is not None:
        intent_total = len(target.plan.tasks)
        key_ids = set(target.key_task_ids)
        key_total = len(key_ids)
        runtime = router.snapshot()
        executed_ids = sorted(runtime["executed_steps"])
        already_ids = sorted(runtime["already_satisfied_steps"])
        realized_ids = sorted((*executed_ids, *already_ids))
        failed_ids = list(runtime["failed_task_ids"])
        unfinished_ids = list(runtime["unfinished_task_ids"])
        key_realized = sum(task_id in key_ids for task_id in realized_ids)
        delays = [max(0, int(value)) for value in runtime["delays"].values()]
    own, other = rewards[seat], rewards[1 - seat]
    post216_steps = [
        value for value in (
            first_post216_unit_failure_step, first_post216_market_failure_step,
        )
        if value >= 0
    ]
    pre_tail_trace_sha256 = hashlib.sha256(json.dumps(
        [trace[0][:v1.STOP], trace[1][:v1.STOP]],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "mode": mode,
        "target": target.block_id if target is not None else "B0_KEEP",
        "seed": int(seed),
        "seat": int(seat),
        "turns": turns,
        "pre_tail_joint_trace_sha256": pre_tail_trace_sha256,
        "completed": bool(turns == v1.HORIZON and env.done),
        "finite_rewards": bool(np.isfinite(rewards).all()),
        "reward": own,
        "opponent_reward": other,
        "outcome": _outcome(own, other),
        "margin": own - other,
        "macro_unit_failures": float(unit_failures),
        "macro_market_failures": float(market_failures),
        "macro_failures": float(unit_failures + market_failures),
        "first_macro_failure_step": min(valid_first) if valid_first else -1,
        "first_failure_in_block": bool(
            valid_first and v1.START <= min(valid_first) < v1.STOP
        ),
        "first_post216_unit_failure_step": first_post216_unit_failure_step,
        "first_post216_market_failure_step": first_post216_market_failure_step,
        "first_post216_failure_step": min(post216_steps) if post216_steps else -1,
        "audit": metric,
        "first_failures": first_failures,
        "intent_total": intent_total,
        "intent_realized": len(realized_ids),
        "intent_executed": len(executed_ids),
        "intent_already_satisfied": len(already_ids),
        "intent_failed": len(failed_ids),
        "intent_unfinished": len(unfinished_ids),
        "key_intent_total": key_total,
        "key_intent_realized": key_realized,
        "intent_delays": delays,
        "realized_task_ids": realized_ids,
        "executed_task_ids": executed_ids,
        "already_satisfied_task_ids": already_ids,
        "failed_task_ids": failed_ids,
        "unfinished_task_ids": unfinished_ids,
        "router_runtime": _router_runtime(router),
    }


def _rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "raw_win_rate": float(np.mean([int(row["outcome"]) == 2 for row in rows])),
        "score_rate": float(np.mean([int(row["outcome"]) for row in rows]) * .5),
        "mean_margin": float(np.mean([float(row["margin"]) for row in rows])),
        "mean_macro_unit_failures": float(np.mean([
            float(row["macro_unit_failures"]) for row in rows
        ])),
        "mean_macro_market_failures": float(np.mean([
            float(row["macro_market_failures"]) for row in rows
        ])),
        "mean_macro_failures": float(np.mean([
            float(row["macro_failures"]) for row in rows
        ])),
        "unit_failure_state_rate": float(np.mean([
            float(row["macro_unit_failures"]) > 0 for row in rows
        ])),
        "market_failure_state_rate": float(np.mean([
            float(row["macro_market_failures"]) > 0 for row in rows
        ])),
        "failure_state_rate": float(np.mean([
            float(row["macro_failures"]) > 0 for row in rows
        ])),
    }


def summarize_results(
    rows: Sequence[Mapping[str, Any]], phase: str,
) -> dict[str, Any]:
    keep_rows = [row for row in rows if row["mode"] == "KEEP"]
    keep = {_state_key(row): row for row in keep_rows}
    target_reports: dict[str, Any] = {}
    eligible_targets: list[str] = []
    for target_id in TARGET_IDS:
        raw = {
            _state_key(row): row for row in rows
            if row["mode"] == "RAW" and row["target"] == target_id
        }
        routed = {
            _state_key(row): row for row in rows
            if row["mode"] == "ROUTED" and row["target"] == target_id
        }
        if set(keep) != set(raw) or set(keep) != set(routed):
            raise ValueError(f"unpaired KEEP/RAW/ROUTED rows for {target_id}")
        keys = sorted(keep)
        raw_excess = np.asarray([
            max(0.0, float(raw[key]["macro_unit_failures"])
                - float(keep[key]["macro_unit_failures"]))
            for key in keys
        ])
        routed_excess = np.asarray([
            max(0.0, float(routed[key]["macro_unit_failures"])
                - float(keep[key]["macro_unit_failures"]))
            for key in keys
        ])
        raw_market_excess = np.asarray([
            max(0.0, float(raw[key]["macro_market_failures"])
                - float(keep[key]["macro_market_failures"]))
            for key in keys
        ])
        routed_market_excess = np.asarray([
            max(0.0, float(routed[key]["macro_market_failures"])
                - float(keep[key]["macro_market_failures"]))
            for key in keys
        ])
        raw_total = float(np.sum(raw_excess))
        reduction = (
            float(1.0 - np.sum(routed_excess) / raw_total) if raw_total > 0 else None
        )
        routed_rows = [routed[key] for key in keys]
        non_pass_override_counts = [
            int(row.get("router_runtime", {}).get(
                "non_pass_base_unit_overrides", 0
            ))
            for row in routed_rows
        ]
        intent_total = sum(int(row["intent_total"]) for row in routed_rows)
        intent_realized = sum(int(row["intent_realized"]) for row in routed_rows)
        intent_executed = sum(int(row["intent_executed"]) for row in routed_rows)
        intent_already = sum(
            int(row["intent_already_satisfied"]) for row in routed_rows
        )
        intent_failed = sum(int(row["intent_failed"]) for row in routed_rows)
        intent_unfinished = sum(int(row["intent_unfinished"]) for row in routed_rows)
        key_total = sum(int(row["key_intent_total"]) for row in routed_rows)
        key_realized = sum(int(row["key_intent_realized"]) for row in routed_rows)
        delays = [
            int(value) for row in routed_rows for value in row["intent_delays"]
        ]
        intent_rate = intent_realized / max(1, intent_total)
        executed_rate = intent_executed / max(1, intent_total)
        already_rate = intent_already / max(1, intent_total)
        key_rate = key_realized / max(1, key_total)
        p90_delay = float(np.quantile(delays, .90)) if delays else math.inf
        raw_metrics = _rates([raw[key] for key in keys])
        routed_metrics = _rates(routed_rows)
        finite_complete = all(
            bool(row["completed"] and row["finite_rewards"])
            for key in keys for row in (raw[key], routed[key])
        )
        intent_gate = bool(
            intent_rate >= .85 and key_rate >= .90 and p90_delay <= 24.0
        )
        if target_id == "RB96_C02":
            continuity_gate = bool(
                routed_metrics["mean_macro_unit_failures"]
                <= raw_metrics["mean_macro_unit_failures"] + 1e-12
                and routed_metrics["raw_win_rate"]
                >= raw_metrics["raw_win_rate"] - .01 - 1e-12
            )
        else:
            continuity_gate = bool(
                reduction is not None and reduction >= .90
                and float(np.mean(routed_excess > 0)) <= .05
                and float(np.mean(routed_excess)) <= .25
            )
        eligible = bool(finite_complete and intent_gate and continuity_gate)
        if eligible:
            eligible_targets.append(target_id)
        target_reports[target_id] = {
            "raw": raw_metrics,
            "routed": routed_metrics,
            "gating_metric": "whole_episode_paired_excess_positional_unit_failures",
            "raw_excess_unit_failures_vs_keep": raw_total,
            "routed_excess_unit_failures_vs_keep": float(np.sum(routed_excess)),
            "raw_new_unit_failure_state_rate_vs_keep": float(np.mean(raw_excess > 0)),
            "routed_new_unit_failure_state_rate_vs_keep": float(np.mean(routed_excess > 0)),
            "mean_routed_excess_unit_failures_vs_keep": float(np.mean(routed_excess)),
            "failure_reduction_fraction": reduction,
            "intent_completion_rate": float(intent_rate),
            "intent_executed_rate": float(executed_rate),
            "intent_already_satisfied_rate": float(already_rate),
            "intent_failed": intent_failed,
            "intent_failed_rate": float(intent_failed / max(1, intent_total)),
            "intent_unfinished": intent_unfinished,
            "intent_unfinished_rate": float(intent_unfinished / max(1, intent_total)),
            "key_intent_completion_rate": float(key_rate),
            "p90_intent_delay": p90_delay if math.isfinite(p90_delay) else None,
            "non_gating_market_diagnostic": {
                "raw_excess_failures_vs_keep": float(np.sum(raw_market_excess)),
                "routed_excess_failures_vs_keep": float(np.sum(routed_market_excess)),
                "raw_new_failure_state_rate_vs_keep": float(np.mean(raw_market_excess > 0)),
                "routed_new_failure_state_rate_vs_keep": float(np.mean(
                    routed_market_excess > 0
                )),
            },
            "non_gating_runtime_overlay_diagnostic": {
                "non_pass_base_unit_overrides_total": int(sum(
                    non_pass_override_counts
                )),
                "mean_non_pass_base_unit_overrides_per_episode": float(np.mean(
                    non_pass_override_counts
                )),
            },
            "finite_and_719": finite_complete,
            "intent_gate": intent_gate,
            "continuity_gate": continuity_gate,
            "eligible_for_exploratory_oracle": eligible,
        }
    keys = sorted(keep)
    block_ids = ["B0_KEEP", *eligible_targets]
    outcome_rows = [[int(keep[key]["outcome"]) for key in keys]]
    margin_rows = [[float(keep[key]["margin"]) for key in keys]]
    for target_id in eligible_targets:
        by_key = {
            _state_key(row): row for row in rows
            if row["mode"] == "ROUTED" and row["target"] == target_id
        }
        outcome_rows.append([int(by_key[key]["outcome"]) for key in keys])
        margin_rows.append([float(by_key[key]["margin"]) for key in keys])
    oracle = v1.portfolio_metrics(
        np.asarray(outcome_rows, np.uint8), np.asarray(margin_rows, np.float32),
        list(range(len(block_ids))), reference_index=0,
    )
    oracle["block_names"] = block_ids
    value_gate = bool(
        oracle["oracle_gain_pp"] >= 1.0
        or oracle["constant_loss_fix_fraction"] >= .15
    )
    global_complete = all(
        bool(row["completed"] and row["finite_rewards"]) for row in rows
    )
    gates = {
        "R1_finite_719": global_complete,
        "R2_c00_c01_unit_failure_reduction_90pct_state_5pct_mean_0_25": all(
            bool(target_reports[target_id]["continuity_gate"])
            for target_id in ("RB96_C00", "RB96_C01")
        ),
        "R3_intent_85_key_90_p90_delay_24": all(
            bool(target_reports[target_id]["intent_gate"])
            for target_id in TARGET_IDS
        ),
        "R4_c02_no_harm": bool(target_reports["RB96_C02"]["continuity_gate"]),
        "R5_exploratory_positional_routed_oracle_value": value_gate,
    }
    return {
        "schema": "block-mvp-routed-intent-v2",
        "phase": phase,
        "status": (
            "passed_positional_subgate"
            if all(gates.values()) else "falsified_positional_subgate"
        ),
        "states": len(keys),
        "games": len(rows),
        "keep": _rates(keep_rows),
        "targets": target_reports,
        "positional_gate_target_ids": eligible_targets,
        "exploratory_positional_routed_oracle": oracle,
        "gates": gates,
        "deployment_eligible": False,
        "value_gate_interpretation": "exploratory_only",
        "scope_limitations": [
            "Market continuity is non-gating; the target static market tape is retained unchanged in steps [96,216).",
            "ROUTED uses B0_KEEP unit micro-actions; the target contributes positional intents and its static market schedule only.",
            "The Python router adds no market orders and waits for the static schedule when land, seeds, or animals are unavailable.",
            "Nearest-actor rebinding does not preserve a donor actor-index permutation; overwritten non-PASS B0 unit actions are diagnostic, so full micro-tape continuity is not claimed.",
            "The continuity metric counts paired excess positional unit-macro failures across all 719 steps, including post-216 positional failures caused by handoff state drift; it is not limited to the routed task list or [96,216) window.",
            "FastEnv executes the exact episode and trace replay must match, but the value gate is exploratory and cannot authorize deployment.",
            "This v2 does not prove HIRE, BUY_LAND, or bulk-buy continuity.",
        ],
    }


def _write_report(path: Path, result: Mapping[str, Any], panel: Mapping[str, Any]) -> None:
    lines = [
        "# BLOCK-MVP-ROUTED-INTENT-v2",
        "",
        f"- phase: `{result['phase']}`",
        f"- status: **{str(result['status']).upper()}**",
        f"- panel: {result['states']} states / {result['games']} games",
        f"- pairing: {panel['pairing']}",
        "",
        "Main continuity gate uses whole-episode paired excess positional unit-macro failures across all 719 steps. It includes post-216 positional failures caused by handoff state drift and is not limited to the routed task list or [96,216) window; market failures are diagnostic.",
        "",
        "| target | RAW unit excess | ROUTED unit excess | reduction | intent | executed | already | failed | unfinished | key | p90 delay | non-PASS B0 overrides total / mean | exploratory eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target_id in TARGET_IDS:
        row = result["targets"][target_id]
        reduction = row["failure_reduction_fraction"]
        overlay = row["non_gating_runtime_overlay_diagnostic"]
        lines.append(
            f"| {target_id} | {row['raw_excess_unit_failures_vs_keep']:.0f} | "
            f"{row['routed_excess_unit_failures_vs_keep']:.0f} | "
            f"{'n/a' if reduction is None else f'{100 * reduction:.1f}%'} | "
            f"{100 * row['intent_completion_rate']:.1f}% | "
            f"{100 * row['intent_executed_rate']:.1f}% | "
            f"{100 * row['intent_already_satisfied_rate']:.1f}% | "
            f"{100 * row['intent_failed_rate']:.1f}% | "
            f"{100 * row['intent_unfinished_rate']:.1f}% | "
            f"{100 * row['key_intent_completion_rate']:.1f}% | "
            f"{row['p90_intent_delay'] if row['p90_intent_delay'] is not None else 'n/a'} | "
            f"{overlay['non_pass_base_unit_overrides_total']} / "
            f"{overlay['mean_non_pass_base_unit_overrides_per_episode']:.2f} | "
            f"{row['eligible_for_exploratory_oracle']} |"
        )
    oracle = result["exploratory_positional_routed_oracle"]
    lines.extend((
        "",
        f"Exploratory positional-routed oracle gain: {oracle['oracle_gain_pp']:.2f}pp; "
        f"KEEP-loss repair: {100 * oracle['constant_loss_fix_fraction']:.2f}%.",
        "",
        "## Gates",
        "",
    ))
    lines.extend(f"- {name}: {value}" for name, value in result["gates"].items())
    lines.extend(("", "## Scope limitations", ""))
    lines.extend(f"- {value}" for value in result["scope_limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    baseline, blocks, v1_args, manifest = load_formal_targets(args.v1_root)
    targets = {
        block_id: build_routed_target(
            baseline, blocks[block_id],
            lookahead=args.lookahead, grace_steps=args.grace_steps,
        )
        for block_id in TARGET_IDS
    }
    splits, split_lineages = v1._opponent_splits(
        v1_args.base_metadata, v1_args.random_seed
    )
    states, panel = build_panel(
        args.phase, splits, v1_args.base_metadata, args.seed_base
    )
    if args.limit_states:
        states = states[:args.limit_states]
        panel["states"] = len(states)
        panel["games"] = len(states) * (1 + 2 * len(TARGET_IDS))
        panel["smoke_limit_states"] = args.limit_states
    additional = {baseline.block_id: baseline.tape}
    additional.update({block_id: blocks[block_id].tape for block_id in TARGET_IDS})
    additional.update({target.routed_id: target.masked_tape for target in targets.values()})
    opponents = list(dict.fromkeys(str(row["opponent"]) for row in states))
    bundle = v1.NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*opponents, *additional)),
    )
    baseline_index = bundle.index(baseline.block_id)
    rows: list[dict[str, Any]] = []
    total_games = len(states) * (1 + 2 * len(TARGET_IDS))
    completed_games = 0
    for state in states:
        common = {
            "opponent": str(state["opponent"]),
            "lineage": str(state["lineage"]),
        }
        opponent_index = bundle.index(common["opponent"])
        keep = run_episode(
            bundle, baseline_index, opponent_index, int(state["seed"]),
            int(state["seat"]), "KEEP", None, None, None,
        )
        rows.append({**keep, **common})
        completed_games += 1
        for block_id in TARGET_IDS:
            target = targets[block_id]
            raw_index = bundle.index(block_id)
            routed_index = bundle.index(target.routed_id)
            for mode in ("RAW", "ROUTED"):
                row = run_episode(
                    bundle, baseline_index, opponent_index, int(state["seed"]),
                    int(state["seat"]), mode, target,
                    raw_index, routed_index,
                )
                rows.append({**row, **common})
                completed_games += 1
        if completed_games == total_games or completed_games % 70 == 0:
            print(f"completed {completed_games}/{total_games} games", flush=True)
    result = summarize_results(rows, args.phase)
    result.update({
        "elapsed_seconds": time.perf_counter() - started,
        "panel": panel,
        "v1_root": str(args.v1_root.resolve()),
        "v1_candidate_actions_sha256": str(manifest["actions_sha256"]),
        "opponent_split_lineages": split_lineages,
        "router": {
            "executor": "ConcurrentIntentRouter",
            "carrier_micro": "B0_KEEP",
            "target_payload": "positional_intents_plus_static_market_only",
            "concurrency": "one active intent per runtime actor; duplicate target_xy serialized",
            "actor_binding": "nearest free runtime actor; replay actor is coordinate provenance only",
            "same_step_seed_reservation": True,
            "lookahead": args.lookahead,
            "grace_steps": args.grace_steps,
            "positional_operations": [
                "PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE_ANIMAL",
            ],
            "rewrites_non_positional_micro_actions": True,
            "static_carrier_rewrites_non_positional_micro_actions": False,
            "runtime_overlay_rewrites_non_positional_micro_actions": True,
            "market_continuity_gate": False,
            "target_bulk_market_tape_unchanged": True,
            "router_adds_market_orders": False,
            "native_overlay_market_schedule": "target_static_schedule_via_action_at",
        },
    })
    output = args.output_root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    route_manifest = {
        "schema": "block-mvp-routed-intent-routes-v2",
        "targets": {
            block_id: {
                "routed_id": target.routed_id,
                "task_count": len(target.plan.tasks),
                "key_task_count": len(target.key_task_ids),
                "masked_tape_sha256": hashlib.sha256(json.dumps(
                    target.masked_tape, ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "tasks": list(target.task_rows),
            }
            for block_id, target in targets.items()
        },
    }
    (output / "route_manifest.json").write_text(
        json.dumps(route_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", result, panel)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase", choices=tuple(PHASE_SPECS), default="dev")
    parser.add_argument("--seed-base", type=int, default=2026082800)
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--grace-steps", type=int, default=24)
    parser.add_argument(
        "--limit-states", type=int, default=0,
        help="Run only the first N paired states for an explicitly labelled smoke run.",
    )
    args = parser.parse_args()
    args.v1_root = args.v1_root.resolve()
    args.output_root = args.output_root.resolve()
    if not (args.v1_root / "candidate_manifest.json").is_file():
        parser.error(f"formal v1 artifacts are missing: {args.v1_root}")
    if args.lookahead < 0 or args.grace_steps < 0 or args.limit_states < 0:
        parser.error("lookahead, grace-steps, and limit-states must be non-negative")
    run_experiment(args)


if __name__ == "__main__":
    main()
