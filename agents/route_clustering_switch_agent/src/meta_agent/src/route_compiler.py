"""Compile RouteGenome mutations into executable replay-anchored plans."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .route_plan import (
    CarrierRoute,
    CompileReport,
    CompiledRoutePlan,
    ProductionTransform,
    RouteTask,
    normalized_action,
)


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_COST = (1000, 2000, 4000)
_RELOCATE = re.compile(
    r"relocate\s+(?P<kind>[A-Z_]+)\s+\((?P<ox>\d+),(?P<oy>\d+)\)"
    r"->\((?P<nx>\d+),(?P<ny>\d+)\)"
)
_SUBSTITUTE = re.compile(
    r"substitute\s+\((?P<x>\d+),(?P<y>\d+)\)\s+"
    r"(?P<old>[A-Z_]+)->(?P<new>[A-Z_]+)"
)
_EVENT_SHIFT = re.compile(
    r"event\[(?P<index>\d+)\]\s+step\s+(?P<old>\d+)->(?P<new>\d+)"
)


def _genome_id(value: Mapping[str, Any]) -> str:
    return str(value.get("genome_id") or value.get("id") or "unknown")


def _payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return value.get("genetic_payload") or value


def _market_rows(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = _payload(value)
    return list(
        payload.get("market_actions")
        or payload.get("market_profile")
        or value.get("market_profile")
        or []
    )


def _market_totals(value: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in _market_rows(value):
        key = (str(row.get("operation") or ""), str(row.get("item") or ""))
        totals[key] += max(0, int(row.get("quantity", 0) or 0))
    return dict(totals)


def _order_quantity(order: Sequence[Any]) -> int:
    if len(order) >= 3:
        try:
            return max(0, int(order[2] or 0))
        except (TypeError, ValueError):
            return 0
    return 1


def _set_order_quantity(order: list[Any], quantity: int) -> None:
    while len(order) < 3:
        order.append(1)
    order[2] = max(0, int(quantity))


def _market_locations(
    actions: list[dict[str, Any]], operation: str, item: str
) -> list[tuple[int, int, int]]:
    found = []
    for step, action in enumerate(actions):
        for index, order in enumerate(action.get("market", []) or []):
            if not order or str(order[0]) != operation:
                continue
            order_item = str(order[1]) if len(order) >= 2 else ""
            if order_item == item:
                found.append((step, index, _order_quantity(order)))
    return found


def _adjust_market_total(
    actions: list[dict[str, Any]], operation: str, item: str, delta: int
) -> int:
    """Adjust aggregate tape quantity, preserving order count when possible."""

    remaining = int(delta)
    locations = _market_locations(actions, operation, item)
    if remaining < 0:
        for step, index, quantity in sorted(locations, key=lambda value: -value[2]):
            take = min(quantity, -remaining)
            order = actions[step]["market"][index]
            _set_order_quantity(order, quantity - take)
            remaining += take
            if remaining == 0:
                break
        for action in actions:
            action["market"] = [
                order for order in action.get("market", []) or []
                if _order_quantity(order) > 0
            ]
    elif remaining > 0:
        if locations:
            step, index, quantity = max(locations, key=lambda value: value[2])
            _set_order_quantity(actions[step]["market"][index], quantity + remaining)
            remaining = 0
        else:
            order = [operation]
            if item:
                order.extend([item, remaining])
            actions[0]["market"].append(order)
            remaining = 0
    return int(delta) - remaining


def _unit_orders(action: Mapping[str, Any]) -> list[list[Any]]:
    return [list(action.get("farmer") or ["PASS"]), *[
        list(value or ["PASS"]) for value in action.get("hands", []) or []
    ]]


def _set_unit_order(action: dict[str, Any], actor: int, order: Sequence[Any]) -> bool:
    if actor == 0:
        action["farmer"] = list(order)
        return True
    hands = action.get("hands", []) or []
    if 0 < actor <= len(hands):
        hands[actor - 1] = list(order)
        action["hands"] = hands
        return True
    return False


def _kind_for_order(order: Sequence[Any]) -> str | None:
    if not order:
        return None
    operation = str(order[0])
    item = str(order[1]) if len(order) >= 2 else ""
    if operation == "PLANT" and item in CROPS:
        return item
    if operation == "BUILD_COOP":
        return "COOP"
    if operation == "BUILD_PASTURE":
        return "PASTURE"
    if operation == "PLACE" and item in ANIMALS:
        return item
    return None


def _order_for_kind(kind: str) -> list[str]:
    if kind in CROPS:
        return ["PLANT", kind]
    if kind == "COOP":
        return ["BUILD_COOP"]
    if kind == "PASTURE":
        return ["BUILD_PASTURE"]
    if kind in ANIMALS:
        return ["PLACE", kind]
    raise ValueError(f"unsupported production kind: {kind}")


def _production_transforms(detail: str) -> list[ProductionTransform]:
    """Compose sequential mutation details into carrier-to-target edits."""

    transforms: list[ProductionTransform] = []
    matches = []
    for match in _RELOCATE.finditer(detail):
        matches.append((match.start(), "relocate", match))
    for match in _SUBSTITUTE.finditer(detail):
        matches.append((match.start(), "substitute", match))
    for _offset, kind, match in sorted(matches):
        if kind == "relocate":
            item = match.group("kind")
            old_xy = (int(match.group("ox")), int(match.group("oy")))
            new_xy = (int(match.group("nx")), int(match.group("ny")))
            index = next((
                i for i, value in enumerate(transforms)
                if value.target_xy == old_xy and value.target_kind == item
            ), None)
            if index is None:
                transforms.append(ProductionTransform(old_xy, item, new_xy, item))
            else:
                value = transforms[index]
                transforms[index] = ProductionTransform(
                    value.source_xy, value.source_kind, new_xy, value.target_kind
                )
        else:
            xy = (int(match.group("x")), int(match.group("y")))
            old_kind, new_kind = match.group("old"), match.group("new")
            index = next((
                i for i, value in enumerate(transforms)
                if value.target_xy == xy and value.target_kind == old_kind
            ), None)
            if index is None:
                transforms.append(ProductionTransform(xy, old_kind, xy, new_kind))
            else:
                value = transforms[index]
                transforms[index] = ProductionTransform(
                    value.source_xy, value.source_kind, value.target_xy, new_kind
                )
    return transforms


def _append_market_order(action: dict[str, Any], order: Sequence[Any]) -> bool:
    market = action.setdefault("market", [])
    if len(market) >= 10:
        return False
    market.append(list(order))
    return True


def _move_market_event(
    actions: list[dict[str, Any]], old_step: int, new_step: int, operation: str
) -> tuple[bool, bool]:
    """Move one market event, swapping the last target order if it is full."""

    if not 0 <= old_step < len(actions) or not 0 <= new_step < len(actions):
        return False, False
    old_market = actions[old_step].get("market", []) or []
    located = next((
        index for index, order in enumerate(old_market)
        if order and str(order[0]) == operation
    ), None)
    if located is None:
        return False, False
    moved = old_market.pop(located)
    new_market = actions[new_step].get("market", []) or []
    displaced = None
    if len(new_market) >= 10:
        displaced = new_market.pop()
    new_market.append(moved)
    if displaced is not None:
        old_market.append(displaced)
    actions[old_step]["market"] = old_market
    actions[new_step]["market"] = new_market
    return True, displaced is not None


def _find_unit_event(
    carrier: CarrierRoute,
    actions: list[dict[str, Any]],
    *,
    step: int,
    operation: str,
    item: str,
    xy: tuple[int, int],
) -> tuple[int, list[Any]] | None:
    if not 0 <= step < len(actions):
        return None
    for actor, order in enumerate(_unit_orders(actions[step])):
        if not order or str(order[0]) != operation:
            continue
        order_item = str(order[1]) if len(order) >= 2 else ""
        if item and order_item != item:
            continue
        if carrier.actor_position(step, actor) == xy:
            return actor, order
    return None


def compile_route_genome(
    carrier: CarrierRoute,
    parent: Mapping[str, Any],
    target: Mapping[str, Any],
) -> CompiledRoutePlan:
    """Compile one mutation using a carrier tape and dynamic macro detours."""

    actions = [normalized_action(value) for value in carrier.actions]
    report = CompileReport(_genome_id(parent), _genome_id(target))
    parent_payload = _payload(parent)
    experiment = dict(target.get("experiment") or {})
    detail = str(experiment.get("detail") or "")
    tasks: list[RouteTask] = []

    parent_market = _market_totals(parent)
    target_market = _market_totals(target)
    for key in sorted(set(parent_market) | set(target_market)):
        delta = int(target_market.get(key, 0)) - int(parent_market.get(key, 0))
        if not delta:
            continue
        operation, item = key
        applied = _adjust_market_total(actions, operation, item, delta)
        label = f"{operation}:{item or '-'}"
        report.static_market_quantity_delta[label] = applied
        report.static_patches += int(applied != 0)
        if applied != delta:
            report.warnings.append(f"market delta only partly applied for {label}")

    transforms = _production_transforms(detail)
    for transform_index, transform in enumerate(transforms):
        changed_actions = 0
        target_order = _order_for_kind(transform.target_kind)
        for step, action in enumerate(actions):
            for actor, order in enumerate(_unit_orders(action)):
                if _kind_for_order(order) != transform.source_kind:
                    continue
                if carrier.actor_position(step, actor) != transform.source_xy:
                    continue
                if (
                    transform.source_xy == transform.target_xy
                    and transform.source_kind == transform.target_kind
                ):
                    changed_actions += 1
                    continue
                else:
                    _set_unit_order(action, actor, ["PASS"])
                    tasks.append(RouteTask(
                        task_id=f"production-{transform_index}-{step}-{actor}",
                        due_step=step,
                        activation_step=step,
                        operation=target_order[0],
                        item=target_order[1] if len(target_order) > 1 else None,
                        target_xy=transform.target_xy,
                        actor_index=actor,
                        return_xy=transform.source_xy,
                        source_step=step,
                    ))
                changed_actions += 1

        if changed_actions == 0:
            report.unsupported.append(
                f"no carrier action matched {transform.source_kind}@{transform.source_xy}"
            )
        if transform.source_kind in CROPS and transform.target_kind in CROPS:
            count = changed_actions
            if transform.source_kind != transform.target_kind and count:
                old_delta = _adjust_market_total(
                    actions, "BUY_SEED", transform.source_kind, -count
                )
                old_label = f"BUY_SEED:{transform.source_kind}"
                new_label = f"BUY_SEED:{transform.target_kind}"
                report.prerequisite_market_delta[old_label] = (
                    report.prerequisite_market_delta.get(old_label, 0) + old_delta
                )
                # New seed is bought on demand by the dynamic task.  Bulk-buying
                # it at the carrier's largest order caused avoidable cash cascades.
                report.prerequisite_market_delta.setdefault(new_label, 0)
        report.production_transforms.append({
            "source_xy": list(transform.source_xy),
            "source_kind": transform.source_kind,
            "target_xy": list(transform.target_xy),
            "target_kind": transform.target_kind,
            "matched_actions": changed_actions,
        })

    parent_events = list(parent_payload.get("structural_events", []) or [])
    event_matches = list(_EVENT_SHIFT.finditer(detail))
    for event_match in event_matches:
        index = int(event_match.group("index"))
        old_step = int(event_match.group("old"))
        new_step = int(event_match.group("new"))
        if not 0 <= index < len(parent_events):
            report.unsupported.append(f"event index out of range: {index}")
            continue
        event = parent_events[index]
        operation = str(event.get("operation") or "")
        item = str(event.get("item") or "")
        event_report = {
            "event_index": index,
            "operation": operation,
            "item": item or None,
            "old_step": old_step,
            "new_step": new_step,
        }
        if operation in ("HIRE", "BUY_LAND"):
            moved, displaced = _move_market_event(
                actions, old_step, new_step, operation
            )
            if not moved:
                report.unsupported.append(
                    f"could not move market event {operation} {old_step}->{new_step}"
                )
            else:
                report.static_patches += 2
            event_report["mode"] = (
                "static_market_swap" if displaced else "static_market_move"
            )
        else:
            xy = (int(event.get("x", -1)), int(event.get("y", -1)))
            located = _find_unit_event(
                carrier, actions, step=old_step, operation=operation,
                item=item, xy=xy,
            )
            if located is None:
                report.unsupported.append(
                    f"could not locate unit event {index} at step {old_step}"
                )
            else:
                actor, order = located
                _set_unit_order(actions[old_step], actor, ["PASS"])
                tasks.append(RouteTask(
                    task_id=f"event-shift-{index}",
                    due_step=new_step,
                    activation_step=max(0, new_step - 12),
                    operation=str(order[0]),
                    item=str(order[1]) if len(order) > 1 else None,
                    target_xy=xy,
                    actor_index=None,
                    return_xy=None,
                    source_step=old_step,
                    grace_steps=48,
                ))
                if operation == "PLACE":
                    report.warnings.append(
                        f"shifted animal event {index} may require a fresh pickup"
                    )
                event_report["mode"] = "state_aware_task"
        report.event_shifts.append(event_report)

    if detail and not transforms and not event_matches:
        if not report.static_market_quantity_delta:
            report.unsupported.append(f"unrecognized mutation detail: {detail[:160]}")
    tasks.sort(key=lambda value: (value.activation_step, value.due_step, value.task_id))
    report.tasks = len(tasks)
    return CompiledRoutePlan(
        carrier=carrier,
        actions=tuple(actions),
        tasks=tuple(tasks),
        report=report,
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _seat(observation: Any) -> int:
    return 1 if int(_get(observation, "player", 0) or 0) == 1 else 0


def _farm(observation: Any) -> Any:
    seat = _seat(observation)
    farms = list(_get(observation, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _positions(observation: Any) -> list[tuple[int, int]]:
    farm = _farm(observation)
    values = [_get(farm, "farmer", (0, 0)), *list(_get(farm, "hands", []) or [])]
    return [(int(value[0]), int(value[1])) for value in values]


def _tile(observation: Any, xy: tuple[int, int]) -> Any:
    try:
        x, y = xy
        return list(_get(_farm(observation), "tiles", []) or [])[y][x]
    except (IndexError, TypeError):
        return "LOCKED"


def _private_dict(observation: Any, key: str) -> dict[str, int]:
    private = _get(observation, "private", {}) or {}
    return {
        str(name): max(0, int(value or 0))
        for name, value in dict(_get(private, key, {}) or {}).items()
    }


def _inventory(observation: Any, actor: int) -> dict[str, int]:
    private = _get(observation, "private", {}) or {}
    values = list(_get(private, "inventories", []) or [])
    if not 0 <= actor < len(values):
        return {}
    return {
        str(name): max(0, int(value or 0))
        for name, value in dict(values[actor] or {}).items()
    }


def _move_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _nearest_actor(positions: Sequence[tuple[int, int]], target: tuple[int, int]) -> int:
    return min(
        range(len(positions)),
        key=lambda index: (
            abs(positions[index][0] - target[0]) + abs(positions[index][1] - target[1]),
            index,
        ),
    )


def _expected(tile: Any, operation: str, item: str | None) -> bool:
    if not isinstance(tile, Mapping):
        return False
    if operation == "PLANT":
        return str(tile.get("kind") or "") == "PLANT" and tile.get("crop") == item
    if operation == "BUILD_COOP":
        return str(tile.get("kind") or "") == "COOP"
    if operation == "BUILD_PASTURE":
        return str(tile.get("kind") or "") == "PASTURE"
    if operation == "PLACE":
        return tile.get("animal") == item
    return False


def _align_action(action: Mapping[str, Any], hand_count: int) -> dict[str, Any]:
    value = normalized_action(action)
    hands = value["hands"][:hand_count]
    hands.extend([["PASS"] for _ in range(hand_count - len(hands))])
    value["hands"] = hands
    value["market"] = value["market"][:10]
    return value


def _ensure_market_order(action: dict[str, Any], order: Sequence[Any]) -> bool:
    """Insert a prerequisite order, replacing the last low-priority slot."""

    operation = str(order[0]) if order else ""
    item = str(order[1]) if len(order) > 1 else ""
    market = action.setdefault("market", [])
    if any(
        value and str(value[0]) == operation
        and (not item or (len(value) > 1 and str(value[1]) == item))
        for value in market
    ):
        return False
    if len(market) >= 10:
        market.pop()
    market.append(list(order))
    return True


def _money(observation: Any) -> float:
    return float(_get(_farm(observation), "money", 0.0) or 0.0)


def _market_prices(observation: Any) -> dict[str, int]:
    market = _get(observation, "market", {}) or {}
    return {
        str(name): max(0, int(value or 0))
        for name, value in dict(_get(market, "prices", {}) or {}).items()
    }


def _ensure_liquidity(
    observation: Any, action: dict[str, Any], required_cash: float
) -> int:
    """Prepend enough shed sales to fund a compiler prerequisite if possible."""

    prices = _market_prices(observation)
    shed = _private_dict(observation, "shed")
    market = action.setdefault("market", [])
    planned: dict[str, int] = defaultdict(int)
    projected_cash = _money(observation)
    for order in market:
        if len(order) >= 3 and str(order[0]) == "SELL":
            product = str(order[1])
            quantity = max(0, int(order[2] or 0))
            planned[product] += quantity
            projected_cash += quantity * prices.get(product, 0)
    deficit = max(0.0, float(required_cash) - projected_cash)
    if deficit <= 0:
        return 0

    rescue = []
    for product in sorted(
        prices,
        key=lambda name: (prices.get(name, 0), shed.get(name, 0), name),
        reverse=True,
    ):
        available = max(0, shed.get(product, 0) - planned.get(product, 0))
        price = prices.get(product, 0)
        if available <= 0 or price <= 0:
            continue
        quantity = min(available, max(1, int((deficit + price - 1) // price)))
        rescue.append(["SELL", product, quantity])
        deficit -= quantity * price
        if deficit <= 0:
            break
    if not rescue:
        return 0
    while len(market) + len(rescue) > 9:
        market.pop()
    action["market"] = rescue + market
    return len(rescue)


class CompiledRouteAgent:
    """Execute a compiled plan with state-aware prerequisite and retry logic."""

    def __init__(self, plan: CompiledRoutePlan) -> None:
        self.plan = plan
        self.runtime: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.runtime = {
            "last_step": -1,
            "next_task": 0,
            "active": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "task_retries": 0,
            "prerequisite_orders": 0,
            "actor_rebinds": 0,
        }

    def _activate(self, observation: Any, step: int) -> None:
        if self.runtime["active"] is not None:
            return
        index = int(self.runtime["next_task"])
        if index >= len(self.plan.tasks):
            return
        task = self.plan.tasks[index]
        if step < task.activation_step:
            return
        positions = _positions(observation)
        if not positions:
            return
        actor = task.actor_index
        if actor is None or actor >= len(positions):
            actor = _nearest_actor(positions, task.target_xy)
            if task.actor_index is not None:
                self.runtime["actor_rebinds"] += 1
        self.runtime["active"] = {
            "task": task,
            "actor": actor,
            "origin": task.return_xy or positions[actor],
            "phase": "approach",
            "attempts": 0,
        }

    def _finish(self, success: bool) -> None:
        active = self.runtime.get("active")
        if active is None:
            return
        key = "completed_tasks" if success else "failed_tasks"
        self.runtime[key].append(active["task"].task_id)
        self.runtime["next_task"] += 1
        self.runtime["active"] = None

    def _task_action(
        self, observation: Any, action: dict[str, Any], step: int
    ) -> dict[str, Any]:
        active = self.runtime.get("active")
        if active is None:
            return action
        task: RouteTask = active["task"]
        positions = _positions(observation)
        actor = int(active["actor"])
        if actor >= len(positions):
            if not positions:
                return action
            actor = _nearest_actor(positions, task.target_xy)
            active["actor"] = actor
            self.runtime["actor_rebinds"] += 1
        position = positions[actor]
        target = task.target_xy

        if step > task.due_step + task.grace_steps:
            self._finish(False)
            return action
        if _expected(_tile(observation, target), task.operation, task.item):
            active["phase"] = "return"

        if active["phase"] == "return":
            origin = tuple(active["origin"])
            if position == origin:
                self._finish(True)
                return action
            _set_unit_order(action, actor, _move_toward(position, origin))
            return action

        if _tile(observation, target) == "LOCKED":
            unlocked = len(_get(_farm(observation), "unlocked_quadrants", []) or [])
            land_index = max(0, min(len(LAND_COST) - 1, unlocked - 1))
            self.runtime["prerequisite_orders"] += _ensure_liquidity(
                observation, action, LAND_COST[land_index]
            )
            if _ensure_market_order(action, ["BUY_LAND"]):
                self.runtime["prerequisite_orders"] += 1
            if position != target:
                _set_unit_order(action, actor, _move_toward(position, target))
            else:
                _set_unit_order(action, actor, ["PASS"])
            return action

        if task.operation == "PLACE" and task.item:
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
                    self.runtime["prerequisite_orders"] += _ensure_liquidity(
                        observation, action, ANIMAL_COST.get(task.item, 0)
                    )
                    if _ensure_market_order(action, ["BUY_ANIMAL", task.item, 1]):
                        self.runtime["prerequisite_orders"] += 1
                return action

        if position != target:
            _set_unit_order(action, actor, _move_toward(position, target))
            return action
        if step < task.due_step:
            _set_unit_order(action, actor, ["PASS"])
            return action

        tile = _tile(observation, target)
        if (
            task.operation in ("PLANT", "BUILD_COOP", "BUILD_PASTURE")
            and tile is not None
            and not _expected(tile, task.operation, task.item)
        ):
            _set_unit_order(action, actor, ["DIG"])
            active["attempts"] += 1
            self.runtime["task_retries"] += 1
            return action
        if task.operation == "PLANT" and task.item:
            if _private_dict(observation, "seeds").get(task.item, 0) <= 0:
                _set_unit_order(action, actor, ["PASS"])
                self.runtime["prerequisite_orders"] += _ensure_liquidity(
                    observation, action, SEED_COST.get(task.item, 0)
                )
                if _ensure_market_order(action, ["BUY_SEED", task.item, 1]):
                    self.runtime["prerequisite_orders"] += 1
                return action

        order = [task.operation]
        if task.item:
            order.append(task.item)
        _set_unit_order(action, actor, order)
        active["attempts"] += 1
        if active["attempts"] > 1:
            self.runtime["task_retries"] += 1
        return action

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        step = min(
            max(0, int(_get(observation, "step", 0) or 0)),
            len(self.plan.actions) - 1,
        )
        if step == 0 or step < int(self.runtime.get("last_step", -1)):
            self.reset()
        self.runtime["last_step"] = step
        hand_count = len(_get(_farm(observation), "hands", []) or [])
        action = _align_action(self.plan.actions[step], hand_count)
        self._activate(observation, step)
        action = self._task_action(observation, action, step)
        action["market"] = list(action.get("market", []) or [])[:10]
        return action


class TapeAgent:
    """Replay a static tape while aligning the daily farm-hand count."""

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self.actions = tuple(actions)

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        step = min(
            max(0, int(_get(observation, "step", 0) or 0)),
            len(self.actions) - 1,
        )
        hand_count = len(_get(_farm(observation), "hands", []) or [])
        return _align_action(self.actions[step], hand_count)
