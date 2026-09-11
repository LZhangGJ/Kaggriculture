"""Audit execution capabilities demonstrated by the latest local gold Replays.

This is diagnostic-only.  It reads public Replay observations/actions and does
not compile Replay actions into the route controller.  The purpose is to find
high-frequency logistics capabilities that the current M3.7 rule executor
cannot yet represent reliably (multi-stop routes, bulk pickup/use, direct
inventory hand-off and end-of-day auto banking).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = (
    ROOT / "replay" / "gold_top20_latest_2026-08-18_082723" / "latest_per_gold"
)
DEFAULT_RECEIPT = DEFAULT_REPLAY_DIR / "latest_gold_replay_receipt.csv"
DEFAULT_JSON = PROJECT / "receipts" / "gold_execution_capability_audit_v1.json"
DEFAULT_MD = PROJECT / "reports" / "GOLD_EXECUTION_CAPABILITY_GAP_AUDIT_V1_ZH.md"

MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
PASS_OPS = {"PASS", "NONE"}
DEPOT_OPS = {"PICKUP", "DROP"}
PRODUCTIVE_OPS = {
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}
SELLABLE = {
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
}
ANIMALS = {"GOOSE", "COW", "SHEEP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def action_parts(value: Any) -> tuple[str, list[Any]]:
    if not value:
        return "PASS", []
    return str(value[0]), list(value[1:])


def tile_at(farm: dict[str, Any], pos: tuple[int, int] | None) -> dict[str, Any]:
    if pos is None:
        return {}
    x, y = pos
    tiles = farm.get("tiles") or []
    if y < 0 or y >= len(tiles) or x < 0 or x >= len(tiles[y]):
        return {}
    tile = tiles[y][x]
    return tile if isinstance(tile, dict) else {}


def inventory_total(inventory: dict[str, Any] | None, names: Iterable[str]) -> int:
    inventory = inventory or {}
    return sum(max(0, int(inventory.get(name, 0) or 0)) for name in names)


def compact_event(event: dict[str, Any]) -> str:
    pos = event.get("pos")
    pos_text = "?" if pos is None else f"{pos[0]},{pos[1]}"
    arg = "" if not event.get("args") else ":" + ":".join(map(str, event["args"]))
    return f"s{event['step']}:{event['op']}{arg}@{pos_text}"


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    key: str,
    context: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> None:
    if len(examples[key]) >= limit:
        return
    examples[key].append(
        {
            **context,
            "events": [compact_event(event) for event in events],
        }
    )


def finalize_route(
    segment: list[dict[str, Any]],
    counters: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    context: dict[str, Any],
) -> None:
    productive = [event for event in segment if event["op"] in PRODUCTIVE_OPS]
    if not productive:
        return
    positions = {tuple(event["pos"]) for event in productive if event.get("pos") is not None}
    ops = [event["op"] for event in productive]
    op_set = set(ops)
    crop_events = [
        event
        for event in productive
        if event["op"] in {"PLANT", "WATER", "FERTILIZE"}
        or (event["op"] == "HARVEST" and event.get("crop"))
    ]
    animal_events = [
        event
        for event in productive
        if event["op"]
        in {
            "BUILD_COOP",
            "BUILD_PASTURE",
            "PLACE",
            "FEED",
            "CARE",
            "COLLECT_FERTILIZER",
        }
        or (event["op"] == "HARVEST" and event.get("animal"))
    ]
    if crop_events and animal_events:
        counters["crop_animal_mixed_route_count"] += 1
        counters["crop_animal_mixed_route_ops"] += len(productive)
        counters["crop_animal_mixed_route_max_ops"] = max(
            counters["crop_animal_mixed_route_max_ops"], len(productive)
        )
        add_example(
            examples,
            "crop_animal_mixed_route",
            context,
            productive,
        )
    if len(productive) >= 2 and len(positions) >= 2:
        counters["multi_target_route_count"] += 1
        counters["multi_target_route_productive_ops"] += len(productive)
        counters["multi_target_route_max_ops"] = max(
            counters["multi_target_route_max_ops"], len(productive)
        )
        counters["multi_target_route_max_tiles"] = max(
            counters["multi_target_route_max_tiles"], len(positions)
        )
        add_example(examples, "multi_target_route", context, productive)

    harvest = [event for event in productive if event["op"] == "HARVEST"]
    harvest_positions = {
        tuple(event["pos"]) for event in harvest if event.get("pos") is not None
    }
    if len(harvest) >= 2 and len(harvest_positions) >= 2:
        counters["multi_harvest_before_drop_count"] += 1
        counters["multi_harvest_before_drop_units"] += len(harvest)
        counters["multi_harvest_before_drop_max_units"] = max(
            counters["multi_harvest_before_drop_max_units"], len(harvest)
        )
        add_example(examples, "multi_harvest_before_drop", context, harvest)

    feeds = [event for event in productive if event["op"] == "FEED"]
    feed_positions = {
        tuple(event["pos"]) for event in feeds if event.get("pos") is not None
    }
    if len(feeds) >= 2 and len(feed_positions) >= 2:
        counters["multi_feed_before_depot_count"] += 1
        counters["multi_feed_before_depot_units"] += len(feeds)
        counters["multi_feed_before_depot_max_units"] = max(
            counters["multi_feed_before_depot_max_units"], len(feeds)
        )
        add_example(examples, "multi_feed_before_depot", context, feeds)

    # PLACE is overloaded by the official API: at a structure it places an
    # animal, while at shed access it can deposit an ordinary item.  Only the
    # former is evidence for a multi-animal placement route.
    places = [
        event
        for event in productive
        if event["op"] == "PLACE"
        and event.get("args")
        and str(event["args"][0]) in ANIMALS
    ]
    place_positions = {
        tuple(event["pos"]) for event in places if event.get("pos") is not None
    }
    if len(places) >= 2 and len(place_positions) >= 2:
        counters["multi_place_before_depot_count"] += 1
        counters["multi_place_before_depot_units"] += len(places)
        counters["multi_place_before_depot_max_units"] = max(
            counters["multi_place_before_depot_max_units"], len(places)
        )
        add_example(examples, "multi_place_before_depot", context, places)

    if len(op_set) >= 2 and len(positions) >= 2:
        counters["mixed_operation_route_count"] += 1
        add_example(examples, "mixed_operation_route", context, productive)

    for op, metric_name, example_name in (
        ("PLANT", "multi_plant_route_count", "multi_plant_route"),
        ("WATER", "multi_water_route_count", "multi_water_route"),
        (
            "COLLECT_FERTILIZER",
            "multi_collect_fertilizer_route_count",
            "multi_collect_fertilizer_route",
        ),
        ("FERTILIZE", "multi_fertilize_route_count", "multi_fertilize_route"),
    ):
        selected_ops = [event for event in productive if event["op"] == op]
        selected_positions = {
            tuple(event["pos"])
            for event in selected_ops
            if event.get("pos") is not None
        }
        if len(selected_ops) >= 2 and len(selected_positions) >= 2:
            counters[metric_name] += 1
            counters[f"{metric_name}_units"] += len(selected_ops)
            counters[f"{metric_name}_max_units"] = max(
                counters[f"{metric_name}_max_units"], len(selected_ops)
            )
            add_example(examples, example_name, context, selected_ops)

    by_position: dict[tuple[int, int], set[str]] = defaultdict(set)
    by_position_events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in productive:
        if event.get("pos") is not None:
            pos = tuple(event["pos"])
            by_position[pos].add(event["op"])
            by_position_events[pos].append(event)
    for pos, same_tile_ops in by_position.items():
        event_list = by_position_events[pos]
        if {"PLANT", "WATER"}.issubset(same_tile_ops):
            counters["same_tile_plant_water_count"] += 1
        if {"HARVEST", "PLANT", "WATER"}.issubset(same_tile_ops):
            counters["same_tile_harvest_replant_water_count"] += 1
            add_example(
                examples,
                "same_tile_harvest_replant_water",
                context,
                event_list,
            )
        if {"FERTILIZE", "WATER"}.issubset(same_tile_ops):
            counters["same_tile_fertilize_water_count"] += 1
        if {"HARVEST", "COLLECT_FERTILIZER"}.issubset(same_tile_ops):
            counters["same_tile_animal_harvest_collect_count"] += 1
            add_example(
                examples,
                "same_tile_animal_harvest_collect",
                context,
                event_list,
            )
        if {"FEED", "CARE"}.issubset(same_tile_ops):
            counters["same_tile_feed_care_count"] += 1
            add_example(examples, "same_tile_feed_care", context, event_list)
        if {"FEED", "COLLECT_FERTILIZER"}.issubset(same_tile_ops):
            counters["same_tile_feed_collect_fertilizer_count"] += 1
        if {"CARE", "COLLECT_FERTILIZER"}.issubset(same_tile_ops):
            counters["same_tile_care_collect_fertilizer_count"] += 1

    wheat_harvest_indices = [
        index
        for index, event in enumerate(productive)
        if event["op"] == "HARVEST" and event.get("crop") == "WHEAT"
    ]
    feed_indices = [
        index for index, event in enumerate(productive) if event["op"] == "FEED"
    ]
    if wheat_harvest_indices and feed_indices and min(feed_indices) > min(wheat_harvest_indices):
        counters["direct_wheat_harvest_to_feed_count"] += 1
        add_example(examples, "direct_wheat_harvest_to_feed", context, productive)

    collect_indices = [
        index
        for index, event in enumerate(productive)
        if event["op"] == "COLLECT_FERTILIZER"
    ]
    fertilize_indices = [
        index
        for index, event in enumerate(productive)
        if event["op"] == "FERTILIZE"
    ]
    if collect_indices and fertilize_indices and min(fertilize_indices) > min(collect_indices):
        counters["direct_collect_to_fertilize_count"] += 1
        add_example(examples, "direct_collect_to_fertilize", context, productive)


def record_batch_chains(
    segment: list[dict[str, Any]],
    counters: Counter[str],
    route_signatures: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    context: dict[str, Any],
) -> None:
    """Record bulk pickup only when the same route consumes it repeatedly."""

    pickups = [
        event
        for event in segment
        if event["op"] == "PICKUP" and event.get("batch_amount", 1) > 1
    ]
    for pickup in pickups:
        item = pickup["batch_item"]
        use_op = (
            "FEED"
            if item == "WHEAT"
            else "PLACE"
            if item in ANIMALS
            else "FERTILIZE"
            if item == "FERTILIZER"
            else None
        )
        if use_op is None:
            continue
        uses = [
            event
            for event in segment
            if event["step"] > pickup["step"]
            and event["op"] == use_op
            and (
                use_op != "PLACE"
                or (event.get("args") and str(event["args"][0]) == item)
            )
        ]
        if len(uses) >= 2:
            counters["productive_batch_pickup_chain_count"] += 1
            counters["productive_batch_pickup_chain_max_uses"] = max(
                counters["productive_batch_pickup_chain_max_uses"], len(uses)
            )
            route_signatures[f"PICKUP_{item}->{use_op}x{len(uses)}"] += 1
            add_example(
                examples,
                "productive_batch_pickup_chain",
                context,
                [pickup, *uses],
            )


def analyze_player(
    replay: dict[str, Any], row: dict[str, str], player: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    counters: Counter[str] = Counter()
    pickup_items: Counter[str] = Counter()
    batch_pickup_items: Counter[str] = Counter()
    route_signatures: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_day_unit: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    steps = replay.get("steps") or []
    for step, frame in enumerate(steps):
        agent = frame[player]
        observation = agent.get("observation") or {}
        action = agent.get("action") or {}
        farms = observation.get("farms") or []
        if player >= len(farms):
            continue
        farm = farms[player]
        day = int(observation.get("day", step // 24) or 0)
        hour = int(observation.get("hour", step % 24) or 0)
        positions = [farm.get("farmer") or [0, 0], *(farm.get("hands") or [])]
        unit_actions = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        inventories = (observation.get("private") or {}).get("inventories") or []
        unit_count = max(len(positions), len(unit_actions), len(inventories))
        while len(unit_actions) < unit_count:
            unit_actions.append(["PASS"])
        while len(positions) < unit_count:
            positions.append(None)

        for unit in range(unit_count):
            op, args = action_parts(unit_actions[unit])
            if op in MOVE_OPS:
                counters["movement_action_count"] += 1
            elif op in PASS_OPS:
                counters["pass_action_count"] += 1
            else:
                counters["non_move_action_count"] += 1
            if op == "PICKUP":
                counters["pickup_count"] += 1
                item = str(args[0]) if args else "UNKNOWN"
                amount = int(args[1]) if len(args) >= 2 else 1
                pickup_items[item] += 1
                if amount > 1:
                    counters["batch_pickup_count"] += 1
                    counters["batch_pickup_units"] += amount
                    counters["batch_pickup_max_amount"] = max(
                        counters["batch_pickup_max_amount"], amount
                    )
                    batch_pickup_items[item] += 1
            if op == "DROP":
                counters["drop_count"] += 1

            if op not in MOVE_OPS | PASS_OPS:
                pos_value = positions[unit]
                pos = None if pos_value is None else (int(pos_value[0]), int(pos_value[1]))
                tile = tile_at(farm, pos)
                event = {
                    "step": step,
                    "day": day,
                    "hour": hour,
                    "unit": unit,
                    "op": op,
                    "args": args,
                    "pos": pos,
                    "crop": tile.get("crop"),
                    "animal": tile.get("animal"),
                    "tile_kind": tile.get("kind"),
                }
                events_by_day_unit[(day, unit)].append(event)

        market = action.get("market") or []
        market_ops = [str(order[0]) for order in market if order]
        counters["market_order_count"] += len(market_ops)
        counters["market_max_orders_per_turn"] = max(
            counters["market_max_orders_per_turn"], len(market_ops)
        )
        if len(market_ops) >= 2:
            counters["multi_market_turn_count"] += 1
        sell_indices = [i for i, op in enumerate(market_ops) if op == "SELL"]
        invest_indices = [
            i
            for i, op in enumerate(market_ops)
            if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND", "HIRE"}
        ]
        if sell_indices and invest_indices and min(sell_indices) < max(invest_indices):
            counters["sell_then_reinvest_turn_count"] += 1
            if len(examples["sell_then_reinvest"]) < 8:
                examples["sell_then_reinvest"].append(
                    {
                        "rank": int(row["rank"]),
                        "team": row["team_name"],
                        "episode_id": int(row["episode_id"]),
                        "day": day + 1,
                        "step": step,
                        "orders": market,
                    }
                )

        if hour == 23:
            for unit in range(unit_count):
                inventory = inventories[unit] if unit < len(inventories) else {}
                carried = inventory_total(inventory, SELLABLE)
                op, _ = action_parts(unit_actions[unit])
                if carried > 0 and op != "DROP":
                    counters["end_day_auto_bank_carrier_count"] += 1
                    counters["end_day_auto_bank_pre_action_units"] += carried
                    add_example(
                        examples,
                        "end_day_auto_bank_carrier",
                        {
                            "rank": int(row["rank"]),
                            "team": row["team_name"],
                            "episode_id": int(row["episode_id"]),
                            "day": day + 1,
                            "unit": unit,
                        },
                        [
                            {
                                "step": step,
                                "op": op,
                                "args": [],
                                "pos": (
                                    None
                                    if positions[unit] is None
                                    else (int(positions[unit][0]), int(positions[unit][1]))
                                ),
                            }
                        ],
                    )
                if op in {"HARVEST", "COLLECT_FERTILIZER"}:
                    counters["end_day_terminal_collection_count"] += 1

    context_base = {
        "rank": int(row["rank"]),
        "team": row["team_name"],
        "episode_id": int(row["episode_id"]),
    }
    for (day, unit), events in events_by_day_unit.items():
        segment: list[dict[str, Any]] = []
        for event in events:
            if event["op"] == "PICKUP":
                finalize_route(
                    segment,
                    counters,
                    examples,
                    {**context_base, "day": day + 1, "unit": unit},
                )
                record_batch_chains(
                    segment,
                    counters,
                    route_signatures,
                    examples,
                    {**context_base, "day": day + 1, "unit": unit},
                )
                segment = [event]
                item = str(event["args"][0]) if event["args"] else "UNKNOWN"
                amount = int(event["args"][1]) if len(event["args"]) >= 2 else 1
                if amount > 1:
                    event["batch_item"] = item
                    event["batch_amount"] = amount
            elif event["op"] == "DROP":
                segment.append(event)
                finalize_route(
                    segment,
                    counters,
                    examples,
                    {**context_base, "day": day + 1, "unit": unit},
                )
                record_batch_chains(
                    segment,
                    counters,
                    route_signatures,
                    examples,
                    {**context_base, "day": day + 1, "unit": unit},
                )
                segment = []
            else:
                segment.append(event)
        finalize_route(
            segment,
            counters,
            examples,
            {**context_base, "day": day + 1, "unit": unit},
        )
        record_batch_chains(
            segment,
            counters,
            route_signatures,
            examples,
            {**context_base, "day": day + 1, "unit": unit},
        )

    return (
        {
            "rank": int(row["rank"]),
            "team_name": row["team_name"],
            "score": float(row["score"]),
            "submission_id": int(row["submission_id"]),
            "episode_id": int(row["episode_id"]),
            "player": player,
            "metrics": dict(sorted(counters.items())),
            "pickup_items": dict(sorted(pickup_items.items())),
            "batch_pickup_items": dict(sorted(batch_pickup_items.items())),
            "batch_route_signatures": dict(route_signatures.most_common()),
        },
        examples,
    )


def metric(player: dict[str, Any], key: str) -> int:
    return int(player["metrics"].get(key, 0))


def aggregate_player_metrics(players: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted(
        {
            key
            for player in players
            for key in player["metrics"]
        }
    )
    totals = {
        key: sum(metric(player, key) for player in players)
        for key in metric_keys
        if not key.endswith("_max_amount")
        and not key.endswith("_max_ops")
        and not key.endswith("_max_tiles")
        and not key.endswith("_max_units")
        and not key.endswith("_max_uses")
        and key != "market_max_orders_per_turn"
    }
    maxima = {
        key: max((metric(player, key) for player in players), default=0)
        for key in metric_keys
        if key.endswith("_max_amount")
        or key.endswith("_max_ops")
        or key.endswith("_max_tiles")
        or key.endswith("_max_units")
        or key.endswith("_max_uses")
        or key == "market_max_orders_per_turn"
    }
    pickup_items: Counter[str] = Counter()
    batch_pickup_items: Counter[str] = Counter()
    for player in players:
        pickup_items.update(player["pickup_items"])
        batch_pickup_items.update(player["batch_pickup_items"])
    return {
        "totals": dict(sorted(totals.items())),
        "maxima": dict(sorted(maxima.items())),
        "pickup_items": dict(sorted(pickup_items.items())),
        "batch_pickup_items": dict(
            sorted(batch_pickup_items.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def capability_rows(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (
            "显式多目标任务链",
            "multi_target_route_count",
            "PARTIAL_OPPORTUNISTIC",
            "一名工人只有一个当前任务；完成后才重新贪心分配，没有目标队列和整条路线资源锁定。",
            "P0",
        ),
        (
            "一次取粮连续喂多只动物",
            "multi_feed_before_depot_count",
            "EXPERIMENTAL_NOT_ACCEPTED",
            "原子执行器支持批量 PICKUP，但正式基线没有可靠的后续目标预留；V19 单独放大取粮已反证。",
            "P0",
        ),
        (
            "一次取多只动物连续放置",
            "multi_place_before_depot_count",
            "MISSING",
            "PLACE 流水线按单只动物建卡，每只通常重新回仓取货。",
            "P2",
        ),
        (
            "连续收获多个地块后统一回仓",
            "multi_harvest_before_drop_count",
            "MISSING",
            "当前 HARVEST 任务立即转入 MOVE_TO_DEPOT，无法先访问第二个成熟地块。",
            "P0",
        ),
        (
            "日终自动回仓/自动入库",
            "end_day_auto_bank_carrier_count",
            "MISSING_AS_PLANNED_ROUTE_MODE",
            "当前收获任务硬性预留返回 shed 和 DROP 路径，没有利用官方日终自动回收库存。",
            "P0",
        ),
        (
            "同格喂养、护理、收肥组合",
            "same_tile_feed_care_count",
            "PARTIAL_OPPORTUNISTIC",
            "可在前一任务完成后偶然续接，但没有把同格动作打包成不可拆的服务卡。",
            "P1",
        ),
        (
            "同格收获后立即补种并浇水",
            "same_tile_harvest_replant_water_count",
            "MISSING",
            "当前作物 HARVEST 先转入回仓链，无法在同一次到访中原地补种并完成当天浇水。",
            "P0",
        ),
        (
            "动物产品收获后同格收肥",
            "same_tile_animal_harvest_collect_count",
            "MISSING",
            "当前动物 HARVEST 成功后立即返仓，没有在同一动物格继续 COLLECT_FERTILIZER。",
            "P1",
        ),
        (
            "一名工人连续浇水多个地块",
            "multi_water_route_count",
            "PARTIAL_OPPORTUNISTIC",
            "逐任务重规划可能偶然连续浇水，但没有预先锁定巡回目标和单位分区。",
            "P1",
        ),
        (
            "一名工人连续收肥多个动物格",
            "multi_collect_fertilizer_route_count",
            "PARTIAL",
            "已有携带肥料后的续接逻辑，但只能从碰巧站在动物格开始，缺少完整收集巡回路线。",
            "P1",
        ),
        (
            "直接用刚收的小麦喂动物",
            "direct_wheat_harvest_to_feed_count",
            "NO_GOLD_EVIDENCE",
            "当前控制器确实不能跨项目直接消费刚收的小麦，但本批 20 名金牌没有出现该链，暂不应投入实现。",
            "DEFER",
        ),
        (
            "收肥后直接给作物施肥",
            "direct_collect_to_fertilize_count",
            "MISSING",
            "动物收肥和作物施肥账本在 shed 处衔接，缺少携带物的跨项目任务交接。",
            "P1",
        ),
        (
            "同回合先卖后买/雇工",
            "sell_then_reinvest_turn_count",
            "IMPLEMENTED",
            "动态事务账本和显式市场顺序已经覆盖，主要风险是十槽截断而非能力缺失。",
            "KEEP",
        ),
        (
            "同格种植后立即浇水",
            "same_tile_plant_water_count",
            "IMPLEMENTED",
            "CROP_PRODUCTION 的 OPERATE 阶段已保持 same-day WATER 连续性。",
            "KEEP",
        ),
    ]
    rows = []
    for name, key, support, gap, priority in specs:
        values = [metric(player, key) for player in players]
        rows.append(
            {
                "capability": name,
                "metric": key,
                "players_with_evidence": sum(value > 0 for value in values),
                "gold_player_count": len(players),
                "observed_count": sum(values),
                "max_per_player": max(values, default=0),
                "current_support": support,
                "gap": gap,
                "priority": priority,
            }
        )
    return rows


def render_markdown(receipt: dict[str, Any]) -> str:
    totals = receipt["aggregate_metrics"]["totals"]
    maxima = receipt["aggregate_metrics"]["maxima"]
    lines = [
        "# 金牌 Replay 执行效率能力缺口审计 v1",
        "",
        f"生成时间：{receipt['measured_at']}",
        "",
        "## 结论",
        "",
        "当前最关键的缺口不是再增加一个买牛/种草莓阈值，而是把单工人的执行表示从“一个当前任务”升级为“有界、多目标、带库存的任务链”。金牌玩家普遍会在一次出发中连续完成多个地块，并利用批量携带、同格组合动作和日终自动入库减少往返。当前控制器只有任务完成后的逐步重规划，能够偶然续接，却不能在出发前锁定后续目标，因此无法稳定复现金牌物流效率。",
        "",
        "本报告只把 Replay 用作能力证据，不播放 Replay 原始动作，也不把观察到的次数当作收益因果。",
        "",
        "关键规模证据：",
        "",
        f"- 共观察到 {totals.get('batch_pickup_count', 0)} 次批量 PICKUP，单次最多 {maxima.get('batch_pickup_max_amount', 0)} 件；其中小麦批量取用占绝大多数。",
        f"- 共观察到 {totals.get('multi_feed_before_depot_count', 0)} 条连续喂养路线，单条最多覆盖 {maxima.get('multi_feed_before_depot_max_units', 0)} 只动物。",
        f"- 共观察到 {totals.get('multi_harvest_before_drop_count', 0)} 条多地块收获路线，单条最多连续收获 {maxima.get('multi_harvest_before_drop_max_units', 0)} 个目标。",
        f"- 连续喂养路线共覆盖 {totals.get('multi_feed_before_depot_units', 0)} 次 FEED，相比逐只回仓少了约 {totals.get('multi_feed_before_depot_units', 0) - totals.get('multi_feed_before_depot_count', 0)} 次重复取粮行程；多地块收获路线同理合并了约 {totals.get('multi_harvest_before_drop_units', 0) - totals.get('multi_harvest_before_drop_count', 0)} 次重复返仓。该数值是动作结构差，不是最终收益估计。",
        f"- 共观察到 {totals.get('end_day_auto_bank_carrier_count', 0)} 个日终仍携带可入库物品且未执行 DROP 的单位时刻，以及 {totals.get('end_day_terminal_collection_count', 0)} 次最后一回合收获/收肥。",
        "",
        "## 覆盖",
        "",
        f"- 金牌玩家：{receipt['scope']['gold_players']} 名。",
        f"- 独立 Replay：{receipt['scope']['unique_replays']} 局。",
        f"- 总帧数：{receipt['scope']['frames']}。",
        f"- 来源：本地已冻结的 `{receipt['scope']['source_snapshot']}` 金牌前 20 最新 Replay 清单。",
        "",
        "## 能力对照",
        "",
        "| 优先级 | 金牌能力 | 有证据玩家 | 观察次数 | 当前支持 | 关键缺口 |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in receipt["capabilities"]:
        lines.append(
            "| {priority} | {capability} | {players_with_evidence}/{gold_player_count} | {observed_count} | {current_support} | {gap} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 每位金牌玩家的核心计数",
            "",
            "| Rank | 玩家 | 多目标路线 | 多次收获后回仓 | 连续喂养 | 连续放置 | 日终自动入库携带者 | 先卖后投资 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for player in sorted(receipt["players"], key=lambda item: item["rank"]):
        m = player["metrics"]
        lines.append(
            f"| {player['rank']} | {player['team_name']} | "
            f"{m.get('multi_target_route_count', 0)} | "
            f"{m.get('multi_harvest_before_drop_count', 0)} | "
            f"{m.get('multi_feed_before_depot_count', 0)} | "
            f"{m.get('multi_place_before_depot_count', 0)} | "
            f"{m.get('end_day_auto_bank_carrier_count', 0)} | "
            f"{m.get('sell_then_reinvest_turn_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 推荐实现顺序",
            "",
            "1. 先实现静态形状的类型化路线卡，而不是任意任务队列：`FEED_TOUR`、`HARVEST_TOUR`、`WATER_TOUR`、`FERTILIZER_TOUR`、`PLACE_TOUR`。容量按 Replay 实测上限留余量：FEED/HARVEST/PLANT/COLLECT/FERTILIZE 各 8 个目标，WATER 12 个目标，PLACE 4 个目标；超出时确定性拆成下一张卡。",
            "2. `FEED_TOUR` 一次锁定 N 个动物目标并 PICKUP N 份小麦；每个目标可带 `FEED/CARE/COLLECT` 位掩码。这样携带量和后续消费目标严格一一对应。",
            "3. `HARVEST_TOUR` 锁定多个成熟目标，并显式选择 `RETURN_AND_DROP` 或 `AUTO_BANK_AT_DAY_END`；仍以官方 1.32.7 逐步一致性验收。",
            "4. `PLACE_TOUR` 批量取同类动物后连续放置，目标结构在出发前一次性预留，避免多个单位抢同一地块。",
            "5. 最后做跨项目交接：`COLLECT_FERTILIZER -> FERTILIZE`。本批 Replay 没有观察到 `HARVEST_WHEAT -> FEED`，因此后者不应排在近期实现队列。",
            "",
            "不要直接实现任意长度通用队列。按任务类型设置固定容量，既覆盖本批 Replay 的观测最大值（FEED 6、HARVEST 8、PLANT 7、WATER 10、COLLECT 5、FERTILIZE 8），也能保持 JAX 静态形状和可控编译成本。",
            "",
            "## 边界",
            "",
            "- `DIAGNOSTIC_ONLY`：计数证明高手使用了这些能力，不证明单独加入某项必然提高最终现金。",
        "- 当前 V19 的批量取粮实验不算实现成功：它没有绑定后续目标，导致重复预留与携带资源滞留。",
        "- 20 个玩家侧并不等于 20 套独立架构；部分 Replay 路线高度相似。玩家覆盖率说明能力普遍存在，不是独立因果投票。",
            "- 下一阶段必须逐项做官方逐步一致性、零硬错误、固定种子收益和动作效率消融。",
            "",
            "## 复现",
            "",
            "```powershell",
            receipt["reproduce_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.receipt.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row.get("status") in {"downloaded", "existing_valid"}
    ]

    players: list[dict[str, Any]] = []
    merged_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replay_meta: list[dict[str, Any]] = []
    grouped: dict[Path, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[ROOT / row["path"]].append(row)

    frame_count = 0
    for path, path_rows in sorted(
        grouped.items(),
        key=lambda item: min(int(row["rank"]) for row in item[1]),
    ):
        replay = json.loads(path.read_text(encoding="utf-8"))
        names = list((replay.get("info") or {}).get("TeamNames") or [])
        frame_count += len(replay.get("steps") or [])
        replay_meta.append(
            {
                "episode_id": int((replay.get("info") or {}).get("EpisodeId", 0)),
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "team_names": names,
                "steps": len(replay.get("steps") or []),
            }
        )
        for row in path_rows:
            if row["team_name"] not in names:
                raise RuntimeError(
                    f"team {row['team_name']!r} absent from episode {row['episode_id']}"
                )
            player_index = names.index(row["team_name"])
            player, examples = analyze_player(replay, row, player_index)
            players.append(player)
            for key, values in examples.items():
                remaining = max(0, 8 - len(merged_examples[key]))
                merged_examples[key].extend(values[:remaining])

    receipt = {
        "receipt_id": "GOLD_EXECUTION_CAPABILITY_AUDIT_V1",
        "status": "PASS",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "gold_players": len(players),
            "unique_replays": len(grouped),
            "frames": frame_count,
            "source_snapshot": args.receipt.resolve().parents[1].name,
            "diagnostic_only": True,
            "raw_action_playback": False,
        },
        "source_receipt": {
            "path": str(args.receipt.relative_to(ROOT)),
            "sha256": sha256(args.receipt),
        },
        "replays": replay_meta,
        "players": sorted(players, key=lambda item: item["rank"]),
        "aggregate_metrics": aggregate_player_metrics(players),
        "capabilities": capability_rows(players),
        "examples": dict(sorted(merged_examples.items())),
        "reproduce_command": (
            "& 'E:\\ai_coding\\kaggle\\kaggriculture\\.venv\\python.exe' "
            "experiments\\project_route_search_v2\\tools\\analyze_gold_execution_capabilities_v1.py"
        ),
        "boundary": "DIAGNOSTIC_ONLY_GOLD_REPLAY_CAPABILITY_EVIDENCE",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "gold_players": len(players),
                "unique_replays": len(grouped),
                "capabilities": receipt["capabilities"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
