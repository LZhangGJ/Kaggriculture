"""Read-only Replay diagnostics for Kaggriculture 1.32.7.

The official Replay schema stores the action that transitions from state
``steps[t - 1]`` to state ``steps[t]`` on ``steps[t]``.  The helpers below
explain that transition without mutating or re-simulating the episode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


TURNS_PER_DAY = 24
PLANT_CRITICAL_HOUR = 16
ANIMAL_CRITICAL_HOUR = 15
SHED_HIGH_WATERMARK = 90
SHED_CAPACITY = 100
ANIMAL_MAX_HELD = {"GOOSE": 4, "COW": 6, "SHEEP": 6}

MOVE_DELTAS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
    "EAST": (1, 0),
}

ACTION_ZH = {
    "PASS": "等待",
    "NORTH": "向北移动",
    "SOUTH": "向南移动",
    "WEST": "向西移动",
    "EAST": "向东移动",
    "PLANT": "种植",
    "WATER": "浇水",
    "HARVEST": "收获",
    "FERTILIZE": "施肥",
    "DIG": "挖除",
    "BUILD_COOP": "建鸡舍",
    "BUILD_PASTURE": "建牧场",
    "FEED": "喂养",
    "CARE": "照料",
    "COLLECT_FERTILIZER": "收集肥料",
    "PICKUP": "从仓库取货",
    "PLACE": "放置/入仓",
    "DROP": "全部入仓",
    "BUY_SEED": "购买种子",
    "BUY_PRODUCT": "购买商品",
    "BUY_ANIMAL": "购买动物",
    "SELL": "出售",
    "HIRE": "雇工",
    "BUY_LAND": "购买土地",
}

ACTION_BADGES = {
    "PLANT": "种",
    "WATER": "水",
    "HARVEST": "收",
    "FERTILIZE": "肥",
    "DIG": "挖",
    "BUILD_COOP": "舍",
    "BUILD_PASTURE": "场",
    "FEED": "喂",
    "CARE": "护",
    "COLLECT_FERTILIZER": "肥+",
    "PICKUP": "取",
    "PLACE": "放",
    "DROP": "仓",
    "NORTH": "↑",
    "SOUTH": "↓",
    "WEST": "←",
    "EAST": "→",
}


class ReplayValidationError(ValueError):
    """Raised when a file is not a usable Kaggriculture Replay."""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _tile_kind(tile: Any) -> str:
    return str(tile.get("kind", "")) if isinstance(tile, dict) else ""


def _sum_inventory(inventory: Any) -> int:
    if not isinstance(inventory, dict):
        return 0
    return sum(max(0, int(value or 0)) for value in inventory.values())


def _inventory_delta(before: Any, after: Any) -> dict[str, int]:
    left = _as_dict(before)
    right = _as_dict(after)
    keys = set(left) | set(right)
    return {
        key: int(right.get(key, 0) or 0) - int(left.get(key, 0) or 0)
        for key in sorted(keys)
        if int(right.get(key, 0) or 0) != int(left.get(key, 0) or 0)
    }


def _delta_text(delta: dict[str, int]) -> str:
    if not delta:
        return "无变化"
    return "，".join(
        f"{item}{value:+d}" for item, value in delta.items()
    )


def _tile_label(tile: Any) -> str:
    if tile is None:
        return "空地"
    if tile == "LOCKED":
        return "未解锁土地"
    if not isinstance(tile, dict):
        return str(tile)
    kind = str(tile.get("kind", "未知"))
    if kind == "PLANT":
        return f"{tile.get('crop', '作物')}"
    if "animal" in tile:
        return f"{tile.get('animal', '动物')}"
    return {"WEED": "杂草", "COOP": "空鸡舍", "PASTURE": "空牧场"}.get(
        kind, kind
    )


def _command(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return ["PASS"]
    return value


@dataclass(frozen=True)
class ReplayMeta:
    path: str
    episode_id: Any
    version: str
    module_version: str
    steps: int
    players: list[str]
    rewards: list[Any]
    statuses: list[Any]
    seed: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "episode_id": self.episode_id,
            "version": self.version,
            "module_version": self.module_version,
            "steps": self.steps,
            "players": self.players,
            "rewards": self.rewards,
            "statuses": self.statuses,
            "seed": self.seed,
        }


class ReplayDiagnostics:
    """Validated Replay plus per-frame explanations and alarm timeline."""

    def __init__(self, replay: dict[str, Any], source_path: str = "") -> None:
        self.replay = replay
        self.source_path = source_path
        self._validate()
        self._timeline = self._build_timeline()

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayDiagnostics":
        replay_path = Path(path).expanduser().resolve()
        if not replay_path.is_file():
            raise ReplayValidationError(f"Replay 文件不存在：{replay_path}")
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayValidationError(f"Replay JSON 读取失败：{exc}") from exc
        return cls(replay, str(replay_path))

    def _validate(self) -> None:
        if not isinstance(self.replay, dict):
            raise ReplayValidationError("Replay 顶层必须是 JSON object")
        if self.replay.get("name") != "kaggriculture":
            raise ReplayValidationError(
                f"不是 Kaggriculture Replay：name={self.replay.get('name')!r}"
            )
        steps = self.replay.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ReplayValidationError("Replay 缺少 steps")
        for index in (0, len(steps) - 1):
            row = steps[index]
            if not isinstance(row, list) or len(row) != 2:
                raise ReplayValidationError(f"steps[{index}] 不是双座位状态")
            if not all(isinstance(item, dict) and "observation" in item for item in row):
                raise ReplayValidationError(f"steps[{index}] 缺少 observation")

    @property
    def step_count(self) -> int:
        return len(self.replay["steps"])

    def meta(self) -> dict[str, Any]:
        info = _as_dict(self.replay.get("info"))
        names = _as_list(info.get("TeamNames"))
        if len(names) < 2:
            agents = _as_list(info.get("Agents"))
            names = [str(_as_dict(agent).get("Name", f"Player {i + 1}")) for i, agent in enumerate(agents)]
        while len(names) < 2:
            names.append(f"Player {len(names) + 1}")
        return ReplayMeta(
            path=self.source_path,
            episode_id=info.get("EpisodeId", self.replay.get("id")),
            version=str(self.replay.get("version", "")),
            module_version=str(self.replay.get("module_version", "")),
            steps=self.step_count,
            players=[str(names[0]), str(names[1])],
            rewards=_as_list(self.replay.get("rewards")),
            statuses=_as_list(self.replay.get("statuses")),
            seed=info.get("seed", _as_dict(self.replay.get("configuration")).get("seed")),
        ).as_dict()

    def _obs(self, step: int, seat: int) -> dict[str, Any]:
        return _as_dict(self.replay["steps"][step][seat].get("observation"))

    def _farm(self, step: int, seat: int) -> dict[str, Any]:
        farms = _as_list(self._obs(step, seat).get("farms"))
        return _as_dict(farms[seat]) if seat < len(farms) else {}

    def _private(self, step: int, seat: int) -> dict[str, Any]:
        return _as_dict(self._obs(step, seat).get("private"))

    def _tile(self, step: int, seat: int, x: int, y: int) -> Any:
        rows = _as_list(self._farm(step, seat).get("tiles"))
        if not (0 <= y < len(rows)):
            return None
        row = _as_list(rows[y])
        return row[x] if 0 <= x < len(row) else None

    def _positions(self, step: int, seat: int) -> list[list[int]]:
        farm = self._farm(step, seat)
        return [
            _as_list(farm.get("farmer")) or [0, 0],
            *[_as_list(pos) for pos in _as_list(farm.get("hands"))],
        ]

    def _inventories(self, step: int, seat: int) -> list[dict[str, Any]]:
        return [_as_dict(inv) for inv in _as_list(self._private(step, seat).get("inventories"))]

    def _shed(self, step: int, seat: int) -> dict[str, Any]:
        return _as_dict(self._private(step, seat).get("shed"))

    def _seeds(self, step: int, seat: int) -> dict[str, Any]:
        return _as_dict(self._private(step, seat).get("seeds"))

    def _current_alarms(self, step: int, seat: int) -> list[dict[str, Any]]:
        obs = self._obs(step, seat)
        farm = self._farm(step, seat)
        hour = int(obs.get("hour", step % TURNS_PER_DAY))
        day = int(obs.get("day", step // TURNS_PER_DAY))
        previous_tiles = _as_list(self._farm(step - 1, seat).get("tiles")) if step > 0 else []
        plant_due: list[dict[str, int]] = []
        plant_at_risk: list[dict[str, int]] = []
        plant_critical: list[dict[str, int]] = []
        animal_due: list[dict[str, int]] = []
        animal_at_risk: list[dict[str, int]] = []
        animal_critical: list[dict[str, int]] = []
        weeds: list[dict[str, int]] = []
        new_weeds: list[dict[str, int]] = []
        animal_full: list[dict[str, int]] = []

        for y, row in enumerate(_as_list(farm.get("tiles"))):
            for x, tile in enumerate(_as_list(row)):
                kind = _tile_kind(tile)
                coord = {"x": x, "y": y}
                if kind == "PLANT" and not bool(tile.get("watered_today", False)):
                    plant_due.append(coord)
                    if int(tile.get("consecutive_unwatered", 0) or 0) >= 1:
                        if hour >= PLANT_CRITICAL_HOUR:
                            plant_critical.append(coord)
                        else:
                            plant_at_risk.append(coord)
                elif kind in {"COOP", "PASTURE"} and isinstance(tile, dict) and "animal" in tile:
                    if not bool(tile.get("fed_today", False)):
                        animal_due.append(coord)
                        if int(tile.get("consecutive_unfed", 0) or 0) >= 1:
                            if hour >= ANIMAL_CRITICAL_HOUR:
                                animal_critical.append(coord)
                            else:
                                animal_at_risk.append(coord)
                    animal = str(tile.get("animal", ""))
                    cap = ANIMAL_MAX_HELD.get(animal)
                    if cap is not None and int(tile.get("yield_units", 0) or 0) >= cap:
                        animal_full.append(coord)
                elif kind == "WEED":
                    weeds.append(coord)
                    before = None
                    if y < len(previous_tiles):
                        previous_row = _as_list(previous_tiles[y])
                        if x < len(previous_row):
                            before = previous_row[x]
                    if _tile_kind(before) != "WEED":
                        new_weeds.append(coord)

        alarms: list[dict[str, Any]] = []

        def add(key: str, title: str, severity: str, coords: list[dict[str, int]], detail: str) -> None:
            if coords:
                alarms.append({
                    "key": key,
                    "title": title,
                    "severity": severity,
                    "count": len(coords),
                    "coords": coords,
                    "detail": detail,
                })

        add("plant_critical", "植物缺水临界", "critical", plant_critical, f"第{day + 1}天第{hour + 1}回合，今日仍未浇水且已有缺水累计")
        add("plant_at_risk", "植物本日不浇将枯死", "warning", plant_at_risk, f"已有缺水累计；距日终还剩 {TURNS_PER_DAY - 1 - hour} 个行动回合")
        noncritical_plants = [coord for coord in plant_due if coord not in plant_critical and coord not in plant_at_risk]
        add("plant_due", "植物今日待浇水", "warning", noncritical_plants, "尚未浇水；越接近日终风险越高")
        add("animal_critical", "动物逃跑风险", "critical", animal_critical, "今日仍未喂养且已有断粮累计")
        add("animal_at_risk", "动物本日不喂将逃跑", "warning", animal_at_risk, f"已有断粮累计；距日终还剩 {TURNS_PER_DAY - 1 - hour} 个行动回合")
        noncritical_animals = [coord for coord in animal_due if coord not in animal_critical and coord not in animal_at_risk]
        add("animal_due", "动物今日待喂养", "warning", noncritical_animals, "尚未喂养；终局放弃喂养可能是有意决策")
        add("new_weed", "新出现杂草", "warning", new_weeds, "本回合首次出现，需要确认是否会挡住路线或生产")
        old_weeds = [coord for coord in weeds if coord not in new_weeds]
        add("weed", "未处理杂草", "info", old_weeds, "仍占用地块")
        add("animal_full", "动物产物已满", "warning", animal_full, "继续生产前应考虑收获，避免浪费新增产量")

        shed_total = _sum_inventory(self._shed(step, seat))
        carried_total = sum(_sum_inventory(inv) for inv in self._inventories(step, seat))
        if shed_total >= SHED_HIGH_WATERMARK:
            alarms.append({
                "key": "shed_near_capacity",
                "title": "仓库接近容量",
                "severity": "critical" if shed_total >= SHED_CAPACITY else "warning",
                "count": 1,
                "coords": [],
                "detail": f"仓库 {shed_total}/{SHED_CAPACITY}",
            })
        if hour == TURNS_PER_DAY - 1 and shed_total + carried_total > SHED_CAPACITY:
            alarms.append({
                "key": "eod_capacity",
                "title": "日终入仓溢出风险",
                "severity": "critical",
                "count": 1,
                "coords": [],
                "detail": f"仓库+随身={shed_total + carried_total}，容量={SHED_CAPACITY}",
            })
        if step == self.step_count - 1 and shed_total + carried_total > 0:
            alarms.append({
                "key": "terminal_unsold",
                "title": "终局仍有未售库存",
                "severity": "critical",
                "count": 1,
                "coords": [],
                "detail": f"未售 {shed_total + carried_total} 件（仓库 {shed_total}，随身 {carried_total}）",
            })
        return alarms

    def _unit_result(
        self,
        step: int,
        seat: int,
        actor_index: int,
        command: list[Any],
        before_pos: list[int],
        after_pos: list[int] | None,
    ) -> tuple[str, str]:
        op = str(command[0])
        args = command[1:]
        x, y = int(before_pos[0]), int(before_pos[1])
        before_tile = self._tile(step - 1, seat, x, y)
        after_tile = self._tile(step, seat, x, y)
        before_invs = self._inventories(step - 1, seat)
        after_invs = self._inventories(step, seat)
        before_inv = before_invs[actor_index] if actor_index < len(before_invs) else {}
        after_inv = after_invs[actor_index] if actor_index < len(after_invs) else {}
        inv_delta = _inventory_delta(before_inv, after_inv)
        day_boundary = int(self._obs(step - 1, seat).get("hour", 0)) == TURNS_PER_DAY - 1

        if op == "PASS":
            return "pass", "等待，无状态操作"
        if op in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[op]
            expected = [x + dx, y + dy]
            if after_pos == expected:
                return "success", f"({x},{y}) → ({expected[0]},{expected[1]})"
            return "noop", f"未移动，仍在 ({x},{y})"
        if op == "PLANT":
            crop = str(args[0]) if args else ""
            ok = _tile_kind(after_tile) == "PLANT" and str(after_tile.get("crop", "")) == crop
            return ("success", f"空地 → {crop}") if ok else ("noop", f"未种下；当前地块={_tile_label(after_tile)}")
        if op == "WATER":
            ok = isinstance(after_tile, dict) and after_tile.get("kind") == "PLANT" and bool(after_tile.get("watered_today", False))
            if ok:
                return "success", f"{_tile_label(after_tile)} 已浇水"
            if day_boundary and _tile_kind(before_tile) == "PLANT":
                return "uncertain", "跨日结算已重置 watered_today，终态无法单独确认"
            return "noop", f"未生效；动作前={_tile_label(before_tile)}"
        if op == "HARVEST":
            before_yield = int(before_tile.get("yield_units", 0) or 0) if isinstance(before_tile, dict) else 0
            after_yield = int(after_tile.get("yield_units", 0) or 0) if isinstance(after_tile, dict) else 0
            ok = before_yield > 0 and (after_yield < before_yield or after_tile is None)
            return ("success", f"收获 {before_yield} 单位；随身变化：{_delta_text(inv_delta)}") if ok else ("noop", f"未收获；动作前产量={before_yield}")
        if op == "FERTILIZE":
            before_until = int(before_tile.get("fertilized_until_day", -1)) if isinstance(before_tile, dict) else -1
            after_until = int(after_tile.get("fertilized_until_day", -1)) if isinstance(after_tile, dict) else -1
            ok = after_until > before_until
            return ("success", f"施肥有效至第 {after_until + 1} 天") if ok else ("noop", "未生效：需站在植物上且随身有肥料")
        if op == "DIG":
            ok = before_tile not in (None, "LOCKED") and after_tile is None
            return ("success", f"已移除 {_tile_label(before_tile)}") if ok else ("noop", f"未移除；动作前={_tile_label(before_tile)}")
        if op in {"BUILD_COOP", "BUILD_PASTURE"}:
            expected = "COOP" if op == "BUILD_COOP" else "PASTURE"
            ok = _tile_kind(after_tile) == expected
            return ("success", f"已建成 {_tile_label(after_tile)}") if ok else ("noop", f"未建成；当前地块={_tile_label(after_tile)}")
        if op == "FEED":
            ok = isinstance(after_tile, dict) and "animal" in after_tile and bool(after_tile.get("fed_today", False))
            if ok:
                return "success", f"{after_tile.get('animal')} 已喂养；随身变化：{_delta_text(inv_delta)}"
            if day_boundary and isinstance(before_tile, dict) and "animal" in before_tile:
                return "uncertain", "跨日结算已重置 fed_today，终态无法单独确认"
            return "noop", "未生效：需站在动物上、动物今日未进食且随身有小麦"
        if op == "CARE":
            ok = isinstance(after_tile, dict) and "animal" in after_tile and bool(after_tile.get("cared_today", False))
            if ok:
                return "success", f"{after_tile.get('animal')} 已照料"
            if day_boundary and isinstance(before_tile, dict) and "animal" in before_tile:
                return "uncertain", "跨日结算已重置 cared_today，终态无法单独确认"
            return "noop", "未生效：需站在动物上且今日尚未照料"
        if op == "COLLECT_FERTILIZER":
            gained = int(inv_delta.get("FERTILIZER", 0))
            return ("success", f"肥料 +{gained}") if gained > 0 else ("noop", "未取得肥料")
        if op == "PICKUP":
            return ("success", f"随身变化：{_delta_text(inv_delta)}") if any(v > 0 for v in inv_delta.values()) else ("noop", "未取到货物：位置、品类或仓库数量不满足")
        if op == "DROP":
            before_total = _sum_inventory(before_inv)
            after_total = _sum_inventory(after_inv)
            shed_delta = _sum_inventory(self._shed(step, seat)) - _sum_inventory(self._shed(step - 1, seat))
            removed = max(0, before_total - after_total)
            if removed <= 0:
                return "noop", "随身物品未减少：可能不在仓库入口"
            discarded = max(0, removed - max(0, shed_delta))
            detail = f"随身减少 {removed}，仓库净变化 {shed_delta:+d}"
            if discarded:
                detail += f"；最多 {discarded} 件未进入仓库（容量或同回合流转）"
            return "success", detail
        if op == "PLACE":
            item = str(args[0]) if args else ""
            if item in {"GOOSE", "COW", "SHEEP"} and isinstance(after_tile, dict) and after_tile.get("animal") == item:
                return "success", f"已放置 {item}"
            return ("success", f"随身变化：{_delta_text(inv_delta)}") if any(v < 0 for v in inv_delta.values()) else ("noop", "未放置：结构、位置、随身数量或仓库容量不满足")
        return "uncertain", f"已提交 {op}；未配置专用结果解释器"

    def _workers(self, step: int, seat: int) -> list[dict[str, Any]]:
        if step <= 0:
            positions = self._positions(0, seat)
            return [
                {
                    "id": "farmer" if index == 0 else f"hand_{index}",
                    "label": "农场主" if index == 0 else f"雇工 {index}",
                    "op": "PASS",
                    "action": "初始状态",
                    "status": "pass",
                    "result": "尚未执行动作",
                    "start": pos,
                    "end": pos,
                    "coord": {"x": int(pos[0]), "y": int(pos[1])},
                    "badge": "",
                }
                for index, pos in enumerate(positions)
            ]

        record = _as_dict(self.replay["steps"][step][seat])
        action = _as_dict(record.get("action"))
        before_positions = self._positions(step - 1, seat)
        after_positions = self._positions(step, seat)
        commands = [_command(action.get("farmer")), *[_command(value) for value in _as_list(action.get("hands"))]]
        rows: list[dict[str, Any]] = []
        for index, before_pos in enumerate(before_positions):
            command = commands[index] if index < len(commands) else ["PASS"]
            op = str(command[0])
            after_pos = after_positions[index] if index < len(after_positions) else None
            status, result = self._unit_result(step, seat, index, command, before_pos, after_pos)
            args = " ".join(str(value) for value in command[1:])
            title = ACTION_ZH.get(op, op) + (f" {args}" if args else "")
            rows.append({
                "id": "farmer" if index == 0 else f"hand_{index}",
                "label": "农场主" if index == 0 else f"雇工 {index}",
                "op": op,
                "action": title,
                "command": command,
                "status": status,
                "result": result,
                "start": before_pos,
                "end": after_pos,
                "target": _tile_label(self._tile(step - 1, seat, int(before_pos[0]), int(before_pos[1]))),
                "coord": {"x": int(before_pos[0]), "y": int(before_pos[1])},
                "badge": ACTION_BADGES.get(op, ""),
            })
        return rows

    def _market(self, step: int, seat: int) -> list[dict[str, Any]]:
        if step <= 0:
            return []
        before_farm = self._farm(step - 1, seat)
        after_farm = self._farm(step, seat)
        before_private = self._private(step - 1, seat)
        after_private = self._private(step, seat)
        action = _as_dict(self.replay["steps"][step][seat].get("action"))
        orders = [_command(value) for value in _as_list(action.get("market"))]
        money_delta = float(after_farm.get("money", 0) or 0) - float(before_farm.get("money", 0) or 0)
        hands_delta = len(_as_list(after_farm.get("hands"))) - len(_as_list(before_farm.get("hands")))
        land_delta = len(_as_list(after_farm.get("unlocked_quadrants"))) - len(_as_list(before_farm.get("unlocked_quadrants")))
        shed_delta = _inventory_delta(before_private.get("shed"), after_private.get("shed"))
        seed_delta = _inventory_delta(before_private.get("seeds"), after_private.get("seeds"))
        rows = []
        for index, order in enumerate(orders):
            op = str(order[0])
            args = " ".join(str(value) for value in order[1:])
            status = "uncertain"
            if op == "HIRE":
                status = "success" if hands_delta > 0 else "noop"
                detail = f"本回合雇工净增 {hands_delta:+d}"
            elif op == "BUY_LAND":
                status = "success" if land_delta > 0 else "noop"
                detail = f"已解锁区域净增 {land_delta:+d}"
            elif op == "BUY_SEED":
                item = str(order[1]) if len(order) > 1 else ""
                status = "success" if seed_delta.get(item, 0) > 0 or money_delta < 0 else "noop"
                detail = f"种子净变化：{_delta_text(seed_delta)}；现金 {money_delta:+,.0f}"
            elif op in {"BUY_PRODUCT", "BUY_ANIMAL"}:
                item = str(order[1]) if len(order) > 1 else ""
                status = "success" if shed_delta.get(item, 0) > 0 or money_delta < 0 else "noop"
                detail = f"仓库净变化：{_delta_text(shed_delta)}；现金 {money_delta:+,.0f}"
            elif op == "SELL":
                status = "success" if money_delta > 0 else "noop"
                detail = f"仓库净变化：{_delta_text(shed_delta)}；现金 {money_delta:+,.0f}"
            else:
                detail = "无法识别的市场指令"
            rows.append({
                "index": index + 1,
                "op": op,
                "action": ACTION_ZH.get(op, op) + (f" {args}" if args else ""),
                "command": order,
                "status": status,
                "result": detail + "（多订单及单位动作同回合发生，按整回合净变化确认）",
            })
        if len(orders) > int(_as_dict(self.replay.get("configuration")).get("maxMarketOrdersPerTurn", 10)):
            rows.append({
                "index": len(rows) + 1,
                "op": "ORDER_LIMIT",
                "action": "市场订单超限",
                "command": [],
                "status": "noop",
                "result": f"提交 {len(orders)} 条，官方只处理前 10 条",
            })
        return rows

    def frame(self, step: int) -> dict[str, Any]:
        if not 0 <= step < self.step_count:
            raise ReplayValidationError(f"step 超出范围：{step}")
        seats = []
        for seat in (0, 1):
            obs = self._obs(step, seat)
            farm = self._farm(step, seat)
            alarms = self._current_alarms(step, seat)
            workers = self._workers(step, seat)
            market = self._market(step, seat)
            status_counts = Counter(row["status"] for row in workers + market)
            seats.append({
                "seat": seat,
                "money": float(farm.get("money", 0) or 0),
                "shed_total": _sum_inventory(self._shed(step, seat)),
                "carried_total": sum(_sum_inventory(inv) for inv in self._inventories(step, seat)),
                "workers": workers,
                "market": market,
                "alarms": alarms,
                "action_summary": dict(status_counts),
            })
        obs0 = self._obs(step, 0)
        return {
            "step": step,
            "day": int(obs0.get("day", step // TURNS_PER_DAY)),
            "hour": int(obs0.get("hour", step % TURNS_PER_DAY)),
            "seats": seats,
        }

    def _build_timeline(self) -> list[dict[str, Any]]:
        timeline = []
        for step in range(self.step_count):
            seat_rows = []
            for seat in (0, 1):
                alarms = self._current_alarms(step, seat)
                severity = "none"
                if any(row["severity"] == "critical" for row in alarms):
                    severity = "critical"
                elif any(row["severity"] == "warning" for row in alarms):
                    severity = "warning"
                elif alarms:
                    severity = "info"
                seat_rows.append({"count": sum(int(row["count"]) for row in alarms), "severity": severity})
            timeline.append({"step": step, "seats": seat_rows})
        return timeline

    def timeline(self) -> list[dict[str, Any]]:
        return self._timeline
