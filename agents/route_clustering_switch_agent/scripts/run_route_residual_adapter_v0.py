#!/usr/bin/env python3
"""Train and validate a per-turn raw-action adapter on one frozen route genome.

The adapter never emits a route ID.  At each turn it ranks six raw-action
residuals, applies at most one before the existing native safety overlays, and
then returns to the exact same frozen 21-gene route schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fast_kaggriculture
import fast_kaggriculture._fast_kaggriculture as native_extension
from fast_kaggriculture import Config, FastEnv, NativeAgentState
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_switch_features import (
    RouteSwitchHistory,
    route_switch_feature_names,
    route_switch_vector,
)
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_block_sequence_oracle_v1 as sequence


SCHEMA = "route-residual-adapter-v0"
HORIZON = sequence.HORIZON
ANCHORS = sequence.ANCHORS
STOPS = sequence.STOPS
KEEP_AUDIT_STEPS = tuple(sorted({
    *ANCHORS, *(int(stop) - 1 for stop in STOPS),
}))
TARGET_OPPONENTS = sequence.TARGET_OPPONENTS
TRAIN_SEEDS = tuple(range(2026085800, 2026085810))
VALIDATION_SEEDS = tuple(range(2026086200, 2026086208))
SEALED_SEEDS = frozenset(sequence.TEST_SEEDS)
DEFAULT_FROZEN_RUN = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\block_sequence_oracle_v1\run"
)
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\route_residual_adapter_v0\run"
)

EDIT_CODES = (
    "KEEP",
    "B0_UNITS",
    "B0_MARKET",
    "B0_FULL",
    "DROP_BUYS",
    "CLEAR_MARKET",
)
EDIT_INDEX = {name: index for index, name in enumerate(EDIT_CODES)}
BUY_OPS = frozenset({
    "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
})
OPS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL",
)
OP_INDEX = {name: index for index, name in enumerate(OPS)}
ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
ITEM_INDEX = {name: index for index, name in enumerate(ITEMS)}
TRADE_ITEMS = ITEMS[:9]
TRADE_INDEX = {name: index for index, name in enumerate(TRADE_ITEMS)}
MARKET_OPS = OPS[18:]
MARKET_INDEX = {name: index for index, name in enumerate(MARKET_OPS)}
ROLE_BY_OP = {
    **{name: 0 for name in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")},
    **{name: 1 for name in ("PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG")},
    **{name: 2 for name in ("BUILD_COOP", "BUILD_PASTURE", "PLACE")},
    **{name: 3 for name in ("FEED", "COLLECT_FERTILIZER", "CARE")},
    **{name: 4 for name in ("DROP", "PICKUP")},
}


@dataclass(frozen=True)
class Candidate:
    code: str
    raw_action: dict[str, Any] | None
    effective_raw: dict[str, Any]
    aliases: tuple[str, ...]
    raw_sha256: str


@dataclass
class TrainingPanel:
    features: list[np.ndarray]
    target: list[float]
    delta_margin: list[float]
    rewards: list[tuple[float, float]]
    seed: list[int]
    seat: list[int]
    opponent: list[int]
    step: list[int]
    edit: list[int]
    decision: list[int]
    selected: list[bool]

    @classmethod
    def empty(cls) -> "TrainingPanel":
        return cls(*([] for _ in range(11)))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.features:
            raise ValueError("training panel is empty")
        return {
            "features": np.stack(self.features).astype(np.float32),
            "target_signed_log_margin": np.asarray(self.target, np.float32),
            "delta_margin": np.asarray(self.delta_margin, np.float64),
            "terminal_rewards": np.asarray(self.rewards, np.float64),
            "seed": np.asarray(self.seed, np.int64),
            "seat": np.asarray(self.seat, np.int8),
            "opponent": np.asarray(self.opponent, np.int8),
            "step": np.asarray(self.step, np.int16),
            "edit": np.asarray(self.edit, np.int8),
            "decision": np.asarray(self.decision, np.int32),
            "selected_by_oracle": np.asarray(self.selected, np.bool_),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    return adaptive._sha256_file(path)


def _atomic_text(path: Path, text: str) -> None:
    adaptive._atomic_text(path, text)


def _atomic_joblib(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _action(raw: Any) -> list[Any]:
    values = list(raw or ["PASS"])
    if not values or str(values[0]) not in OP_INDEX:
        return ["PASS"]
    result: list[Any] = [str(values[0])]
    if len(values) >= 2 and str(values[1]) in ITEM_INDEX:
        result.append(str(values[1]))
    if len(values) >= 3:
        result.append(int(values[2]))
    return result


def normalize_player_action(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "farmer": _action(raw.get("farmer")),
        "hands": [_action(value) for value in list(raw.get("hands", ()) or ())],
        "market": [_action(value) for value in list(raw.get("market", ()) or ())],
    }


def residual_candidates(
    base_tape: Sequence[Mapping[str, Any]],
    baseline_tape: Sequence[Mapping[str, Any]],
    step: int,
) -> list[Candidate]:
    """Build the frozen six-code vocabulary and canonicalize aliases."""

    if not (0 <= step < HORIZON):
        raise ValueError("residual step is outside the 719-step tape")
    base = normalize_player_action(base_tape[step])
    baseline = normalize_player_action(baseline_tape[step])
    values: list[tuple[str, dict[str, Any]]] = [
        ("KEEP", base),
        ("B0_UNITS", {
            "farmer": baseline["farmer"],
            "hands": baseline["hands"],
            "market": base["market"],
        }),
        ("B0_MARKET", {
            "farmer": base["farmer"],
            "hands": base["hands"],
            "market": baseline["market"],
        }),
        ("B0_FULL", baseline),
        ("DROP_BUYS", {
            "farmer": base["farmer"],
            "hands": base["hands"],
            "market": [
                order for order in base["market"]
                if str(order[0]) not in BUY_OPS
            ],
        }),
        ("CLEAR_MARKET", {
            "farmer": base["farmer"],
            "hands": base["hands"],
            "market": [],
        }),
    ]
    canonical: dict[bytes, tuple[str, dict[str, Any], list[str]]] = {}
    order: list[bytes] = []
    for code, raw in values:
        normalized = normalize_player_action(raw)
        payload = _canonical_bytes(normalized)
        if payload not in canonical:
            canonical[payload] = (code, normalized, [code])
            order.append(payload)
        else:
            canonical[payload][2].append(code)
    result = []
    base_payload = _canonical_bytes(base)
    for payload in order:
        code, raw, aliases = canonical[payload]
        result.append(Candidate(
            code=code,
            raw_action=None if payload == base_payload else raw,
            effective_raw=raw,
            aliases=tuple(aliases),
            raw_sha256=_sha256_bytes(payload),
        ))
    if not result or result[0].code != "KEEP" or result[0].raw_action is not None:
        raise AssertionError("KEEP must be the first canonical residual")
    return result


def _packed_action(raw: Sequence[Any]) -> tuple[int, int, int]:
    values = _action(raw)
    op = OP_INDEX[str(values[0])]
    item = ITEM_INDEX.get(str(values[1]), -1) if len(values) >= 2 else -1
    quantity = int(values[2]) if len(values) >= 3 else 1
    if quantity < 0:
        raise ValueError("negative raw action quantity")
    return op, item, quantity


def pack_candidates(
    candidates: Sequence[Candidate],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("candidate batch is empty")
    explicit = [candidate.effective_raw for candidate in candidates]
    max_units = max(1, max(1 + len(raw["hands"]) for raw in explicit))
    max_market = max(1, max(len(raw["market"]) for raw in explicit))
    units = np.zeros((len(candidates), max_units, 3), np.int32)
    market = np.zeros((len(candidates), max_market, 3), np.int32)
    unit_counts = np.empty(len(candidates), np.int32)
    market_counts = np.empty(len(candidates), np.int32)
    for index, candidate in enumerate(candidates):
        if candidate.raw_action is None:
            unit_counts[index] = market_counts[index] = -1
            continue
        raw = candidate.effective_raw
        unit_values = [raw["farmer"], *raw["hands"]]
        unit_counts[index] = len(unit_values)
        market_counts[index] = len(raw["market"])
        for slot, action in enumerate(unit_values):
            units[index, slot] = _packed_action(action)
        for slot, action in enumerate(raw["market"]):
            market[index, slot] = _packed_action(action)
    return units, unit_counts, market, market_counts


def _cash_flow(
    raw: Mapping[str, Any], observation: Mapping[str, Any],
) -> tuple[float, float]:
    farm = dict(list(observation.get("farms", ()) or ())[int(observation["player"])])
    hires = int(farm.get("hires_today", 0) or 0)
    land = len(farm.get("unlocked_quadrants", ()) or ())
    prices = dict((observation.get("market", {}) or {}).get("prices", {}) or {})
    expense = income = 0.0
    # Keep this aligned with route_switch_features' existing capital estimator.
    from meta_agent.src import route_switch_features as route_features
    for order in raw.get("market", ()) or ():
        spend, revenue, hires, land, *_ = route_features._order_flow(
            order, prices, hires, land,
        )
        expense += spend
        income += revenue
    return expense, income


def action_vector(
    raw: Mapping[str, Any], observation: Mapping[str, Any],
) -> np.ndarray:
    """Encode one raw player action into the frozen 64-D action panel."""

    raw = normalize_player_action(raw)
    unit_ops = np.zeros(18, np.float32)
    unit_items = np.zeros(12, np.float32)
    roles = np.zeros(5, np.float32)
    for value in [raw["farmer"], *raw["hands"]]:
        op, item, quantity = _packed_action(value)
        if op >= 18:
            raise ValueError("market operation appeared in a unit slot")
        unit_ops[op] += 1
        if item >= 0:
            unit_items[item] += max(1, quantity)
        roles[ROLE_BY_OP[OPS[op]]] += 1
    market_ops = np.zeros(6, np.float32)
    trades = np.zeros(18, np.float32)
    for value in raw["market"]:
        op, item, quantity = _packed_action(value)
        name = OPS[op]
        if name not in MARKET_INDEX:
            raise ValueError("unit operation appeared in a market slot")
        market_ops[MARKET_INDEX[name]] += 1
        if item in range(9):
            side = 1 if name == "SELL" else 0
            trades[side * 9 + item] += max(1, quantity)
    expense, income = _cash_flow(raw, observation)
    scalars = np.asarray((
        1 + len(raw["hands"]), len(raw["market"]),
        expense, income, income - expense,
    ), np.float32)
    result = np.concatenate((
        unit_ops, unit_items, roles, market_ops, trades, scalars,
    )).astype(np.float32)
    if result.shape != (64,) or not np.isfinite(result).all():
        raise ValueError("invalid 64-D raw action vector")
    return result


def _layouts(observation: Mapping[str, Any], seat: int) -> np.ndarray:
    values = []
    for player in (seat, 1 - seat):
        view = dict(observation)
        view["player"] = player
        layout, mask = continuation.own_layout(view)
        values.extend((layout.astype(np.float32), mask.astype(np.float32)))
    result = np.concatenate(values).astype(np.float32)
    if result.shape != (208,):
        raise AssertionError("two-board layout feature changed shape")
    return result


def feature_names() -> tuple[str, ...]:
    state = route_switch_feature_names()
    layout = [
        *[f"self_tile_{index}" for index in range(100)],
        *[f"self_unlock_{index}" for index in range(4)],
        *[f"opponent_tile_{index}" for index in range(100)],
        *[f"opponent_unlock_{index}" for index in range(4)],
    ]
    action = [
        *[f"unit_op_{name.lower()}" for name in OPS[:18]],
        *[f"unit_item_{name.lower()}" for name in ITEMS],
        *[f"unit_role_{index}" for index in range(5)],
        *[f"market_op_{name.lower()}" for name in MARKET_OPS],
        *[f"buy_{name.lower()}" for name in TRADE_ITEMS],
        *[f"sell_{name.lower()}" for name in TRADE_ITEMS],
        "actor_count", "market_order_count", "estimated_expense",
        "estimated_income", "estimated_net_cash",
    ]
    names = [*state, *layout]
    for prefix in ("base", "b0", "candidate_minus_base"):
        names.extend(f"{prefix}_{name}" for name in action)
    names.extend(f"edit_{code.lower()}" for code in EDIT_CODES)
    names.extend(("seat", "segment", "segment_offset"))
    return tuple(names)


FEATURE_NAMES = feature_names()


def candidate_features(
    observation: Mapping[str, Any],
    history: RouteSwitchHistory,
    base_tape: Sequence[Mapping[str, Any]],
    base_raw: Mapping[str, Any],
    baseline_raw: Mapping[str, Any],
    candidate: Candidate,
    seat: int,
    step: int,
) -> np.ndarray:
    return candidate_feature_rows(
        observation, history, base_tape, base_raw, baseline_raw,
        [candidate], seat, step,
    )[0]


def candidate_feature_rows(
    observation: Mapping[str, Any],
    history: RouteSwitchHistory,
    base_tape: Sequence[Mapping[str, Any]],
    base_raw: Mapping[str, Any],
    baseline_raw: Mapping[str, Any],
    candidates: Sequence[Candidate],
    seat: int,
    step: int,
) -> np.ndarray:
    if not candidates:
        raise ValueError("cannot encode an empty residual candidate set")
    if int(observation.get("step", -1)) != int(step):
        raise ValueError("observation and residual feature step disagree")
    state = route_switch_vector(observation, history, base_tape)
    board = _layouts(observation, seat)
    base = action_vector(base_raw, observation)
    b0 = action_vector(baseline_raw, observation)
    segment = int(np.searchsorted(STOPS, step, side="right"))
    context = np.asarray((
        seat, segment / max(1, len(ANCHORS) - 1),
        (step - ANCHORS[segment]) / 23.0,
    ), np.float32)
    rows = []
    for candidate in candidates:
        proposed = action_vector(candidate.effective_raw, observation)
        edit = np.zeros(len(EDIT_CODES), np.float32)
        edit[EDIT_INDEX[candidate.code]] = 1
        rows.append(np.concatenate((
            state, board, base, b0, proposed - base, edit, context,
        )))
    result = np.stack(rows).astype(np.float32)
    if result.shape != (len(candidates), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("invalid residual feature rows")
    return result


def _signed_log(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value)))


def _full_schedule(
    bundle: NativeTeammateBundle,
    genome: Sequence[str],
    opponent_id: str,
    seat: int,
) -> np.ndarray:
    routes = [[bundle.index(route_id) for route_id in genome]]
    return sequence.native_schedules(
        routes, bundle.index(opponent_id), seat,
    )[0]


def scheduled_tape(
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Compose the actual 21-gene future used by capital-plan features."""

    if len(genome) != len(ANCHORS):
        raise ValueError("scheduled tape requires exactly 21 route genes")
    result: list[Mapping[str, Any]] = []
    for step in range(HORIZON):
        if step < ANCHORS[0]:
            result.append(baseline_tape[step])
        else:
            segment = int(np.searchsorted(STOPS, step, side="right"))
            result.append(tapes[str(genome[segment])][step])
    return result


def execution_route_ids(genome: Sequence[str]) -> list[str]:
    return [
        str(genome[int(np.searchsorted(STOPS, step, side="right"))])
        for step in range(ANCHORS[0], HORIZON)
    ]


def prefix_with_history(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    opponent_id: str,
    seed: int,
    seat: int,
) -> tuple[FastEnv, list[NativeAgentState], RouteSwitchHistory]:
    env = FastEnv(Config(), int(seed))
    states = [NativeAgentState(), NativeAgentState()]
    history = RouteSwitchHistory()
    baseline = bundle.index(baseline_id)
    opponent = bundle.index(opponent_id)
    pair = (baseline, opponent) if seat == 0 else (opponent, baseline)
    while int(env.step_count) < ANCHORS[0]:
        step = int(env.step_count)
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        actions = [
            bundle.executor.action_at(env, player, pair[player], states[player])
            for player in (0, 1)
        ]
        env.step(actions)
    if int(env.step_count) != ANCHORS[0]:
        raise RuntimeError("residual prefix did not stop at step 216")
    return env, states, history


def _terminal_rank(rewards: Sequence[float], seat: int) -> tuple[int, float]:
    return adaptive.reward_rank(rewards, seat)


def fixed_result(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    opponent_id: str,
    genome: Sequence[str],
    genome_sha256: str,
    seed: int,
    seat: int,
) -> dict[str, Any]:
    baseline_rewards = adaptive._baseline_rewards(
        bundle, bundle.index(baseline_id), bundle.index(opponent_id), seed, seat,
    )
    env, states = sequence.prefix_snapshot(
        bundle, bundle.index(baseline_id), bundle.index(opponent_id), seed, seat,
    )
    schedule = _full_schedule(bundle, genome, opponent_id, seat)
    rewards = np.asarray(bundle.executor.rollout_schedule_batch(
        env, states[0], states[1], schedule[None, :, :], STOPS,
    ), np.float64)[0]
    rank = _terminal_rank(rewards, seat)
    baseline_rank = _terminal_rank(baseline_rewards, seat)
    executed_routes = execution_route_ids(genome)
    return {
        "arm": "fixed_route",
        "opponent": opponent_id,
        "seed": int(seed),
        "seat": int(seat),
        "rewards": rewards.tolist(),
        "outcome": rank[0],
        "margin": rank[1],
        "baseline_rewards": list(baseline_rewards),
        "baseline_outcome": baseline_rank[0],
        "baseline_margin": baseline_rank[1],
        "genome_sha256": genome_sha256,
        "execution_route_sha256": _json_sha256(executed_routes),
        "route_change_count": sum(
            actual != expected
            for actual, expected in zip(
                executed_routes, execution_route_ids(genome), strict=True,
            )
        ),
        "edits": 0,
        "completed": True,
    }


def _commit_candidate(
    bundle: NativeTeammateBundle,
    env: FastEnv,
    states: Sequence[NativeAgentState],
    route_pair: Sequence[int],
    seat: int,
    candidate: Candidate,
) -> None:
    actions = []
    for player in (0, 1):
        if player == seat:
            actions.append(bundle.executor.action_at_with_raw_override(
                env, player, int(route_pair[player]), states[player],
                candidate.raw_action,
            ))
        else:
            actions.append(bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            ))
    env.step(actions)


def _residual_rollouts(
    bundle: NativeTeammateBundle,
    env: FastEnv,
    states: Sequence[NativeAgentState],
    schedule: np.ndarray,
    stops: np.ndarray,
    seat: int,
    candidates: Sequence[Candidate],
) -> np.ndarray:
    units, unit_counts, market, market_counts = pack_candidates(candidates)
    rewards = np.asarray(bundle.executor.rollout_schedule_raw_override_batch(
        env, states[0], states[1], schedule, stops, seat,
        units, unit_counts, market, market_counts,
    ), np.float64)
    if rewards.shape != (len(candidates), 2) or not np.isfinite(rewards).all():
        raise RuntimeError("native residual rollout returned invalid rewards")
    return rewards


def oracle_scenario(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    opponent_id: str,
    genome: Sequence[str],
    genome_sha256: str,
    seed: int,
    seat: int,
    fixed: Mapping[str, Any],
    panel: TrainingPanel | None,
    decision_start: int,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    scenario_started = time.perf_counter()
    env, states, history = prefix_with_history(
        bundle, baseline_id, opponent_id, seed, seat,
    )
    schedule = _full_schedule(bundle, genome, opponent_id, seat)
    plan_tape = scheduled_tape(baseline_tape, tapes, genome)
    decisions: list[dict[str, Any]] = []
    executed_routes: list[str] = []
    edits = aliases = keep_checks = candidate_rollouts = candidate_steps = 0
    first_keep: tuple[float, float] | None = None
    decision_id = decision_start
    for step in range(ANCHORS[0], HORIZON):
        if int(env.step_count) != step:
            raise RuntimeError("oracle residual trajectory missed a step")
        segment = int(np.searchsorted(STOPS, step, side="right"))
        base_id = str(genome[segment])
        executed_routes.append(base_id)
        base_tape = tapes[base_id]
        candidates = residual_candidates(base_tape, baseline_tape, step)
        candidate_rollouts += len(candidates)
        candidate_steps += len(candidates) * (HORIZON - step)
        aliases += sum(len(candidate.aliases) - 1 for candidate in candidates)
        rewards = _residual_rollouts(
            bundle, env, states, schedule[segment:], STOPS[segment:],
            seat, candidates,
        )
        keep = tuple(map(float, rewards[0]))
        if first_keep is None:
            first_keep = keep
            if keep != tuple(map(float, fixed["rewards"])):
                raise RuntimeError("step216 KEEP disagrees with frozen route")
        if step in KEEP_AUDIT_STEPS:
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1],
                schedule[segment:][None, :, :], STOPS[segment:],
            ), np.float64)[0]
            if tuple(map(float, reference)) != keep:
                raise RuntimeError("raw KEEP disagrees with schedule rollout")
            keep_checks += 1
        best = 0
        best_rank = _terminal_rank(rewards[0], seat)
        for index in range(1, len(candidates)):
            rank = _terminal_rank(rewards[index], seat)
            if rank > best_rank:
                best, best_rank = index, rank
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        base_raw = normalize_player_action(base_tape[step])
        baseline_raw = normalize_player_action(baseline_tape[step])
        keep_margin = _terminal_rank(rewards[0], seat)[1]
        if panel is not None:
            feature_rows = candidate_feature_rows(
                observation, history, plan_tape, base_raw, baseline_raw,
                candidates, seat, step,
            )
            for index, candidate in enumerate(candidates):
                margin = _terminal_rank(rewards[index], seat)[1]
                delta = margin - keep_margin
                panel.features.append(feature_rows[index])
                panel.target.append(_signed_log(delta))
                panel.delta_margin.append(delta)
                panel.rewards.append(tuple(map(float, rewards[index])))
                panel.seed.append(int(seed))
                panel.seat.append(int(seat))
                panel.opponent.append(TARGET_OPPONENTS.index(opponent_id))
                panel.step.append(step)
                panel.edit.append(EDIT_INDEX[candidate.code])
                panel.decision.append(decision_id)
                panel.selected.append(index == best)
        chosen = candidates[best]
        edits += int(chosen.code != "KEEP")
        decisions.append({
            "split": split,
            "opponent": opponent_id,
            "seed": int(seed),
            "seat": int(seat),
            "step": step,
            "segment": segment,
            "base_route_id": base_id,
            "genome_sha256": genome_sha256,
            "candidate_codes": [candidate.code for candidate in candidates],
            "candidate_aliases": [list(candidate.aliases) for candidate in candidates],
            "candidate_raw_sha256": [candidate.raw_sha256 for candidate in candidates],
            "terminal_rewards": rewards.tolist(),
            "selected_code": chosen.code,
            "selected_index": best,
            "selected_raw_sha256": chosen.raw_sha256,
            "keep_margin": keep_margin,
            "selected_margin": best_rank[1],
        })
        _commit_candidate(
            bundle, env, states, schedule[segment], seat, chosen,
        )
        decision_id += 1
    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("oracle residual trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    rank = _terminal_rank(rewards, seat)
    fixed_rank = (int(fixed["outcome"]), float(fixed["margin"]))
    if rank < fixed_rank:
        raise AssertionError("receding-horizon oracle regressed against fixed route")
    if decisions and rewards != tuple(map(float, decisions[-1]["terminal_rewards"][
        decisions[-1]["selected_index"]
    ])):
        raise RuntimeError("last oracle rollout disagrees with committed action")
    return ({
        "arm": "greedy_residual_oracle",
        "split": split,
        "opponent": opponent_id,
        "seed": int(seed),
        "seat": int(seat),
        "rewards": list(rewards),
        "outcome": rank[0],
        "margin": rank[1],
        "genome_sha256": genome_sha256,
        "execution_route_sha256": _json_sha256(executed_routes),
        "route_change_count": sum(
            actual != expected
            for actual, expected in zip(
                executed_routes, execution_route_ids(genome), strict=True,
            )
        ),
        "edits": edits,
        "canonical_aliases": aliases,
        "keep_equivalence_checks": keep_checks,
        "candidate_rollouts": candidate_rollouts,
        "candidate_simulated_steps": candidate_steps,
        "elapsed_seconds": time.perf_counter() - scenario_started,
        "completed": True,
    }, decisions, decision_id)


def _model_params(trees: int, random_seed: int) -> dict[str, Any]:
    return {
        "n_estimators": int(trees),
        "max_depth": 14,
        "min_samples_leaf": 16,
        "max_features": 0.5,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": int(random_seed),
    }


def _tree_predictions(model: ExtraTreesRegressor, values: np.ndarray) -> np.ndarray:
    return np.stack([tree.predict(values) for tree in model.estimators_], axis=1)


def _decision_metrics(
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    positive: np.ndarray,
    beta: float,
    threshold: float,
    min_positive: float,
) -> dict[str, Any]:
    realized = []
    selected = 0
    correct = 0
    decision_ids = arrays["decision"]
    if np.any(decision_ids[1:] < decision_ids[:-1]):
        raise ValueError("training decisions must be stored contiguously")
    starts = np.flatnonzero(np.r_[True, decision_ids[1:] != decision_ids[:-1]])
    stops = np.r_[starts[1:], len(decision_ids)]
    for start, stop in zip(starts, stops, strict=True):
        indices = np.arange(start, stop, dtype=np.int64)
        keep = indices[arrays["edit"][indices] == EDIT_INDEX["KEEP"]]
        if len(keep) != 1:
            raise RuntimeError("OOF decision does not have exactly one KEEP")
        chosen = int(keep[0])
        candidates = indices[arrays["edit"][indices] != EDIT_INDEX["KEEP"]]
        if len(candidates):
            score = mean[candidates] - beta * std[candidates]
            candidate = int(candidates[int(np.argmax(score))])
            if float(np.max(score)) > threshold and positive[candidate] >= min_positive:
                chosen = candidate
        delta = float(arrays["delta_margin"][chosen])
        realized.append(delta)
        selected += int(chosen != keep[0])
        oracle = indices[int(np.argmax(arrays["delta_margin"][indices]))]
        correct += int(chosen == oracle)
    values = np.asarray(realized, np.float64)
    return {
        "decisions": len(values),
        "selected": selected,
        "selection_rate": selected / max(1, len(values)),
        "mean_realized_delta_margin": float(np.mean(values)),
        "sum_realized_delta_margin": float(np.sum(values)),
        "harmful_rate": float(np.mean(values < 0)),
        "beneficial_rate": float(np.mean(values > 0)),
        "top1_accuracy": correct / max(1, len(values)),
        "beta": float(beta),
        "threshold": float(threshold),
        "min_positive_fraction": float(min_positive),
    }


def fit_ranker(
    arrays: Mapping[str, np.ndarray],
    trees: int,
    cv_trees: int,
    cv_folds: int,
    random_seed: int,
) -> tuple[ExtraTreesRegressor, dict[str, Any], dict[str, np.ndarray]]:
    x = arrays["features"]
    y = arrays["target_signed_log_margin"]
    groups = arrays["seed"]
    unique_groups = np.unique(groups)
    folds = min(int(cv_folds), len(unique_groups))
    if folds < 2:
        raise ValueError("ranker needs at least two train seed groups")
    mean = np.full(len(y), np.nan, np.float64)
    std = np.full(len(y), np.nan, np.float64)
    positive = np.full(len(y), np.nan, np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, valid) in enumerate(splitter.split(x, y, groups)):
        model = ExtraTreesRegressor(**_model_params(
            cv_trees, random_seed + fold * 1009,
        ))
        counts = np.bincount(arrays["decision"][train])
        weights = 1.0 / np.maximum(1, counts[arrays["decision"][train]])
        model.fit(x[train], y[train], sample_weight=weights)
        predictions = _tree_predictions(model, x[valid])
        mean[valid] = predictions.mean(axis=1)
        std[valid] = predictions.std(axis=1)
        positive[valid] = (predictions > 0).mean(axis=1)
        print(json.dumps({
            "event": "oof_fold_complete", "fold": fold,
            "train_rows": len(train), "valid_rows": len(valid),
        }, sort_keys=True), flush=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("OOF prediction panel is incomplete")
    positive_scores = mean[(arrays["edit"] != 0) & (mean > 0)]
    thresholds = [0.0]
    if len(positive_scores):
        thresholds.extend(map(float, np.quantile(
            positive_scores, [0.25, 0.5, 0.75, 0.9],
        )))
        # A train-only conservative candidate guarantees calibration can
        # choose exact KEEP when every firing rule has negative OOF value.
        thresholds.append(float(np.max(positive_scores) + 1.0))
    trials = [
        _decision_metrics(arrays, mean, std, positive, beta, threshold, minimum)
        for beta in (0.0, 0.5, 1.0)
        for threshold in sorted(set(thresholds))
        for minimum in (0.5, 0.6, 0.7)
    ]
    chosen = max(trials, key=lambda row: (
        row["mean_realized_delta_margin"], -row["harmful_rate"],
        row["beneficial_rate"], -row["selection_rate"],
    ))
    final = ExtraTreesRegressor(**_model_params(trees, random_seed))
    counts = np.bincount(arrays["decision"])
    weights = 1.0 / np.maximum(1, counts[arrays["decision"]])
    final.fit(x, y, sample_weight=weights)
    return final, {
        "folds": folds,
        "grouping": "seed keeps both opponents, both seats and all steps together",
        "chosen": chosen,
        "trials": trials,
        "model_params": _model_params(trees, random_seed),
    }, {"mean": mean, "std": std, "positive_fraction": positive}


def model_choice(
    model: ExtraTreesRegressor,
    values: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    predictions = _tree_predictions(model, values)
    mean = predictions.mean(axis=1)
    std = predictions.std(axis=1)
    positive = (predictions > 0).mean(axis=1)
    chosen = 0
    if len(values) > 1:
        beta = float(calibration["beta"])
        score = mean[1:] - beta * std[1:]
        index = int(np.argmax(score)) + 1
        if (
            float(score[index - 1]) > float(calibration["threshold"])
            and positive[index] >= float(calibration["min_positive_fraction"])
        ):
            chosen = index
    return chosen, mean, std, positive


def learned_scenario(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    opponent_id: str,
    genome: Sequence[str],
    genome_sha256: str,
    seed: int,
    seat: int,
    model: ExtraTreesRegressor,
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, states, history = prefix_with_history(
        bundle, baseline_id, opponent_id, seed, seat,
    )
    schedule = _full_schedule(bundle, genome, opponent_id, seat)
    plan_tape = scheduled_tape(baseline_tape, tapes, genome)
    decisions = []
    executed_routes: list[str] = []
    edits = 0
    for step in range(ANCHORS[0], HORIZON):
        segment = int(np.searchsorted(STOPS, step, side="right"))
        base_id = str(genome[segment])
        executed_routes.append(base_id)
        base_tape = tapes[base_id]
        candidates = residual_candidates(base_tape, baseline_tape, step)
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        base_raw = normalize_player_action(base_tape[step])
        baseline_raw = normalize_player_action(baseline_tape[step])
        values = candidate_feature_rows(
            observation, history, plan_tape, base_raw, baseline_raw,
            candidates, seat, step,
        )
        chosen, mean, std, positive = model_choice(model, values, calibration)
        candidate = candidates[chosen]
        edits += int(candidate.code != "KEEP")
        if candidate.code != "KEEP" or step % 24 == 0:
            decisions.append({
                "opponent": opponent_id, "seed": int(seed), "seat": seat,
                "step": step, "segment": segment, "base_route_id": base_id,
                "genome_sha256": genome_sha256,
                "candidate_codes": [value.code for value in candidates],
                "selected_code": candidate.code,
                "selected_raw_sha256": candidate.raw_sha256,
                "prediction_mean": mean.tolist(),
                "prediction_std": std.tolist(),
                "positive_tree_fraction": positive.tolist(),
            })
        _commit_candidate(
            bundle, env, states, schedule[segment], seat, candidate,
        )
    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("learned residual trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    rank = _terminal_rank(rewards, seat)
    return ({
        "arm": "learned_residual_adapter",
        "split": "validation",
        "opponent": opponent_id,
        "seed": int(seed),
        "seat": int(seat),
        "rewards": list(rewards),
        "outcome": rank[0],
        "margin": rank[1],
        "genome_sha256": genome_sha256,
        "execution_route_sha256": _json_sha256(executed_routes),
        "route_change_count": sum(
            actual != expected
            for actual, expected in zip(
                executed_routes, execution_route_ids(genome), strict=True,
            )
        ),
        "edits": edits,
        "completed": True,
    }, decisions)


def _paired_rows(
    arm: Sequence[Mapping[str, Any]], fixed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = {
        (row["opponent"], row["seed"], row["seat"]): row for row in fixed
    }
    result = []
    for row in arm:
        base = reference[(row["opponent"], row["seed"], row["seat"])]
        value = dict(row)
        value.update({
            "fixed_outcome": int(base["outcome"]),
            "fixed_margin": float(base["margin"]),
            "delta_margin_vs_fixed": float(row["margin"]) - float(base["margin"]),
            "loss_repaired_to_win": int(base["outcome"]) == 0 and int(row["outcome"]) == 2,
            "outcome_regression_vs_fixed": int(row["outcome"]) < int(base["outcome"]),
        })
        result.append(value)
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    opponents = sorted({str(row["opponent"]) for row in rows})
    candidate_rollouts = sum(int(row.get("candidate_rollouts", 0)) for row in rows)
    candidate_steps = sum(int(row.get("candidate_simulated_steps", 0)) for row in rows)
    elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in rows)
    result = {
        "states": len(rows),
        "raw_win_rate": float(np.mean([int(row["outcome"]) == 2 for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "mean_delta_margin_vs_fixed": float(np.mean([
            row.get("delta_margin_vs_fixed", 0) for row in rows
        ])),
        "loss_repairs_to_win": sum(bool(row.get("loss_repaired_to_win")) for row in rows),
        "outcome_regressions_vs_fixed": sum(
            bool(row.get("outcome_regression_vs_fixed")) for row in rows
        ),
        "edits": sum(int(row.get("edits", 0)) for row in rows),
        "completed": sum(bool(row.get("completed")) for row in rows),
        "per_opponent_raw_win_rate": {
            opponent: float(np.mean([
                int(row["outcome"]) == 2 for row in rows
                if str(row["opponent"]) == opponent
            ]))
            for opponent in opponents
        },
    }
    if candidate_rollouts:
        result["candidate_rollouts"] = candidate_rollouts
        result["candidate_simulated_steps"] = candidate_steps
        result["candidate_rollouts_per_second"] = candidate_rollouts / max(elapsed, 1e-9)
        result["simulated_steps_per_second"] = candidate_steps / max(elapsed, 1e-9)
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result = {"file": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _report_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    rows = []
    for arm in ("fixed", "learned", "oracle"):
        summary = validation[arm]
        states = int(summary["states"])
        wins = round(float(summary["raw_win_rate"]) * states)
        rows.append(
            f"| {arm} | {wins}/{states} ({100.0 * float(summary['raw_win_rate']):.2f}%) "
            f"| {float(summary['mean_margin']):.2f} "
            f"| {float(summary['mean_delta_margin_vs_fixed']):+.2f} "
            f"| {int(summary['loss_repairs_to_win'])} "
            f"| {int(summary['outcome_regressions_vs_fixed'])} "
            f"| {int(summary['edits'])} |"
        )
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in report["gates"].items()
    )
    return (
        "# ROUTE-RESIDUAL-ADAPTER-v0\n\n"
        f"Status: `{report['status']}`\n\n"
        "A learned one-turn raw-action residual adapter improves a frozen "
        "21-gene route without changing any route gene. This is a feasibility "
        "result, not the full-pool 90/80 championship claim.\n\n"
        "## Fresh validation\n\n"
        "| arm | wins | mean margin | delta vs fixed | loss-to-win | regressions | edits |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        f"Train seeds: `{len(report['splits']['train_seeds'])}`; fresh validation "
        f"seeds: `{len(report['splits']['validation_seeds'])}`; opponents: "
        f"`{', '.join(report['splits']['opponents'])}`; sealed test executed: "
        f"`{str(bool(report['sealed_test_executed'])).lower()}`.\n\n"
        "## Gates\n\n"
        + gates
        + "\n\n"
        "## Evidence boundary\n\n"
        "The model was hashed and held immutable in memory before the validation "
        "oracle was opened. The v0 artifacts retain the resulting hashes and file "
        "timestamps, but do not retain a separate before/after identity snapshot.\n"
    )


def load_experiment(
    args: argparse.Namespace,
) -> tuple[
    NativeTeammateBundle, str, list[dict[str, Any]],
    dict[str, Sequence[Mapping[str, Any]]], dict[str, tuple[str, ...]],
    dict[str, Any], dict[str, Any],
]:
    root = args.frozen_run.resolve()
    report_path = root / "FINAL_REPORT.json"
    frozen_path = root / "frozen_genomes.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    pinned = dict(report.get("artifacts", {}).get("frozen_genomes") or {})
    if pinned and _sha256_file(frozen_path) != str(pinned.get("sha256")):
        raise ValueError("frozen genome artifact digest mismatch")
    if (
        frozen.get("schema") != "block-sequence-oracle-v1-frozen-genomes"
        or tuple(map(int, frozen.get("anchors", ()))) != ANCHORS
        or tuple(map(int, frozen.get("stops", ()))) != tuple(map(int, STOPS))
        or int(frozen.get("horizon", -1)) != HORIZON
    ):
        raise ValueError("invalid frozen genome runtime contract")
    if bool(frozen.get("split_manifest", {}).get("test_executed", True)):
        raise ValueError("source search unexpectedly opened its sealed test")
    opponents = tuple(args.opponent or TARGET_OPPONENTS)
    records = dict(frozen.get("genomes") or {})
    if any(opponent not in records for opponent in opponents):
        raise ValueError("requested opponent is missing a frozen genome")
    prepared = args.prepared_root or [
        adaptive.DEFAULT_OLD_PREPARED, adaptive.DEFAULT_NEW_PREPARED,
    ]
    bank, bank_signatures = adaptive._load_banks([
        Path(value).resolve() for value in prepared
    ])
    baseline_id, baseline_tape, _, _, v1_args, v1_manifest = (
        continuation.load_frozen_v1_openings(args.v1_root)
    )
    if str(frozen.get("baseline_route_id")) != baseline_id:
        raise ValueError("frozen baseline route changed")
    genomes: dict[str, tuple[str, ...]] = {}
    selected_routes = set()
    for opponent in opponents:
        record = dict(records[opponent])
        genome = tuple(map(str, record.get("route_ids", ())))
        if len(genome) != len(ANCHORS) or _json_sha256(list(genome)) != str(record.get("genome_sha256")):
            raise ValueError(f"invalid frozen genome for {opponent}")
        if any(route_id != baseline_id and route_id not in bank for route_id in genome):
            raise ValueError(f"frozen genome route is absent from replay bank: {opponent}")
        genomes[opponent] = genome
        selected_routes.update(genome)
    tapes: dict[str, Sequence[Mapping[str, Any]]] = {baseline_id: baseline_tape}
    tapes.update({route_id: bank[route_id] for route_id in selected_routes if route_id != baseline_id})
    additional = {route_id: tape for route_id, tape in tapes.items()}
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*opponents, *additional)),
    )
    for method in (
        "action_at_with_raw_override", "rollout_schedule_raw_override_batch",
        "rollout_schedule_batch", "advance_segment",
    ):
        if not hasattr(bundle.executor, method):
            raise RuntimeError(f"native extension is missing {method}")
    inputs = {
        "source_search_report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        "frozen_genomes": {"path": str(frozen_path), "sha256": _sha256_file(frozen_path)},
        "prepared_banks": bank_signatures,
        "v1_manifest_sha256": _json_sha256(v1_manifest),
        "baseline_tape_sha256": _json_sha256(baseline_tape),
        "selected_route_tapes": {
            route_id: _json_sha256(tape) for route_id, tape in sorted(tapes.items())
        },
        "native_extension": {
            "path": str(Path(native_extension.__file__).resolve()),
            "sha256": _sha256_file(Path(native_extension.__file__).resolve()),
        },
        "fast_package": {
            "path": str(Path(fast_kaggriculture.__file__).resolve()),
            "sha256": _sha256_file(Path(fast_kaggriculture.__file__).resolve()),
        },
    }
    source = {
        "report_status": report.get("status"),
        "holdout_eligible": bool(frozen.get("holdout_eligible")),
        "sealed_test_executed": False,
    }
    return bundle, baseline_id, baseline_tape, tapes, genomes, inputs, source


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    train_seeds = tuple(range(args.train_seed_start, args.train_seed_start + args.train_seed_count))
    validation_seeds = tuple(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_seed_count,
    ))
    if set(train_seeds) & set(validation_seeds) or (set(train_seeds) | set(validation_seeds)) & SEALED_SEEDS:
        raise ValueError("train/validation seeds overlap each other or the sealed test")
    opponents = tuple(args.opponent or TARGET_OPPONENTS)
    if any(opponent not in TARGET_OPPONENTS for opponent in opponents):
        raise ValueError("v0 is scoped to the two frozen hard opponents")
    split_manifest = {
        "schema": "route-residual-adapter-split-v0",
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "sealed_test_seeds": "not enumerated by this runner",
        "opponents": list(opponents),
        "seats": [0, 1],
        "grouping": "one seed keeps all opponents, seats, steps and candidates together",
        "test_executed": False,
    }
    split_manifest["sha256"] = _json_sha256(split_manifest)
    split_path = output / "split_manifest.json"
    vocabulary = {
        "schema": "route-residual-vocabulary-v0",
        "codes": list(EDIT_CODES),
        "semantics": {
            "KEEP": "null override; execute frozen route raw action",
            "B0_UNITS": "B0 farmer/hands plus frozen-route market",
            "B0_MARKET": "frozen-route farmer/hands plus B0 market",
            "B0_FULL": "B0 full raw action for this turn only",
            "DROP_BUYS": "frozen-route units and sells; delete all purchases",
            "CLEAR_MARKET": "frozen-route units and empty market",
        },
        "application_point": "raw tape action before native stateful overlays",
        "causality": (
            "online at step t: hidden state produced before t is retained; "
            "the residual does not retroactively rewrite earlier lookahead"
        ),
        "duration_steps": 1,
        "future_schedule": "unchanged frozen 21-gene genome",
        "alias_rule": "canonicalize identical raw actions; KEEP wins exact ties",
    }
    vocabulary_path = output / "residual_vocabulary.json"
    bundle, baseline_id, baseline_tape, tapes, genomes, inputs, source = load_experiment(args)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_text(split_path, json.dumps(
        split_manifest, ensure_ascii=False, indent=2,
    ) + "\n")
    _atomic_text(vocabulary_path, json.dumps(
        vocabulary, ensure_ascii=False, indent=2,
    ) + "\n")
    genome_hashes = {
        opponent: _json_sha256(list(genome)) for opponent, genome in genomes.items()
    }
    execution_hashes = {
        opponent: _json_sha256(execution_route_ids(genome))
        for opponent, genome in genomes.items()
    }
    panel = TrainingPanel.empty()
    train_fixed: list[dict[str, Any]] = []
    train_oracle: list[dict[str, Any]] = []
    train_decisions: list[dict[str, Any]] = []
    decision_id = 0
    train_scenarios = [
        (opponent, seed, seat)
        for seed in train_seeds for opponent in opponents for seat in (0, 1)
    ]
    for index, (opponent, seed, seat) in enumerate(train_scenarios, 1):
        fixed = fixed_result(
            bundle, baseline_id, opponent, genomes[opponent],
            genome_hashes[opponent], seed, seat,
        )
        fixed["split"] = "train"
        oracle, decisions, decision_id = oracle_scenario(
            bundle, baseline_id, baseline_tape, tapes, opponent,
            genomes[opponent], genome_hashes[opponent], seed, seat,
            fixed, panel, decision_id, "train",
        )
        train_fixed.append(fixed)
        train_oracle.append(oracle)
        train_decisions.extend(decisions)
        print(json.dumps({
            "event": "train_oracle_scenario_complete", "index": index,
            "total": len(train_scenarios), "opponent": opponent,
            "seed": seed, "seat": seat, "edits": oracle["edits"],
            "delta_margin": oracle["margin"] - fixed["margin"],
        }, sort_keys=True), flush=True)
    arrays = panel.arrays()
    dataset_path = output / "training_panel.npz"
    _atomic_npz(dataset_path, arrays)
    model, cv, oof = fit_ranker(
        arrays, args.trees, args.cv_trees, args.cv_folds, args.random_seed,
    )
    model_path = output / "residual_ranker.joblib"
    _atomic_joblib(model_path, model)
    oof_path = output / "oof_predictions.npz"
    _atomic_npz(oof_path, oof)
    model_manifest = {
        "schema": "route-residual-ranker-v0",
        "model": "ExtraTreesRegressor candidate-conditioned signed-log delta margin",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "training_rows": len(arrays["features"]),
        "trajectory_groups": len(train_seeds),
        "calibration": cv,
        "model_sha256": _sha256_file(model_path),
        "training_panel_sha256": _sha256_file(dataset_path),
        "split_manifest_sha256": _sha256_file(split_path),
        "frozen_before_validation_oracle": True,
    }
    model_manifest_path = output / "model_manifest.json"
    _atomic_text(model_manifest_path, json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n")
    frozen_model_identity = {
        path: (_sha256_file(path), path.stat().st_mtime_ns)
        for path in (model_path, model_manifest_path, dataset_path, split_path, vocabulary_path)
    }

    validation_fixed: list[dict[str, Any]] = []
    validation_learned: list[dict[str, Any]] = []
    learned_decisions: list[dict[str, Any]] = []
    validation_scenarios = [
        (opponent, seed, seat)
        for seed in validation_seeds for opponent in opponents for seat in (0, 1)
    ]
    for index, (opponent, seed, seat) in enumerate(validation_scenarios, 1):
        fixed = fixed_result(
            bundle, baseline_id, opponent, genomes[opponent],
            genome_hashes[opponent], seed, seat,
        )
        fixed["split"] = "validation"
        learned, decisions = learned_scenario(
            bundle, baseline_id, baseline_tape, tapes, opponent,
            genomes[opponent], genome_hashes[opponent], seed, seat,
            model, cv["chosen"],
        )
        validation_fixed.append(fixed)
        validation_learned.append(learned)
        learned_decisions.extend(decisions)
        print(json.dumps({
            "event": "learned_validation_scenario_complete", "index": index,
            "total": len(validation_scenarios), "opponent": opponent,
            "seed": seed, "seat": seat, "edits": learned["edits"],
            "delta_margin": learned["margin"] - fixed["margin"],
        }, sort_keys=True), flush=True)

    # Validation oracle is deliberately opened only after the model and all
    # learned closed-loop trajectories are frozen.  Its labels never refit or
    # recalibrate the adapter.
    validation_oracle: list[dict[str, Any]] = []
    validation_oracle_decisions: list[dict[str, Any]] = []
    fixed_by_key = {
        (row["opponent"], row["seed"], row["seat"]): row
        for row in validation_fixed
    }
    for index, (opponent, seed, seat) in enumerate(validation_scenarios, 1):
        fixed = fixed_by_key[(opponent, seed, seat)]
        oracle, decisions, decision_id = oracle_scenario(
            bundle, baseline_id, baseline_tape, tapes, opponent,
            genomes[opponent], genome_hashes[opponent], seed, seat,
            fixed, None, decision_id, "validation_oracle_only",
        )
        validation_oracle.append(oracle)
        validation_oracle_decisions.extend(decisions)
        print(json.dumps({
            "event": "validation_oracle_scenario_complete", "index": index,
            "total": len(validation_scenarios), "opponent": opponent,
            "seed": seed, "seat": seat, "edits": oracle["edits"],
            "delta_margin": oracle["margin"] - fixed["margin"],
        }, sort_keys=True), flush=True)
    after_model_identity = {
        path: (_sha256_file(path), path.stat().st_mtime_ns)
        for path in frozen_model_identity
    }
    if after_model_identity != frozen_model_identity:
        raise RuntimeError("frozen model artifacts changed during validation oracle")

    train_oracle_paired = _paired_rows(train_oracle, train_fixed)
    validation_learned_paired = _paired_rows(validation_learned, validation_fixed)
    validation_oracle_paired = _paired_rows(validation_oracle, validation_fixed)
    artifacts: dict[str, dict[str, Any]] = {}
    rows_by_file = {
        "train_fixed.jsonl": train_fixed,
        "train_oracle.jsonl": train_oracle_paired,
        "train_oracle_decisions.jsonl": train_decisions,
        "validation_fixed.jsonl": validation_fixed,
        "validation_learned.jsonl": validation_learned_paired,
        "validation_learned_decisions.jsonl": learned_decisions,
        "validation_oracle.jsonl": validation_oracle_paired,
        "validation_oracle_decisions.jsonl": validation_oracle_decisions,
    }
    for name, rows in rows_by_file.items():
        path = output / name
        _write_jsonl(path, rows)
        artifacts[name] = _artifact(path, len(rows))
    for path in (
        split_path, vocabulary_path, dataset_path, oof_path,
        model_path, model_manifest_path,
    ):
        artifacts[path.name] = _artifact(path)

    learned_summary = _summary(validation_learned_paired)
    oracle_summary = _summary(validation_oracle_paired)
    fixed_summary = _summary(validation_fixed)
    gates = {
        "G0_keep_matches_same_frozen_schedule": all(
            int(row["keep_equivalence_checks"]) == len(KEEP_AUDIT_STEPS)
            for row in (*train_oracle, *validation_oracle)
        ),
        "G1_all_arms_keep_identical_genome_and_zero_route_changes": all(
            row["genome_sha256"] == genome_hashes[row["opponent"]]
            and row["execution_route_sha256"] == execution_hashes[row["opponent"]]
            and int(row["route_change_count"]) == 0
            for row in (
                *train_fixed, *train_oracle, *validation_fixed,
                *validation_learned, *validation_oracle,
            )
        ),
        "G2_validation_oracle_has_headroom": (
            oracle_summary["mean_delta_margin_vs_fixed"] > 0
            and oracle_summary["edits"] > 0
        ),
        "G3_validation_oracle_repairs_a_loss_to_win": oracle_summary["loss_repairs_to_win"] > 0,
        "G4_learned_adapter_fires": learned_summary["edits"] > 0,
        "G5_learned_adapter_generalizes_positive_mean_or_repairs_loss": (
            learned_summary["mean_delta_margin_vs_fixed"] > 0
            or learned_summary["loss_repairs_to_win"] > 0
        ),
        "G6_learned_adapter_has_no_outcome_regression": learned_summary["outcome_regressions_vs_fixed"] == 0,
        "G7_all_trajectories_complete": all(
            bool(row["completed"]) for row in (
                *train_fixed, *train_oracle, *validation_fixed,
                *validation_learned, *validation_oracle,
            )
        ),
        "G8_model_was_frozen_before_validation_oracle": after_model_identity == frozen_model_identity,
    }
    if not gates["G2_validation_oracle_has_headroom"]:
        status = "vocabulary_insufficient"
    elif not gates["G5_learned_adapter_generalizes_positive_mean_or_repairs_loss"]:
        status = "oracle_only_model_failed"
    elif not gates["G6_learned_adapter_has_no_outcome_regression"]:
        status = "learned_but_unsafe"
    elif all(gates.values()):
        status = "residual_adapter_feasible"
    else:
        status = "partial_residual_feasibility"
    report = {
        "schema": SCHEMA,
        "status": status,
        "claim": "fixed 21-gene route with a learned one-turn raw-action residual adapter",
        "not_claimed": "this v0 does not meet the full-pool 90/80 championship gate",
        "source": source,
        "inputs": inputs,
        "splits": split_manifest,
        "baseline_route_id": baseline_id,
        "genomes": {
            opponent: {
                "route_ids": list(genome),
                "genome_sha256": genome_hashes[opponent],
                "execution_route_sha256": execution_hashes[opponent],
            }
            for opponent, genome in genomes.items()
        },
        "vocabulary": vocabulary,
        "model": model_manifest,
        "train": {
            "fixed": _summary(train_fixed),
            "oracle": _summary(train_oracle_paired),
        },
        "validation": {
            "fixed": fixed_summary,
            "learned": learned_summary,
            "oracle": oracle_summary,
        },
        "gates": gates,
        "sealed_test_executed": False,
        "artifacts": artifacts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "FINAL_REPORT.json"
    _atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(output / "FINAL_REPORT.md", _report_markdown(report))
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--opponent", action="append", default=None)
    result.add_argument("--train-seed-start", type=int, default=TRAIN_SEEDS[0])
    result.add_argument("--train-seed-count", type=int, default=len(TRAIN_SEEDS))
    result.add_argument("--validation-seed-start", type=int, default=VALIDATION_SEEDS[0])
    result.add_argument("--validation-seed-count", type=int, default=len(VALIDATION_SEEDS))
    result.add_argument("--trees", type=int, default=96)
    result.add_argument("--cv-trees", type=int, default=24)
    result.add_argument("--cv-folds", type=int, default=5)
    result.add_argument("--random-seed", type=int, default=2026082802)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.train_seed_count < 2 or args.validation_seed_count < 1
        or args.trees < 8 or args.cv_trees < 4 or args.cv_folds < 2
    ):
        raise ValueError("invalid ROUTE-RESIDUAL-ADAPTER-v0 budget")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
