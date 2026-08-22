"""V5 task-card union, task state machines, real ETA, and reserved execution.

V3 stores each task once and scores a worker-by-task matrix. V4 adds persistent
task phases, acquisition-aware ETA, and prioritized space-time A* reservations.
V5 additionally reserves candidate capacity for learned and expert-injected cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from math import log1p
from typing import Any, Mapping, Sequence

import numpy as np

from .board_policy import BOARD_SIZE, _canonical_farms, _positions
from .decision_schema import (
    CANDIDATE_FEATURES,
    TASK_PRIOR,
    CandidateSource,
    DecisionCandidate,
    DecisionMemory,
    TaskType,
    _candidate_features,
    _compile_unit_candidate,
    _distance,
    _inventory,
    _nearest_shed,
    _plan_relevant,
    _tile_at,
    _unit_options,
)
from .gpu_policy import (
    ANIMALS,
    CROPS,
    ITEMS,
    MARKET_ACTIONS,
    MARKET_INDEX,
    MAX_UNITS,
    PRODUCTS,
    _get,
    _mapping,
)
from .task_planning import (
    bfs_shortest_path,
    movement_for_path,
    plan_prioritized_routes,
)


MAX_TASK_CARDS = 64
PAIR_FEATURES = 12
MAX_MARKET_ORDERS = 10


class BudgetCategory(IntEnum):
    """Mutually exclusive envelopes for cash-consuming market orders."""

    LABOR = 0
    CROP = 1
    ANIMAL = 2
    LAND = 3
    SUPPLIES = 4


BUDGET_CATEGORIES = tuple(category.name for category in BudgetCategory)
BUDGET_FEATURES = 2 + len(BUDGET_CATEGORIES)

STRATEGY_MODES = (
    "BALANCED",
    "CROP_EXPANSION",
    "ANIMAL_CASHFLOW",
    "PREMIUM_MARKET",
    "CONSERVATIVE",
    "TERMINAL_LIQUIDATION",
    "ANTI_MIRROR",
    "OPPONENT_COUNTER",
    "ROBUST_OPENING",
)

MODE_INDEX = {name: index for index, name in enumerate(STRATEGY_MODES)}
_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
_ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_LAND_PRICES = (1_000, 2_000, 4_000)
_MARKET_BASE = dict(zip(PRODUCTS, (25, 35, 60, 120, 250, 50, 160, 200, 100), strict=True))
_MARKET_T = dict(zip(PRODUCTS, (400, 450, 200, 100, 300, 332, 122, 105, 200), strict=True))
_MARKET_BELOW_FUNC = dict(zip(PRODUCTS, (2, 3, 0, 2, 3, 0, 2, 3, 0), strict=True))
_MARKET_BELOW_TARGET = dict(zip(PRODUCTS, (0.80, 0.20, 0.40, 0.70, 0.20, 0.40, 0.60, 0.20, 0.40), strict=True))
_MARKET_ABOVE_FUNC = dict(zip(PRODUCTS, (3, 2, 2, 0, 1, 3, 0, 1, 0), strict=True))
_MARKET_ABOVE_TARGET = dict(zip(PRODUCTS, (0.20, 0.70, 0.60, 1.60, 3.60, 0.20, 1.60, 3.20, 0.40), strict=True))


@dataclass(frozen=True)
class BudgetPlan:
    """One persistent daily cash plan produced by the learned budget head.

    ``category_limits`` are absolute cash envelopes fixed when the plan is
    created.  Sales later in the day do not silently enlarge them.  Required
    task resources may use the separately bounded emergency allowance.
    """

    day: int
    cash_at_plan: float
    cash_floor: float
    reserve_fraction: float
    category_fractions: tuple[float, ...]
    category_limits: tuple[float, ...]
    emergency_fraction: float = 0.0
    emergency_limit: float = 0.0
    minimum_reserve_fraction: float = 0.0
    effective_reserve_fraction: float = 0.0

    def __post_init__(self) -> None:
        expected = len(BUDGET_CATEGORIES)
        if len(self.category_fractions) != expected:
            raise ValueError("budget category fractions have an incompatible length")
        if len(self.category_limits) != expected:
            raise ValueError("budget category limits have an incompatible length")
        if self.cash_at_plan < 0 or self.cash_floor < 0:
            raise ValueError("budget cash values must be non-negative")
        if self.cash_floor > self.cash_at_plan + 1e-6:
            raise ValueError("budget cash floor cannot exceed planned cash")
        if any(value < 0 for value in self.category_limits):
            raise ValueError("budget category limits must be non-negative")

    @classmethod
    def from_fractions(
        cls,
        *,
        day: int,
        money: float,
        reserve_fraction: float,
        category_fractions: Sequence[float],
        emergency_fraction: float = 0.0,
        minimum_reserve_fraction: float = 0.0,
    ) -> "BudgetPlan":
        money = max(float(money), 0.0)
        reserve = float(np.clip(reserve_fraction, 0.0, 1.0))
        minimum_reserve = float(np.clip(minimum_reserve_fraction, 0.0, 1.0))
        effective_reserve = max(reserve, minimum_reserve)
        emergency = float(np.clip(emergency_fraction, 0.0, 1.0))
        shares = np.asarray(category_fractions, dtype=np.float64)
        if shares.shape != (len(BUDGET_CATEGORIES),):
            raise ValueError("budget category fractions have an incompatible shape")
        shares = np.maximum(shares, 0.0)
        total = float(shares.sum())
        shares = (
            shares / total
            if total > 1e-12
            else np.full(len(BUDGET_CATEGORIES), 1.0 / len(BUDGET_CATEGORIES))
        )
        cash_floor = money * effective_reserve
        spendable = max(money - cash_floor, 0.0)
        return cls(
            day=int(day),
            cash_at_plan=money,
            cash_floor=cash_floor,
            reserve_fraction=reserve,
            category_fractions=tuple(float(value) for value in shares),
            category_limits=tuple(float(value * spendable) for value in shares),
            emergency_fraction=emergency,
            emergency_limit=money * emergency,
            minimum_reserve_fraction=minimum_reserve,
            effective_reserve_fraction=effective_reserve,
        )

    @property
    def features(self) -> tuple[float, ...]:
        return (
            self.reserve_fraction,
            *self.category_fractions,
            self.emergency_fraction,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "cash_at_plan": self.cash_at_plan,
            "cash_floor": self.cash_floor,
            "reserve_fraction": self.reserve_fraction,
            "minimum_reserve_fraction": self.minimum_reserve_fraction,
            "effective_reserve_fraction": self.effective_reserve_fraction,
            "category_fractions": {
                name: self.category_fractions[index]
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "category_limits": {
                name: self.category_limits[index]
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "emergency_fraction": self.emergency_fraction,
            "emergency_limit": self.emergency_limit,
        }


def market_budget_category(action_index: int) -> BudgetCategory | None:
    """Map a primitive purchase to exactly one spending envelope."""

    operation, _item, _quantity = MARKET_ACTIONS[int(action_index)]
    return {
        "HIRE": BudgetCategory.LABOR,
        "BUY_SEED": BudgetCategory.CROP,
        "BUY_ANIMAL": BudgetCategory.ANIMAL,
        "BUY_LAND": BudgetCategory.LAND,
        "BUY_PRODUCT": BudgetCategory.SUPPLIES,
    }.get(operation)


def _fib(index: int) -> int:
    left, right = 1, 1
    for _ in range(max(index, 0)):
        left, right = right, left + right
    return left


def _market_shape(code: int, value: float) -> float:
    value = max(value, 0.0)
    if code == 0:
        return value
    if code == 1:
        return value * value
    if code == 2:
        return value**0.5
    return log1p(value)


def _market_quote(item: str, inventory: int) -> float:
    base = float(_MARKET_BASE[item])
    below = inventory < 10_000
    code = _MARKET_BELOW_FUNC[item] if below else _MARKET_ABOVE_FUNC[item]
    target = _MARKET_BELOW_TARGET[item] if below else _MARKET_ABOVE_TARGET[item]
    scale = _market_shape(code, float(_MARKET_T[item]))
    amplitude = target * base / max(scale, 1e-12)
    delta = amplitude * _market_shape(code, abs(float(inventory - 10_000)))
    price = base + delta if below else base - delta
    return max(float(np.rint(price)), 1.0)


class CandidateOrigin(IntEnum):
    """How a task card entered the deployable candidate union."""

    RULE = 0
    CONTINUATION = 1
    LEARNED = 2
    EXPERT = 3


@dataclass(frozen=True)
class TaskCard:
    candidate: DecisionCandidate
    preferred_owner: int = -1
    capacity: int = 1
    origin: CandidateOrigin = CandidateOrigin.RULE

    @property
    def task_type(self) -> TaskType:
        return self.candidate.task_type

    @property
    def item(self) -> str | None:
        return self.candidate.item

    @property
    def target(self) -> tuple[int, int]:
        return self.candidate.target_x, self.candidate.target_y


class ExecutorFailureReason(str, Enum):
    INVALID_TASK_INDEX = "INVALID_TASK_INDEX"
    PAIR_MASKED = "PAIR_MASKED"
    INVALID_MARKET_INDEX = "INVALID_MARKET_INDEX"
    MISSING_RESOURCE = "MISSING_RESOURCE"
    TARGET_RESERVED = "TARGET_RESERVED"
    PATH_BLOCKED = "PATH_BLOCKED"
    PRECONDITION_CHANGED = "PRECONDITION_CHANGED"


@dataclass(frozen=True)
class ExecutorFailure:
    step: int
    day: int
    unit: int
    task_type: str
    target: tuple[int, int]
    reason: str
    action: tuple[Any, ...]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "day": self.day,
            "unit": self.unit,
            "task_type": self.task_type,
            "target": list(self.target),
            "reason": self.reason,
            "action": list(self.action),
            "detail": self.detail,
        }


class TaskPhase(str, Enum):
    IDLE = "IDLE"
    ASSIGNED = "ASSIGNED"
    ACQUIRE_RESOURCE = "ACQUIRE_RESOURCE"
    NAVIGATE = "NAVIGATE"
    EXECUTE = "EXECUTE"
    WAITING = "WAITING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TaskTerminationReason(str, Enum):
    NONE = "NONE"
    STAGE_ADVANCED = "STAGE_ADVANCED"
    GOAL_SATISFIED = "GOAL_SATISFIED"
    POLICY_IDLED = "POLICY_IDLED"
    ASSIGNMENT_REPLACED = "ASSIGNMENT_REPLACED"
    PRECONDITION_INVALID = "PRECONDITION_INVALID"


_CHAIN_FAMILY = {
    TaskType.CROP_PRODUCTION: "CROP_LIFECYCLE",
    TaskType.WATER_CROP: "CROP_LIFECYCLE",
    TaskType.APPLY_FERTILIZER: "CROP_LIFECYCLE",
    TaskType.BUILD_ANIMAL_STRUCTURE: "ANIMAL_LIFECYCLE",
    TaskType.ANIMAL_PLACE: "ANIMAL_LIFECYCLE",
    TaskType.ANIMAL_FEED: "ANIMAL_LIFECYCLE",
    TaskType.ANIMAL_CARE: "ANIMAL_LIFECYCLE",
    TaskType.ANIMAL_COLLECT_PRODUCT: "ANIMAL_LIFECYCLE",
    TaskType.ANIMAL_COLLECT_FERTILIZER: "ANIMAL_LIFECYCLE",
}


def task_chain_family(candidate: DecisionCandidate) -> str:
    """Return the persistent objective family used by the runtime state machine."""

    return _CHAIN_FAMILY.get(candidate.task_type, candidate.task_type.name)


def _same_task(left: DecisionCandidate | None, right: DecisionCandidate) -> bool:
    return bool(
        left is not None
        and left.task_type == right.task_type
        and left.target_x == right.target_x
        and left.target_y == right.target_y
        and left.item_id == right.item_id
    )


def same_task_chain(left: DecisionCandidate | None, right: DecisionCandidate) -> bool:
    """Whether two cards are consecutive stages of one spatial objective."""

    if left is None or right.task_type == TaskType.IDLE_OR_PASS:
        return False
    if _same_task(left, right):
        return True
    left_family = task_chain_family(left)
    return bool(
        left_family in ("CROP_LIFECYCLE", "ANIMAL_LIFECYCLE")
        and left_family == task_chain_family(right)
        and left.target_x == right.target_x
        and left.target_y == right.target_y
    )


@dataclass
class UnitTaskState:
    task: DecisionCandidate | None = None
    phase: TaskPhase = TaskPhase.IDLE
    started_step: int = -1
    updated_step: int = -1
    eta_steps: int = 0
    route: tuple[tuple[int, int], ...] = ()
    failure_count: int = 0
    chain_id: str = ""
    chain_family: str = ""
    chain_tasks: tuple[DecisionCandidate, ...] = ()
    stage_index: int = 0
    stage_started_step: int = -1
    completed_stages: int = 0
    termination_reason: str = TaskTerminationReason.NONE.value


@dataclass(frozen=True)
class TaskTransition:
    step: int
    day: int
    unit: int
    task_type: str
    previous_phase: str
    phase: str
    eta_steps: int
    chain_id: str
    chain_family: str
    stage_index: int
    completed_stages: int
    termination_reason: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "day": self.day,
            "unit": self.unit,
            "task_type": self.task_type,
            "previous_phase": self.previous_phase,
            "phase": self.phase,
            "eta_steps": self.eta_steps,
            "chain_id": self.chain_id,
            "chain_family": self.chain_family,
            "stage_index": self.stage_index,
            "completed_stages": self.completed_stages,
            "termination_reason": self.termination_reason,
            "detail": self.detail,
        }


@dataclass
class HierarchicalMemory:
    decision: DecisionMemory = field(default_factory=DecisionMemory)
    strategy_mode: int = -1
    strategy_day: int = -1
    mode_age: int = 0
    mode_switches: int = 0
    mode_history: list[dict[str, Any]] = field(default_factory=list)
    budget_day: int = -1
    budget_plan: BudgetPlan | None = None
    budget_spent_by_category: list[float] = field(
        default_factory=lambda: [0.0] * len(BUDGET_CATEGORIES)
    )
    budget_emergency_spent: float = 0.0
    opponent_observation_history: list[tuple[int, tuple[float, ...]]] = field(
        default_factory=list
    )
    task_states: list[UnitTaskState] = field(
        default_factory=lambda: [UnitTaskState() for _ in range(MAX_UNITS)]
    )
    task_transition_counts: dict[str, int] = field(default_factory=dict)
    recent_task_transitions: list[TaskTransition] = field(default_factory=list)
    executor_failure_counts: dict[str, int] = field(default_factory=dict)
    executor_task_failure_counts: dict[str, int] = field(default_factory=dict)
    recent_executor_failures: list[ExecutorFailure] = field(default_factory=list)
    internal_trace_history: list[dict[str, Any]] = field(default_factory=list)
    max_executor_failures: int = 128
    max_task_transitions: int = 256
    max_internal_traces: int = 256
    next_chain_serial: int = 0

    def transition_task(
        self,
        unit: int,
        phase: TaskPhase,
        *,
        observation: Any,
        candidate: DecisionCandidate | None = None,
        eta_steps: int | None = None,
        route: Sequence[tuple[int, int]] | None = None,
        continue_chain: bool = False,
        termination_reason: TaskTerminationReason | str = TaskTerminationReason.NONE,
        detail: str = "",
    ) -> None:
        state = self.task_states[unit]
        previous = state.phase
        step = int(_get(observation, "step", 0) or 0)
        if candidate is not None:
            old = state.task
            same = _same_task(old, candidate)
            if not same:
                if candidate.task_type == TaskType.IDLE_OR_PASS:
                    state.chain_id = ""
                    state.chain_family = ""
                    state.chain_tasks = ()
                    state.stage_index = 0
                    state.stage_started_step = -1
                    state.completed_stages = 0
                elif continue_chain and state.chain_id:
                    state.stage_index += 1
                    state.stage_started_step = step
                    state.chain_tasks = state.chain_tasks + (candidate,)
                else:
                    serial = self.next_chain_serial
                    self.next_chain_serial += 1
                    family = task_chain_family(candidate)
                    state.chain_id = (
                        f"c{serial}:s{step}:u{unit}:{family}:"
                        f"{candidate.target_x},{candidate.target_y}"
                    )
                    state.chain_family = family
                    state.chain_tasks = (candidate,)
                    state.stage_index = 0
                    state.stage_started_step = step
                    state.completed_stages = 0
                    state.started_step = step
                state.failure_count = 0
                state.termination_reason = TaskTerminationReason.NONE.value
            state.task = candidate
        if eta_steps is not None:
            state.eta_steps = int(eta_steps)
        if route is not None:
            state.route = tuple(route)
        state.updated_step = step
        state.phase = phase
        if phase in (TaskPhase.COMPLETED, TaskPhase.CANCELLED):
            reason = (
                termination_reason.value
                if isinstance(termination_reason, TaskTerminationReason)
                else str(termination_reason)
            )
            state.termination_reason = reason
            if phase == TaskPhase.COMPLETED:
                state.completed_stages += 1
        if previous == phase and not detail:
            return
        task = state.task
        task_name = task.task_type.name if task is not None else TaskType.IDLE_OR_PASS.name
        key = f"{task_name}:{phase.value}"
        self.task_transition_counts[key] = self.task_transition_counts.get(key, 0) + 1
        transition = TaskTransition(
            step=state.updated_step,
            day=int(_get(observation, "day", state.updated_step // 24) or 0),
            unit=unit,
            task_type=task_name,
            previous_phase=previous.value,
            phase=phase.value,
            eta_steps=state.eta_steps,
            chain_id=state.chain_id,
            chain_family=state.chain_family,
            stage_index=state.stage_index,
            completed_stages=state.completed_stages,
            termination_reason=state.termination_reason,
            detail=detail,
        )
        self.recent_task_transitions.append(transition)
        overflow = len(self.recent_task_transitions) - self.max_task_transitions
        if overflow > 0:
            del self.recent_task_transitions[:overflow]

    def record_mode_transition(
        self,
        *,
        day: int,
        previous_mode: int,
        mode: int,
        switched: bool,
    ) -> None:
        if switched:
            self.mode_switches += 1
        self.mode_history.append(
            {
                "day": int(day),
                "decision": "SWITCH" if switched else "KEEP",
                "previous_mode": int(previous_mode),
                "mode": int(mode),
            }
        )
        if len(self.mode_history) > 64:
            del self.mode_history[:-64]

    def record_executor_failure(self, failure: ExecutorFailure) -> None:
        self.executor_failure_counts[failure.reason] = (
            self.executor_failure_counts.get(failure.reason, 0) + 1
        )
        key = f"{failure.task_type}:{failure.reason}"
        self.executor_task_failure_counts[key] = (
            self.executor_task_failure_counts.get(key, 0) + 1
        )
        self.recent_executor_failures.append(failure)
        overflow = len(self.recent_executor_failures) - self.max_executor_failures
        if overflow > 0:
            del self.recent_executor_failures[:overflow]

    def executor_diagnostics(self) -> dict[str, Any]:
        return {
            "failure_count": int(sum(self.executor_failure_counts.values())),
            "by_reason": dict(sorted(self.executor_failure_counts.items())),
            "by_task_and_reason": dict(
                sorted(self.executor_task_failure_counts.items())
            ),
            "recent_failures": [
                failure.as_dict() for failure in self.recent_executor_failures
            ],
            "task_state_machine": {
                "transition_count": int(sum(self.task_transition_counts.values())),
                "by_task_and_phase": dict(sorted(self.task_transition_counts.items())),
                "current": [
                    {
                        "unit": unit,
                        "task_type": (
                            state.task.task_type.name
                            if state.task is not None
                            else TaskType.IDLE_OR_PASS.name
                        ),
                        "phase": state.phase.value,
                        "eta_steps": state.eta_steps,
                        "route": [list(position) for position in state.route],
                        "failure_count": state.failure_count,
                        "chain_id": state.chain_id,
                        "chain_family": state.chain_family,
                        "stage_index": state.stage_index,
                        "chain_length": len(state.chain_tasks),
                        "completed_stages": state.completed_stages,
                        "termination_reason": state.termination_reason,
                    }
                    for unit, state in enumerate(self.task_states)
                ],
                "recent_transitions": [
                    transition.as_dict()
                    for transition in self.recent_task_transitions
                ],
            },
            "strategy": {
                "mode": self.strategy_mode,
                "day": self.strategy_day,
                "switches": self.mode_switches,
                "history": list(self.mode_history),
            },
            "budget": (
                {
                    **self.budget_plan.as_dict(),
                    "spent_by_category": {
                        name: self.budget_spent_by_category[index]
                        for index, name in enumerate(BUDGET_CATEGORIES)
                    },
                    "emergency_spent": self.budget_emergency_spent,
                }
                if self.budget_plan is not None
                else None
            ),
            "agent_trace": {
                "count": len(self.internal_trace_history),
                "latest": (
                    self.internal_trace_history[-1]
                    if self.internal_trace_history
                    else None
                ),
            },
            "opponent_history_steps": len(self.opponent_observation_history),
        }

    def record_internal_trace(
        self,
        observation: Any,
        action: Mapping[str, Any],
    ) -> None:
        """Record the policy's unambiguous task state before the env advances."""

        workers: list[dict[str, Any]] = []
        for unit, state in enumerate(self.task_states):
            task = state.task
            if task is None:
                continue
            workers.append(
                {
                    "unit": unit,
                    "source": "internal",
                    "confidence": 1.0,
                    "phase": state.phase.value,
                    "hypotheses": [
                        {
                            "task_type": task.task_type.name,
                            "target": [task.target_x, task.target_y],
                            "item_id": task.item_id,
                            "probability": 1.0,
                            "eta_steps": state.eta_steps,
                            "route": [list(position) for position in state.route],
                            "evidence": "native_task_state_machine",
                            "phase": state.phase.value,
                            "chain_id": state.chain_id,
                            "chain_family": state.chain_family,
                            "stage_index": state.stage_index,
                            "chain_length": len(state.chain_tasks),
                            "completed_stages": state.completed_stages,
                            "termination_reason": state.termination_reason,
                            "chain": [
                                {
                                    "task_type": stage.task_type.name,
                                    "target": [stage.target_x, stage.target_y],
                                    "item_id": stage.item_id,
                                    "operation": stage.task_type.name,
                                    "eta_steps": (
                                        state.eta_steps
                                        if index == state.stage_index
                                        else 0
                                    ),
                                }
                                for index, stage in enumerate(state.chain_tasks)
                            ],
                        }
                    ],
                }
            )
        trace = {
            "schema": "kaggriculture.agent-trace.v2",
            "step": int(_get(observation, "step", 0) or 0),
            "day": int(_get(observation, "day", 0) or 0),
            "strategy_mode": self.strategy_mode,
            "source": "internal",
            "action": dict(action),
            "workers": workers,
        }
        self.internal_trace_history.append(trace)
        overflow = len(self.internal_trace_history) - self.max_internal_traces
        if overflow > 0:
            del self.internal_trace_history[:overflow]

    def reset(self) -> None:
        self.decision.reset()
        self.strategy_mode = -1
        self.strategy_day = -1
        self.mode_age = 0
        self.mode_switches = 0
        self.mode_history.clear()
        self.budget_day = -1
        self.budget_plan = None
        self.budget_spent_by_category[:] = [0.0] * len(BUDGET_CATEGORIES)
        self.budget_emergency_spent = 0.0
        self.opponent_observation_history.clear()
        self.task_states[:] = [UnitTaskState() for _ in range(MAX_UNITS)]
        self.task_transition_counts.clear()
        self.recent_task_transitions.clear()
        self.executor_failure_counts.clear()
        self.executor_task_failure_counts.clear()
        self.recent_executor_failures.clear()
        self.internal_trace_history.clear()
        self.next_chain_serial = 0


@dataclass(frozen=True)
class EncodedTaskSet:
    tasks: tuple[TaskCard, ...]
    features: np.ndarray
    task_ids: np.ndarray
    item_ids: np.ndarray
    source_ids: np.ndarray
    origin_ids: np.ndarray
    target_xy: np.ndarray
    preferred_owner_ids: np.ndarray
    task_mask: np.ndarray
    pair_features: np.ndarray
    eta_steps: np.ndarray
    pair_mask: np.ndarray
    capacity: np.ndarray
    required_market_indices: np.ndarray


def _task_key(card: TaskCard) -> tuple[int, int, int, int, int]:
    candidate = card.candidate
    owner_key = card.preferred_owner if card.preferred_owner >= 0 else -1
    return (
        int(candidate.task_type),
        candidate.target_x,
        candidate.target_y,
        candidate.item_id,
        owner_key,
    )


def _strip_owner(
    candidate: DecisionCandidate,
    *,
    preferred_owner: int = -1,
    continuation: bool | None = None,
    origin: CandidateOrigin = CandidateOrigin.RULE,
) -> TaskCard:
    value = replace(
        candidate,
        owner_unit=-1,
        path_steps=0,
        continuation=(candidate.continuation if continuation is None else continuation),
    )
    return TaskCard(
        candidate=value,
        preferred_owner=preferred_owner,
        capacity=MAX_UNITS if value.task_type == TaskType.IDLE_OR_PASS else 1,
        origin=origin,
    )


def _select_task_cards(
    cards: Sequence[tuple[TaskCard, int]],
    *,
    expert_quota: int = 8,
    learned_quota: int = 24,
) -> list[TaskCard]:
    """Merge rule, learned, and expert cards under explicit reserved quotas."""

    idle = next(card for card, _ in cards if card.task_type == TaskType.IDLE_OR_PASS)
    fixed: list[tuple[TaskCard, int]] = []
    experts: list[tuple[TaskCard, int]] = []
    learned: list[tuple[TaskCard, int]] = []
    optional: dict[TaskType, list[tuple[TaskCard, int]]] = {}
    for card, best_distance in cards:
        if card.task_type == TaskType.IDLE_OR_PASS:
            continue
        if card.candidate.mandatory or card.candidate.continuation:
            fixed.append((card, best_distance))
        elif card.origin == CandidateOrigin.EXPERT:
            experts.append((card, best_distance))
        elif card.origin == CandidateOrigin.LEARNED:
            learned.append((card, best_distance))
        else:
            optional.setdefault(card.task_type, []).append((card, best_distance))
    ranking = lambda value: (
        -value[0].candidate.prior,
        value[1],
        value[0].candidate.target_y,
        value[0].candidate.target_x,
        value[0].candidate.item_id,
    )
    fixed.sort(key=ranking)
    selected = [idle, *(card for card, _ in fixed[: MAX_TASK_CARDS - 1])]

    experts.sort(key=ranking)
    selected.extend(
        card
        for card, _ in experts[: min(max(expert_quota, 0), MAX_TASK_CARDS - len(selected))]
    )

    learned_by_family: dict[TaskType, list[tuple[TaskCard, int]]] = {}
    for value in learned:
        learned_by_family.setdefault(value[0].task_type, []).append(value)
    for values in learned_by_family.values():
        values.sort(key=ranking)
    learned_families = sorted(learned_by_family, key=int)
    learned_limit = min(
        max(learned_quota, 0), MAX_TASK_CARDS - len(selected)
    )
    cursor = 0
    while learned_limit > 0 and learned_families:
        family = learned_families[cursor % len(learned_families)]
        values = learned_by_family[family]
        if values:
            selected.append(values.pop(0)[0])
            learned_limit -= 1
        if not values:
            learned_families.remove(family)
            if not learned_families:
                break
            cursor %= len(learned_families)
        else:
            cursor += 1

    families = sorted(optional, key=int)
    for family in families:
        optional[family].sort(key=ranking)
    cursor = 0
    while len(selected) < MAX_TASK_CARDS and families:
        family = families[cursor % len(families)]
        values = optional[family]
        if values:
            selected.append(values.pop(0)[0])
        if not values:
            families.remove(family)
            if not families:
                break
            cursor %= len(families)
        else:
            cursor += 1
    return selected


@dataclass(frozen=True)
class TaskEta:
    movement_steps: int
    interaction_steps: int
    wait_steps: int
    total_steps: int
    reachable: bool
    deadline_feasible: bool
    immediate_goal: tuple[int, int]
    waypoints: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TaskResourcePlan:
    feasible: bool
    available: bool
    required_market_index: int = 0
    detail: str = ""


def _structural_candidate_legal(
    observation: Any,
    own: Any,
    unit: int,
    candidate: DecisionCandidate,
) -> bool:
    """Strict operation precondition without requiring the resource to exist yet."""

    positions = _positions(own)
    if not 0 <= unit < len(positions):
        return False
    task = candidate.task_type
    if task == TaskType.IDLE_OR_PASS:
        return True
    inventory = _inventory(_get(observation, "private", {}) or {}, unit)
    if task in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT):
        return (
            sum(max(int(value), 0) for value in inventory.values()) > 0
            and (candidate.target_x, candidate.target_y)
            in ((4, 4), (5, 4), (4, 5), (5, 5))
        )
    x, y = candidate.target_x, candidate.target_y
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    raw_tile = _tile_at(own, x, y)
    if raw_tile == "LOCKED":
        return False
    tile = _mapping(raw_tile)
    kind = tile.get("kind")
    day = int(_get(observation, "day", 0) or 0)
    item = candidate.item
    if task == TaskType.CLEAR_OR_REMOVE_TILE:
        return kind == "WEED"
    if task == TaskType.CROP_PRODUCTION:
        if kind == "PLANT":
            return int(tile.get("yield_units", 0)) > 0
        return kind is None and item in CROPS and day < 29
    if task == TaskType.WATER_CROP:
        return kind == "PLANT" and not bool(tile.get("watered_today", False))
    if task == TaskType.APPLY_FERTILIZER:
        return (
            kind == "PLANT"
            and int(tile.get("fertilized_until_day", -1)) < day + 2
        )
    if task == TaskType.BUILD_ANIMAL_STRUCTURE:
        return kind is None and unit == 0 and item in ("GOOSE", "COW")
    animal = tile.get("animal")
    if task == TaskType.ANIMAL_PLACE:
        compatible = (
            ("GOOSE",)
            if kind == "COOP"
            else ("COW", "SHEEP")
            if kind == "PASTURE"
            else ()
        )
        return not animal and item in compatible
    if task == TaskType.ANIMAL_FEED:
        return bool(animal) and not bool(tile.get("fed_today", False))
    if task == TaskType.ANIMAL_CARE:
        return bool(animal) and not bool(tile.get("cared_today", False))
    if task == TaskType.ANIMAL_COLLECT_PRODUCT:
        return bool(animal) and int(tile.get("yield_units", 0)) > 0
    if task == TaskType.ANIMAL_COLLECT_FERTILIZER:
        return bool(animal) and bool(tile.get("fertilizer_available", False))
    if task == TaskType.SHED_PICKUP:
        return (x, y) in ((4, 4), (5, 4), (4, 5), (5, 5)) and item is not None
    return False


def _path_steps(
    start: Sequence[int], target: Sequence[int]
) -> tuple[int, tuple[tuple[int, int], ...]]:
    path = bfs_shortest_path(start, target, board_size=BOARD_SIZE)
    return (len(path) - 1, path) if path else (BOARD_SIZE * BOARD_SIZE, ())


def estimate_task_eta(
    observation: Any,
    own: Any,
    unit: int,
    candidate: DecisionCandidate,
) -> TaskEta:
    """Estimate acquisition + route + execution/wait time for one worker-task edge."""

    positions = _positions(own)
    if unit < 0 or unit >= len(positions):
        return TaskEta(0, 0, 0, BOARD_SIZE * BOARD_SIZE, False, False, (-1, -1), ())
    start = tuple(int(value) for value in positions[unit])
    if candidate.task_type == TaskType.IDLE_OR_PASS:
        return TaskEta(0, 0, 0, 0, True, True, start, (start,))
    private = _get(observation, "private", {}) or {}
    inventory = _inventory(private, unit)
    shed = _mapping(_get(private, "shed", {}))
    item = candidate.item
    target = (candidate.target_x, candidate.target_y)
    needs_item = candidate.task_type in (
        TaskType.ANIMAL_FEED,
        TaskType.ANIMAL_PLACE,
        TaskType.APPLY_FERTILIZER,
        TaskType.SHED_PICKUP,
    )
    waypoints: list[tuple[int, int]] = [start]
    movement = 0
    interactions = 1
    wait = 0
    reachable = True
    immediate_goal = target
    cursor = start

    if needs_item and item and int(inventory.get(item, 0)) <= 0:
        depot = _nearest_shed(start)
        immediate_goal = depot
        if int(shed.get(item, 0)) <= 0:
            reachable = False
        steps, path = _path_steps(cursor, depot)
        reachable &= bool(path)
        movement += steps
        if path:
            waypoints.extend(path[1:])
        interactions += 1  # pickup before the final operation
        cursor = depot

    if candidate.task_type in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT):
        target = _nearest_shed(start)
        immediate_goal = target
        interactions = int(sum(int(value) for value in inventory.values()) > 0)

    if target[0] < 0 or target[1] < 0:
        reachable = False
    else:
        steps, path = _path_steps(cursor, target)
        reachable &= bool(path)
        movement += steps
        if path:
            waypoints.extend(path[1:])

    tile = _mapping(_tile_at(own, *target)) if target[0] >= 0 else {}
    if (
        candidate.task_type == TaskType.CROP_PRODUCTION
        and tile.get("kind") == "PLANT"
        and int(tile.get("yield_units", 0)) <= 0
        and bool(tile.get("watered_today", False))
    ):
        hour = int(_get(observation, "hour", int(_get(observation, "step", 0) or 0) % 24) or 0)
        wait = max(24 - hour, 1)
    total = movement + interactions + wait
    step = int(_get(observation, "step", 0) or 0)
    deadline_feasible = (
        candidate.deadline_step >= 719
        or step + total <= candidate.deadline_step
    )
    return TaskEta(
        movement_steps=movement,
        interaction_steps=interactions,
        wait_steps=wait,
        total_steps=total,
        reachable=reachable,
        deadline_feasible=deadline_feasible,
        immediate_goal=immediate_goal,
        waypoints=tuple(waypoints),
    )


def generate_task_cards(
    observation: Any,
    memory: HierarchicalMemory | None = None,
    *,
    extra_cards: Sequence[TaskCard] = (),
    expert_quota: int = 8,
    learned_quota: int = 24,
) -> EncodedTaskSet:
    """Generate the rule/state pool and merge optional learned/expert cards."""

    memory = HierarchicalMemory() if memory is None else memory
    own, opponent = _canonical_farms(observation)
    positions = _positions(own)
    idle = TaskCard(
        DecisionCandidate(
            TaskType.IDLE_OR_PASS,
            prior=TASK_PRIOR[TaskType.IDLE_OR_PASS],
        ),
        capacity=MAX_UNITS,
    )
    deduplicated: dict[tuple[int, int, int, int, int], tuple[TaskCard, int]] = {
        _task_key(idle): (idle, 0)
    }
    for unit in range(min(len(positions), MAX_UNITS)):
        state = memory.task_states[unit]
        plan = state.task or memory.decision.unit_plans[unit]
        if plan is not None and _plan_relevant(observation, plan, own):
            continued = _strip_owner(
                replace(plan, continuation=True, prior=min(plan.prior + 250.0, 12_000.0)),
                preferred_owner=unit,
                origin=CandidateOrigin.CONTINUATION,
            )
            deduplicated[_task_key(continued)] = (
                continued,
                estimate_task_eta(
                    observation,
                    own,
                    unit,
                    replace(continued.candidate, owner_unit=unit),
                ).total_steps,
            )
        else:
            memory.decision.unit_plans[unit] = None
            if state.task is not None and state.phase not in (
                TaskPhase.COMPLETED,
                TaskPhase.CANCELLED,
                TaskPhase.IDLE,
            ):
                execution_awaiting_classification = bool(
                    state.phase == TaskPhase.EXECUTE
                    and state.updated_step
                    < int(_get(observation, "step", 0) or 0)
                )
                # The compiler needs the executed stage intact so it can either
                # advance a compatible chain or close the completed objective.
                if not execution_awaiting_classification:
                    memory.transition_task(
                        unit,
                        TaskPhase.CANCELLED,
                        observation=observation,
                        termination_reason=TaskTerminationReason.PRECONDITION_INVALID,
                        detail="task precondition changed before verified execution",
                    )

        for candidate in _unit_options(observation, own, unit):
            preferred = (
                unit
                if candidate.task_type
                in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT)
                else -1
            )
            card = _strip_owner(candidate, preferred_owner=preferred)
            key = _task_key(card)
            distance = estimate_task_eta(
                observation, own, unit, candidate
            ).total_steps
            current = deduplicated.get(key)
            if current is None or (
                card.candidate.prior,
                -distance,
            ) > (
                current[0].candidate.prior,
                -current[1],
            ):
                deduplicated[key] = (card, distance)

    origin_priority = {
        CandidateOrigin.RULE: 0,
        CandidateOrigin.LEARNED: 1,
        CandidateOrigin.CONTINUATION: 2,
        CandidateOrigin.EXPERT: 3,
    }
    for raw_card in extra_cards:
        if raw_card.task_type in (TaskType.NONE, TaskType.IDLE_OR_PASS):
            continue
        card = _strip_owner(
            raw_card.candidate,
            preferred_owner=raw_card.preferred_owner,
            continuation=raw_card.candidate.continuation,
            origin=raw_card.origin,
        )
        eligible_units = (
            [card.preferred_owner]
            if 0 <= card.preferred_owner < min(len(positions), MAX_UNITS)
            else list(range(min(len(positions), MAX_UNITS)))
        )
        if card.origin in (CandidateOrigin.LEARNED, CandidateOrigin.EXPERT):
            deployable = False
            for unit in eligible_units:
                owned = replace(card.candidate, owner_unit=unit)
                eta = estimate_task_eta(observation, own, unit, owned)
                resource = task_resource_plan(
                    observation, own, unit, owned
                )
                if (
                    _structural_candidate_legal(
                        observation, own, unit, owned
                    )
                    and resource.feasible
                    and (eta.reachable or resource.required_market_index > 0)
                    and eta.deadline_feasible
                ):
                    deployable = True
                    break
            if not deployable:
                continue
        distances = [
            estimate_task_eta(
                observation,
                own,
                unit,
                replace(card.candidate, owner_unit=unit),
            ).total_steps
            for unit in eligible_units
        ]
        best_distance = min(distances, default=BOARD_SIZE * BOARD_SIZE)
        key = _task_key(card)
        current = deduplicated.get(key)
        if current is None or (
            origin_priority[card.origin],
            card.candidate.prior,
            -best_distance,
        ) > (
            origin_priority[current[0].origin],
            current[0].candidate.prior,
            -current[1],
        ):
            deduplicated[key] = (card, best_distance)

    tasks = _select_task_cards(
        tuple(deduplicated.values()),
        expert_quota=expert_quota,
        learned_quota=learned_quota,
    )
    task_count = len(tasks)
    workload = sum(card.candidate.prior >= 6_000.0 for card in tasks)
    features = np.zeros((MAX_TASK_CARDS, CANDIDATE_FEATURES), dtype=np.float32)
    task_ids = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    item_ids = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    source_ids = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    origin_ids = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    target_xy = np.full((MAX_TASK_CARDS, 2), -1, dtype=np.int64)
    preferred_owner_ids = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    task_mask = np.zeros(MAX_TASK_CARDS, dtype=np.bool_)
    capacity = np.zeros(MAX_TASK_CARDS, dtype=np.int64)
    item_index = {item: index for index, item in enumerate(ITEMS)}
    for index, card in enumerate(tasks):
        features[index] = _candidate_features(
            observation, card.candidate, own, opponent, workload
        )
        task_ids[index] = int(card.task_type)
        item_ids[index] = item_index.get(card.item, -1) + 1
        source_ids[index] = int(card.candidate.source)
        origin_ids[index] = int(card.origin)
        target_xy[index] = card.target
        preferred_owner_ids[index] = card.preferred_owner + 1
        task_mask[index] = True
        capacity[index] = card.capacity

    pair_features = np.zeros(
        (MAX_UNITS, MAX_TASK_CARDS, PAIR_FEATURES), dtype=np.float32
    )
    eta_steps = np.full(
        (MAX_UNITS, MAX_TASK_CARDS), BOARD_SIZE * BOARD_SIZE, dtype=np.int16
    )
    pair_mask = np.zeros((MAX_UNITS, MAX_TASK_CARDS), dtype=np.bool_)
    required_market_indices = np.zeros(
        (MAX_UNITS, MAX_TASK_CARDS), dtype=np.int64
    )
    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    market_budget = MarketBudget.from_observation(observation)
    step = int(_get(observation, "step", 0) or 0)
    for unit, position in enumerate(positions[:MAX_UNITS]):
        inventory = _inventory(private, unit)
        carried = sum(max(int(value), 0) for value in inventory.values())
        plan = memory.task_states[unit].task or memory.decision.unit_plans[unit]
        for task_index, card in enumerate(tasks):
            candidate = card.candidate
            preferred_ok = card.preferred_owner in (-1, unit)
            target_valid = candidate.target_x >= 0 and candidate.target_y >= 0
            eta = estimate_task_eta(
                observation,
                own,
                unit,
                replace(candidate, owner_unit=unit),
            )
            eta_steps[unit, task_index] = min(eta.total_steps, np.iinfo(np.int16).max)
            same_plan = bool(
                plan is not None
                and int(plan.task_type) == int(candidate.task_type)
                and plan.target_x == candidate.target_x
                and plan.target_y == candidate.target_y
                and plan.item_id == candidate.item_id
            )
            resource = task_resource_plan(
                observation,
                own,
                unit,
                replace(candidate, owner_unit=unit),
                budget=market_budget,
            )
            required_market_indices[unit, task_index] = (
                resource.required_market_index
            )
            structurally_legal = _structural_candidate_legal(
                observation,
                own,
                unit,
                replace(candidate, owner_unit=unit),
            )
            pair_mask[unit, task_index] = (
                preferred_ok
                and (structurally_legal or same_plan or card.candidate.continuation)
                and resource.feasible
                and (eta.reachable or resource.required_market_index > 0)
                and eta.deadline_feasible
            )
            item = card.item
            needs_item = candidate.task_type in (
                TaskType.ANIMAL_FEED,
                TaskType.ANIMAL_PLACE,
                TaskType.APPLY_FERTILIZER,
                TaskType.SHED_PICKUP,
            )
            pair_features[unit, task_index] = np.asarray(
                (
                    1.0,
                    min(eta.total_steps / 48.0, 2.0),
                    float(target_valid and tuple(position) == card.target),
                    float(same_plan),
                    float(card.preferred_owner == unit),
                    float(item is not None and int(inventory.get(item, 0)) > 0),
                    float(item is not None and int(shed.get(item, 0)) > 0),
                    min(carried / 100.0, 2.0),
                    np.clip(
                        (candidate.deadline_step - step - eta.total_steps) / 48.0,
                        -2.0,
                        2.0,
                    ),
                    float(unit == 0),
                    float(needs_item),
                    float(card.target in ((4, 4), (5, 4), (4, 5), (5, 5))),
                ),
                dtype=np.float32,
            )
    return EncodedTaskSet(
        tasks=tuple(tasks),
        features=features,
        task_ids=task_ids,
        item_ids=item_ids,
        source_ids=source_ids,
        origin_ids=origin_ids,
        target_xy=target_xy,
        preferred_owner_ids=preferred_owner_ids,
        task_mask=task_mask,
        pair_features=pair_features,
        eta_steps=eta_steps,
        pair_mask=pair_mask,
        capacity=capacity,
        required_market_indices=required_market_indices,
    )


@dataclass
class MarketBudget:
    money: float
    shed: dict[str, int]
    seeds: dict[str, int]
    hands: int
    hires: int
    unlocked: int
    prices: dict[str, float]
    market_inventory: dict[str, int]

    @classmethod
    def from_observation(cls, observation: Any) -> "MarketBudget":
        own, _ = _canonical_farms(observation)
        private = _get(observation, "private", {}) or {}
        return cls(
            money=float(_get(own, "money", 0.0)),
            shed={item: int(_mapping(_get(private, "shed", {})).get(item, 0)) for item in ITEMS},
            seeds={crop: int(_mapping(_get(private, "seeds", {})).get(crop, 0)) for crop in CROPS},
            hands=len(_get(own, "hands", []) or []),
            hires=int(_get(own, "hires_today", 0) or 0),
            unlocked=len(_get(own, "unlocked_quadrants", []) or []),
            prices={
                item: float(
                    _mapping(_get(_get(observation, "market", {}) or {}, "prices", {})).get(item, 0)
                )
                for item in PRODUCTS
            },
            market_inventory={
                item: int(
                    _mapping(_get(_get(observation, "market", {}) or {}, "inventory", {})).get(item, 10_000)
                )
                for item in PRODUCTS
            },
        )

    @property
    def room(self) -> int:
        return max(0, 100 - sum(max(value, 0) for value in self.shed.values()))

    def action_spend(self, action_index: int) -> float:
        """Return the cash consumed by a full primitive order at current quotes."""

        operation, item, quantity = MARKET_ACTIONS[int(action_index)]
        if operation == "HIRE":
            return float(_fib(self.hires))
        if operation == "BUY_LAND" and 1 <= self.unlocked <= 3:
            return float(_LAND_PRICES[self.unlocked - 1])
        if operation == "BUY_SEED" and item in CROPS:
            return float(_SEED_COST[item] * quantity)
        if operation == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
            return float(
                sum(
                    _market_quote(item, self.market_inventory[item] - unit - 1)
                    for unit in range(quantity)
                )
            )
        if operation == "BUY_ANIMAL" and item in ANIMALS:
            return float(_ANIMAL_COST[item])
        return 0.0

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(len(MARKET_ACTIONS), dtype=np.bool_)
        mask[0] = True
        if self.hands < MAX_UNITS - 1 and self.money >= _fib(self.hires):
            mask[MARKET_INDEX[("HIRE", None, 1)]] = True
        if 1 <= self.unlocked <= 3 and self.money >= _LAND_PRICES[self.unlocked - 1]:
            mask[MARKET_INDEX[("BUY_LAND", None, 1)]] = True
        for crop in CROPS:
            for quantity in (1, 4, 8, 16):
                if self.money >= _SEED_COST[crop] * quantity:
                    mask[MARKET_INDEX[("BUY_SEED", crop, quantity)]] = True
        for item in ("WHEAT", "FERTILIZER"):
            for quantity in (1, 4, 8, 16):
                inventory = self.market_inventory[item]
                cost = sum(
                    _market_quote(item, inventory - unit - 1)
                    for unit in range(quantity)
                )
                if self.room >= quantity and self.money >= cost:
                    mask[MARKET_INDEX[("BUY_PRODUCT", item, quantity)]] = True
        for animal in ANIMALS:
            if self.room >= 1 and self.money >= _ANIMAL_COST[animal]:
                mask[MARKET_INDEX[("BUY_ANIMAL", animal, 1)]] = True
        for item in PRODUCTS:
            if self.shed[item] > 0:
                mask[MARKET_INDEX[("SELL", item, 100)]] = True
        return mask

    def apply(self, action_index: int) -> None:
        operation, item, quantity = MARKET_ACTIONS[int(action_index)]
        if operation == "NONE":
            return
        if operation == "HIRE":
            self.money -= _fib(self.hires)
            self.hires += 1
            self.hands += 1
        elif operation == "BUY_LAND":
            self.money -= _LAND_PRICES[self.unlocked - 1]
            self.unlocked += 1
        elif operation == "BUY_SEED" and item in CROPS:
            self.money -= _SEED_COST[item] * quantity
            self.seeds[item] += quantity
        elif operation == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
            for _ in range(quantity):
                quote = _market_quote(item, self.market_inventory[item] - 1)
                if self.room <= 0 or self.money < quote:
                    break
                self.money -= quote
                self.shed[item] += 1
                self.market_inventory[item] -= 1
            self.prices[item] = _market_quote(item, self.market_inventory[item])
        elif operation == "BUY_ANIMAL" and item in ANIMALS:
            self.money -= _ANIMAL_COST[item]
            self.shed[item] += 1
        elif operation == "SELL" and item in PRODUCTS:
            for _ in range(quantity):
                if self.shed[item] <= 0:
                    break
                quote = _market_quote(item, self.market_inventory[item])
                self.shed[item] -= 1
                self.money += quote
                if quote > 1:
                    self.market_inventory[item] += 1
            self.prices[item] = _market_quote(item, self.market_inventory[item])


@dataclass
class PlannedMarketBudget:
    """A physical market ledger guarded by a fixed daily :class:`BudgetPlan`."""

    ledger: MarketBudget
    plan: BudgetPlan
    spent_by_category: np.ndarray = field(
        default_factory=lambda: np.zeros(len(BUDGET_CATEGORIES), dtype=np.float64)
    )
    emergency_spent: float = 0.0

    @classmethod
    def from_observation(
        cls,
        observation: Any,
        plan: BudgetPlan,
        *,
        spent_by_category: Sequence[float] | None = None,
        emergency_spent: float = 0.0,
    ) -> "PlannedMarketBudget":
        spent = np.zeros(len(BUDGET_CATEGORIES), dtype=np.float64)
        if spent_by_category is not None:
            spent = np.asarray(spent_by_category, dtype=np.float64).copy()
            if spent.shape != (len(BUDGET_CATEGORIES),):
                raise ValueError("spent budget categories have an incompatible shape")
        return cls(
            ledger=MarketBudget.from_observation(observation),
            plan=plan,
            spent_by_category=spent,
            emergency_spent=max(float(emergency_spent), 0.0),
        )

    @property
    def money(self) -> float:
        return self.ledger.money

    @property
    def emergency_remaining(self) -> float:
        return max(self.plan.emergency_limit - self.emergency_spent, 0.0)

    def _budget_violation(self, action_index: int) -> float:
        category = market_budget_category(action_index)
        if category is None:
            return 0.0
        cost = self.ledger.action_spend(action_index)
        category_remaining = max(
            self.plan.category_limits[int(category)]
            - self.spent_by_category[int(category)],
            0.0,
        )
        cash_above_floor = max(self.ledger.money - self.plan.cash_floor, 0.0)
        return max(cost - category_remaining, cost - cash_above_floor, 0.0)

    def legal_mask(self, *, emergency: bool = False) -> np.ndarray:
        mask = self.ledger.legal_mask()
        for action_index in np.flatnonzero(mask):
            if market_budget_category(int(action_index)) is None:
                continue
            violation = self._budget_violation(int(action_index))
            allowed = violation <= 1e-6 or (
                emergency and violation <= self.emergency_remaining + 1e-6
            )
            if not allowed:
                mask[int(action_index)] = False
        return mask

    def apply(self, action_index: int, *, emergency: bool = False) -> None:
        action_index = int(action_index)
        if not bool(self.legal_mask(emergency=emergency)[action_index]):
            raise ValueError("market order violates the daily budget plan")
        category = market_budget_category(action_index)
        violation = self._budget_violation(action_index)
        before = self.ledger.money
        self.ledger.apply(action_index)
        spend = max(before - self.ledger.money, 0.0)
        if category is not None:
            self.spent_by_category[int(category)] += spend
            if emergency:
                self.emergency_spent += min(violation, spend)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "spent_by_category": {
                name: float(self.spent_by_category[index])
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "emergency_spent": self.emergency_spent,
            "emergency_remaining": self.emergency_remaining,
            "money": self.ledger.money,
        }


def task_resource_plan(
    observation: Any,
    own: Any,
    unit: int,
    candidate: DecisionCandidate,
    *,
    budget: MarketBudget | None = None,
) -> TaskResourcePlan:
    """Resolve an immediate resource or one legal market acquisition order."""

    private = _get(observation, "private", {}) or {}
    inventory = _inventory(private, unit)
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))
    item = candidate.item
    required: tuple[str, str, int] | None = None
    available = True
    tile = (
        _mapping(_tile_at(own, candidate.target_x, candidate.target_y))
        if 0 <= candidate.target_x < BOARD_SIZE
        and 0 <= candidate.target_y < BOARD_SIZE
        else {}
    )
    if candidate.task_type == TaskType.CROP_PRODUCTION and tile.get("kind") is None:
        available = bool(item in CROPS and int(seeds.get(item, 0)) > 0)
        if item in CROPS:
            required = ("BUY_SEED", item, 1)
    elif candidate.task_type in (
        TaskType.ANIMAL_FEED,
        TaskType.APPLY_FERTILIZER,
        TaskType.ANIMAL_PLACE,
        TaskType.SHED_PICKUP,
    ):
        available = bool(
            item is not None
            and (
                int(inventory.get(item, 0)) > 0
                or int(shed.get(item, 0)) > 0
            )
        )
        if item in ANIMALS:
            required = ("BUY_ANIMAL", item, 1)
        elif item in ("WHEAT", "FERTILIZER"):
            required = ("BUY_PRODUCT", item, 1)
    if available or required is None:
        return TaskResourcePlan(
            feasible=available or required is None,
            available=available,
            detail="resource already available" if available else "no acquisition action",
        )
    action_index = MARKET_INDEX.get(required, 0)
    budget = MarketBudget.from_observation(observation) if budget is None else budget
    market_feasible = bool(action_index and budget.legal_mask()[action_index])
    return TaskResourcePlan(
        feasible=market_feasible,
        available=False,
        required_market_index=action_index if market_feasible else 0,
        detail=(
            f"requires market order {required}"
            if market_feasible
            else f"market cannot satisfy {required}"
        ),
    )
def market_prior_features(observation: Any) -> np.ndarray:
    """Rule-derived, normalized priors for every primitive market candidate."""

    step = int(_get(observation, "step", 0) or 0)
    terminal = step >= 718
    values = np.zeros(len(MARKET_ACTIONS), dtype=np.float32)
    for index, (operation, _item, _quantity) in enumerate(MARKET_ACTIONS):
        prior = {
            "NONE": TASK_PRIOR[TaskType.IDLE_OR_PASS],
            "HIRE": TASK_PRIOR[TaskType.HIRE_WORKER],
            "BUY_LAND": TASK_PRIOR[TaskType.BUY_LAND],
            "BUY_SEED": TASK_PRIOR[TaskType.CROP_PRODUCTION],
            "BUY_PRODUCT": TASK_PRIOR[TaskType.BUY_PRODUCT],
            "BUY_ANIMAL": TASK_PRIOR[TaskType.ANIMAL_PURCHASE],
            "SELL": TASK_PRIOR[
                TaskType.TERMINAL_LIQUIDATION if terminal else TaskType.SELL_INVENTORY
            ],
        }[operation]
        values[index] = min(prior / 12_000.0, 1.25)
    return values


def _failure_event(
    observation: Any,
    unit: int,
    candidate: DecisionCandidate,
    reason: ExecutorFailureReason,
    action: Sequence[Any],
    detail: str,
) -> ExecutorFailure:
    return ExecutorFailure(
        step=int(_get(observation, "step", 0) or 0),
        day=int(_get(observation, "day", 0) or 0),
        unit=unit,
        task_type=candidate.task_type.name,
        target=(candidate.target_x, candidate.target_y),
        reason=reason.value,
        action=tuple(action),
        detail=detail,
    )


def _diagnose_executor_pass(
    observation: Any,
    own: Any,
    candidate: DecisionCandidate,
    position: Sequence[int],
    action: Sequence[Any],
    *,
    target_was_reserved: bool,
) -> tuple[ExecutorFailureReason, str] | None:
    """Classify definite non-idle executor failures without logging valid waits."""

    if not action or str(action[0]) != "PASS":
        return None
    task = candidate.task_type
    if task == TaskType.IDLE_OR_PASS:
        return None
    private = _get(observation, "private", {}) or {}
    inventory = _inventory(private, candidate.owner_unit)
    shed = _mapping(_get(private, "shed", {}))
    item = candidate.item
    needs_item = task in (
        TaskType.ANIMAL_FEED,
        TaskType.ANIMAL_PLACE,
        TaskType.APPLY_FERTILIZER,
        TaskType.SHED_PICKUP,
    )
    if (
        needs_item
        and item
        and int(inventory.get(item, 0)) <= 0
        and int(shed.get(item, 0)) <= 0
    ):
        return (
            ExecutorFailureReason.MISSING_RESOURCE,
            f"required item {item} absent from unit inventory and shed",
        )
    if needs_item and item and int(inventory.get(item, 0)) <= 0:
        depot = _nearest_shed(position)
        if tuple(position) != depot:
            return (
                ExecutorFailureReason.PATH_BLOCKED,
                f"no legal first move toward shed {depot}",
            )
    if task in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT):
        depot = _nearest_shed(position)
        if tuple(position) == depot and not sum(int(value) for value in inventory.values()):
            return None
        if tuple(position) != depot:
            return (
                ExecutorFailureReason.PATH_BLOCKED,
                f"no legal first move toward shed {depot}",
            )

    target = (candidate.target_x, candidate.target_y)
    if tuple(position) != target:
        return (
            ExecutorFailureReason.PATH_BLOCKED,
            f"no legal first move from {tuple(position)} toward {target}",
        )
    if target_was_reserved:
        return (
            ExecutorFailureReason.TARGET_RESERVED,
            f"another unit reserved target {target} in the same step",
        )
    tile = _mapping(_tile_at(own, *target))
    if (
        task == TaskType.CROP_PRODUCTION
        and tile.get("kind") == "PLANT"
        and int(tile.get("yield_units", 0)) <= 0
        and bool(tile.get("watered_today", False))
    ):
        return None
    return (
        ExecutorFailureReason.PRECONDITION_CHANGED,
        "task reached its target but no valid primitive operation remained",
    )


def compile_hierarchical_action(
    observation: Any,
    encoded: EncodedTaskSet,
    assignment: Sequence[int],
    market_indices: Sequence[int],
    memory: HierarchicalMemory | None = None,
) -> dict[str, Any]:
    """Compile a task matching and ordered market sequence into one action."""

    memory = HierarchicalMemory() if memory is None else memory
    own, _ = _canonical_farms(observation)
    positions = _positions(own)
    market: list[list[Any]] = []
    for raw_index in market_indices[:MAX_MARKET_ORDERS]:
        index = int(raw_index)
        if not 0 <= index < len(MARKET_ACTIONS):
            candidate = DecisionCandidate(TaskType.NONE)
            memory.record_executor_failure(
                _failure_event(
                    observation,
                    -1,
                    candidate,
                    ExecutorFailureReason.INVALID_MARKET_INDEX,
                    ("PASS",),
                    f"market index {index} is outside [0, {len(MARKET_ACTIONS)})",
                )
            )
            break
        operation, item, quantity = MARKET_ACTIONS[index]
        if operation == "NONE":
            break
        market.append(
            [operation] if item is None else [operation, item, quantity]
        )

    occupied = {tuple(int(value) for value in position) for position in positions}
    candidates: dict[int, DecisionCandidate] = {}
    etas: dict[int, TaskEta] = {}
    priorities: dict[int, tuple[int, ...]] = {}
    for unit in range(min(len(positions), MAX_UNITS)):
        task_index = int(assignment[unit]) if unit < len(assignment) else -1
        if not 0 <= task_index < len(encoded.tasks):
            candidate = DecisionCandidate(TaskType.IDLE_OR_PASS, owner_unit=unit)
            memory.record_executor_failure(
                _failure_event(
                    observation,
                    unit,
                    candidate,
                    ExecutorFailureReason.INVALID_TASK_INDEX,
                    ("PASS",),
                    f"task index {task_index} is outside [0, {len(encoded.tasks)})",
                )
            )
        elif not bool(encoded.pair_mask[unit, task_index]):
            rejected = encoded.tasks[task_index].candidate
            memory.record_executor_failure(
                _failure_event(
                    observation,
                    unit,
                    replace(rejected, owner_unit=unit),
                    ExecutorFailureReason.PAIR_MASKED,
                    ("PASS",),
                    "selected worker-task edge is masked by the candidate generator",
                )
            )
            candidate = DecisionCandidate(TaskType.IDLE_OR_PASS, owner_unit=unit)
        else:
            card = encoded.tasks[task_index]
            eta_steps = int(encoded.eta_steps[unit, task_index])
            candidate = replace(
                card.candidate,
                owner_unit=unit,
                path_steps=eta_steps,
            )
        eta = estimate_task_eta(observation, own, unit, candidate)
        candidates[unit] = candidate
        etas[unit] = eta
        priorities[unit] = (
            0 if candidate.mandatory else 1,
            candidate.deadline_step,
            eta.total_steps,
        )

    starts = {
        unit: tuple(int(value) for value in positions[unit])
        for unit in candidates
    }
    goals = {unit: etas[unit].immediate_goal for unit in candidates}
    routes, _ = plan_prioritized_routes(
        starts,
        goals,
        priorities,
        board_size=BOARD_SIZE,
        horizon=64,
    )
    reserved_moves: set[tuple[int, int]] = set()
    reserved_targets: set[tuple[int, int]] = set()
    unit_actions: list[list[Any]] = []
    for unit in range(min(len(positions), MAX_UNITS)):
        candidate = candidates[unit]
        eta = etas[unit]
        route = routes.get(unit, (starts[unit],))
        target_was_reserved = (
            tuple(int(value) for value in positions[unit])
            == (candidate.target_x, candidate.target_y)
            and (candidate.target_x, candidate.target_y) in reserved_targets
        )
        if tuple(int(value) for value in positions[unit]) != eta.immediate_goal:
            action = movement_for_path(route)
        else:
            action = _compile_unit_candidate(
                observation,
                own,
                candidate,
                occupied,
                reserved_moves,
                reserved_targets,
            )
        reservation_wait = bool(
            len(route) >= 2
            and route[0] == route[1]
            and route[0] != eta.immediate_goal
        )
        diagnosed = (
            None
            if reservation_wait
            else _diagnose_executor_pass(
                observation,
                own,
                candidate,
                positions[unit],
                action,
                target_was_reserved=target_was_reserved,
            )
        )
        if diagnosed is not None:
            reason, detail = diagnosed
            memory.record_executor_failure(
                _failure_event(
                    observation, unit, candidate, reason, action, detail
                )
            )
            phase = TaskPhase.BLOCKED
        elif candidate.task_type == TaskType.IDLE_OR_PASS:
            phase = TaskPhase.IDLE
        elif str(action[0]) in ("NORTH", "SOUTH", "EAST", "WEST"):
            phase = (
                TaskPhase.ACQUIRE_RESOURCE
                if eta.immediate_goal != (candidate.target_x, candidate.target_y)
                else TaskPhase.NAVIGATE
            )
        elif str(action[0]) == "PASS":
            phase = TaskPhase.WAITING
        else:
            phase = TaskPhase.EXECUTE
        static_immediate_path = bfs_shortest_path(
            starts[unit], eta.immediate_goal, board_size=BOARD_SIZE
        )
        static_immediate_steps = max(len(static_immediate_path) - 1, 0)
        reserved_steps = max(len(route) - 1, 0)
        adjusted_eta = eta.total_steps + max(
            reserved_steps - static_immediate_steps, 0
        )
        previous_state = memory.task_states[unit]
        previous_task = previous_state.task
        same_task = _same_task(previous_task, candidate)
        same_chain = same_task_chain(previous_task, candidate)
        continued_chain = False
        if (
            previous_task is not None
            and previous_task.task_type != TaskType.IDLE_OR_PASS
            and not same_task
            and previous_state.phase
            not in (TaskPhase.COMPLETED, TaskPhase.CANCELLED, TaskPhase.IDLE)
        ):
            previous_legal = _structural_candidate_legal(
                observation, own, unit, previous_task
            )
            prior_execution_observed = bool(
                previous_state.phase == TaskPhase.EXECUTE
                and previous_state.updated_step
                < int(_get(observation, "step", 0) or 0)
                and not previous_legal
            )
            if prior_execution_observed:
                continued_chain = same_chain
                memory.transition_task(
                    unit,
                    TaskPhase.COMPLETED,
                    observation=observation,
                    termination_reason=(
                        TaskTerminationReason.STAGE_ADVANCED
                        if same_chain
                        else TaskTerminationReason.GOAL_SATISFIED
                    ),
                    detail=(
                        "executed stage advanced within persistent chain"
                        if same_chain
                        else "executed task precondition is now satisfied"
                    ),
                )
            else:
                memory.transition_task(
                    unit,
                    TaskPhase.CANCELLED,
                    observation=observation,
                    termination_reason=(
                        TaskTerminationReason.POLICY_IDLED
                        if candidate.task_type == TaskType.IDLE_OR_PASS
                        else TaskTerminationReason.ASSIGNMENT_REPLACED
                    ),
                    detail="assignment replaced before verified completion",
                )
        if candidate.task_type != TaskType.IDLE_OR_PASS and not same_task:
            memory.transition_task(
                unit,
                TaskPhase.ASSIGNED,
                observation=observation,
                candidate=candidate,
                eta_steps=adjusted_eta,
                route=route,
                continue_chain=continued_chain,
                detail=(
                    "next stage in persistent task chain"
                    if continued_chain
                    else "new task assignment"
                ),
            )
        memory.transition_task(
            unit,
            phase,
            observation=observation,
            candidate=candidate,
            eta_steps=adjusted_eta,
            route=route,
            detail=(
                diagnosed[1]
                if diagnosed is not None
                else "space-time reservation wait"
                if reservation_wait
                else ""
            ),
        )
        if diagnosed is not None:
            memory.task_states[unit].failure_count += 1
        memory.decision.unit_plans[unit] = (
            None
            if candidate.task_type == TaskType.IDLE_OR_PASS
            else replace(candidate, continuation=False)
        )
        unit_actions.append(action)
    memory.mode_age += 1
    compiled = {
        "farmer": unit_actions[0] if unit_actions else ["PASS"],
        "hands": unit_actions[1:],
        "market": market,
    }
    memory.record_internal_trace(observation, compiled)
    return compiled


def strategy_mode_from_name(name: str) -> int:
    """Stable initial family label for expert-conditioned behavior cloning."""

    lowered = name.lower()
    if any(token in lowered for token in ("animal", "cow", "sheep", "goose")):
        return MODE_INDEX["ANIMAL_CASHFLOW"]
    if any(token in lowered for token in ("premium", "strawberry", "melon")):
        return MODE_INDEX["PREMIUM_MARKET"]
    if any(token in lowered for token in ("mirror", "clone")):
        return MODE_INDEX["ANTI_MIRROR"]
    if any(token in lowered for token in ("terminal", "liquid")):
        return MODE_INDEX["TERMINAL_LIQUIDATION"]
    if any(token in lowered for token in ("safe", "cashflow", "conservative")):
        return MODE_INDEX["CONSERVATIVE"]
    if any(token in lowered for token in ("counter", "adaptive", "fusion")):
        return MODE_INDEX["OPPONENT_COUNTER"]
    if any(token in lowered for token in ("crop", "route", "land")):
        return MODE_INDEX["CROP_EXPANSION"]
    return MODE_INDEX["BALANCED"]
