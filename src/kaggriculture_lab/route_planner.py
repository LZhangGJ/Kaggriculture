"""Parameterized route plans and a small resource-constrained farm scheduler.

The route learner should not emit 719 low-level actions.  A :class:`PlanSpec`
describes strategic intent, while the scheduler answers whether the farmer and
hired hands can execute a concrete task graph before its deadlines.  The module
is deliberately independent from the Kaggle interpreter so it can be used by
scripted agents, offline data generation, and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence


Position = tuple[int, int]


@dataclass(frozen=True)
class PlanSpec:
    """Compact numeric description of one strategic route candidate.

    The fields are intentionally few.  New route families should first be
    expressed by changing these values instead of adding a new 719-step script.
    They are also the default feature vector consumed by the route-value model.
    """

    name: str
    cow_target: int = 0
    sheep_target: int = 0
    goose_target: int = 0
    land_target: int = 25
    max_hands: int = 0
    cash_reserve: int = 0
    first_hire_step: int = 0
    sale_offset: int = 0
    liquidation_step: int = 672

    def __post_init__(self) -> None:
        nonnegative = {
            "cow_target": self.cow_target,
            "sheep_target": self.sheep_target,
            "goose_target": self.goose_target,
            "land_target": self.land_target,
            "max_hands": self.max_hands,
            "cash_reserve": self.cash_reserve,
            "first_hire_step": self.first_hire_step,
            "liquidation_step": self.liquidation_step,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.land_target > 100:
            raise ValueError("land_target cannot exceed the 10x10 board")
        if self.max_hands > 32:
            raise ValueError("max_hands cannot exceed the official hand bound")
        if self.liquidation_step > 719:
            raise ValueError("liquidation_step must be within the season")

    def normalized_features(self) -> tuple[float, ...]:
        """Return a stable feature vector for route-value training."""
        return (
            self.cow_target / 16.0,
            self.sheep_target / 16.0,
            self.goose_target / 16.0,
            self.land_target / 100.0,
            self.max_hands / 16.0,
            self.cash_reserve / 20_000.0,
            self.first_hire_step / 719.0,
            self.sale_offset / 24.0,
            self.liquidation_step / 719.0,
        )

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, object] | None) -> "PlanSpec":
        data = dict(values or {})
        data.pop("name", None)
        known = {field_.name for field_ in cls.__dataclass_fields__.values()} - {"name"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown PlanSpec fields: {unknown}")
        return cls(name=name, **{key: int(value) for key, value in data.items()})


PLAN_FEATURE_NAMES = (
    "cow_target",
    "sheep_target",
    "goose_target",
    "land_target",
    "max_hands",
    "cash_reserve",
    "first_hire_step",
    "sale_offset",
    "liquidation_step",
)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    position: Position
    available_step: int = 0
    role: str = "hand"

    def __post_init__(self) -> None:
        if self.available_step < 0:
            raise ValueError("available_step must be non-negative")


@dataclass(frozen=True)
class FarmTask:
    """One schedulable farm job.

    ``value`` is an offline planning utility, not the competition reward.  A
    mandatory task should represent maintenance or a prerequisite whose failure
    makes the route invalid (feeding, deadline-critical harvest, liquidation,
    and similar operations).
    """

    task_id: str
    position: Position
    release_step: int
    deadline_step: int
    duration: int = 1
    value: float = 0.0
    mandatory: bool = False
    predecessors: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ("farmer", "hand")

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if self.release_step < 0 or self.deadline_step < self.release_step:
            raise ValueError(f"Invalid task window for {self.task_id}")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not self.allowed_roles:
            raise ValueError("allowed_roles cannot be empty")


@dataclass(frozen=True)
class ScheduleAssignment:
    task_id: str
    worker_id: str
    start_step: int
    finish_step: int
    travel_steps: int
    tardiness: int


@dataclass(frozen=True)
class SchedulerConfig:
    horizon_end: int = 719
    beam_width: int = 64
    mandatory_miss_penalty: float = 1_000_000.0
    tardiness_penalty: float = 1_000.0
    travel_penalty: float = 0.25
    makespan_penalty: float = 0.01

    def __post_init__(self) -> None:
        if self.horizon_end <= 0:
            raise ValueError("horizon_end must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")


@dataclass(frozen=True)
class ScheduleResult:
    assignments: tuple[ScheduleAssignment, ...]
    unscheduled: tuple[str, ...]
    missed_mandatory: tuple[str, ...]
    total_value: float
    total_travel: int
    total_tardiness: int
    makespan: int
    utility: float

    @property
    def feasible(self) -> bool:
        return not self.missed_mandatory


@dataclass(frozen=True)
class HireDecision:
    should_hire: bool
    marginal_utility: float
    hire_cost: int
    without_hire: ScheduleResult
    with_hire: ScheduleResult


@dataclass(frozen=True)
class _BeamState:
    completed: frozenset[str]
    worker_positions: tuple[Position, ...]
    worker_available: tuple[int, ...]
    assignments: tuple[ScheduleAssignment, ...] = ()
    total_value: float = 0.0
    total_travel: int = 0
    total_tardiness: int = 0


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _validate_tasks(tasks: Sequence[FarmTask]) -> dict[str, FarmTask]:
    task_map: dict[str, FarmTask] = {}
    for task in tasks:
        if task.task_id in task_map:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        task_map[task.task_id] = task
    for task in tasks:
        missing = sorted(set(task.predecessors) - set(task_map))
        if missing:
            raise ValueError(f"Task {task.task_id} has missing predecessors: {missing}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("Task graph contains a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for predecessor in task_map[task_id].predecessors:
            visit(predecessor)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_map:
        visit(task_id)
    return task_map


def _partial_rank(state: _BeamState) -> tuple[float, ...]:
    """Lower is better while the beam is still expanding."""
    makespan = max(state.worker_available, default=0)
    return (
        state.total_tardiness,
        -state.total_value,
        state.total_travel,
        makespan,
    )


def _deduplicate(states: Iterable[_BeamState], beam_width: int) -> list[_BeamState]:
    best: dict[tuple[object, ...], _BeamState] = {}
    for state in states:
        key = (
            state.completed,
            state.worker_positions,
            state.worker_available,
        )
        previous = best.get(key)
        if previous is None or _partial_rank(state) < _partial_rank(previous):
            best[key] = state
    return sorted(best.values(), key=_partial_rank)[:beam_width]


def _finish_result(
    state: _BeamState,
    task_map: Mapping[str, FarmTask],
    config: SchedulerConfig,
) -> ScheduleResult:
    unscheduled_ids = tuple(sorted(set(task_map) - set(state.completed)))
    missed = tuple(task_id for task_id in unscheduled_ids if task_map[task_id].mandatory)
    makespan = max(state.worker_available, default=0)
    utility = (
        state.total_value
        - len(missed) * config.mandatory_miss_penalty
        - state.total_tardiness * config.tardiness_penalty
        - state.total_travel * config.travel_penalty
        - makespan * config.makespan_penalty
    )
    return ScheduleResult(
        assignments=state.assignments,
        unscheduled=unscheduled_ids,
        missed_mandatory=missed,
        total_value=state.total_value,
        total_travel=state.total_travel,
        total_tardiness=state.total_tardiness,
        makespan=makespan,
        utility=utility,
    )


def schedule_tasks(
    tasks: Sequence[FarmTask],
    workers: Sequence[WorkerSpec],
    *,
    config: SchedulerConfig | None = None,
) -> ScheduleResult:
    """Schedule a small task DAG with beam search.

    The scheduler is designed for route-level lookahead, where tens of meaningful
    tasks are present in a 24--96 step window.  It deliberately avoids solving
    every movement action in the entire season at once.
    """

    cfg = config or SchedulerConfig()
    task_map = _validate_tasks(tasks)
    if not workers:
        empty = _BeamState(frozenset(), (), ())
        return _finish_result(empty, task_map, cfg)

    beam = [
        _BeamState(
            completed=frozenset(),
            worker_positions=tuple(worker.position for worker in workers),
            worker_available=tuple(worker.available_step for worker in workers),
        )
    ]

    for _ in range(len(tasks)):
        expanded: list[_BeamState] = []
        any_expansion = False
        for state in beam:
            ready = [
                task
                for task in tasks
                if task.task_id not in state.completed
                and set(task.predecessors).issubset(state.completed)
            ]
            for task in ready:
                for worker_index, worker in enumerate(workers):
                    if worker.role not in task.allowed_roles:
                        continue
                    travel = manhattan(state.worker_positions[worker_index], task.position)
                    start = max(
                        state.worker_available[worker_index] + travel,
                        task.release_step,
                    )
                    finish = start + task.duration
                    if finish > cfg.horizon_end:
                        continue
                    tardiness = max(0, finish - task.deadline_step)
                    positions = list(state.worker_positions)
                    availability = list(state.worker_available)
                    positions[worker_index] = task.position
                    availability[worker_index] = finish
                    assignment = ScheduleAssignment(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        start_step=start,
                        finish_step=finish,
                        travel_steps=travel,
                        tardiness=tardiness,
                    )
                    expanded.append(
                        _BeamState(
                            completed=state.completed | {task.task_id},
                            worker_positions=tuple(positions),
                            worker_available=tuple(availability),
                            assignments=state.assignments + (assignment,),
                            total_value=state.total_value + task.value,
                            total_travel=state.total_travel + travel,
                            total_tardiness=state.total_tardiness + tardiness,
                        )
                    )
                    any_expansion = True
        if not any_expansion:
            break
        beam = _deduplicate([*beam, *expanded], cfg.beam_width)

    results = [_finish_result(state, task_map, cfg) for state in beam]
    return max(
        results,
        key=lambda result: (
            result.feasible,
            result.utility,
            -result.total_tardiness,
            -result.total_travel,
            -result.makespan,
        ),
    )


def fibonacci_hire_cost(hires_today: int, multiplier: int = 1) -> int:
    if hires_today < 0:
        raise ValueError("hires_today must be non-negative")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    left, right = 1, 1
    for _ in range(hires_today):
        left, right = right, left + right
    return left * multiplier


def evaluate_one_hire(
    tasks: Sequence[FarmTask],
    workers: Sequence[WorkerSpec],
    *,
    hires_today: int,
    hire_step: int,
    spawn: Position = (4, 4),
    multiplier: int = 1,
    config: SchedulerConfig | None = None,
    minimum_margin: float = 0.0,
) -> HireDecision:
    """Compare the best schedule with and without one additional hand."""

    cfg = config or SchedulerConfig()
    without = schedule_tasks(tasks, workers, config=cfg)
    hire_cost = fibonacci_hire_cost(hires_today, multiplier)
    worker_ids = {worker.worker_id for worker in workers}
    suffix = 1
    while f"hire_{suffix}" in worker_ids:
        suffix += 1
    hired = WorkerSpec(
        worker_id=f"hire_{suffix}",
        position=spawn,
        available_step=hire_step,
        role="hand",
    )
    with_hire_raw = schedule_tasks(tasks, [*workers, hired], config=cfg)
    with_hire = replace(with_hire_raw, utility=with_hire_raw.utility - hire_cost)
    marginal = with_hire.utility - without.utility
    return HireDecision(
        should_hire=marginal > minimum_margin,
        marginal_utility=marginal,
        hire_cost=hire_cost,
        without_hire=without,
        with_hire=with_hire,
    )


def mutate_plan(plan: PlanSpec) -> tuple[PlanSpec, ...]:
    """Generate a controlled local neighborhood instead of a Cartesian product."""

    candidates: list[PlanSpec] = [plan]
    mutations = (
        ("cow_target", -2, 0, 16),
        ("cow_target", 2, 0, 16),
        ("sheep_target", -2, 0, 16),
        ("sheep_target", 2, 0, 16),
        ("land_target", -25, 25, 100),
        ("land_target", 25, 25, 100),
        ("max_hands", -1, 0, 16),
        ("max_hands", 1, 0, 16),
        ("sale_offset", -4, -24, 24),
        ("sale_offset", 4, -24, 24),
        ("liquidation_step", -24, 600, 719),
        ("liquidation_step", 24, 600, 719),
    )
    for field_name, delta, minimum, maximum in mutations:
        current = int(getattr(plan, field_name))
        updated = max(minimum, min(maximum, current + delta))
        if updated == current:
            continue
        candidates.append(
            replace(
                plan,
                name=f"{plan.name}:{field_name}{delta:+d}",
                **{field_name: updated},
            )
        )

    unique: dict[tuple[float, ...], PlanSpec] = {}
    for candidate in candidates:
        unique.setdefault(candidate.normalized_features(), candidate)
    return tuple(unique.values())
