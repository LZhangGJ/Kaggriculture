"""V7 internal traces, persistent task chains, and inverse-planning soft labels.

The trace schema is deliberately JSON-friendly so opaque Python/JAX teachers and
the trainable PyTorch policy can share one behavior-cloning data path.  Native
policy traces have confidence 1.0; inverse traces keep a normalized Top-M
distribution instead of pretending an ambiguous movement prefix has one label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import exp
from pathlib import Path
from typing import Any, Mapping, Sequence

from .board_policy import _canonical_farms, _positions
from .decision_schema import TaskType
from .gpu_policy import MAX_UNITS, _get
from .hierarchical_bc import IntentLabel, _intent_from_primitive, _unit_action


TRACE_SCHEMA = "kaggriculture.agent-trace.v2"


class TraceSource(str, Enum):
    INTERNAL = "internal"
    INVERSE_PLANNING = "inverse_planning"


class IntentPhase(str, Enum):
    IDLE = "IDLE"
    ACQUIRE_RESOURCE = "ACQUIRE_RESOURCE"
    NAVIGATE = "NAVIGATE"
    EXECUTE = "EXECUTE"
    WAITING = "WAITING"


INTENT_PHASES = tuple(IntentPhase)
INTENT_PHASE_INDEX = {phase: index for index, phase in enumerate(INTENT_PHASES)}


@dataclass(frozen=True)
class SoftIntentHypothesis:
    intent: IntentLabel
    probability: float
    eta_steps: int
    route: tuple[tuple[int, int], ...] = ()
    evidence: str = ""
    phase: IntentPhase = IntentPhase.IDLE
    chain_id: str = ""
    stage_index: int = 0
    chain: tuple[IntentLabel, ...] = ()
    chain_operations: tuple[str, ...] = ()
    chain_etas: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.intent.task_type.name,
            "target": [self.intent.target_x, self.intent.target_y],
            "item_id": self.intent.item_id,
            "probability": self.probability,
            "eta_steps": self.eta_steps,
            "route": [list(position) for position in self.route],
            "evidence": self.evidence,
            "phase": self.phase.value,
            "chain_id": self.chain_id,
            "stage_index": self.stage_index,
            "chain": [
                {
                    "task_type": stage.task_type.name,
                    "target": [stage.target_x, stage.target_y],
                    "item_id": stage.item_id,
                    "operation": (
                        self.chain_operations[index]
                        if index < len(self.chain_operations)
                        else ""
                    ),
                    "eta_steps": (
                        self.chain_etas[index]
                        if index < len(self.chain_etas)
                        else -1
                    ),
                }
                for index, stage in enumerate(self.chain)
            ],
        }


@dataclass(frozen=True)
class SoftIntentDistribution:
    unit: int
    hypotheses: tuple[SoftIntentHypothesis, ...]
    confidence: float
    source: TraceSource = TraceSource.INVERSE_PLANNING

    @property
    def best(self) -> IntentLabel:
        if not self.hypotheses:
            return IntentLabel(TaskType.IDLE_OR_PASS)
        return self.hypotheses[0].intent

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "source": self.source.value,
            "confidence": self.confidence,
            "hypotheses": [value.as_dict() for value in self.hypotheses],
        }


@dataclass(frozen=True)
class AgentStepTrace:
    step: int
    day: int
    strategy_mode: int
    action: Mapping[str, Any]
    workers: tuple[SoftIntentDistribution, ...]
    source: TraceSource
    teacher: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "step": self.step,
            "day": self.day,
            "strategy_mode": self.strategy_mode,
            "source": self.source.value,
            "teacher": self.teacher,
            "action": dict(self.action),
            "workers": [value.as_dict() for value in self.workers],
        }


def _intent_key(intent: IntentLabel) -> tuple[int, int, int, int]:
    return (
        int(intent.task_type),
        intent.target_x,
        intent.target_y,
        intent.item_id,
    )


def _position_at(observation: Any, unit: int) -> tuple[int, int] | None:
    own, _ = _canonical_farms(observation)
    positions = _positions(own)
    if unit >= len(positions):
        return None
    return tuple(int(value) for value in positions[unit])


_CROP_CHAIN = {
    TaskType.CROP_PRODUCTION,
    TaskType.WATER_CROP,
    TaskType.APPLY_FERTILIZER,
}
_ANIMAL_CHAIN = {
    TaskType.BUILD_ANIMAL_STRUCTURE,
    TaskType.ANIMAL_PLACE,
    TaskType.ANIMAL_FEED,
    TaskType.ANIMAL_CARE,
    TaskType.ANIMAL_COLLECT_PRODUCT,
    TaskType.ANIMAL_COLLECT_FERTILIZER,
}


def _chain_key(intent: IntentLabel) -> tuple[str, int, int, int]:
    if intent.task_type in _CROP_CHAIN:
        family = "CROP"
        item = -1
    elif intent.task_type in _ANIMAL_CHAIN:
        family = "ANIMAL"
        item = -1
    else:
        family = intent.task_type.name
        item = intent.item_id
    return family, intent.target_x, intent.target_y, item


def infer_episode_soft_intents(
    observations: Sequence[Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    horizon: int = 24,
    top_m: int = 3,
) -> list[list[SoftIntentDistribution]]:
    """Infer Top-M alternative task chains and supervise only each active stage."""

    if len(observations) != len(actions):
        raise ValueError("observations and actions must have equal length")
    if horizon <= 0 or top_m <= 0:
        raise ValueError("horizon and top_m must be positive")
    result: list[list[SoftIntentDistribution]] = []
    for step, observation in enumerate(observations):
        own, _ = _canonical_farms(observation)
        unit_count = min(len(_positions(own)), MAX_UNITS)
        row: list[SoftIntentDistribution] = []
        for unit in range(unit_count):
            start = _position_at(observation, unit)
            chains: dict[
                tuple[str, int, int, int],
                list[
                    tuple[
                        IntentLabel,
                        int,
                        tuple[tuple[int, int], ...],
                        str,
                        int,
                    ]
                ],
            ] = {}
            pickups: list[
                tuple[IntentLabel, int, tuple[tuple[int, int], ...], str, int]
            ] = []
            route: list[tuple[int, int]] = [start] if start is not None else []
            moved = 0
            raw_now = _unit_action(actions[step], unit)
            for future in range(step, min(len(actions), step + horizon + 1)):
                position = _position_at(observations[future], unit)
                if position is None:
                    break
                if not route or route[-1] != position:
                    # Farm hands expire overnight and are re-created in hire
                    # order.  The same list index can therefore refer to a new
                    # worker at a distant shed on the next day.  Never turn that
                    # identity discontinuity into a teleporting route/ETA label.
                    if route and (
                        abs(route[-1][0] - position[0])
                        + abs(route[-1][1] - position[1])
                        > 1
                    ):
                        break
                    route.append(position)
                    moved += 1
                raw = _unit_action(actions[future], unit)
                intent = _intent_from_primitive(observations[future], unit, raw)
                if intent is None:
                    continue
                operation = str(raw[0]) if raw else "PASS"
                eta = future - step
                distance = (
                    abs(start[0] - intent.target_x) + abs(start[1] - intent.target_y)
                    if start is not None and intent.target_x >= 0
                    else moved
                )
                path_error = abs(moved - distance)
                event = (
                    intent,
                    eta,
                    tuple(route),
                    operation,
                    path_error,
                )
                if intent.task_type == TaskType.SHED_PICKUP:
                    pickups.append(event)
                else:
                    chains.setdefault(_chain_key(intent), []).append(event)

            current_operation = str(raw_now[0]) if raw_now else "PASS"
            standalone_pickup = not chains and bool(pickups)
            if standalone_pickup:
                for event in pickups:
                    chains.setdefault(_chain_key(event[0]), []).append(event)
            idle_score = 0.2 if current_operation == "PASS" else 0.05
            if not chains:
                idle_score = 1.0
            scored: list[tuple[float, SoftIntentHypothesis]] = []
            for key, events in chains.items():
                events.sort(key=lambda value: value[1])
                active = events[0]
                intent, eta, event_route, operation, path_error = active
                acquisition = (
                    []
                    if standalone_pickup
                    else [value for value in pickups if value[1] <= eta]
                )
                stages = [*acquisition, *events]
                if eta == 0:
                    phase = IntentPhase.EXECUTE
                elif current_operation == "PASS":
                    phase = IntentPhase.WAITING
                elif acquisition or intent.task_type == TaskType.SHED_PICKUP:
                    phase = IntentPhase.ACQUIRE_RESOURCE
                else:
                    phase = IntentPhase.NAVIGATE
                immediate_boost = 1.5 if eta == 0 else 1.0
                coherence = 1.0 + 0.10 * min(len(events) - 1, 3)
                score = immediate_boost * coherence * exp(
                    -0.075 * eta - 0.35 * path_error
                )
                chain_id = f"u{unit}:{key[0]}:{key[1]},{key[2]}:{key[3]}"
                scored.append(
                    (
                        score,
                        SoftIntentHypothesis(
                            intent=intent,
                            probability=0.0,
                            eta_steps=eta,
                            route=event_route,
                            evidence=(
                                f"active_stage={operation}@+{eta};"
                                f"remaining_stages={len(stages)};path_error={path_error}"
                            ),
                            phase=phase,
                            chain_id=chain_id,
                            stage_index=len(acquisition),
                            chain=tuple(value[0] for value in stages),
                            chain_operations=tuple(value[3] for value in stages),
                            chain_etas=tuple(value[1] for value in stages),
                        ),
                    )
                )
            idle = IntentLabel(TaskType.IDLE_OR_PASS)
            scored.append(
                (
                    idle_score,
                    SoftIntentHypothesis(
                        intent=idle,
                        probability=0.0,
                        eta_steps=0,
                        route=tuple(route[:1]),
                        evidence=(
                            "no_nearby_semantic_chain"
                            if not chains
                            else "idle_alternative"
                        ),
                        phase=IntentPhase.IDLE,
                        chain_id=f"u{unit}:IDLE",
                        stage_index=0,
                        chain=(idle,),
                        chain_operations=("PASS",),
                        chain_etas=(0,),
                    ),
                )
            )
            ranked = sorted(scored, key=lambda value: value[0], reverse=True)[:top_m]
            normalizer = sum(value[0] for value in ranked)
            hypotheses = tuple(
                SoftIntentHypothesis(
                    intent=value[1].intent,
                    probability=value[0] / max(normalizer, 1e-12),
                    eta_steps=value[1].eta_steps,
                    route=value[1].route,
                    evidence=value[1].evidence,
                    phase=value[1].phase,
                    chain_id=value[1].chain_id,
                    stage_index=value[1].stage_index,
                    chain=value[1].chain,
                    chain_operations=value[1].chain_operations,
                    chain_etas=value[1].chain_etas,
                )
                for value in ranked
            )
            top_probability = hypotheses[0].probability if hypotheses else 0.0
            second_probability = hypotheses[1].probability if len(hypotheses) > 1 else 0.0
            confidence = min(max(top_probability - 0.5 * second_probability, 0.0), 1.0)
            row.append(
                SoftIntentDistribution(
                    unit=unit,
                    hypotheses=hypotheses,
                    confidence=confidence,
                )
            )
        result.append(row)
    return result


def trace_quality_audit(
    traces: Sequence[AgentStepTrace | Mapping[str, Any]],
    *,
    low_confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    """Audit inverse/internal labels before they are trusted for BC."""

    confidence_values: list[float] = []
    phase_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    teacher_counts: dict[str, int] = {}
    hypothesis_count = 0
    ambiguous_workers = 0
    invalid_probability_rows = 0
    missing_chain_ids = 0
    invalid_stage_indices = 0
    discontinuous_routes = 0
    workers = 0
    for raw_trace in traces:
        trace = raw_trace.as_dict() if isinstance(raw_trace, AgentStepTrace) else raw_trace
        teacher = str(trace.get("teacher", ""))
        teacher_counts[teacher] = teacher_counts.get(teacher, 0) + 1
        for distribution in trace.get("workers", []):
            workers += 1
            confidence = float(distribution.get("confidence", 0.0))
            confidence_values.append(confidence)
            hypotheses = list(distribution.get("hypotheses", []))
            hypothesis_count += len(hypotheses)
            ambiguous_workers += int(len(hypotheses) > 1)
            probability_sum = sum(
                float(hypothesis.get("probability", 0.0))
                for hypothesis in hypotheses
            )
            if hypotheses and abs(probability_sum - 1.0) > 1e-5:
                invalid_probability_rows += 1
            for hypothesis in hypotheses:
                phase = str(hypothesis.get("phase", ""))
                task = str(hypothesis.get("task_type", ""))
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                task_counts[task] = task_counts.get(task, 0) + 1
                chain = list(hypothesis.get("chain", []))
                stage_index = int(hypothesis.get("stage_index", 0))
                missing_chain_ids += int(not hypothesis.get("chain_id"))
                invalid_stage_indices += int(
                    not chain or not 0 <= stage_index < len(chain)
                )
                route = list(hypothesis.get("route", []))
                if any(
                    len(left) != 2
                    or len(right) != 2
                    or abs(int(left[0]) - int(right[0]))
                    + abs(int(left[1]) - int(right[1]))
                    > 1
                    for left, right in zip(route, route[1:])
                ):
                    discontinuous_routes += 1
    count = max(workers, 1)
    confidence_array = (
        [float(value) for value in confidence_values] or [0.0]
    )
    return {
        "schema": "kaggriculture.trace-quality-audit.v1",
        "steps": len(traces),
        "workers": workers,
        "hypotheses": hypothesis_count,
        "mean_confidence": sum(confidence_array) / len(confidence_array),
        "low_confidence_rate": sum(
            value < low_confidence_threshold for value in confidence_values
        )
        / count,
        "ambiguous_worker_rate": ambiguous_workers / count,
        "invalid_probability_rows": invalid_probability_rows,
        "missing_chain_ids": missing_chain_ids,
        "invalid_stage_indices": invalid_stage_indices,
        "discontinuous_routes": discontinuous_routes,
        "phases": dict(sorted(phase_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "teachers": dict(sorted(teacher_counts.items())),
    }


def merge_trace_quality_audits(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge precomputed live/offline audits without retaining every trace row."""

    if not audits:
        return trace_quality_audit(())
    totals = {
        "steps": 0,
        "workers": 0,
        "hypotheses": 0,
        "invalid_probability_rows": 0,
        "missing_chain_ids": 0,
        "invalid_stage_indices": 0,
        "discontinuous_routes": 0,
    }
    weighted_confidence = 0.0
    low_confidence_workers = 0.0
    ambiguous_workers = 0.0
    counters: dict[str, dict[str, int]] = {
        "phases": {},
        "tasks": {},
        "teachers": {},
    }
    for audit in audits:
        workers = int(audit.get("workers", 0))
        for key in totals:
            totals[key] += int(audit.get(key, 0))
        weighted_confidence += float(audit.get("mean_confidence", 0.0)) * workers
        low_confidence_workers += float(audit.get("low_confidence_rate", 0.0)) * workers
        ambiguous_workers += float(audit.get("ambiguous_worker_rate", 0.0)) * workers
        for field, output in counters.items():
            for key, value in dict(audit.get(field, {})).items():
                output[str(key)] = output.get(str(key), 0) + int(value)
    worker_denominator = max(totals["workers"], 1)
    return {
        "schema": "kaggriculture.trace-quality-audit.v1",
        **totals,
        "mean_confidence": weighted_confidence / worker_denominator,
        "low_confidence_rate": low_confidence_workers / worker_denominator,
        "ambiguous_worker_rate": ambiguous_workers / worker_denominator,
        **{
            field: dict(sorted(values.items()))
            for field, values in counters.items()
        },
    }


def inverse_episode_traces(
    observations: Sequence[Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    strategy_mode: int = -1,
    teacher: str = "",
    horizon: int = 24,
    top_m: int = 3,
) -> list[AgentStepTrace]:
    soft = infer_episode_soft_intents(
        observations, actions, horizon=horizon, top_m=top_m
    )
    return [
        AgentStepTrace(
            step=int(_get(observation, "step", index) or 0),
            day=int(_get(observation, "day", 0) or 0),
            strategy_mode=strategy_mode,
            action=action,
            workers=tuple(workers),
            source=TraceSource.INVERSE_PLANNING,
            teacher=teacher,
        )
        for index, (observation, action, workers) in enumerate(
            zip(observations, actions, soft, strict=True)
        )
    ]


def write_traces_jsonl(path: str | Path, traces: Sequence[AgentStepTrace | Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for trace in traces:
            row = trace.as_dict() if isinstance(trace, AgentStepTrace) else dict(trace)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
