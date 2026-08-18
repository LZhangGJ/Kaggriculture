"""Agent-native counterfactual route rollouts on the fast official interpreter.

One scenario runs the common prefix exactly once.  At the decision point the
trusted local environment is deep-copied, and each candidate agent continues
from the identical state against an independently warmed opponent instance.
This keeps arbitrary Python agents usable while avoiding repeated 120--600 step
prefixes for every route.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

from .route_learning import CounterfactualRecord, encode_route_context
from .route_planner import PlanSpec


Agent = Callable[[Any, Any], Mapping[str, Any]]
AgentSpec = str | os.PathLike[str] | Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class RouteCandidate:
    name: str
    agent: AgentSpec
    plan: PlanSpec

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RouteCandidate name cannot be empty")
        if self.plan.name != self.name:
            raise ValueError("RouteCandidate name and PlanSpec name must match")


@dataclass(frozen=True)
class RolloutScenario:
    seed: int
    seat: int
    decision_step: int
    prefix_agent: AgentSpec
    opponent: AgentSpec

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if not 0 <= self.decision_step <= 718:
            raise ValueError("decision_step must be in [0, 718]")


@dataclass(frozen=True)
class _EnvironmentSnapshot:
    payload: Any
    step: int
    seed: int


def _spec_name(spec: AgentSpec) -> str:
    if isinstance(spec, (str, os.PathLike)):
        return os.fspath(spec)
    module = getattr(spec, "__module__", "callable")
    qualname = getattr(spec, "__qualname__", getattr(spec, "__name__", "agent"))
    return f"{module}:{qualname}"


def _call_style(agent: Callable[..., Mapping[str, Any]]) -> int:
    try:
        signature = inspect.signature(agent)
    except (TypeError, ValueError):
        return 1
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return 2 if has_varargs or len(positional) >= 2 else 1


def _adapt(agent: Callable[..., Mapping[str, Any]]) -> Agent:
    if _call_style(agent) == 2:
        return lambda observation, configuration: agent(observation, configuration)
    return lambda observation, configuration: agent(observation)


def _load_fresh_file_agent(path: Path) -> Agent:
    module_name = f"kaggriculture_route_agent_{uuid.uuid4().hex}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load agent from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, "make_agent", None)
    if callable(factory):
        agent = factory()
    else:
        agent = getattr(module, "agent", None)
    if not callable(agent):
        raise AttributeError(f"{path} does not expose agent(obs) or make_agent()")
    return _adapt(agent)


def resolve_fresh_agent(spec: AgentSpec) -> Agent:
    """Resolve a separate agent instance for one counterfactual branch.

    File agents are imported under a unique module name, preventing one route's
    module-level state from contaminating another.  Built-in names are stateless;
    ``module:callable`` references should expose an object that resets on step 0.
    """

    if callable(spec):
        return _adapt(spec)
    raw = os.fspath(spec)
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return _load_fresh_file_agent(candidate.resolve())

    from .fast_env import resolve_agent

    if ":" in raw:
        module_name, attribute = raw.split(":", 1)
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        if inspect.isclass(value):
            value = value()
        return _adapt(value)
    return resolve_agent(raw)


def _snapshot_fast_env(environment: Any) -> _EnvironmentSnapshot:
    """Copy the interpreter and state together so shared references survive."""

    payload = copy.deepcopy((environment._env, environment._state))  # noqa: SLF001
    return _EnvironmentSnapshot(
        payload=payload,
        step=int(environment._step),  # noqa: SLF001
        seed=int(environment._seed),  # noqa: SLF001
    )


def _restore_fast_env(environment: Any, snapshot: _EnvironmentSnapshot) -> tuple[Any, Any]:
    environment._env, environment._state = copy.deepcopy(snapshot.payload)  # noqa: SLF001
    environment._step = snapshot.step  # noqa: SLF001
    environment._seed = snapshot.seed  # noqa: SLF001
    try:
        environment._env.state = environment._state  # noqa: SLF001
    except Exception:
        pass
    environment._sync_shared_step()  # noqa: SLF001
    return environment.observations()


def _scenario_id(scenario: RolloutScenario) -> str:
    payload = {
        "seed": scenario.seed,
        "seat": scenario.seat,
        "decision_step": scenario.decision_step,
        "prefix": _spec_name(scenario.prefix_agent),
        "opponent": _spec_name(scenario.opponent),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _context_hash(values: Sequence[float]) -> str:
    serialized = json.dumps(
        [round(float(value), 8) for value in values], separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _safe_action(agent: Agent, observation: Any, configuration: Any) -> Mapping[str, Any]:
    action = agent(observation, configuration)
    if not isinstance(action, Mapping):
        raise TypeError(f"Agent returned {type(action).__name__}, expected a mapping")
    return action


def _error_record(
    scenario: RolloutScenario,
    candidate: RouteCandidate,
    *,
    scenario_id: str,
    context: Sequence[float],
    context_hash: str,
    steps: int,
    error: BaseException | str,
) -> CounterfactualRecord:
    message = str(error) if isinstance(error, str) else "".join(
        traceback.format_exception_only(type(error), error)
    ).strip()
    return CounterfactualRecord(
        scenario_id=scenario_id,
        context_hash=context_hash,
        route_name=candidate.name,
        seed=scenario.seed,
        seat=scenario.seat,
        decision_step=scenario.decision_step,
        opponent=_spec_name(scenario.opponent),
        prefix_agent=_spec_name(scenario.prefix_agent),
        context=tuple(float(value) for value in context),
        plan=candidate.plan.normalized_features(),
        completed=0.0,
        win=0.0,
        margin=0.0,
        terminal_cash=0.0,
        opponent_cash=0.0,
        steps=steps,
        candidate_status="ERROR",
        opponent_status="UNKNOWN",
        error=message,
    )


def collect_counterfactual_group(
    scenario: RolloutScenario,
    candidates: Sequence[RouteCandidate],
    *,
    configuration: Mapping[str, Any] | None = None,
) -> list[CounterfactualRecord]:
    """Evaluate all candidates from one identical decision state."""

    if not candidates:
        raise ValueError("At least one route candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Route candidate names must be unique")

    from .fast_env import FastKaggricultureEnv

    environment = FastKaggricultureEnv(configuration=configuration)
    observations = environment.reset(scenario.seed)
    prefix = resolve_fresh_agent(scenario.prefix_agent)
    route_agents: list[Agent | None] = []
    opponent_agents: list[Agent | None] = []
    warm_errors: list[BaseException | None] = []
    for candidate in candidates:
        try:
            route_agents.append(resolve_fresh_agent(candidate.agent))
            opponent_agents.append(resolve_fresh_agent(scenario.opponent))
            warm_errors.append(None)
        except BaseException as exc:
            route_agents.append(None)
            opponent_agents.append(None)
            warm_errors.append(exc)

    while environment.step_index < scenario.decision_step:
        candidate_observation = observations[scenario.seat]
        opponent_observation = observations[1 - scenario.seat]
        prefix_action = _safe_action(prefix, candidate_observation, environment.configuration)
        opponent_action: Mapping[str, Any] | None = None
        for index, (route_agent, opponent_agent) in enumerate(
            zip(route_agents, opponent_agents, strict=True)
        ):
            if warm_errors[index] is not None:
                continue
            try:
                assert route_agent is not None and opponent_agent is not None
                _safe_action(route_agent, candidate_observation, environment.configuration)
                branch_opponent_action = _safe_action(
                    opponent_agent, opponent_observation, environment.configuration
                )
                if opponent_action is None:
                    opponent_action = branch_opponent_action
            except BaseException as exc:
                warm_errors[index] = exc
        if opponent_action is None:
            fallback_opponent = resolve_fresh_agent(scenario.opponent)
            opponent_action = _safe_action(
                fallback_opponent, opponent_observation, environment.configuration
            )
        actions: list[Mapping[str, Any]] = [{}, {}]
        actions[scenario.seat] = prefix_action
        actions[1 - scenario.seat] = opponent_action
        result = environment.step(actions)
        observations = result.observations

    decision_observation = observations[scenario.seat]
    context_array = encode_route_context(decision_observation)
    context = tuple(float(value) for value in context_array)
    context_digest = _context_hash(context)
    scenario_digest = _scenario_id(scenario)
    snapshot = _snapshot_fast_env(environment)
    records: list[CounterfactualRecord] = []

    for index, candidate in enumerate(candidates):
        if warm_errors[index] is not None:
            records.append(
                _error_record(
                    scenario,
                    candidate,
                    scenario_id=scenario_digest,
                    context=context,
                    context_hash=context_digest,
                    steps=environment.step_index,
                    error=warm_errors[index] or "route warm-up failed",
                )
            )
            continue
        try:
            branch_observations = _restore_fast_env(environment, snapshot)
            route_agent = route_agents[index]
            opponent_agent = opponent_agents[index]
            assert route_agent is not None and opponent_agent is not None
            while not environment.done:
                candidate_observation = branch_observations[scenario.seat]
                opponent_observation = branch_observations[1 - scenario.seat]
                candidate_action = _safe_action(
                    route_agent, candidate_observation, environment.configuration
                )
                opponent_action = _safe_action(
                    opponent_agent, opponent_observation, environment.configuration
                )
                actions = [{}, {}]
                actions[scenario.seat] = candidate_action
                actions[1 - scenario.seat] = opponent_action
                step_result = environment.step(actions)
                branch_observations = step_result.observations

            state = environment._state  # noqa: SLF001
            statuses = (str(state[0].status), str(state[1].status))
            raw_rewards = (state[0].reward, state[1].reward)
            candidate_cash = float(raw_rewards[scenario.seat] or 0.0)
            opponent_cash = float(raw_rewards[1 - scenario.seat] or 0.0)
            completed = float(statuses[scenario.seat] == "DONE")
            if candidate_cash > opponent_cash:
                win = 1.0
            elif candidate_cash == opponent_cash:
                win = 0.5
            else:
                win = 0.0
            records.append(
                CounterfactualRecord(
                    scenario_id=scenario_digest,
                    context_hash=context_digest,
                    route_name=candidate.name,
                    seed=scenario.seed,
                    seat=scenario.seat,
                    decision_step=scenario.decision_step,
                    opponent=_spec_name(scenario.opponent),
                    prefix_agent=_spec_name(scenario.prefix_agent),
                    context=context,
                    plan=candidate.plan.normalized_features(),
                    completed=completed,
                    win=win,
                    margin=candidate_cash - opponent_cash,
                    terminal_cash=candidate_cash,
                    opponent_cash=opponent_cash,
                    steps=environment.step_index,
                    candidate_status=statuses[scenario.seat],
                    opponent_status=statuses[1 - scenario.seat],
                    error=None,
                )
            )
        except BaseException as exc:
            records.append(
                _error_record(
                    scenario,
                    candidate,
                    scenario_id=scenario_digest,
                    context=context,
                    context_hash=context_digest,
                    steps=environment.step_index,
                    error=exc,
                )
            )
    return records


def _collect_worker(
    payload: tuple[
        RolloutScenario,
        tuple[RouteCandidate, ...],
        dict[str, Any],
    ]
) -> list[CounterfactualRecord]:
    scenario, candidates, configuration = payload
    return collect_counterfactual_group(
        scenario,
        candidates,
        configuration=configuration,
    )


def collect_counterfactual_grid(
    scenarios: Iterable[RolloutScenario],
    candidates: Sequence[RouteCandidate],
    *,
    configuration: Mapping[str, Any] | None = None,
    workers: int = 1,
) -> list[CounterfactualRecord]:
    """Collect a scenario grid, optionally with process-level parallelism."""

    scenario_list = list(scenarios)
    candidate_tuple = tuple(candidates)
    config = dict(configuration or {})
    if workers <= 1:
        return [
            record
            for scenario in scenario_list
            for record in collect_counterfactual_group(
                scenario, candidate_tuple, configuration=config
            )
        ]
    if any(callable(candidate.agent) for candidate in candidate_tuple):
        raise ValueError("workers > 1 requires string/file route agent specifications")
    if any(callable(scenario.prefix_agent) or callable(scenario.opponent) for scenario in scenario_list):
        raise ValueError("workers > 1 requires string/file prefix and opponent specs")
    payloads = [(scenario, candidate_tuple, config) for scenario in scenario_list]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        groups = executor.map(
            _collect_worker,
            payloads,
            chunksize=max(1, len(payloads) // max(1, workers * 4)),
        )
        return [record for group in groups for record in group]
