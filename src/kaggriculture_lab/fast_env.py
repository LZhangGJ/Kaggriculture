"""A low-overhead runner around Kaggle's official Kaggriculture interpreter.

The public ``kaggle-environments`` runner is the source of truth for submissions.
It also performs schema validation, deep copies agent observations, converts nested
dictionaries to ``Struct`` objects, records every replay step, and captures logs.
Those safeguards are valuable for evaluation but dominate runtime during trusted
local self-play.

This module keeps the official 1.32.6 game initializer and interpreter, while
removing that framework work from the inner step loop.  Use it for training and
large local tournaments; use the official runner for packaging and final checks.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ENGINE_VERSION = "1.32.6"
Agent = Callable[..., Mapping[str, Any]]
AgentSpec = Agent | str | os.PathLike[str]


@contextmanager
def _silence_native_output():
    """Silence import-time native-library diagnostics in worker processes."""
    try:
        stdout_fd = sys.__stdout__.fileno()
        stderr_fd = sys.__stderr__.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except (AttributeError, OSError, io.UnsupportedOperation):
        yield
        return

    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def _load_engine() -> tuple[Any, Any, str]:
    """Import the official package without unrelated OpenSpiel discovery noise."""
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        with (
            _silence_native_output(),
            redirect_stdout(captured_out),
            redirect_stderr(captured_err),
        ):
            import kaggle_environments
            from kaggle_environments import make
            from kaggle_environments.envs.kaggriculture import kaggriculture
    except Exception:
        print(captured_out.getvalue(), end="")
        print(captured_err.getvalue(), end="", file=sys.stderr)
        raise

    actual = kaggle_environments.__version__
    if actual != ENGINE_VERSION:
        raise RuntimeError(
            f"Expected kaggle-environments {ENGINE_VERSION}, found {actual}. "
            "Engine drift invalidates local results."
        )
    return make, kaggriculture, actual


_MAKE, _ENGINE, _ACTUAL_ENGINE_VERSION = _load_engine()


@dataclass(frozen=True)
class StepResult:
    """Result from one fast environment step."""

    observations: tuple[Any, Any]
    rewards: tuple[float | None, float | None]
    statuses: tuple[str, str]
    done: bool
    step: int


@dataclass(frozen=True)
class EpisodeResult:
    """Compact episode result suitable for tournament aggregation."""

    seed: int
    rewards: tuple[float | None, float | None]
    statuses: tuple[str, str]
    steps: int
    duration_seconds: float


def _call_style(agent: Agent) -> int:
    """Return 1 or 2 depending on whether an agent accepts configuration."""
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


def _adapt_agent(agent: Agent) -> Callable[[Any, Any], Mapping[str, Any]]:
    style = _call_style(agent)
    if style == 2:
        return lambda observation, configuration: agent(observation, configuration)
    return lambda observation, configuration: agent(observation)


@lru_cache(maxsize=128)
def _resolve_string_agent(spec: str) -> Callable[[Any, Any], Mapping[str, Any]]:
    if spec in _ENGINE.agents:
        return _adapt_agent(_ENGINE.agents[spec])

    candidate = Path(spec).expanduser().resolve()
    if candidate.is_file():
        module_name = f"kaggriculture_agent_{abs(hash(str(candidate)))}"
        module_spec = importlib.util.spec_from_file_location(module_name, candidate)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load agent from {candidate}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        agent = getattr(module, "agent", None)
        if not callable(agent):
            raise AttributeError(f"{candidate} does not expose callable agent(obs)")
        return _adapt_agent(agent)

    if ":" in spec:
        module_name, attribute = spec.split(":", 1)
        agent = getattr(importlib.import_module(module_name), attribute)
        if not callable(agent):
            raise TypeError(f"{spec} is not callable")
        return _adapt_agent(agent)

    raise ValueError(
        f"Unknown agent {spec!r}; use pass/random/starter, a .py path, "
        "or module:callable."
    )


def resolve_agent(spec: AgentSpec) -> Callable[[Any, Any], Mapping[str, Any]]:
    """Resolve a built-in name, Python file, module reference, or callable."""
    if callable(spec):
        return _adapt_agent(spec)
    return _resolve_string_agent(os.fspath(spec))


class FastKaggricultureEnv:
    """Trusted-agent environment using Kaggle's official state transition.

    Fast mode deliberately omits action-schema validation, timeout enforcement,
    log capture, replay history, and defensive deep copies.  An exception from an
    agent is raised immediately.  This makes failures visible during training and
    keeps the hot loop small.
    """

    def __init__(
        self,
        configuration: Mapping[str, Any] | None = None,
        *,
        copy_observations: bool = False,
    ) -> None:
        self.base_configuration = dict(configuration or {})
        self.base_configuration.setdefault("episodeSteps", 720)
        self.copy_observations = copy_observations
        self._env: Any | None = None
        self._state: Any | None = None
        self._step = 0
        self._seed = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._state is not None and all(
            state.status not in {"ACTIVE", "INACTIVE"} for state in self._state
        )

    @property
    def configuration(self) -> Any:
        if self._env is None:
            raise RuntimeError("Environment has not been reset")
        return self._env.configuration

    def reset(self, seed: int = 0) -> tuple[Any, Any]:
        configuration = dict(self.base_configuration)
        configuration["seed"] = int(seed)
        self._env = _MAKE("kaggriculture", configuration=configuration, debug=False)
        self._state = self._env.state
        self._step = 0
        self._seed = int(seed)
        self._sync_shared_step()
        return self.observations()

    def _require_state(self) -> Any:
        if self._state is None or self._env is None:
            raise RuntimeError("Call reset(seed) before step(actions)")
        return self._state

    def _sync_shared_step(self) -> None:
        state = self._require_state()
        for player_state in state:
            player_state.observation.step = self._step

    def observations(self) -> tuple[Any, Any]:
        state = self._require_state()
        observations = (state[0].observation, state[1].observation)
        if self.copy_observations:
            return copy.deepcopy(observations)
        return observations

    def step(self, actions: Sequence[Mapping[str, Any]]) -> StepResult:
        state = self._require_state()
        if self.done:
            raise RuntimeError("Episode is done; call reset(seed) before stepping again")
        if len(actions) != 2:
            raise ValueError(f"Expected two actions, got {len(actions)}")

        state[0].action = actions[0]
        state[1].action = actions[1]
        _ENGINE.interpreter(state, self._env)
        self._step += 1
        self._sync_shared_step()

        rewards = (state[0].reward, state[1].reward)
        statuses = (state[0].status, state[1].status)
        return StepResult(
            observations=self.observations(),
            rewards=rewards,
            statuses=statuses,
            done=self.done,
            step=self._step,
        )

    def run(self, agents: Sequence[AgentSpec], seed: int = 0) -> EpisodeResult:
        if len(agents) != 2:
            raise ValueError(f"Expected two agents, got {len(agents)}")
        resolved = (resolve_agent(agents[0]), resolve_agent(agents[1]))
        self.reset(seed)
        started = time.perf_counter()

        while not self.done:
            observations = self.observations()
            actions = (
                resolved[0](observations[0], self.configuration),
                resolved[1](observations[1], self.configuration),
            )
            self.step(actions)

        state = self._require_state()
        return EpisodeResult(
            seed=int(seed),
            rewards=(state[0].reward, state[1].reward),
            statuses=(state[0].status, state[1].status),
            steps=self._step,
            duration_seconds=time.perf_counter() - started,
        )


class VectorFastEnv:
    """Synchronous in-process environments for batched policy inference.

    Gather observations from all environments, run one batched CPU/GPU policy
    call, then pass the resulting ``[environment][player]`` actions to ``step``.
    The game transition remains on CPU; only policy inference should move to GPU.
    """

    def __init__(
        self,
        num_envs: int,
        configuration: Mapping[str, Any] | None = None,
        *,
        copy_observations: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [
            FastKaggricultureEnv(
                configuration=configuration,
                copy_observations=copy_observations,
            )
            for _ in range(num_envs)
        ]

    def reset(self, seeds: Iterable[int]) -> list[tuple[Any, Any]]:
        seed_list = list(seeds)
        if len(seed_list) != len(self.envs):
            raise ValueError(f"Expected {len(self.envs)} seeds, got {len(seed_list)}")
        return [env.reset(seed) for env, seed in zip(self.envs, seed_list, strict=True)]

    def observations(self) -> list[tuple[Any, Any]]:
        return [env.observations() for env in self.envs]

    def step(self, actions: Sequence[Sequence[Mapping[str, Any]]]) -> list[StepResult]:
        if len(actions) != len(self.envs):
            raise ValueError(f"Expected actions for {len(self.envs)} environments")
        return [
            env.step(env_actions)
            for env, env_actions in zip(self.envs, actions, strict=True)
        ]


def run_fast_episode(
    agent_a: AgentSpec = "starter",
    agent_b: AgentSpec = "starter",
    *,
    seed: int = 0,
    configuration: Mapping[str, Any] | None = None,
    copy_observations: bool = False,
) -> EpisodeResult:
    """Run one low-overhead episode."""
    return FastKaggricultureEnv(
        configuration=configuration,
        copy_observations=copy_observations,
    ).run((agent_a, agent_b), seed=seed)


def _duel_worker(task: tuple[str, str, int, int, dict[str, Any]]) -> EpisodeResult:
    agent_a, agent_b, seed, seat_a, configuration = task
    pair = (agent_a, agent_b) if seat_a == 0 else (agent_b, agent_a)
    raw = run_fast_episode(*pair, seed=seed, configuration=configuration)
    if seat_a == 0:
        return raw
    return EpisodeResult(
        seed=raw.seed,
        rewards=(raw.rewards[1], raw.rewards[0]),
        statuses=(raw.statuses[1], raw.statuses[0]),
        steps=raw.steps,
        duration_seconds=raw.duration_seconds,
    )


def run_duel(
    agent_a: str,
    agent_b: str,
    seeds: Iterable[int],
    *,
    both_seats: bool = True,
    workers: int = 1,
    configuration: Mapping[str, Any] | None = None,
) -> list[EpisodeResult]:
    """Run a deterministic duel, optionally distributing episodes over processes.

    Multiprocessing intentionally accepts string agent specifications so tasks are
    spawn-safe on Windows.  Use file paths or ``module:callable`` for custom agents.
    Returned rewards are always from ``agent_a`` then ``agent_b`` perspective.
    """
    seats = (0, 1) if both_seats else (0,)
    config = dict(configuration or {})
    tasks = [
        (agent_a, agent_b, int(seed), seat, config)
        for seed in seeds
        for seat in seats
    ]
    if workers <= 1:
        return [_duel_worker(task) for task in tasks]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_duel_worker, tasks, chunksize=max(1, len(tasks) // (workers * 4))))
