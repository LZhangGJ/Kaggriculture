#!/usr/bin/env python3
"""Collect one execution-gated v3 label on a live student state.

The played policy handle is never handed to the teacher.  At step 312 this
collector clones it twice: one clone runs frozen R1 to obtain final executable
jobs, and the other replays their controlled RELEASE/placement projection to
produce causal resources and fixed-point checks.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import build_midgame_action_events_v3 as contract
from experiments.student_action_event_agent import (
    MAX_TOKENS, ROOT, StudentActionEventAgent, canonical_observation, production,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observations(state):
    result = []
    for player in (0, 1):
        observation = json.loads(json.dumps(state[player]))
        if observation.get("step") is None:
            observation["step"] = observation["day"] * 24 + observation["hour"]
        observation["player"] = player
        result.append(observation)
    return result


class DaggerActionEventAgent(StudentActionEventAgent):
    """Student rollout plus read-only teacher clones at selected day boundaries."""

    def __init__(self, *args, dagger_step: int = 312, dagger_steps=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dagger_steps = tuple(sorted(set(map(
            int, dagger_steps if dagger_steps is not None else (dagger_step,)))))
        if (not self.dagger_steps or any(
                step < 312 or step >= 719 or step % 24
                for step in self.dagger_steps)):
            raise ValueError("DAgger steps must be day boundaries in [312, 696]")
        self.dagger_audits: list[dict] = []
        self.dagger_failures: list[dict] = []
        self.dagger_samples: list[dict[str, np.ndarray]] = []
        self._dagger_attempted_steps: set[int] = set()
        self.lib.td_clone.argtypes = [ctypes.c_void_p]
        self.lib.td_clone.restype = ctypes.c_void_p
        self.lib.td_student_live_meta.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        self.lib.td_student_live_meta.restype = ctypes.c_int

    def _live_meta(self, handle) -> list[int]:
        values = (ctypes.c_int32 * 6)()
        if self.lib.td_student_live_meta(handle, values, len(values)) != 6:
            raise RuntimeError("failed to export student live meta")
        return list(values)

    def _state_arrays(self, observation, step, context_values, events,
                      teacher_jobs, canonical_jobs):
        canonical = canonical_observation(observation)
        exact = np.asarray(list(production.policy._pack(canonical)), dtype="<f8")
        contract.validate_day_boundary_pack(canonical, exact.size)
        width = int(self.dimensions["packed_observation"])
        if exact.size > width:
            raise RuntimeError(f"packed observation {exact.size}>{width}")
        packed = np.zeros((1, width), dtype="<f8")
        packed[0, :exact.size] = exact
        encoded = self.tokenizer.encode(canonical)
        if encoded.num_tokens > MAX_TOKENS:
            raise RuntimeError(f"token count {encoded.num_tokens}>{MAX_TOKENS}")
        arrays = {
            "causal_context": np.asarray([context_values], dtype="<f4"),
            "packed_observation": packed,
            "observation_length": np.asarray([exact.size], dtype="<u2"),
            "token_continuous": np.zeros((1, MAX_TOKENS, 24), dtype="<f4"),
            "token_type": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_category_a": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_category_b": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_category_c": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_x": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_y": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_owner": np.zeros((1, MAX_TOKENS), dtype="u1"),
            "token_count": np.asarray([encoded.num_tokens], dtype="<u2"),
            "state_event_offsets": np.asarray([0, len(events)], dtype="<i8"),
            "state_step": np.asarray([step], dtype="<u2"),
            "event_stage": np.asarray([row["stage"] for row in events], dtype="u1"),
            "event_cell": np.asarray([row["cell"] for row in events], dtype="u1"),
            "event_resources": np.asarray(
                [row["resources"] for row in events], dtype="<f4").reshape(-1, 347),
            "event_legal_mask": np.asarray(
                [row["mask"] for row in events], dtype="<u2"),
            "event_label_class": np.asarray(
                [row["label"] for row in events], dtype="u1"),
            "teacher_job_actions": np.asarray(teacher_jobs, dtype="<i4"),
            "canonical_job_actions": np.asarray(canonical_jobs, dtype="<i4"),
        }
        arrays["token_continuous"][0, :encoded.num_tokens] = encoded.continuous.numpy()
        for target, source in (
                ("token_type", "token_type"),
                ("token_category_a", "category_a"),
                ("token_category_b", "category_b"),
                ("token_category_c", "category_c"),
                ("token_x", "x"), ("token_y", "y"), ("token_owner", "owner")):
            arrays[target][0, :encoded.num_tokens] = getattr(encoded, source).numpy()
        labels = arrays["event_label_class"].astype(np.int64)
        masks = arrays["event_legal_mask"].astype(np.int64)
        stages = arrays["event_stage"].astype(np.int64)
        bad_label = np.flatnonzero(((masks >> labels) & 1) != 1)
        bad_release = np.flatnonzero((stages == 0) & ((masks & ~0b110) != 0))
        bad_placement = np.flatnonzero(
            (stages == 1) & ((masks & (1 << 2)) != 0))
        if len(bad_label) or len(bad_release) or len(bad_placement):
            raise RuntimeError(
                "collected event mask/label contract failed: "
                f"events={len(events)} bad_label={bad_label[:4].tolist()} "
                f"bad_release={bad_release[:4].tolist()} "
                f"bad_placement={bad_placement[:4].tolist()}")
        return arrays

    def _collect_dagger(self, observation) -> dict:
        step = production.policy.observed_step(observation)
        if step not in self.dagger_steps:
            raise ValueError("unexpected DAgger collection step")
        if self.external:
            raise RuntimeError("student-state DAgger requires an active live handle")
        packed = production.policy._pack(observation)
        original_meta_before = self._live_meta(self.handle)
        original_debug_before = self.lib.td_debug(self.handle).decode()
        teacher_handle = self.lib.td_clone(self.handle)
        canonical_handle = self.lib.td_clone(self.handle)
        if not teacher_handle or not canonical_handle:
            if teacher_handle:
                self.lib.td_delete(teacher_handle)
            if canonical_handle:
                self.lib.td_delete(canonical_handle)
            raise RuntimeError("td_clone failed")
        distinct_handles = len({int(self.handle), int(teacher_handle),
                                int(canonical_handle)}) == 3
        teacher_seconds = replay_seconds = 0.0
        try:
            # From here on the exporter callback types own this diagnostic ABI.
            # The student callback has already run at step 288 and is not called
            # again in this first-handoff-only runtime.
            contract.bind(self.lib)
            teacher_started = time.perf_counter()
            teacher_jobs = contract.actor_owned_teacher_jobs(
                SimpleNamespace(lib=self.lib, handle=teacher_handle), packed)
            teacher_seconds = time.perf_counter() - teacher_started

            context_width = int(self.dimensions["causal_context"])
            context_values = (ctypes.c_double * context_width)()
            if self.lib.td_student_pre_context_observation(
                    canonical_handle, packed, len(packed), context_values,
                    len(context_values)) != context_width:
                raise RuntimeError(self.lib.td_debug(canonical_handle).decode())
            replay_started = time.perf_counter()
            (events, canonical_jobs, resource_exact, fixed_point,
             stop_exact, controlled_exact) = contract.plan_from_teacher(
                SimpleNamespace(lib=self.lib, handle=canonical_handle), packed,
                teacher_jobs)
            replay_seconds = time.perf_counter() - replay_started
            arrays = self._state_arrays(
                observation, step, list(context_values), events,
                teacher_jobs, canonical_jobs)
        finally:
            self.lib.td_delete(teacher_handle)
            self.lib.td_delete(canonical_handle)
            # ``contract.bind`` installs its own equivalent CFUNCTYPE classes.
            # Restore this agent's callback types so the object remains valid
            # if a harness resets and reuses it for another game.
            self._bind_student_abi()

        original_meta_after = self._live_meta(self.handle)
        original_debug_after = self.lib.td_debug(self.handle).decode()
        original_unchanged = (original_meta_before == original_meta_after and
                              original_debug_before == original_debug_after)
        gates = {
            "distinct_clone_handles": distinct_handles,
            "original_handle_observable_state_unchanged": original_unchanged,
            "resource_replay_exact": bool(resource_exact),
            "label_fixed_point": bool(fixed_point),
            "stop_terminal_exact": bool(stop_exact),
            "controlled_job_projection": bool(controlled_exact),
            "actor_owned_placement": bool(controlled_exact),
            "event_mask_label_legal": True,
        }
        status = "PASS" if all(gates.values()) else "FAIL"
        self.dagger_samples.append(arrays)
        audit = {
            "status": status,
            "scope": "student_state_day_boundary_controlled_actions",
            "state_source": "live_student_handle_cloned_before_student_action",
            "teacher_source": "actor_owned_native_v3_on_td_clone",
            "canonical_source": "projected_labels_on_independent_td_clone",
            "original_policy_action": "computed_after_clones_on_unmodified_handle",
            "step": step,
            "events": len(events),
            "release_events": sum(row["stage"] == 0 for row in events),
            "placement_events": sum(row["stage"] == 1 for row in events),
            "class_counts": {
                contract.CLASS_NAMES[index]: int(np.count_nonzero(
                    arrays["event_label_class"] == index))
                for index in range(len(contract.CLASS_NAMES))
                if np.any(arrays["event_label_class"] == index)
            },
            "teacher_seconds": teacher_seconds,
            "canonical_replay_seconds": replay_seconds,
            "original_live_meta_before": original_meta_before,
            "original_live_meta_after": original_meta_after,
            "gates": gates,
        }
        if status != "PASS":
            raise RuntimeError(f"student-state DAgger gate failed: {gates}")
        return audit

    def __call__(self, observation, configuration=None):
        step = production.policy.observed_step(observation)
        if step in self.dagger_steps and step not in self._dagger_attempted_steps:
            self._dagger_attempted_steps.add(step)
            try:
                self.dagger_audits.append(self._collect_dagger(observation))
            except Exception as error:
                self.dagger_failures.append({"step": step, "error": repr(error)})
        return super().__call__(observation, configuration)


def create_agent(seat=0, *, binary_path=None, checkpoint_path=None,
                 manifest_path=None, dagger_step=312, dagger_steps=None,
                 student_steps=None, allow_unattested_steps=False):
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    from experiments.student_action_event_agent import (
        DEFAULT_BINARY, DEFAULT_CHECKPOINT, DEFAULT_MANIFEST,
    )
    collect_steps = tuple(dagger_steps) if dagger_steps is not None else (dagger_step,)
    if student_steps is None:
        student_steps = range(288, max(collect_steps), 24)
    dynamic = DaggerActionEventAgent(
        config, binary_path=binary_path or DEFAULT_BINARY,
        checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT,
        manifest_path=manifest_path or DEFAULT_MANIFEST,
        dagger_steps=collect_steps, student_steps=student_steps,
        allow_unattested_steps=allow_unattested_steps)
    replay = production.replay_deployment()
    route = production.create_replay_agent(
        replay, f"student_v3_dagger_replay_seat_{seat}")
    return production.ReplayThenDynamicAgent(
        route, dynamic, 288, handoff_land=None, handoff_floor=0,
        handoff_delay_days=0, selector=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2620000000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--opponent", default="ahmed_v47")
    parser.add_argument("--step", type=int, default=312)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-student-state-dagger-"
        "step312-seed2620000000.json"))
    parser.add_argument("--sample", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-student-state-dagger-"
        "step312-seed2620000000.npz"))
    args = parser.parse_args()

    from fast_kaggriculture import Config, FastEnv
    opponent_module = _load(
        ROOT / "opponents" / args.opponent / "main.py",
        f"dagger_opponent_{args.opponent}")
    opponent = opponent_module.agent
    opponent_takes_configuration = len(inspect.signature(opponent).parameters) > 1
    if args.step < 312 or args.step >= 719 or args.step % 24:
        parser.error("--step must be a day boundary in [312, 696]")
    candidate = create_agent(
        args.seat, binary_path=args.binary, checkpoint_path=args.checkpoint,
        manifest_path=args.manifest, dagger_step=args.step)
    env = FastEnv(Config(), args.seed)
    state = list(env.reset(args.seed))
    frames = 0
    decision_seconds = 0.0
    started = time.perf_counter()
    try:
        while not env.done:
            actions = []
            for player, observation in enumerate(_observations(state)):
                if player == args.seat:
                    decision_started = time.perf_counter()
                    action = candidate(observation, {})
                    decision_seconds += time.perf_counter() - decision_started
                else:
                    action = (opponent(observation, {})
                              if opponent_takes_configuration
                              else opponent(observation))
                actions.append(action)
            state = env.step(actions)
            frames += 1
        dynamic = candidate.dynamic
        if len(dynamic.dagger_samples) != 1:
            raise RuntimeError(f"no DAgger sample: {dynamic.dagger_failures}")
        args.sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.sample, **dynamic.dagger_samples[0])
        audit = dynamic.dagger_audits[0]
        result = {
            "status": "PASS" if (
                frames == 719 and len(dynamic.dagger_audits) == 1 and
                not dynamic.dagger_failures and audit["status"] == "PASS") else "FAIL",
            "seed": args.seed, "seat": args.seat, "opponent": args.opponent,
            "engine": "fast_kaggriculture:FastEnv", "frames": frames,
            "cash": float(env.rewards[args.seat]),
            "opponent_cash": float(env.rewards[1 - args.seat]),
            "margin": float(env.rewards[args.seat] - env.rewards[1 - args.seat]),
            "wall_seconds": time.perf_counter() - started,
            "decision_seconds": decision_seconds,
            "checkpoint": str(dynamic.checkpoint_path),
            "checkpoint_sha256": _sha256(dynamic.checkpoint_path),
            "binary": str(dynamic.binary_path),
            "binary_sha256": _sha256(dynamic.binary_path),
            "sample": str(args.sample.resolve()),
            "sample_sha256": _sha256(args.sample),
            "sample_arrays": {
                name: {"shape": list(value.shape), "dtype": value.dtype.str}
                for name, value in dynamic.dagger_samples[0].items()
            },
            "student_step288": dynamic.student_summary(),
            "dagger": audit,
            "dagger_failures": dynamic.dagger_failures,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(1)
    finally:
        candidate.close()


if __name__ == "__main__":
    main()
