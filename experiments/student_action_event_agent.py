#!/usr/bin/env python3
"""Experimental first-handoff v3 action-event student.

The network controls only the native plan built at step 288.  The opening is
the unchanged replay route and all later observations use the frozen R1 again.
This module is deliberately separate from the production entry point.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.student_economic_features_v1 import (
    economic_features, prefix_flow_features,
)
from experiments.student_v3_runtime_model import build_model


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ROOT = Path("/root/kaggriculture_transformer_ppo_starter")
PACKED_OBSERVATION_CAPACITY = 3074
DEFAULT_CHECKPOINT = (
    ROOT / "work/student-v1/action-event-v3-first-handoff-128-student.pt")
DEFAULT_BINARY = ROOT / "work/agent-student-contract-v3.so"
DEFAULT_MANIFEST = ROOT / "work/student-v1/action-event-v3-first-handoff-128/manifest.json"
EVENT_CLASSES = (
    "STOP", "NONE_OR_KEEP", "RELEASE", "WHEAT", "CARROT", "TOMATO",
    "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP",
)
MAX_TOKENS = 320


def validate_day_boundary_pack(observation: dict, packed_length: int) -> None:
    step = int(observation["day"]) * 24 + int(observation["hour"])
    farms = observation["farms"]
    inventories = observation["private"]["inventories"]
    shops = observation["town"]["unlocked_shops"]
    if (step % 24 or len(farms) != 2 or
            any(farm.get("hands") or int(farm.get("hires_today", 0))
                for farm in farms) or inventories != [{}] or len(shops) > 8 or
            packed_length != 3066 + len(shops) or
            packed_length > PACKED_OBSERVATION_CAPACITY):
        raise ValueError("observation violates hour-zero packed ABI")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


production = _load_module("student_v3_production_shell", ROOT / "agent/main.py")
policy = production.policy


def canonical_observation(observation: dict) -> dict:
    """Put the acting farm first and erase the seat, as in the v3 shard."""
    result = copy.deepcopy(observation)
    player = int(result.pop("player", 0))
    farms = list(result["farms"])
    if len(farms) != 2 or player not in (0, 1):
        raise ValueError("expected a two-player observation")
    result["farms"] = [farms[player], farms[1 - player]]
    result["player"] = 0
    return result


class StudentActionEventAgent(policy.Agent):
    """Install free-running v3 plans only on checkpoint-attested days."""

    def __init__(self, config: dict, *, binary_path: Path, checkpoint_path: Path,
                 manifest_path: Path = DEFAULT_MANIFEST, sample: bool = False,
                 temperature: float = 1.0, student_steps=(288,),
                 allow_unattested_steps: bool = False):
        self.binary_path = Path(binary_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.sample = bool(sample)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(config=config, binary_path=self.binary_path)
        self._bind_student_abi()
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("event_classes", ())) != EVENT_CLASSES:
            raise ValueError("checkpoint event classes do not match v3 ABI")
        self._validate_training_contract(checkpoint)
        trained_steps = set(map(int, checkpoint.get("training_state_steps", (288,))))
        self.student_steps = tuple(sorted(set(map(int, student_steps))))
        self.unattested_steps = tuple(sorted(set(self.student_steps) - trained_steps))
        if (not self.student_steps or
                (self.unattested_steps and not allow_unattested_steps) or
                any(step < 288 or step >= 719 or step % 24
                    for step in self.student_steps)):
            raise ValueError("requested student days are not attested by the checkpoint")
        dimensions = checkpoint["model_dimensions"]
        self.model = build_model(
            dimensions["causal_context"], dimensions["packed_observation"],
            dimensions["event_resources"], checkpoint.get("model_scale", 1))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.dimensions = dict(dimensions)
        self.economic_v1 = dimensions == {
            "causal_context": 2233, "packed_observation": 3145,
            "event_resources": 374,
        }
        self.normalization = {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in checkpoint["normalization"].items()
        }
        sys.path.insert(0, str(
            ROOT / "experiments/student_v306_vendor" if self.economic_v1
            else TOKENIZER_ROOT))
        from kaggrl.tokenizer import ObservationTokenizer
        self.tokenizer = ObservationTokenizer()
        self.student_days: list[dict] = []
        self.student_failures: list[dict] = []
        self._student_attempted_steps: set[int] = set()

    def _validate_training_contract(self, checkpoint: dict) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        validation = manifest.get("validation", {})
        dimensions = checkpoint.get("model_dimensions", {})
        if dimensions == {"causal_context": 2233,
                          "packed_observation": 3145,
                          "event_resources": 374}:
            if (checkpoint.get("shard_manifest_sha256") !=
                    _sha256(self.manifest_path) or
                    manifest.get("validation_status") != "accepted" or
                    tuple(checkpoint.get("training_state_steps", ())) !=
                    tuple(range(288, 673, 24))):
                raise ValueError("v306 checkpoint contract mismatch")
            return
        if (manifest.get("validation_status") != "accepted" or
                manifest.get("schema", {}).get("schema_name") !=
                "autoregressive-action-event-bc-v3" or
                validation.get("scope") != "first_handoff_only_step288" or
                validation.get("execution_contract") != "PASS" or
                validation.get("actor_owned_placement") != "PASS" or
                validation.get("packed_observation_capacity") != "PASS" or
                dimensions.get("packed_observation") !=
                PACKED_OBSERVATION_CAPACITY or
                manifest.get("dimensions", {}).get("packed_observation") !=
                PACKED_OBSERVATION_CAPACITY):
            raise ValueError("checkpoint manifest lacks accepted first-handoff contract")
        manifest_digest = _sha256(self.manifest_path)
        if checkpoint.get("shard_manifest_sha256") != manifest_digest:
            raise ValueError("checkpoint and v3 shard manifest disagree")
        runtime = manifest.get("model_runtime", {})
        expected_files = {
            relative: _sha256(ROOT / relative)
            for relative in (
                "agent/main.py", "policy/r1/agent.py", "policy/r1/config.json")
        }
        tokenizer_path = TOKENIZER_ROOT / "kaggrl/tokenizer.py"
        if (runtime.get("files") != expected_files or
                runtime.get("tokenizer") != {
                    "path": str(tokenizer_path.resolve()),
                    "sha256": _sha256(tokenizer_path),
                } or runtime.get("scaffold_settings") != self.config):
            raise ValueError("runtime tokenizer/packer/scaffold fingerprint mismatch")
        semantic = validation.get("semantic_smoke", {})
        if semantic.get("binary_sha256") != _sha256(self.binary_path):
            raise ValueError("runtime binary was not attested by the v3 shard")

    def _bind_student_abi(self) -> None:
        self._release_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
        self._slot_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
        lib = self.lib
        lib.td_student_slot_abi_version.argtypes = []
        lib.td_student_slot_abi_version.restype = ctypes.c_int
        lib.td_student_pre_context_observation.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.td_student_pre_context_observation.restype = ctypes.c_int
        lib.td_student_plan_v3_callback_observation.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            self._release_callback_type, self._slot_callback_type,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.td_student_plan_v3_callback_observation.restype = ctypes.c_int
        lib.td_student_install_prepared.argtypes = [ctypes.c_void_p]
        lib.td_student_install_prepared.restype = ctypes.c_int
        if lib.td_student_slot_abi_version() != 3:
            raise RuntimeError("student actor-owned diagnostic ABI v3 is required")

    def _state_tensors(self, observation: dict, context_values):
        canonical = canonical_observation(observation)
        exact = np.asarray(list(policy._pack(canonical)), dtype=np.float32)
        validate_day_boundary_pack(canonical, exact.size)
        observation_width = int(self.dimensions["packed_observation"])
        if exact.size > observation_width:
            raise ValueError(
                f"packed observation {exact.size}>{observation_width}")
        padded = np.zeros(observation_width, dtype=np.float32)
        padded[:exact.size] = exact
        if self.economic_v1:
            padded[PACKED_OBSERVATION_CAPACITY:] = economic_features(exact)
        encoded = self.tokenizer.encode(canonical)
        if encoded.num_tokens > MAX_TOKENS:
            raise ValueError(f"token count {encoded.num_tokens}>{MAX_TOKENS}")
        if self.economic_v1:
            # Starter tokenizer order differs from the native rollout ABI.
            # Keep the whole town-token block in native order and IDs.
            if not bool((encoded.token_type[-8:] == 6).all()):
                raise ValueError("missing eight public shop tokens")
            order = [0, 2, 7, 4, 5, 1, 6, 3]
            for name in ("continuous", "token_type", "category_a", "category_b",
                         "category_c", "x", "y", "owner"):
                values = getattr(encoded, name)
                values[-8:] = values[-8:][order].clone()
            encoded.category_a[-8:] = torch.arange(1, 9)
        continuous = torch.zeros((1, MAX_TOKENS, 24), dtype=torch.float32)
        continuous[0, :encoded.num_tokens] = encoded.continuous
        categories = []
        for name in ("token_type", "category_a", "category_b", "category_c",
                     "x", "y", "owner"):
            value = torch.zeros((1, MAX_TOKENS), dtype=torch.long)
            value[0, :encoded.num_tokens] = getattr(encoded, name)
            categories.append(value)
        context = torch.as_tensor(context_values, dtype=torch.float32).unsqueeze(0)
        packed = torch.from_numpy(padded).unsqueeze(0)
        length = torch.tensor([exact.size], dtype=torch.float32)
        context = ((context - self.normalization["context_mean"]) /
                   self.normalization["context_std"])
        packed = ((packed - self.normalization["observation_mean"]) /
                  self.normalization["observation_std"])
        length = ((length - self.normalization["observation_length_mean"]) /
                  self.normalization["observation_length_std"])
        return (context, packed, length, continuous, categories,
                torch.tensor([encoded.num_tokens], dtype=torch.long))

    def _install_student_plan(self, observation: dict) -> dict:
        step = policy.observed_step(observation)
        if step not in self.student_steps:
            raise ValueError("v3 checkpoint is not attested for this day")
        packed = policy._pack(observation)
        if self.external:
            if self.lib.td_activate_external(self.handle, packed, len(packed)):
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
            self.external = False
        context_width = int(self.dimensions["causal_context"])
        context_values = (ctypes.c_double * context_width)()
        if self.lib.td_student_pre_context_observation(
                self.handle, packed, len(packed), context_values,
                len(context_values)) != context_width:
            raise RuntimeError(self.lib.td_debug(self.handle).decode())
        state = self._state_tensors(observation, list(context_values))
        events: list[dict] = []
        callback_errors: list[Exception] = []
        inference_seconds = 0.0
        previous = len(EVENT_CLASSES)

        with torch.inference_mode():
            hidden = self.model.initial_hidden(*state)

            def choose(stage, cell, suggested, mask_bits, values, width):
                nonlocal hidden, inference_seconds, previous
                try:
                    cell = int(cell)
                    mask_bits = int(mask_bits)
                    width = int(width)
                    if (not 0 <= cell < 100 or not mask_bits or
                            mask_bits >> len(EVENT_CLASSES) or
                            width != 347):
                        raise RuntimeError("invalid native v3 callback event")
                    if stage == 0:
                        if mask_bits & ~0b110 or not mask_bits & 0b110:
                            raise RuntimeError("invalid release legal mask")
                    elif mask_bits & (1 << 2) or not mask_bits & (1 << 1):
                        raise RuntimeError("invalid placement legal mask")
                    resource = np.ctypeslib.as_array(values, shape=(width,)).copy()
                    if self.economic_v1:
                        resource = np.concatenate((
                            resource, prefix_flow_features(resource, step // 24)))
                    resource = torch.from_numpy(resource.astype(np.float32)).unsqueeze(0)
                    resource = ((resource - self.normalization["resource_mean"]) /
                                self.normalization["resource_std"])
                    legal = torch.tensor([[
                        bool(mask_bits & (1 << index))
                        for index in range(len(EVENT_CLASSES))]], dtype=torch.bool)
                    started = time.perf_counter()
                    logits, hidden = self.model.step(
                        hidden, resource, torch.tensor([cell]),
                        torch.tensor([stage]), torch.tensor([previous]), legal)
                    inference_seconds += time.perf_counter() - started
                    if not torch.isfinite(logits[legal]).all():
                        raise RuntimeError("non-finite legal student logits")
                    distribution = torch.distributions.Categorical(
                        logits=logits / self.temperature)
                    selected = (int(distribution.sample().item()) if self.sample
                                else int(logits.argmax(1).item()))
                    if not bool(legal[0, selected]):
                        raise RuntimeError("student selected an illegal class")
                    events.append({
                        "index": len(events), "stage": int(stage),
                        "cell": cell, "suggested_class": int(suggested),
                        "selected_class": selected,
                        "selected_name": EVENT_CLASSES[selected],
                        "legal_mask": mask_bits,
                        "log_prob": float(distribution.log_prob(
                            torch.tensor([selected])).item()),
                        "entropy": float(distribution.entropy().item()),
                    })
                    previous = selected
                    return selected
                except Exception as error:
                    callback_errors.append(error)
                    return -1000

            @self._release_callback_type
            def release(_user, cell, suggested, mask_bits, values, width):
                return choose(0, cell, suggested, mask_bits, values, width)

            @self._slot_callback_type
            def placement(_user, cell, suggested, mask_bits, values, width):
                return choose(1, cell, suggested, mask_bits, values, width)

            count = self.lib.td_student_plan_v3_callback_observation(
                self.handle, packed, len(packed), release, placement, None,
                None, 0)
            if callback_errors:
                raise callback_errors[0]
            if count < 0:
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
            if count != len(events):
                raise RuntimeError(
                    f"native callback count {count}!={len(events)}")
            if self.lib.td_student_install_prepared(self.handle):
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
        return {
            "step": step, "events": events,
            "inference_seconds": inference_seconds,
        }

    def __call__(self, observation, configuration=None):
        step = policy.observed_step(observation)
        if step in self.student_steps and step not in self._student_attempted_steps:
            self._student_attempted_steps.add(step)
            started = time.perf_counter()
            try:
                record = self._install_student_plan(observation)
                record["plan_seconds"] = time.perf_counter() - started
                record["fallback"] = False
                self.student_days.append(record)
            except Exception as error:
                self.student_failures.append({
                    "step": step, "error": repr(error),
                    "plan_seconds": time.perf_counter() - started,
                })
        return super().__call__(observation, configuration)

    def student_summary(self) -> dict:
        events = [event for day in self.student_days for event in day["events"]]
        successful_steps = [int(day["step"]) for day in self.student_days]
        return {
            "scope": "requested_day_boundary_actions",
            "student_steps": list(self.student_steps),
            "successful_steps": successful_steps,
            "checkpoint_attested_action_steps": [
                step for step in self.student_steps
                if step not in self.unattested_steps],
            "unattested_exploration_steps": list(self.unattested_steps),
            "other_days_policy": "frozen_r1",
            "days": len(self.student_days),
            "events": len(events),
            "stage_counts": dict(sorted(Counter(
                "release" if row["stage"] == 0 else "placement"
                for row in events).items())),
            "class_counts": dict(sorted(Counter(
                row["selected_name"] for row in events).items())),
            "illegal": sum(
                not (row["legal_mask"] & (1 << row["selected_class"]))
                for row in events),
            "fallbacks": len(self.student_failures),
            "inference_seconds": sum(
                day["inference_seconds"] for day in self.student_days),
            "plan_seconds": sum(day["plan_seconds"] for day in self.student_days),
            "failures": list(self.student_failures),
            "event_trace": events,
        }


def create_agent(seat=0):
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    checkpoint_path = Path(os.environ.get("STUDENT_CHECKPOINT", DEFAULT_CHECKPOINT))
    if (checkpoint_path.parent.name == "student-v306" or
            checkpoint_path.name.startswith("v3-ppo-native-job-economic-v1-v306-")):
        config["intraday"] = 0
        student_steps = tuple(range(288, 673, 24))
        sample = True
    else:
        student_steps = (288,)
        sample = os.environ.get("STUDENT_SAMPLE", "0") == "1"
    dynamic = StudentActionEventAgent(
        config,
        binary_path=Path(os.environ.get("STUDENT_R1_BINARY", DEFAULT_BINARY)),
        checkpoint_path=checkpoint_path,
        manifest_path=Path(os.environ.get("STUDENT_V3_MANIFEST", DEFAULT_MANIFEST)),
        sample=sample,
        temperature=float(os.environ.get("STUDENT_TEMPERATURE", "1")),
        student_steps=student_steps,
    )
    replay = production.replay_deployment()
    if not replay:
        return dynamic
    route = production.create_replay_agent(replay, f"student_v3_replay_seat_{seat}")
    # The accepted shard is explicitly a step-288 first-handoff contract.
    # Do not let the land-triggered deployment wrapper activate R1 earlier.
    return production.ReplayThenDynamicAgent(
        route, dynamic, 288, handoff_land=None, handoff_floor=0,
        handoff_delay_days=0, selector=None)


_instances = {}


def agent(observation, configuration=None):
    observation = policy.normalize_observation(observation)
    seat = int(policy._get(observation, "player", 0))
    if seat not in _instances:
        _instances[seat] = create_agent(seat)
    return _instances[seat](observation, configuration)
