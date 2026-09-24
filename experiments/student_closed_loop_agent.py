#!/usr/bin/env python3
"""Experimental step>=288 per-cell BC policy on the frozen replay/R1 shell.

This module is intentionally separate from ``agent/main.py``.  It reconstructs
an executable R1 plan from the student's autoregressive prefix and falls back
to the unchanged dynamic policy if the diagnostic seam or network fails.
"""

from __future__ import annotations

import copy
import ctypes
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


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ROOT = Path("/root/kaggriculture_transformer_ppo_starter")
DEFAULT_CHECKPOINT = ROOT / "work/student-v1/live-slot-v2-batch2k-student.pt"
DEFAULT_BINARY = ROOT / "work/agent-student-closed-loop-v1.so"
CLASS_KINDS = (-1, 0, 1, 2, 3, 4, 9, 10, 11)
MAX_TOKENS = 320


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


production = _load_module("student_production_shell", ROOT / "agent/main.py")
policy = production.policy


def canonical_observation(observation: dict) -> dict:
    """Match the acting-first, identity-free representation used by BC."""
    result = copy.deepcopy(observation)
    player = int(result.pop("player", 0))
    farms = list(result["farms"])
    if len(farms) != 2 or player not in (0, 1):
        raise ValueError("expected a two-player observation")
    result["farms"] = [farms[player], farms[1 - player]]
    result["player"] = 0
    return result


class StudentR1Agent(policy.Agent):
    """Use the BC actor to fill R1's daily portfolio one legal cell at a time."""

    def __init__(self, config: dict, *, binary_path: Path, checkpoint_path: Path,
                 sample: bool = False, temperature: float = 1.0):
        super().__init__(config=config, binary_path=binary_path)
        self.sample = bool(sample)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self._bind_student_abi()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("class_kinds", ())) != CLASS_KINDS:
            raise ValueError("checkpoint class order does not match native ABI")
        dimensions = checkpoint["model_dimensions"]
        from experiments.train_midgame_autofill_v1 import build_model
        self.model = build_model(dimensions["causal_context"],
                                 dimensions["packed_observation"],
                                 dimensions["slot_resources"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.dimensions = dict(dimensions)
        self.normalization = {
            name: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for name, value in checkpoint["normalization"].items()
        }
        sys.path.insert(0, str(TOKENIZER_ROOT))
        from kaggrl.tokenizer import ObservationTokenizer
        self.tokenizer = ObservationTokenizer()
        self.student_days = []
        self.student_failures = []

    def _bind_student_abi(self) -> None:
        lib = self.lib
        self._student_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
        lib.td_student_slot_abi_version.argtypes = []
        lib.td_student_slot_abi_version.restype = ctypes.c_int
        lib.td_student_slot_contract_json.argtypes = []
        lib.td_student_slot_contract_json.restype = ctypes.c_char_p
        lib.td_student_pre_context_observation.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.td_student_pre_context_observation.restype = ctypes.c_int
        lib.td_student_prepare_prefix_observation.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.td_student_prepare_prefix_observation.restype = ctypes.c_int
        lib.td_student_candidate_slot_meta.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t]
        lib.td_student_candidate_slot_meta.restype = ctypes.c_int
        lib.td_student_candidate_slot_resources.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t]
        lib.td_student_candidate_slot_resources.restype = ctypes.c_int
        lib.td_student_install_prepared.argtypes = [ctypes.c_void_p]
        lib.td_student_install_prepared.restype = ctypes.c_int
        lib.td_student_plan_callback_observation.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            self._student_callback_type, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.td_student_plan_callback_observation.restype = ctypes.c_int
        if lib.td_student_slot_abi_version() != 2:
            raise RuntimeError("student slot ABI v2 is required")
        contract = json.loads(lib.td_student_slot_contract_json())
        if (tuple(contract.get("class_kinds", ())) != CLASS_KINDS or
                contract.get("meta_width") != 4 or
                contract.get("resource_width") != 347):
            raise RuntimeError("unexpected student slot contract")
        self.slot_contract = contract

    @staticmethod
    def _array(values, ctype):
        return (ctype * len(values))(*values) if values else None

    def _state_tensors(self, observation: dict, packed, context_values):
        canonical = canonical_observation(observation)
        exact = np.asarray(list(policy._pack(canonical)), dtype=np.float32)
        observation_width = self.dimensions["packed_observation"]
        if exact.size > observation_width:
            raise ValueError(f"packed observation {exact.size}>{observation_width}")
        padded = np.zeros(observation_width, dtype=np.float32)
        padded[:exact.size] = exact
        encoded = self.tokenizer.encode(canonical)
        if encoded.num_tokens > MAX_TOKENS:
            raise ValueError(f"token count {encoded.num_tokens}>{MAX_TOKENS}")
        continuous = torch.zeros((1, MAX_TOKENS, 24), dtype=torch.float32)
        continuous[0, :encoded.num_tokens] = encoded.continuous
        category_names = ("token_type", "category_a", "category_b", "category_c",
                          "x", "y", "owner")
        categories = []
        for name in category_names:
            value = torch.zeros((1, MAX_TOKENS), dtype=torch.long)
            value[0, :encoded.num_tokens] = getattr(encoded, name)
            categories.append(value)
        context = torch.as_tensor(context_values, dtype=torch.float32).unsqueeze(0)
        observation_tensor = torch.from_numpy(padded).unsqueeze(0)
        context = ((context - self.normalization["context_mean"]) /
                   self.normalization["context_std"])
        observation_tensor = (
            (observation_tensor - self.normalization["observation_mean"]) /
            self.normalization["observation_std"])
        return context, observation_tensor, continuous, categories, torch.tensor(
            [encoded.num_tokens], dtype=torch.long)

    def _prepare_prefix(self, packed, cells: list[int], kinds: list[int]):
        cell_buffer = self._array(cells, ctypes.c_int32)
        kind_buffer = self._array(kinds, ctypes.c_int32)
        count = self.lib.td_student_prepare_prefix_observation(
            self.handle, packed, len(packed), cell_buffer, kind_buffer, len(cells),
            None, 0)
        if count < 0:
            raise RuntimeError(self.lib.td_debug(self.handle).decode())
        meta = (ctypes.c_int32 * (count * 4))()
        resources = (ctypes.c_double * (count * self.dimensions["slot_resources"]))()
        if (self.lib.td_student_candidate_slot_meta(
                self.handle, 0, meta, len(meta)) != count or
                self.lib.td_student_candidate_slot_resources(
                    self.handle, 0, resources, len(resources)) != count):
            raise RuntimeError("native student slot export failed")
        rows = [tuple(meta[index * 4:(index + 1) * 4]) for index in range(count)]
        return rows, resources

    def _install_student_plan(self, observation: dict) -> dict:
        packed = policy._pack(observation)
        if self.external:
            if self.lib.td_activate_external(self.handle, packed, len(packed)):
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
            self.external = False
        context_width = self.dimensions["causal_context"]
        context_values = (ctypes.c_double * context_width)()
        if self.lib.td_student_pre_context_observation(
                self.handle, packed, len(packed), context_values,
                len(context_values)) != context_width:
            raise RuntimeError(self.lib.td_debug(self.handle).decode())
        state = self._state_tensors(observation, packed, list(context_values))
        with torch.no_grad():
            hidden = self.model.initial_hidden(*state)
            choices = []
            previous = 9
            callback_error = []

            def choose(_user, cell, base_kind, mask_bits, resource_values,
                       resource_count):
                nonlocal hidden, previous
                try:
                    slot = len(choices)
                    if (slot >= 100 or cell in {row["cell"] for row in choices} or
                            not 0 <= cell < 100 or mask_bits & ~0x1FF or
                            not mask_bits or resource_count !=
                            self.dimensions["slot_resources"]):
                        raise RuntimeError("invalid native callback slot")
                    resources = torch.tensor(
                        list(resource_values[:resource_count]), dtype=torch.float32
                    ).unsqueeze(0)
                    resources = ((resources - self.normalization["resource_mean"]) /
                                 self.normalization["resource_std"])
                    legal = torch.tensor(
                        [[bool(mask_bits & (1 << index)) for index in range(9)]],
                        dtype=torch.bool)
                    logits, hidden = self.model.step(
                        hidden, resources, torch.tensor([cell]),
                        torch.tensor([previous]), legal)
                    if not torch.isfinite(logits[legal]).all():
                        raise RuntimeError("non-finite legal student logits")
                    distribution = torch.distributions.Categorical(
                        logits=logits / self.temperature)
                    selected = (int(distribution.sample().item()) if self.sample else
                                int(logits.argmax(1).item()))
                    if not bool(legal[0, selected]):
                        raise RuntimeError("student selected an illegal class")
                    kind = CLASS_KINDS[selected]
                    choices.append({
                        "slot": slot, "cell": int(cell), "kind": int(kind),
                        "class": selected, "base_greedy_kind": int(base_kind),
                        "base_greedy_agree": kind == base_kind,
                        "legal_mask": int(mask_bits),
                        "log_prob": float(distribution.log_prob(
                            torch.tensor([selected])).item()),
                        "entropy": float(distribution.entropy().item()),
                    })
                    previous = selected
                    return int(kind)
                except Exception as error:
                    callback_error.append(error)
                    return -1000

            callback = self._student_callback_type(choose)
            count = self.lib.td_student_plan_callback_observation(
                self.handle, packed, len(packed), callback, None, None, 0)
            if callback_error:
                raise callback_error[0]
            if count < 0:
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
            if count != len(choices):
                raise RuntimeError("native callback count mismatch")
            if self.lib.td_student_install_prepared(self.handle):
                raise RuntimeError(self.lib.td_debug(self.handle).decode())
        return {
            "step": policy.observed_step(observation), "slots": choices,
            "base_greedy_agreement": (
                sum(row["base_greedy_agree"] for row in choices) / len(choices)
                if choices else None),
        }

    def __call__(self, observation, configuration=None):
        step = policy.observed_step(observation)
        if step >= 288 and step % 24 == 0:
            started = time.perf_counter()
            try:
                record = self._install_student_plan(observation)
                record["plan_seconds"] = time.perf_counter() - started
                record["fallback"] = False
                self.student_days.append(record)
            except Exception as error:
                self.student_failures.append({"step": step, "error": repr(error)})
        return super().__call__(observation, configuration)

    def student_summary(self) -> dict:
        slots = [slot for day in self.student_days for slot in day["slots"]]
        return {
            "days": len(self.student_days), "slots": len(slots),
            "fallbacks": len(self.student_failures),
            "illegal": sum(not (slot["legal_mask"] & (1 << slot["class"]))
                           for slot in slots),
            "base_greedy_agreement": (
                sum(slot["base_greedy_agree"] for slot in slots) / len(slots)
                if slots else None),
            "class_counts": dict(sorted(Counter(slot["kind"] for slot in slots).items())),
            "plan_seconds": sum(day["plan_seconds"] for day in self.student_days),
            "failures": list(self.student_failures),
        }


def create_agent(seat=0):
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    checkpoint = Path(os.environ.get("STUDENT_CHECKPOINT", DEFAULT_CHECKPOINT))
    binary = Path(os.environ.get("STUDENT_R1_BINARY", DEFAULT_BINARY))
    dynamic = StudentR1Agent(
        config, binary_path=binary, checkpoint_path=checkpoint,
        sample=os.environ.get("STUDENT_SAMPLE", "0") == "1",
        temperature=float(os.environ.get("STUDENT_TEMPERATURE", "1")))
    replay = production.replay_deployment()
    if not replay:
        return dynamic
    route = production.create_replay_agent(replay, f"student_replay_seat_{seat}")
    handoff = int(os.environ.get("REPLAY_HANDOFF_STEP", replay.get("handoff_step", 288)))
    land = os.environ.get("REPLAY_HANDOFF_LAND", replay.get("handoff_land"))
    floor = os.environ.get("REPLAY_HANDOFF_MIN_STEP", replay.get("handoff_floor"))
    delay = os.environ.get("REPLAY_HANDOFF_LAND_DELAY", replay.get("handoff_land_delay_days", 1))
    return production.ReplayThenDynamicAgent(
        route, dynamic, handoff,
        handoff_land=None if land in (None, "") else int(land),
        handoff_floor=None if floor in (None, "") else int(floor),
        handoff_delay_days=0 if delay in (None, "") else int(delay),
        selector=None)


_instances = {}


def agent(observation, configuration=None):
    observation = policy.normalize_observation(observation)
    seat = int(policy._get(observation, "player", 0))
    if seat not in _instances:
        _instances[seat] = create_agent(seat)
    return _instances[seat](observation, configuration)
