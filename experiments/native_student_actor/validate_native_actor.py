#!/usr/bin/env python3
"""Compare the all-C++ actor suffix with the Python/NPU reference actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.native_student_actor.benchmark_npu_coordinator import counter_uniform
from experiments.native_student_actor.live_npu_prefix_scaling import (
    DEPLOYMENT, MANIFEST, META_ASSET, THOMAS_ASSET,
)
from experiments.native_student_actor.live_npu_smoke import (
    CHECKPOINT, load_extension, state_tensors,
)
from experiments.student_action_event_agent import EVENT_CLASSES
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_midgame_student_v1 import _sha256


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "work/student-v1/native-r3-parent.bin"


def prefix_batch(native, manifest: dict, size: int):
    cases = manifest["cases"]
    caches = [cases[index % len(cases)]["cache"] for index in range(size)]
    prefix = native.PrefixBatch(
        manifest["binary"], caches, str(THOMAS_ASSET), str(META_ASSET),
        str(DEPLOYMENT))
    started = time.perf_counter()
    prefix.run(size)
    return prefix, time.perf_counter() - started


def trace_digest(traces: list[list[tuple]]) -> str:
    rows = [[list(event[:7]) for event in trace] for trace in traces]
    return hashlib.sha256(json.dumps(
        rows, separators=(",", ":")).encode()).hexdigest()


def terminal_rows(prefix) -> list[dict]:
    keys = ("step", "done", "own_cash", "rival_cash", "suffix_steps",
            "action_hash", "actor_days", "actor_events", "actor_hash",
            "error")
    return [{key: row[key] for key in keys} for row in prefix.summary()["cases"]]


def run_npu(native, model, checkpoint: dict, device: torch.device,
            manifest: dict, size: int, policy_seed: int,
            full_suffix: bool) -> dict:
    prefix, prefix_seconds = prefix_batch(native, manifest, size)
    traces: list[list[tuple]] = [[] for _ in range(size)]
    initial_error = token_counts_equal = lengths_equal = None
    plan_seconds = advance_seconds = 0.0
    started = time.perf_counter()
    last_step = 672 if full_suffix else 288
    for step in range(288, last_step + 1, 24):
        if prefix.current_step != step:
            raise RuntimeError("Python/NPU prefix clock drift")
        plan_started = time.perf_counter()
        plan = native.PlanBatch(
            manifest["binary"], prefix.handles, prefix.packed, False)
        try:
            contexts = np.asarray(plan.contexts)
            state = state_tensors(
                list(prefix.observations), contexts,
                checkpoint["normalization"])
            with torch.inference_mode():
                hidden = model.initial_hidden(*(
                    [value.to(device) for value in state[:4]] +
                    [[value.to(device) for value in state[4]],
                     state[5].to(device)]))
                previous = torch.full(
                    (size,), len(EVENT_CLASSES), dtype=torch.long,
                    device=device)
                if step == 288:
                    native_initial = prefix.actor_initial_state(str(WEIGHTS))
                    initial_error = float(np.max(np.abs(
                        np.asarray(native_initial["hidden"]) -
                        hidden.cpu().numpy())))
                    token_counts_equal = bool(np.array_equal(
                        np.asarray(native_initial["token_counts"]),
                        state[5].numpy()))
                    raw_lengths = (
                        state[2] * torch.as_tensor(checkpoint["normalization"][
                            "observation_length_std"]) +
                        torch.as_tensor(checkpoint["normalization"][
                            "observation_length_mean"])).round().numpy()
                    lengths_equal = bool(np.array_equal(
                        np.asarray(native_initial["observation_lengths"]),
                        raw_lengths))
                day_counts = np.zeros(size, dtype=np.int64)
                while True:
                    ready = plan.collect_ready(60_000, True)
                    indices = np.asarray(ready["session_indices"], dtype=np.int32)
                    terminal = np.asarray(ready["terminal"], dtype=np.uint8)
                    if not indices.size:
                        if bool(terminal.all()):
                            break
                        raise TimeoutError(plan.summary())
                    stages = np.asarray(ready["stages"], dtype=np.int64)
                    cells = np.asarray(ready["cells"], dtype=np.int64)
                    suggested = np.asarray(ready["suggested"], dtype=np.int64)
                    masks = np.asarray(ready["legal_masks"], dtype=np.int64)
                    sequences = np.asarray(ready["sequences"], dtype=np.uint64)
                    resources = np.asarray(ready["resources"], dtype=np.float32)
                    legal = ((masks[:, None] &
                              (1 << np.arange(len(EVENT_CLASSES)))) != 0)
                    resource = ((torch.from_numpy(resources) - torch.as_tensor(
                        checkpoint["normalization"]["resource_mean"])) /
                        torch.as_tensor(checkpoint["normalization"][
                            "resource_std"]))
                    active = torch.from_numpy(indices.astype(np.int64)).to(device)
                    logits, next_hidden = model.step(
                        hidden.index_select(0, active), resource.to(device),
                        torch.from_numpy(cells).to(device),
                        torch.from_numpy(stages).to(device),
                        previous.index_select(0, active),
                        torch.from_numpy(legal).to(device))
                    log_probs = torch.log_softmax(logits.float(), dim=1)
                    probabilities = log_probs.exp()
                    cdf = probabilities.cumsum(1)
                    cdf[:, -1] = 1.0
                    counters = [len(traces[int(index)]) for index in indices]
                    uniforms = torch.tensor([
                        counter_uniform(policy_seed + int(index), counter)
                        for index, counter in zip(indices, counters)
                    ], dtype=torch.float32, device=device)
                    actions = (cdf < uniforms[:, None]).sum(1).clamp_max(
                        len(EVENT_CLASSES) - 1)
                    selected = log_probs.gather(1, actions[:, None])[:, 0]
                    entropy = -(probabilities * log_probs).sum(1)
                    hidden.index_copy_(0, active, next_hidden)
                    previous.index_copy_(0, active, actions)
                    host_actions = actions.cpu().numpy().astype(np.int32)
                    host_log_probs = selected.cpu().tolist()
                    host_entropy = entropy.cpu().tolist()
                    for row, index_value in enumerate(indices):
                        index = int(index_value)
                        if int(sequences[row]) != day_counts[index]:
                            raise RuntimeError("Python/NPU event sequence drift")
                        counter = len(traces[index])
                        traces[index].append((
                            step, counter, int(stages[row]), int(cells[row]),
                            int(suggested[row]), int(masks[row]),
                            int(host_actions[row]), float(host_log_probs[row]),
                            float(host_entropy[row])))
                        day_counts[index] += 1
                    plan.apply(indices, host_actions)
            plan.join()
            if plan.summary()["states"] != ["done"] * size:
                raise RuntimeError(plan.summary())
        finally:
            plan.close()
        plan_seconds += time.perf_counter() - plan_started
        advance_started = time.perf_counter()
        prefix.advance_to(step + 24, size, 2 << 20, True)
        advance_seconds += time.perf_counter() - advance_started
    if full_suffix:
        advance_started = time.perf_counter()
        prefix.advance_to(719, size, 2 << 20, True)
        advance_seconds += time.perf_counter() - advance_started
    return {
        "traces": traces,
        "trace_sha256": trace_digest(traces),
        "prefix_seconds": prefix_seconds,
        "suffix_seconds": time.perf_counter() - started,
        "plan_seconds": plan_seconds,
        "advance_seconds": advance_seconds,
        "initial_hidden_max_abs": initial_error,
        "token_counts_equal": token_counts_equal,
        "observation_lengths_equal": lengths_equal,
        "terminal": terminal_rows(prefix),
    }


def run_native(native, manifest: dict, size: int, policy_seed: int,
               full_suffix: bool) -> dict:
    prefix, prefix_seconds = prefix_batch(native, manifest, size)
    started = time.perf_counter()
    if full_suffix:
        metrics = prefix.run_native_actor_suffix(
            str(WEIGHTS), policy_seed, size, 2 << 20, True)
    else:
        metrics = prefix.plan_native_actor(
            str(WEIGHTS), policy_seed, size, 2 << 20, True)
        prefix.advance_to(312, size, 2 << 20, True)
    traces = [[tuple(event) for event in trace]
              for trace in prefix.actor_traces()]
    return {
        "traces": traces,
        "trace_sha256": trace_digest(traces),
        "prefix_seconds": prefix_seconds,
        "suffix_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "terminal": terminal_rows(prefix),
    }


def compare(reference: dict, native: dict, tolerance: float,
            full_suffix: bool) -> dict:
    metadata_mismatches = action_mismatches = length_mismatches = 0
    max_logprob = max_entropy = 0.0
    first_mismatch = None
    for session, (left_trace, right_trace) in enumerate(zip(
            reference["traces"], native["traces"])):
        length_mismatches += abs(len(left_trace) - len(right_trace))
        for event, (left, right) in enumerate(zip(left_trace, right_trace)):
            if left[:6] != right[:6]:
                metadata_mismatches += 1
            if left[6] != right[6]:
                action_mismatches += 1
            max_logprob = max(max_logprob, abs(left[7] - right[7]))
            max_entropy = max(max_entropy, abs(left[8] - right[8]))
            if first_mismatch is None and left[:7] != right[:7]:
                first_mismatch = {
                    "session": session, "event": event,
                    "python_npu": left, "native": right,
                }
    terminal_equal = True
    if full_suffix:
        fields = ("step", "done", "own_cash", "rival_cash", "suffix_steps",
                  "action_hash", "error")
        terminal_equal = all(
            all(left[key] == right[key] for key in fields)
            for left, right in zip(reference["terminal"], native["terminal"]))
    passed = (not metadata_mismatches and not action_mismatches and
              not length_mismatches and max_logprob <= tolerance and
              max_entropy <= tolerance and terminal_equal and
              reference["initial_hidden_max_abs"] <= tolerance and
              reference["token_counts_equal"] and
              reference["observation_lengths_equal"])
    return {
        "status": "PASS" if passed else "FAIL",
        "metadata_mismatches": metadata_mismatches,
        "action_mismatches": action_mismatches,
        "length_mismatches": length_mismatches,
        "max_logprob_abs": max_logprob,
        "max_entropy_abs": max_entropy,
        "terminal_equal": terminal_equal,
        "first_mismatch": first_mismatch,
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--policy-seed", type=int, default=2026092301)
    parser.add_argument("--first-day", action="store_true")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-actor-parity-b4.json"))
    args = parser.parse_args()
    if args.sessions < 1 or args.sessions > 256 or args.output.exists():
        parser.error("sessions must be in [1,256] and output must be new")
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dimensions = checkpoint["model_dimensions"]
    model = build_model(
        dimensions["causal_context"], dimensions["packed_observation"],
        dimensions["event_resources"], checkpoint["model_scale"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    native_module = load_extension()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    full_suffix = not args.first_day
    reference = run_npu(
        native_module, model, checkpoint, device, manifest, args.sessions,
        args.policy_seed, full_suffix)
    native = run_native(
        native_module, manifest, args.sessions, args.policy_seed, full_suffix)
    comparison = compare(reference, native, args.tolerance, full_suffix)
    result = {
        "status": comparison["status"],
        "scope": ("full-native-actor-suffix-parity" if full_suffix else
                  "first-day-native-actor-parity"),
        "sessions": args.sessions,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "weights_sha256": _sha256(WEIGHTS),
        "rng": "splitmix64(per_session_policy_seed,global_event_index)",
        "comparison": comparison,
        "python_npu": {key: value for key, value in reference.items()
                       if key != "traces"},
        "native": {key: value for key, value in native.items()
                   if key != "traces"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
