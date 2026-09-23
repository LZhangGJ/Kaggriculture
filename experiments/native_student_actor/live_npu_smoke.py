#!/usr/bin/env python3
"""Drive real paused v3 planner callbacks with the batched NPU actor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.build_midgame_action_events_v3 import (
    PACKED_OBSERVATION_CAPACITY, validate_day_boundary_pack,
)
from experiments.native_student_actor.benchmark_npu_coordinator import counter_uniform
from experiments.native_student_rollout.smoke import (
    default_trajectory, load_extension, restore,
)
from experiments.student_action_event_agent import (
    EVENT_CLASSES, MAX_TOKENS, TOKENIZER_ROOT, canonical_observation, policy,
)
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_midgame_student_v1 import _sha256
from scripts.extract_midgame_bc import observation


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT / "work/student-v1/action-event-v3-actor-owned-dagger-r3-scale3-e20.pt")
BINARY = ROOT / "work/agent-student-actor-owned-v3.so"


def step288_observation(path: Path) -> dict:
    seat = None
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("type") == "meta":
                seat = int(row["seat"])
            elif row.get("type") == "step" and int(row["step"]) == 288:
                if seat is None:
                    raise RuntimeError("trajectory meta must precede steps")
                return observation(row, seat)
    raise RuntimeError("trajectory does not contain step 288")


def state_tensors(observations: list[dict], contexts: np.ndarray,
                  normalization: dict) -> tuple:
    sys.path.insert(0, str(TOKENIZER_ROOT))
    from kaggrl.tokenizer import ObservationTokenizer

    tokenizer = ObservationTokenizer()
    packed_rows, lengths, continuous_rows, count_rows = [], [], [], []
    category_rows = [[] for _ in range(7)]
    for current in observations:
        canonical = canonical_observation(current)
        exact = np.asarray(list(policy._pack(canonical)), dtype=np.float32)
        validate_day_boundary_pack(canonical, exact.size)
        if exact.size > PACKED_OBSERVATION_CAPACITY:
            raise ValueError("packed observation exceeds the fixed v3 capacity")
        padded = np.zeros(PACKED_OBSERVATION_CAPACITY, dtype=np.float32)
        padded[:exact.size] = exact
        encoded = tokenizer.encode(canonical)
        if encoded.num_tokens > MAX_TOKENS:
            raise ValueError("token count exceeds the fixed v3 capacity")
        token_continuous = torch.zeros((MAX_TOKENS, 24), dtype=torch.float32)
        token_continuous[:encoded.num_tokens] = encoded.continuous
        packed_rows.append(padded)
        lengths.append(exact.size)
        continuous_rows.append(token_continuous)
        count_rows.append(encoded.num_tokens)
        for index, name in enumerate((
                "token_type", "category_a", "category_b", "category_c",
                "x", "y", "owner")):
            value = torch.zeros(MAX_TOKENS, dtype=torch.long)
            value[:encoded.num_tokens] = getattr(encoded, name)
            category_rows[index].append(value)

    def norm(value, mean: str, std: str):
        return ((value - torch.as_tensor(normalization[mean])) /
                torch.as_tensor(normalization[std]))

    return (
        norm(torch.from_numpy(contexts.astype(np.float32)),
             "context_mean", "context_std"),
        norm(torch.from_numpy(np.stack(packed_rows)),
             "observation_mean", "observation_std"),
        norm(torch.tensor(lengths, dtype=torch.float32),
             "observation_length_mean", "observation_length_std"),
        torch.stack(continuous_rows),
        [torch.stack(rows) for rows in category_rows],
        torch.tensor(count_rows, dtype=torch.long),
    )


def run_group(native, model, checkpoint: dict, device: torch.device,
              binary: Path, trajectory: Path, size: int,
              policy_seed: int, verify_installed: bool = True) -> dict:
    agents, batch = [], None
    current = step288_observation(trajectory)
    started = time.perf_counter()
    try:
        warm_started = time.perf_counter()
        packed = []
        for _ in range(size):
            agent, values = restore(binary, trajectory)
            agents.append(agent)
            packed.append(values)
        warm_handle_seconds = time.perf_counter() - warm_started
        batch_started = time.perf_counter()
        batch = native.PlanBatch(
            str(binary), [int(agent.handle) for agent in agents], packed, True)
        contexts = np.asarray(batch.contexts)
        batch_constructor_seconds = time.perf_counter() - batch_started
        state_started = time.perf_counter()
        state = state_tensors([current] * size, contexts,
                              checkpoint["normalization"])
        python_state_pack_seconds = time.perf_counter() - state_started
        with torch.inference_mode():
            initial_started = time.perf_counter()
            hidden = model.initial_hidden(*(
                [value.to(device) for value in state[:4]] +
                [[value.to(device) for value in state[4]], state[5].to(device)]))
            previous = torch.full(
                (size,), len(EVENT_CLASSES), dtype=torch.long, device=device)
            torch.npu.synchronize(device)
            npu_initial_seconds = time.perf_counter() - initial_started
            traces = [[] for _ in range(size)]
            ready_sizes = []
            cpp_collect_seconds = coalesce_seconds = 0.0
            python_event_pack_seconds = npu_event_seconds = 0.0
            python_apply_seconds = 0.0
            plan_started = time.perf_counter()
            while True:
                collect_started = time.perf_counter()
                ready = batch.collect_ready(60_000)
                cpp_collect_seconds += time.perf_counter() - collect_started
                if len(ready["session_indices"]):
                    # Coalesce sibling callbacks that arrived immediately after
                    # the first notification; waiting sessions remain immutable.
                    coalesce_started = time.perf_counter()
                    time.sleep(0.001)
                    ready = batch.collect_ready(0)
                    coalesce_seconds += time.perf_counter() - coalesce_started
                indices = np.asarray(ready["session_indices"], dtype=np.int32)
                terminal = np.asarray(ready["terminal"], dtype=np.uint8)
                if not indices.size:
                    if bool(terminal.all()):
                        break
                    raise TimeoutError(batch.summary())
                pack_started = time.perf_counter()
                stages = np.asarray(ready["stages"], dtype=np.int64)
                cells = np.asarray(ready["cells"], dtype=np.int64)
                masks = np.asarray(ready["legal_masks"], dtype=np.int64)
                sequences = np.asarray(ready["sequences"], dtype=np.uint64)
                resources = np.asarray(ready["resources"], dtype=np.float32)
                legal = (masks[:, None] &
                         (1 << np.arange(len(EVENT_CLASSES)))) != 0
                resource = ((torch.from_numpy(resources) - torch.as_tensor(
                    checkpoint["normalization"]["resource_mean"])) /
                    torch.as_tensor(checkpoint["normalization"]["resource_std"]))
                active = torch.from_numpy(indices.astype(np.int64)).to(device)
                python_event_pack_seconds += time.perf_counter() - pack_started
                inference_started = time.perf_counter()
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
                uniforms = torch.tensor([
                    counter_uniform(policy_seed, int(sequence))
                    for sequence in sequences
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
                npu_event_seconds += time.perf_counter() - inference_started
                apply_started = time.perf_counter()
                for row, session_index in enumerate(indices):
                    if int(sequences[row]) != len(traces[int(session_index)]):
                        raise RuntimeError("native event sequence is not contiguous")
                    traces[int(session_index)].append((
                        int(stages[row]), int(cells[row]), int(masks[row]),
                        int(host_actions[row]), float(host_log_probs[row]),
                        float(host_entropy[row]),
                    ))
                batch.apply(indices, host_actions)
                ready_sizes.append(int(indices.size))
                python_apply_seconds += time.perf_counter() - apply_started
        join_started = time.perf_counter()
        batch.join()
        join_seconds = time.perf_counter() - join_started
        live_plan_seconds = time.perf_counter() - plan_started
        summary = batch.summary()
        if summary["states"] != ["done"] * size:
            raise RuntimeError(summary)
        batch.close()
        batch = None
        installed_actions = []
        if verify_installed:
            for agent in agents:
                agent.external = False
                installed_actions.append(agent(current, {}))
        sizes = np.asarray(ready_sizes, dtype=np.int64)
        histogram = {
            str(int(value)): int((sizes == value).sum())
            for value in np.unique(sizes)
        }
        events = sum(map(len, traces))
        return {
            "sessions": size, "events": events,
            "event_counts": list(map(len, traces)), "traces": traces,
            "callback_rounds": len(ready_sizes),
            "ready_batch_sizes": ready_sizes,
            "max_ready_batch": max(ready_sizes, default=0),
            "ready_batch_distribution": {
                "min": int(sizes.min()) if sizes.size else 0,
                "mean": float(sizes.mean()) if sizes.size else 0.0,
                "p50": float(np.percentile(sizes, 50)) if sizes.size else 0.0,
                "p90": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
                "max": int(sizes.max()) if sizes.size else 0,
                "histogram": histogram,
            },
            "events_per_second_live_plan": events / live_plan_seconds,
            "warm_handle_seconds": warm_handle_seconds,
            "batch_constructor_seconds": batch_constructor_seconds,
            "python_state_pack_seconds": python_state_pack_seconds,
            "npu_initial_seconds": npu_initial_seconds,
            "cpp_wait_collect_seconds": cpp_collect_seconds,
            "coalesce_seconds": coalesce_seconds,
            "python_event_pack_seconds": python_event_pack_seconds,
            "npu_event_seconds": npu_event_seconds,
            "python_apply_seconds": python_apply_seconds,
            "join_seconds": join_seconds,
            "live_plan_seconds": live_plan_seconds,
            # Backward-compatible names used by the B=1/B=4 smoke report.
            "inference_seconds": npu_event_seconds,
            "initial_seconds": npu_initial_seconds,
            "wall_seconds": time.perf_counter() - started,
            "summary": summary, "installed_actions": installed_actions,
        }
    finally:
        if batch is not None:
            batch.close()
        for agent in agents:
            agent.close()


def compare(single: dict, multiple: dict, tolerance: float) -> dict:
    reference = single["traces"][0]
    action_mismatches = metadata_mismatches = 0
    max_logprob = max_entropy = 0.0
    for trace in multiple["traces"]:
        if len(trace) != len(reference):
            metadata_mismatches += abs(len(trace) - len(reference)) + 1
            continue
        for left, right in zip(reference, trace):
            metadata_mismatches += left[:3] != right[:3]
            action_mismatches += left[3] != right[3]
            max_logprob = max(max_logprob, abs(left[4] - right[4]))
            max_entropy = max(max_entropy, abs(left[5] - right[5]))
    installed_equal = all(
        action == single["installed_actions"][0]
        for action in multiple["installed_actions"])
    passed = (not metadata_mismatches and not action_mismatches and
              max_logprob <= tolerance and max_entropy <= tolerance and
              installed_equal and multiple["max_ready_batch"] > 1)
    return {
        "status": "PASS" if passed else "FAIL",
        "metadata_mismatches": metadata_mismatches,
        "action_mismatches": action_mismatches,
        "max_logprob_abs": max_logprob, "max_entropy_abs": max_entropy,
        "installed_actions_equal": installed_equal,
        "observed_real_batch": multiple["max_ready_batch"] > 1,
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--policy-seed", type=int, default=2026092301)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/live-npu-smoke.json"))
    args = parser.parse_args()
    if args.batch_size < 2 or args.output.exists():
        parser.error("batch size must be >=2 and output must not exist")
    trajectory = args.trajectory or default_trajectory()
    for path in (args.checkpoint, args.binary, trajectory):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.device.startswith("npu"):
        parser.error("live smoke requires npu[:index]")
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
    native = load_extension()
    single = run_group(native, model, checkpoint, device, args.binary.resolve(),
                       trajectory.resolve(), 1, args.policy_seed)
    multiple = run_group(native, model, checkpoint, device, args.binary.resolve(),
                         trajectory.resolve(), args.batch_size, args.policy_seed)
    comparison = compare(single, multiple, args.tolerance)
    result = {
        "status": comparison["status"],
        "scope": "live-paused-native-planner-to-npu-actor",
        "checkpoint_sha256": _sha256(args.checkpoint),
        "binary_sha256": _sha256(args.binary), "device": str(device),
        "trajectory": str(trajectory.resolve()),
        "trajectory_sha256": _sha256(trajectory),
        "rng": "splitmix64(policy_seed,native_session_sequence)",
        "comparison": comparison,
        "single": {key: value for key, value in single.items()
                   if key not in ("traces", "installed_actions")},
        "multiple": {key: value for key, value in multiple.items()
                     if key not in ("traces", "installed_actions")},
        "installed_action_sha256": hashlib.sha256(json.dumps(
            single["installed_actions"][0], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
