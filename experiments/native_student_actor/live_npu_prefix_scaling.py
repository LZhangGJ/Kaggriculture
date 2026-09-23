#!/usr/bin/env python3
"""Measure the PrefixBatch -> paused planner -> batched NPU actor seam."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.native_student_actor.benchmark_npu_coordinator import counter_uniform
from experiments.native_student_actor.live_npu_smoke import (
    CHECKPOINT, load_extension, state_tensors,
)
from experiments.student_action_event_agent import EVENT_CLASSES
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_midgame_student_v1 import _sha256


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"
THOMAS_ASSET = (
    ROOT / "experiments/native_opponents/thomas_2945_cpp/thomas_2945.assets.bin")
META_ASSET = ROOT / "experiments/native_opponents/metav4_2965/metav4_2965.assets.bin"
DEPLOYMENT = ROOT / "agent/replay_deployment.json"


def distribution(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()) if array.size else 0,
        "mean": float(array.mean()) if array.size else 0.0,
        "p50": float(np.percentile(array, 50)) if array.size else 0.0,
        "p90": float(np.percentile(array, 90)) if array.size else 0.0,
        "max": int(array.max()) if array.size else 0,
        "histogram": {
            str(int(value)): int((array == value).sum())
            for value in np.unique(array)
        },
    }


def run_group(native, model, checkpoint: dict, device: torch.device,
              manifest: dict, size: int, policy_seed: int,
              prefix_threads: int) -> dict:
    cases = manifest["cases"]
    caches = [cases[index % len(cases)]["cache"] for index in range(size)]
    pipeline_started = time.perf_counter()
    pipeline_cpu_started = time.process_time()
    prefix = native.PrefixBatch(
        manifest["binary"], caches, str(THOMAS_ASSET), str(META_ASSET),
        str(DEPLOYMENT))
    prefix_started = time.perf_counter()
    prefix_cpu_started = time.process_time()
    prefix.run(prefix_threads)
    prefix_seconds = time.perf_counter() - prefix_started
    prefix_cpu_seconds = time.process_time() - prefix_cpu_started

    plan = None
    try:
        observations = list(prefix.observations)
        plan_started = time.perf_counter()
        plan = native.PlanBatch(
            manifest["binary"], prefix.handles, prefix.packed, False)
        contexts = np.asarray(plan.contexts)
        plan_constructor_seconds = time.perf_counter() - plan_started

        pack_started = time.perf_counter()
        state = state_tensors(
            observations, contexts, checkpoint["normalization"])
        python_state_pack_seconds = time.perf_counter() - pack_started
        with torch.inference_mode():
            initial_started = time.perf_counter()
            hidden = model.initial_hidden(*(
                [value.to(device) for value in state[:4]] +
                [[value.to(device) for value in state[4]], state[5].to(device)]))
            previous = torch.full(
                (size,), len(EVENT_CLASSES), dtype=torch.long, device=device)
            torch.npu.synchronize(device)
            npu_initial_seconds = time.perf_counter() - initial_started

            ready_sizes: list[int] = []
            event_counts = np.zeros(size, dtype=np.int64)
            cpp_wait_collect_seconds = 0.0
            python_event_pack_seconds = 0.0
            npu_event_seconds = 0.0
            python_apply_seconds = 0.0
            live_started = time.perf_counter()
            live_cpu_started = time.process_time()
            while True:
                collect_started = time.perf_counter()
                ready = plan.collect_ready(60_000, True)
                cpp_wait_collect_seconds += time.perf_counter() - collect_started
                indices = np.asarray(ready["session_indices"], dtype=np.int32)
                terminal = np.asarray(ready["terminal"], dtype=np.uint8)
                if not indices.size:
                    if bool(terminal.all()):
                        break
                    raise TimeoutError(plan.summary())

                pack_started = time.perf_counter()
                stages = np.asarray(ready["stages"], dtype=np.int64)
                cells = np.asarray(ready["cells"], dtype=np.int64)
                masks = np.asarray(ready["legal_masks"], dtype=np.int64)
                sequences = np.asarray(ready["sequences"], dtype=np.uint64)
                resources = np.asarray(ready["resources"], dtype=np.float32)
                legal = ((masks[:, None] &
                          (1 << np.arange(len(EVENT_CLASSES)))) != 0)
                resource = ((torch.from_numpy(resources) - torch.as_tensor(
                    checkpoint["normalization"]["resource_mean"])) /
                    torch.as_tensor(
                        checkpoint["normalization"]["resource_std"]))
                active = torch.from_numpy(indices.astype(np.int64)).to(device)
                python_event_pack_seconds += time.perf_counter() - pack_started

                inference_started = time.perf_counter()
                logits, next_hidden = model.step(
                    hidden.index_select(0, active), resource.to(device),
                    torch.from_numpy(cells).to(device),
                    torch.from_numpy(stages).to(device),
                    previous.index_select(0, active),
                    torch.from_numpy(legal).to(device))
                probabilities = torch.softmax(logits.float(), dim=1)
                cdf = probabilities.cumsum(1)
                cdf[:, -1] = 1.0
                uniforms = torch.tensor([
                    counter_uniform(policy_seed + int(index), int(sequence))
                    for index, sequence in zip(indices, sequences)
                ], dtype=torch.float32, device=device)
                actions = (cdf < uniforms[:, None]).sum(1).clamp_max(
                    len(EVENT_CLASSES) - 1)
                hidden.index_copy_(0, active, next_hidden)
                previous.index_copy_(0, active, actions)
                host_actions = actions.cpu().numpy().astype(np.int32)
                npu_event_seconds += time.perf_counter() - inference_started

                apply_started = time.perf_counter()
                for row, index in enumerate(indices):
                    if int(sequences[row]) != event_counts[int(index)]:
                        raise RuntimeError("native event sequence is not contiguous")
                    event_counts[int(index)] += 1
                plan.apply(indices, host_actions)
                python_apply_seconds += time.perf_counter() - apply_started
                ready_sizes.append(int(indices.size))

            plan.join()
            live_plan_seconds = time.perf_counter() - live_started
            live_plan_cpu_seconds = time.process_time() - live_cpu_started
        summary = plan.summary()
        if summary["states"] != ["done"] * size:
            raise RuntimeError(summary)
        events = int(event_counts.sum())
        pipeline_seconds = time.perf_counter() - pipeline_started
        pipeline_cpu_seconds = time.process_time() - pipeline_cpu_started
        return {
            "sessions": size,
            "events": events,
            "event_counts": event_counts.tolist(),
            "callback_rounds": len(ready_sizes),
            "ready_batch_distribution": distribution(ready_sizes),
            "prefix_seconds": prefix_seconds,
            "prefix_cpu_seconds": prefix_cpu_seconds,
            "plan_constructor_seconds": plan_constructor_seconds,
            "python_state_pack_seconds": python_state_pack_seconds,
            "npu_initial_seconds": npu_initial_seconds,
            "cpp_planner_wait_advance_seconds": cpp_wait_collect_seconds,
            "python_event_pack_seconds": python_event_pack_seconds,
            "npu_event_forward_seconds": npu_event_seconds,
            "python_apply_seconds": python_apply_seconds,
            "live_plan_seconds": live_plan_seconds,
            "live_plan_cpu_seconds": live_plan_cpu_seconds,
            "pipeline_seconds": pipeline_seconds,
            "pipeline_cpu_seconds": pipeline_cpu_seconds,
            "events_per_second_live": events / live_plan_seconds,
            "events_per_second_including_prefix": events / pipeline_seconds,
            "planner_threads": size,
            "prefix_workers_requested": prefix_threads,
            "summary": summary,
        }
    finally:
        if plan is not None:
            plan.close()
        # PrefixBatch owns the planner handles and native environments. Keep it
        # alive until every paused planner thread has joined.
        del prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batches", default="16,64,128,256")
    parser.add_argument("--policy-seed", type=int, default=2026092301)
    parser.add_argument("--prefix-threads", type=int, default=0)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/live-npu-prefix-scaling.json"))
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if (not batches or min(batches) < 1 or max(batches) > 256 or
            len(set(batches)) != len(batches) or args.output.exists()):
        parser.error("batches must be unique values in [1,256]; output must be new")
    if not args.device.startswith("npu"):
        parser.error("live scaling requires npu[:index]")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for path in (args.checkpoint, Path(manifest["binary"]), THOMAS_ASSET,
                 META_ASSET, DEPLOYMENT):
        if not path.is_file():
            raise FileNotFoundError(path)

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

    # Compile/warm the two actor methods outside every reported scale point.
    warmup = run_group(native, model, checkpoint, device, manifest, 4,
                       args.policy_seed, min(4, args.prefix_threads or 4))
    rows = []
    for batch_size in batches:
        row = run_group(
            native, model, checkpoint, device, manifest, batch_size,
            args.policy_seed, args.prefix_threads)
        denominator = row["live_plan_seconds"]
        row["live_timing_fraction"] = {
            "cpp_planner_wait_advance":
                row["cpp_planner_wait_advance_seconds"] / denominator,
            "python_batch_pack_apply": (
                row["python_event_pack_seconds"] +
                row["python_apply_seconds"]) / denominator,
            "npu_event_forward":
                row["npu_event_forward_seconds"] / denominator,
        }
        rows.append(row)
        print(json.dumps({"event": "live_prefix_scaling", **row}), flush=True)

    result = {
        "status": "PASS",
        "scope": "native-prefix-paused-planner-batched-npu-live-scaling",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "planner_binary": manifest["binary"],
        "planner_binary_sha256": manifest["binary_sha256"],
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "rng": "splitmix64(per_session_policy_seed,native_event_sequence)",
        "one_pthread_per_planner_session": True,
        "max_sessions": native.MAX_SESSIONS,
        "warmup": {
            key: warmup[key] for key in (
                "sessions", "events", "prefix_seconds", "live_plan_seconds",
                "npu_initial_seconds", "npu_event_forward_seconds")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
