#!/usr/bin/env python3
"""Scale the real paused-planner/NPU seam up to its 256-session ABI cap."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

from experiments.native_student_actor.live_npu_smoke import (
    BINARY, CHECKPOINT, default_trajectory, load_extension, run_group,
)
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_midgame_student_v1 import _sha256


ROOT = Path(__file__).resolve().parents[2]


def replicated_parity(row: dict) -> dict:
    reference = row["traces"][0]
    metadata_action_mismatches = length_mismatches = 0
    max_logprob = max_entropy = 0.0
    for trace in row["traces"][1:]:
        length_mismatches += abs(len(trace) - len(reference))
        for left, right in zip(reference, trace):
            metadata_action_mismatches += left[:4] != right[:4]
            if left[:4] == right[:4]:
                max_logprob = max(max_logprob, abs(left[4] - right[4]))
                max_entropy = max(max_entropy, abs(left[5] - right[5]))
    return {
        "metadata_action_mismatches": metadata_action_mismatches,
        "length_mismatches": length_mismatches,
        "max_logprob_abs_on_aligned_events": max_logprob,
        "max_entropy_abs_on_aligned_events": max_entropy,
        "actions_equal": not metadata_action_mismatches and not length_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batches", default="16,64,128,256")
    parser.add_argument("--policy-seed", type=int, default=2026092301)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/live-npu-scaling.json"))
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if (not batches or min(batches) < 1 or max(batches) > 256 or
            len(set(batches)) != len(batches) or args.output.exists()):
        parser.error("batches must be unique values in [1,256]; output must be new")
    trajectory = args.trajectory or default_trajectory()
    for path in (args.checkpoint, args.binary, trajectory):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.device.startswith("npu"):
        parser.error("scaling run requires npu[:index]")
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

    # Exclude one-time NPU graph/kernel startup from the requested scale points.
    warmup = run_group(
        native, model, checkpoint, device, args.binary.resolve(),
        trajectory.resolve(), 1, args.policy_seed, verify_installed=False)
    rows = []
    started = time.perf_counter()
    for batch_size in batches:
        row = run_group(
            native, model, checkpoint, device, args.binary.resolve(),
            trajectory.resolve(), batch_size, args.policy_seed,
            verify_installed=False)
        row["replicated_parity"] = replicated_parity(row)
        row["events_per_second_including_handle_warm"] = (
            row["events"] / row["wall_seconds"])
        row["timing_fraction_of_live_plan"] = {
            "cpp_wait_collect": row["cpp_wait_collect_seconds"] /
                                row["live_plan_seconds"],
            "coalesce": row["coalesce_seconds"] / row["live_plan_seconds"],
            "python_event_pack_apply": (
                row["python_event_pack_seconds"] + row["python_apply_seconds"]
            ) / row["live_plan_seconds"],
            "npu_event": row["npu_event_seconds"] / row["live_plan_seconds"],
        }
        row.pop("traces")
        row.pop("installed_actions")
        rows.append(row)
        print(json.dumps({"event": "live_scaling", **row}), flush=True)

    result = {
        "status": "PASS", "scope": "real-paused-planner-npu-live-scaling",
        "checkpoint_sha256": _sha256(args.checkpoint),
        "binary_sha256": _sha256(args.binary), "device": str(device),
        "trajectory": str(trajectory.resolve()),
        "trajectory_sha256": _sha256(trajectory),
        "policy_seed": args.policy_seed,
        "one_pthread_per_session": True, "max_sessions": native.MAX_SESSIONS,
        "warmup": {
            "sessions": 1, "events": warmup["events"],
            "wall_seconds": warmup["wall_seconds"],
            "npu_initial_seconds": warmup["npu_initial_seconds"],
            "npu_event_seconds": warmup["npu_event_seconds"],
        },
        "rows": rows, "scaling_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": result["status"], "output": str(args.output),
        "scaling_wall_seconds": result["scaling_wall_seconds"],
    }))


if __name__ == "__main__":
    main()
