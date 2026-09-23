#!/usr/bin/env python3
"""Offline proof/benchmark for one NPU serving variable-length actor sessions."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import EVENT_CLASSES, build_model
from experiments.train_midgame_student_v1 import _sha256


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT / "work/student-v1/action-event-v3-actor-owned-dagger-r3-scale3-e20.pt")
ROLLOUT = ROOT / "work/student-v1/v3-ppo-native-strong-r3-192g-v0.rollout.pt"
MASK64 = (1 << 64) - 1


def counter_uniform(policy_seed: int, event_index: int) -> float:
    """SplitMix64 keyed only by (game policy seed, global event index)."""
    value = ((int(policy_seed) & MASK64) +
             0x9E3779B97F4A7C15 * (int(event_index) + 1)) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return ((value >> 11) + 0.5) / float(1 << 53)


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def load_sessions(path: Path, checkpoint_sha: str) -> list[dict]:
    rollout = torch.load(path, map_location="cpu", weights_only=False)
    if rollout.get("policy_version") != checkpoint_sha:
        raise ValueError("rollout was not captured by the requested checkpoint")
    sessions = []
    for game in rollout["games"]:
        event_base = 0
        for day in game["days"]:
            sessions.append({
                "state": day["state"], "events": day["events"],
                "policy_seed": int(game["policy_seed"]),
                "event_base": event_base,
            })
            event_base += len(day["events"])
    return sessions


class BatchedActorCoordinator:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 temperature: float = 1.0):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature

    def _state(self, sessions: list[dict], name: str, dtype: torch.dtype):
        return torch.from_numpy(np.concatenate(
            [session["state"][name] for session in sessions], axis=0)).to(
                device=self.device, dtype=dtype)

    def run(self, sessions: list[dict]) -> dict:
        started = time.perf_counter()
        categories = [torch.from_numpy(np.concatenate([
            session["state"]["token_categories"][index]
            for session in sessions], axis=0)).to(self.device, torch.long)
                      for index in range(7)]
        with torch.inference_mode():
            hidden = self.model.initial_hidden(
                self._state(sessions, "context", torch.float32),
                self._state(sessions, "observation", torch.float32),
                self._state(sessions, "observation_length", torch.float32),
                self._state(sessions, "token_continuous", torch.float32),
                categories, self._state(sessions, "token_count", torch.long))
            previous = torch.full(
                (len(sessions),), len(EVENT_CLASSES), dtype=torch.long,
                device=self.device)
            synchronize(self.device)
            initial_seconds = time.perf_counter() - started
            lengths = np.asarray([len(session["events"]) for session in sessions])
            records = [[] for _ in sessions]
            event_started = time.perf_counter()
            for event_index in range(int(lengths.max(initial=0))):
                active_np = np.flatnonzero(event_index < lengths)
                active = torch.as_tensor(
                    active_np, device=self.device, dtype=torch.long)
                rows = [sessions[index]["events"][event_index]
                        for index in active_np]
                resources = torch.from_numpy(np.concatenate(
                    [row["resources"] for row in rows], axis=0)).to(
                        self.device, torch.float32)
                cells = torch.tensor([row["cell"] for row in rows],
                                     device=self.device, dtype=torch.long)
                stages = torch.tensor([row["stage"] for row in rows],
                                      device=self.device, dtype=torch.long)
                legal = torch.from_numpy(np.concatenate(
                    [row["legal"] for row in rows], axis=0)).to(
                        self.device, torch.bool)
                logits, next_hidden = self.model.step(
                    hidden.index_select(0, active), resources, cells, stages,
                    previous.index_select(0, active), legal)
                log_probs = torch.log_softmax(
                    logits.float() / self.temperature, dim=1)
                probabilities = log_probs.exp()
                cdf = probabilities.cumsum(1)
                cdf[:, -1] = 1.0
                uniforms = torch.tensor([
                    counter_uniform(sessions[index]["policy_seed"],
                                    sessions[index]["event_base"] + event_index)
                    for index in active_np
                ], device=self.device, dtype=torch.float32)
                actions = (cdf < uniforms[:, None]).sum(1).clamp_max(
                    len(EVENT_CLASSES) - 1)
                selected_log_probs = log_probs.gather(1, actions[:, None])[:, 0]
                entropy = -(probabilities * log_probs).sum(1)
                if not bool(legal.gather(1, actions[:, None]).all().item()):
                    raise RuntimeError("counter sampler selected an illegal action")
                hidden.index_copy_(0, active, next_hidden)
                previous.index_copy_(0, active, actions)
                host_actions = actions.cpu().tolist()
                host_log_probs = selected_log_probs.cpu().tolist()
                host_entropy = entropy.cpu().tolist()
                for local, session_index in enumerate(active_np):
                    records[session_index].append((
                        int(host_actions[local]), float(host_log_probs[local]),
                        float(host_entropy[local])))
            synchronize(self.device)
        event_seconds = time.perf_counter() - event_started
        return {
            "records": records, "initial_seconds": initial_seconds,
            "event_seconds": event_seconds,
            "total_seconds": initial_seconds + event_seconds,
            "events": int(lengths.sum()), "rounds": int(lengths.max(initial=0)),
        }


def parity(coordinator: BatchedActorCoordinator, sessions: list[dict],
           tolerance: float) -> dict:
    singles = [coordinator.run([session])["records"][0] for session in sessions]
    batched = coordinator.run(sessions)["records"]
    action_mismatches = 0
    max_logprob = max_entropy = 0.0
    for expected, actual in zip(singles, batched):
        if len(expected) != len(actual):
            raise RuntimeError("batched coordinator changed the event count")
        for left, right in zip(expected, actual):
            action_mismatches += left[0] != right[0]
            max_logprob = max(max_logprob, abs(left[1] - right[1]))
            max_entropy = max(max_entropy, abs(left[2] - right[2]))
    result = {
        "status": "PASS" if (not action_mismatches and
                               max_logprob <= tolerance and
                               max_entropy <= tolerance) else "FAIL",
        "sessions": len(sessions),
        "events": sum(map(len, batched)),
        "action_mismatches": action_mismatches,
        "max_logprob_abs": max_logprob, "max_entropy_abs": max_entropy,
        "tolerance": tolerance,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"sequential/batched parity failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--rollout", type=Path, default=ROLLOUT)
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batches", default="192,512,1024")
    parser.add_argument("--parity-sessions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2,
                        help="first run warms each batch shape; remaining runs are measured")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if (not batches or min(batches) <= 0 or args.parity_sessions <= 1 or
            args.repeats < 2 or not math.isfinite(args.tolerance) or
            args.tolerance <= 0):
        parser.error("invalid batch/parity/repeat/tolerance setting")
    for path in (args.checkpoint, args.rollout):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("NPU is unavailable")
        torch.npu.set_device(args.device)
    elif args.device != "cpu":
        parser.error("device must be cpu or npu[:index]")
    device = torch.device(args.device)
    checkpoint_sha = _sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dimensions = checkpoint["model_dimensions"]
    model = build_model(
        dimensions["causal_context"], dimensions["packed_observation"],
        dimensions["event_resources"], checkpoint["model_scale"])
    model.load_state_dict(checkpoint["model"])
    coordinator = BatchedActorCoordinator(model, device)
    sessions = load_sessions(args.rollout, checkpoint_sha)
    if max(batches) > len(sessions) or args.parity_sessions > len(sessions):
        raise ValueError("not enough captured day sessions")

    parity_result = parity(coordinator, sessions[:args.parity_sessions],
                           args.tolerance)
    benchmarks = []
    for batch in batches:
        rows = sessions[:batch]
        if device.type == "npu":
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats(device)
        try:
            coordinator.run(rows)  # warm exact active-batch shape sequence
            measured = [coordinator.run(rows) for _ in range(args.repeats - 1)]
            total = statistics.median(row["total_seconds"] for row in measured)
            event = statistics.median(row["event_seconds"] for row in measured)
            events = measured[0]["events"]
            benchmarks.append({
                "status": "PASS", "oom": False,
                "batch": batch, "events": events,
                "rounds": measured[0]["rounds"],
                "initial_seconds_median": statistics.median(
                    row["initial_seconds"] for row in measured),
                "event_seconds_median": event, "total_seconds_median": total,
                "events_per_second_event_loop": events / event,
                "events_per_second_including_initial": events / total,
                "peak_memory_allocated_bytes": (
                    int(torch.npu.max_memory_allocated(device))
                    if device.type == "npu" else None),
                "peak_memory_reserved_bytes": (
                    int(torch.npu.max_memory_reserved(device))
                    if device.type == "npu" else None),
            })
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            benchmarks.append({
                "status": "OOM", "oom": True, "batch": batch,
                "error": str(error),
                "peak_memory_allocated_bytes": int(
                    torch.npu.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(
                    torch.npu.max_memory_reserved(device)),
            })
            torch.npu.empty_cache()
        print(json.dumps({"event": "benchmark", **benchmarks[-1]}), flush=True)
    result = {
        "status": "PASS", "device": str(device),
        "checkpoint_sha256": checkpoint_sha,
        "rollout_policy_version": checkpoint_sha,
        "rng": "splitmix64(policy_seed,global_event_index); RNG is post-forward",
        "event_source": "captured rows; offline fixture does not mutate resources",
        "parity": parity_result, "benchmarks": benchmarks,
    }
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.output)
        result["output"] = str(args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
