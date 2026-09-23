#!/usr/bin/env python3
"""Deterministic one/many-session smoke for the paused native planner seam."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.audit_teacher_batch import r1_module
from scripts.extract_midgame_bc import observation


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def default_trajectory() -> Path:
    paths = [path for path in sorted((ROOT / "data/bc/midgame-v1").glob(
        "**/*-seat*.jsonl.gz"))
             if not any(part.startswith("INTERRUPTED") for part in path.parts)]
    if not paths:
        raise FileNotFoundError("no complete BC trajectory found")
    return paths[0]


def load_extension():
    sys.path.insert(0, str(HERE / "build"))
    return importlib.import_module("_paused_plan")


def restore(binary: Path, trajectory: Path):
    module = r1_module()
    agent = module.Agent(binary_path=binary)
    current = None
    seat = None
    with gzip.open(trajectory, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("type") == "meta":
                seat = int(row["seat"])
                continue
            if row.get("type") != "step":
                continue
            if seat is None:
                raise RuntimeError("trajectory meta must precede steps")
            current = observation(row, seat)
            step = int(row["step"])
            if step == 288:
                break
            agent.observe_external(current, row["actions"][seat])
    if current is None or int(current["day"]) * 24 + int(current["hour"]) != 288:
        agent.close()
        raise RuntimeError("trajectory does not contain step 288")
    if agent.last != 287 or not agent.external:
        agent.close()
        raise RuntimeError("R1 handle did not warm consecutively through step 287")
    packed = np.asarray(list(module._pack(current)), dtype=np.float64)
    return agent, packed


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run_group(native, binary: Path, trajectory: Path, size: int,
              stack_bytes: int) -> dict:
    agents = []
    batch = None
    started = time.perf_counter()
    try:
        packed = []
        for _ in range(size):
            agent, values = restore(binary, trajectory)
            agents.append(agent)
            packed.append(values)
        batch = native.PlanBatch(
            str(binary), [int(agent.handle) for agent in agents], packed,
            True, stack_bytes)
        contexts = np.asarray(batch.contexts)
        if contexts.shape != (size, native.CONTEXT_WIDTH):
            raise AssertionError(f"context shape {contexts.shape}")
        traces: list[list[tuple]] = [[] for _ in range(size)]
        rounds = 0
        while True:
            ready = batch.collect_ready(60_000)
            indices = np.asarray(ready["session_indices"], dtype=np.int32)
            stages = np.asarray(ready["stages"], dtype=np.int32)
            cells = np.asarray(ready["cells"], dtype=np.int32)
            suggested = np.asarray(ready["suggested"], dtype=np.int32)
            masks = np.asarray(ready["legal_masks"], dtype=np.int32)
            sequences = np.asarray(ready["sequences"], dtype=np.uint64)
            resources = np.asarray(ready["resources"], dtype=np.float64)
            terminal = np.asarray(ready["terminal"], dtype=np.uint8)
            if resources.shape != (indices.size, native.RESOURCE_WIDTH):
                raise AssertionError(f"resource shape {resources.shape}")
            if indices.size == 0:
                if bool(terminal.all()):
                    break
                raise TimeoutError(f"no ready event: {batch.summary()}")
            for row, index in enumerate(indices):
                index = int(index)
                action = int(suggested[row])
                mask = int(masks[row])
                if int(sequences[row]) != len(traces[index]):
                    raise AssertionError("per-session sequence is not contiguous")
                if not mask & (1 << action):
                    raise AssertionError("Oracle suggestion is illegal")
                traces[index].append((
                    int(stages[row]), int(cells[row]), action, mask,
                    digest(resources[row]),
                ))
            batch.apply(indices, suggested)
            rounds += 1
        batch.join()
        summary = batch.summary()
        if summary["states"] != ["done"] * size:
            raise AssertionError(summary)
        if summary["event_counts"] != [len(trace) for trace in traces]:
            raise AssertionError("callback result count disagrees with trace")
        return {
            "sessions": size,
            "rounds": rounds,
            "wall_seconds": time.perf_counter() - started,
            "stack_bytes": stack_bytes,
            "context_sha256": [digest(row) for row in contexts],
            "event_counts": [len(trace) for trace in traces],
            "traces": traces,
            "summary": summary,
        }
    finally:
        if batch is not None:
            batch.close()
        for agent in agents:
            agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stack-bytes", type=int, default=2 << 20)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/paused-plan-smoke.json"))
    args = parser.parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2")
    trajectory = args.trajectory or default_trajectory()
    native = load_extension()
    single = run_group(
        native, args.binary.resolve(), trajectory.resolve(), 1,
        args.stack_bytes)
    multiple = run_group(
        native, args.binary.resolve(), trajectory.resolve(), args.batch_size,
        args.stack_bytes)
    reference = single["traces"][0]
    deterministic = all(trace == reference for trace in multiple["traces"])
    contexts_equal = (
        len(set(multiple["context_sha256"])) == 1 and
        multiple["context_sha256"][0] == single["context_sha256"][0])
    result = {
        "status": "PASS" if deterministic and contexts_equal else "FAIL",
        "scope": "paused-native-plan-session-oracle",
        "binary": str(args.binary.resolve()),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "trajectory": str(trajectory.resolve()),
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "deterministic_event_traces": deterministic,
        "deterministic_contexts": contexts_equal,
        "single": {key: value for key, value in single.items() if key != "traces"},
        "multiple": {key: value for key, value in multiple.items() if key != "traces"},
        "trace_sha256": hashlib.sha256(json.dumps(
            reference, separators=(",", ":")).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
