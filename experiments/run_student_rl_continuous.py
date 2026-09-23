#!/usr/bin/env python3
"""Continuously run accepted native JobBatch PPO rounds."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "work/student-v1"
HEARTBEAT_SECONDS = 5.0


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _paths(round_number: int) -> tuple[str, Path, Path, Path, Path]:
    representative = "meta" if round_number % 2 else "thomas"
    stem = f"v3-ppo-native-job-v{round_number}-{representative}-fieldcraft-critic-1536g"
    return (
        representative,
        STUDENT / f"{stem}.pt",
        STUDENT / f"{stem}.rollout.npz",
        STUDENT / f"{stem}.metrics.json",
        STUDENT / f"native-v{round_number}-{representative}-fieldcraft-critic-1536g.bin",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-round", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=(
        STUDENT / "action-event-v3-actor-owned-width3074-full/manifest.json"))
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--native-job-threads", type=int, default=192)
    parser.add_argument("--rounds", type=int, default=0,
                        help="0 means continue until the stop file appears")
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.start_round < 1 or args.seed_start < 0 or args.rounds < 0 or
            args.native_job_threads < 1 or not args.checkpoint.is_file() or
            not args.weights.is_file() or not args.manifest.is_file() or
            not args.binary.is_file()):
        parser.error("invalid round, seed, checkpoint, weights, manifest, binary, or threads")

    runtime = ROOT / "work/continuous-rl"
    runtime.mkdir(parents=True, exist_ok=True)
    stop_file = runtime / "stop"
    state_file = runtime / "state.json"
    lock = (runtime / "driver.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another continuous RL driver is already running") from error

    child: subprocess.Popen | None = None
    stop_signal: int | None = None
    state: dict = {}

    def write_state(status: str | None = None, stage: str | None = None,
                    **values: object) -> None:
        if status is not None:
            state["status"] = status
        if stage is not None:
            state["stage"] = stage
        state.update(values)
        state["driver_pid"] = os.getpid()
        state["child_pid"] = (
            child.pid if child is not None and child.poll() is None else None)
        state["heartbeat_at"] = time.time()
        _write_json(state_file, state)

    def stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        stop_signal = signum
        state.update(status="stopping", signal=signum)
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    def run_child(command: list[str], stage: str) -> int:
        nonlocal child
        write_state("training" if stage == "train" else "exporting", stage)
        child = subprocess.Popen(command, cwd=ROOT)
        if stop_signal is not None:
            try:
                child.send_signal(stop_signal)
            except ProcessLookupError:
                pass
        write_state()
        while True:
            try:
                return_code = child.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                write_state()
        child = None
        write_state(child_exit_code=return_code)
        if stop_signal is None and return_code:
            raise subprocess.CalledProcessError(return_code, command)
        return return_code

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    checkpoint, weights = args.checkpoint.resolve(), args.weights.resolve()
    try:
        for offset in range(args.rounds or sys.maxsize):
            if stop_signal is not None or stop_file.exists():
                write_state("stopped", "idle", next_round=args.start_round + offset)
                return
            round_number = args.start_round + offset
            seed_start = args.seed_start + offset * 100_000
            representative, output, rollout, metrics, frozen = _paths(round_number)
            opponents = ("metav4_2965,fieldcraft_2887" if representative == "meta"
                         else "thomas_2945_cpp,fieldcraft_2887")
            state = {
                "status": "starting", "stage": "starting", "round": round_number,
                "seed_start": seed_start, "opponents": opponents.split(","),
            }
            write_state()
            if any(path.exists() for path in (output, metrics, frozen)):
                raise FileExistsError(f"round {round_number} output already exists")
            train = [
                sys.executable, str(ROOT / "experiments/train_student_action_event_rl_v3.py"),
                "--checkpoint", str(checkpoint),
                "--manifest", str(args.manifest.resolve()),
                "--binary", str(args.binary.resolve()),
                "--output", str(output), "--rollout-output", str(rollout),
                "--metrics-output", str(metrics), "--opponents", opponents,
                "--games", "1536", "--seed-start", str(seed_start),
                "--policy-seed", str(2026092300000 + round_number * 1000),
                "--epochs", "1", "--batch-games", "32", "--replay-batch-games", "256",
                "--learning-rate", "1.8e-5", "--native-job-rollout",
                "--native-weights", str(weights), "--native-job-threads",
                str(args.native_job_threads),
                "--native-logprob-max-tolerance", "5e-4",
                "--day-state-crossfit-baseline", "--day-state-critic-workers", "6",
                "--train-threads", "8", "--device", args.device,
            ]
            if rollout.exists():
                train.append("--resume-rollout")
            export = [
                sys.executable, str(ROOT / "experiments/native_student_actor/export_frozen_binary.py"),
                "--checkpoint", str(output), "--output", str(frozen),
            ]
            state.update(
                status="dry-run" if args.dry_run else "training",
                checkpoint_in=str(checkpoint), weights_in=str(weights),
                train_command=train, export_command=export)
            write_state()
            if args.dry_run:
                print(json.dumps(json.loads(state_file.read_text()), indent=2))
                return
            train_exit_code = run_child(train, "train")
            if stop_signal is not None:
                write_state("stopped", "idle", exit_code=train_exit_code)
                return
            write_state("training", "validate")
            result = json.loads(metrics.read_text(encoding="utf-8"))
            if (result.get("status") != "PASS" or result.get("games") != 1536 or
                    result.get("illegal") != 0 or result.get("fallbacks") != 0):
                raise RuntimeError(f"round {round_number} failed its formal gates")
            export_exit_code = run_child(export, "export")
            if stop_signal is not None:
                write_state("stopped", "idle", exit_code=export_exit_code)
                return
            checkpoint, weights = output.resolve(), frozen.resolve()
            state = {
                "status": "round-complete", "stage": "between-rounds",
                "round": round_number, "seed_start": seed_start,
                "checkpoint": str(checkpoint), "weights": str(weights),
                "next_round": round_number + 1,
                "next_seed_start": seed_start + 100_000,
            }
            write_state()
    except Exception as error:
        exit_code = error.returncode if isinstance(
            error, subprocess.CalledProcessError) else 1
        message = f"{type(error).__name__}: {error}"
        write_state("failed", "failed", error=message, last_error=message,
                    exit_code=exit_code)
        raise


if __name__ == "__main__":
    main()
