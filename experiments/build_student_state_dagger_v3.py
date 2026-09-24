#!/usr/bin/env python3
"""Build a fail-closed mmap shard from live student-state DAgger labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from experiments import build_midgame_action_events_v3 as contract
from experiments.collect_student_state_dagger_v3 import (
    ROOT, _load, _observations, create_agent,
)
from experiments.train_midgame_student_v1 import _json_hash, _save_array, _sha256


STATE_ARRAYS = (
    "causal_context", "packed_observation", "observation_length",
    "token_continuous", "token_type", "token_category_a", "token_category_b",
    "token_category_c", "token_x", "token_y", "token_owner", "token_count",
    "state_step",
)
EVENT_ARRAYS = (
    "event_stage", "event_cell", "event_resources", "event_legal_mask",
    "event_label_class",
)
REQUIRED_GATES = (
    "distinct_clone_handles", "original_handle_observable_state_unchanged",
    "resource_replay_exact", "label_fixed_point", "stop_terminal_exact",
    "controlled_job_projection", "actor_owned_placement",
    "event_mask_label_legal",
)


def _worker_init():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def _parse_game(value: str) -> dict:
    try:
        seed, opponent, seat = value.split(":")
        result = {"seed": int(seed), "opponent": opponent, "seat": int(seat)}
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "game must be SEED:OPPONENT:SEAT") from error
    if not opponent or result["seat"] not in (0, 1):
        raise argparse.ArgumentTypeError("game seat must be 0 or 1")
    return result


def _fingerprints(checkpoint: Path, manifest: Path, binary: Path) -> dict:
    checkpoint = checkpoint.resolve()
    manifest = manifest.resolve()
    binary = binary.resolve()
    values = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "manifest": str(manifest), "manifest_sha256": _sha256(manifest),
        "binary": str(binary), "binary_sha256": _sha256(binary),
    }
    payload = json.loads(manifest.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    semantic = payload.get("validation", {}).get("semantic_smoke", {})
    if (saved.get("shard_manifest_sha256") != values["manifest_sha256"] or
            semantic.get("binary_sha256") != values["binary_sha256"] or
            tuple(saved.get("event_classes", ())) != tuple(contract.CLASS_NAMES) or
            set(saved.get("model_inputs", ())) != set(contract.MODEL_INPUTS)):
        raise ValueError("rollout checkpoint/manifest/binary fingerprint contract failed")
    if (payload.get("validation_status") != "accepted" or
            payload.get("validation", {}).get("execution_contract") != "PASS" or
            payload.get("validation", {}).get("actor_owned_placement") != "PASS" or
            payload.get("validation", {}).get("packed_observation_capacity") != "PASS"):
        raise ValueError("rollout manifest is not accepted")
    dimensions = saved.get("model_dimensions", {})
    expected_dimensions = {
        "causal_context": 2233,
        "packed_observation": contract.PACKED_OBSERVATION_CAPACITY,
        "event_resources": 347,
    }
    if (dimensions != expected_dimensions or
            {name: payload.get("dimensions", {}).get(name)
             for name in expected_dimensions} != expected_dimensions):
        raise ValueError(f"unexpected rollout model dimensions: {dimensions}")
    if (len(payload.get("causal_context_names", ())) != 2233 or
            len(set(payload["causal_context_names"])) != 2233 or
            len(payload.get("event_resource_names", ())) != 347 or
            len(set(payload["event_resource_names"])) != 347):
        raise ValueError("rollout manifest feature names are incomplete/non-unique")
    trained_steps = tuple(sorted(set(map(
        int, saved.get("training_state_steps", (288,))))))
    if (not trained_steps or any(
            step < 288 or step >= 719 or step % 24 for step in trained_steps)):
        raise ValueError("checkpoint training_state_steps are invalid")
    values["checkpoint_training_state_steps"] = list(trained_steps)
    values["source_manifest"] = payload
    return values


def _validate_student_rollout(summary: dict, expected_steps,
                              attested_steps, exploratory_steps) -> None:
    expected = list(map(int, expected_steps))
    attested = list(map(int, attested_steps))
    exploratory = list(map(int, exploratory_steps))
    if (summary.get("student_steps") != expected or
            summary.get("successful_steps") != expected or
            summary.get("days") != len(expected) or
            summary.get("checkpoint_attested_action_steps") != attested or
            summary.get("unattested_exploration_steps") != exploratory or
            summary.get("fallbacks") != 0 or summary.get("failures") != [] or
            summary.get("illegal") != 0):
        raise RuntimeError(f"invalid student rollout summary: {summary}")


def collect_one(job: dict) -> dict:
    _worker_init()
    from fast_kaggriculture import Config, FastEnv

    started = time.perf_counter()
    candidate = None
    try:
        opponent_path = ROOT / "opponents" / job["opponent"] / "main.py"
        if not opponent_path.is_file():
            raise FileNotFoundError(opponent_path)
        opponent = _load(
            opponent_path,
            f'dagger_batch_{job["opponent"]}_{job["seed"]}_{job["seat"]}').agent
        import inspect
        takes_configuration = len(inspect.signature(opponent).parameters) > 1
        candidate = create_agent(
            job["seat"], binary_path=Path(job["binary"]),
            checkpoint_path=Path(job["checkpoint"]),
            manifest_path=Path(job["manifest"]), dagger_steps=job["steps"],
            student_steps=job["student_steps"],
            allow_unattested_steps=job["allow_unattested_steps"])
        env = FastEnv(Config(), job["seed"])
        state = list(env.reset(job["seed"]))
        frames = 0
        while frames <= job["through_step"]:
            actions = []
            for player, observation in enumerate(_observations(state)):
                if player == job["seat"]:
                    action = candidate(observation, {})
                else:
                    action = (opponent(observation, {}) if takes_configuration
                              else opponent(observation))
                actions.append(action)
            state = env.step(actions)
            frames += 1
        dynamic = candidate.dynamic
        if (len(dynamic.dagger_audits) != len(job["steps"]) or
                len(dynamic.dagger_samples) != len(job["steps"]) or
                dynamic.dagger_failures):
            raise RuntimeError(
                f"DAgger collection failed: {dynamic.dagger_failures}")
        if [audit["step"] for audit in dynamic.dagger_audits] != list(job["steps"]):
            raise RuntimeError("DAgger collection steps do not match the request")
        for audit in dynamic.dagger_audits:
            if (audit.get("status") != "PASS" or
                    any(audit.get("gates", {}).get(name) is not True
                        for name in REQUIRED_GATES)):
                raise RuntimeError(f"DAgger state gate failed: {audit}")
        student_rollout = dynamic.student_summary()
        if dynamic.student_failures:
            raise RuntimeError(
                f"student rollout used fallback: {dynamic.student_failures}")
        _validate_student_rollout(
            student_rollout, job["student_steps"],
            job["attested_student_steps"], job["exploratory_student_steps"])
        student_days = {int(day["step"]): day["events"]
                        for day in dynamic.student_days}
        for sample, audit in zip(dynamic.dagger_samples, dynamic.dagger_audits):
            teacher_trace = list(zip(
                map(int, sample["event_stage"]),
                map(int, sample["event_cell"]),
                map(int, sample["event_label_class"])))
            student_trace = [
                (int(event["stage"]), int(event["cell"]),
                 int(event["selected_class"]))
                for event in student_days[audit["step"]]
            ]
            prefix = 0
            for teacher_event, student_event in zip(teacher_trace, student_trace):
                if teacher_event != student_event:
                    break
                prefix += 1
            audit["student_teacher_trace_exact"] = student_trace == teacher_trace
            audit["student_teacher_common_prefix"] = prefix
            audit["student_events"] = len(student_trace)
        return {
            "status": "PASS", "seed": job["seed"],
            "opponent": job["opponent"], "seat": job["seat"],
            "steps": job["steps"], "through_step": job["through_step"],
            "frames": frames, "samples": dynamic.dagger_samples,
            "audits": dynamic.dagger_audits,
            "student_rollout": {
                key: value for key, value in student_rollout.items()
                if key != "event_trace"
            },
            "fingerprints": {
                name: job[name] for name in (
                    "checkpoint_sha256", "manifest_sha256", "binary_sha256")
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "status": "FAIL", "seed": job["seed"],
            "opponent": job["opponent"], "seat": job["seat"],
            "steps": job["steps"], "through_step": job["through_step"],
            "error": repr(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if candidate is not None:
            candidate.close()


def _merge(results: list[dict]):
    samples = [
        (row, arrays, audit)
        for row in results
        for arrays, audit in zip(row["samples"], row["audits"])
    ]
    state_count = len(samples)
    event_counts = [len(arrays["event_cell"]) for _row, arrays, _audit in samples]
    event_count = sum(event_counts)
    arrays = {
        name: np.concatenate([sample[name] for _row, sample, _audit in samples], axis=0)
        for name in STATE_ARRAYS
    }
    for name in ("teacher_job_actions", "canonical_job_actions"):
        arrays[name] = np.stack([
            sample[name] for _row, sample, _audit in samples])
    arrays.update({
        name: np.concatenate([sample[name] for _row, sample, _audit in samples], axis=0)
        for name in EVENT_ARRAYS
    })
    arrays["state_event_offsets"] = np.concatenate((
        np.asarray([0], dtype="<i8"), np.cumsum(event_counts, dtype="<i8")))
    arrays["split"] = np.empty(state_count, dtype="u1")
    arrays["group_hash128"] = np.empty((state_count, 16), dtype="u1")
    for index, (row, _sample, _audit) in enumerate(samples):
        opponent_family = contract.OPPONENT_FAMILIES.get(
            row["opponent"], row["opponent"])
        digest = hashlib.sha256(
            f'{row["seed"]}:{opponent_family}'.encode()).digest()[:16]
        arrays["group_hash128"][index] = np.frombuffer(digest, dtype=np.uint8)
        arrays["split"][index] = int(digest[0] < 51)
    if (arrays["state_event_offsets"].shape != (state_count + 1,) or
            arrays["state_event_offsets"][-1] != event_count):
        raise RuntimeError("merged event offsets are invalid")
    labels = arrays["event_label_class"].astype(np.int64)
    masks = arrays["event_legal_mask"].astype(np.int64)
    stages = arrays["event_stage"].astype(np.int64)
    terminal = all(
        0 not in labels[start:stop][:-1]
        for start, stop in zip(arrays["state_event_offsets"][:-1],
                               arrays["state_event_offsets"][1:]))
    if (np.any(((masks >> labels) & 1) != 1) or
            np.any((stages == 0) & ((masks & ~0b110) != 0)) or
            np.any((stages == 1) & ((masks & (1 << 2)) != 0)) or
            np.any((stages == 1) & ((masks & (1 << 1)) == 0)) or not terminal):
        raise RuntimeError("merged label/mask/stage/STOP contract failed")
    return arrays, samples


def build(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    through_step = args.through_step if args.through_step is not None else args.step
    if (args.step < 312 or through_step >= 719 or args.step % 24 or
            through_step % 24 or through_step < args.step):
        raise ValueError("step range must contain day boundaries in [312, 696]")
    steps = tuple(range(args.step, through_step + 1, 24))
    full_rollout = len(steps) > 1
    student_steps = tuple(range(
        288, through_step + (1 if full_rollout else 0), 24))
    fingerprints = _fingerprints(args.checkpoint, args.manifest, args.binary)
    source_manifest = fingerprints.pop("source_manifest")
    checkpoint_steps = set(fingerprints["checkpoint_training_state_steps"])
    attested_student_steps = tuple(
        step for step in student_steps if step in checkpoint_steps)
    exploratory_student_steps = tuple(
        step for step in student_steps if step not in checkpoint_steps)
    if exploratory_student_steps and not full_rollout:
        raise ValueError(
            "single-step DAgger requires every preceding student action to be "
            "checkpoint-attested")
    jobs = [{**game, "steps": steps, "through_step": through_step,
             "student_steps": student_steps,
             "attested_student_steps": attested_student_steps,
             "exploratory_student_steps": exploratory_student_steps,
             "allow_unattested_steps": bool(exploratory_student_steps),
             **fingerprints}
            for game in args.game]
    if len({(row["seed"], row["opponent"], row["seat"]) for row in jobs}) != len(jobs):
        raise ValueError("duplicate rollout game")
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            initializer=_worker_init) as pool:
        futures = [pool.submit(collect_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    results.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"]))
    failures = [{key: row.get(key) for key in
                 ("seed", "opponent", "seat", "steps", "through_step", "error")}
                for row in results if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"refusing entire DAgger shard: {failures}")
    if any(row["fingerprints"] != {
            name: fingerprints[name] for name in
            ("checkpoint_sha256", "manifest_sha256", "binary_sha256")}
            for row in results):
        raise RuntimeError("worker rollout fingerprint mismatch")
    for row in results:
        _validate_student_rollout(
            row["student_rollout"], student_steps,
            attested_student_steps, exploratory_student_steps)
    arrays, samples = _merge(results)
    args.output.mkdir(parents=True)
    array_manifest = {
        name: _save_array(args.output, name, value)
        for name, value in arrays.items()
    }
    schema = {
        "schema_name": "student_state_dagger_action_event_v3",
        "container": "numpy .npy mmap",
        "class_names": list(contract.CLASS_NAMES),
        "model_input_arrays": list(contract.MODEL_INPUTS),
        "target_only_arrays": ["event_label_class"],
        "audit_only_arrays": [
            "state_step", "split", "group_hash128", "teacher_job_actions",
            "canonical_job_actions"],
        "identity_policy": (
            "seed/opponent/seat are audit-only; identity is absent from forward"),
    }
    counts = Counter(map(int, arrays["event_label_class"]))
    audit_groups = [{
        "seed": row["seed"], "opponent": row["opponent"],
        "opponent_family": contract.OPPONENT_FAMILIES.get(
            row["opponent"], row["opponent"]), "seat": row["seat"],
        "step": audit["step"], "events": audit["events"],
        "student_events": audit["student_events"],
        "student_teacher_trace_exact": audit["student_teacher_trace_exact"],
        "student_teacher_common_prefix": audit["student_teacher_common_prefix"],
        "gates": audit["gates"],
        "teacher_seconds": audit["teacher_seconds"],
        "canonical_replay_seconds": audit["canonical_replay_seconds"],
        "rollout_seconds": row["elapsed_seconds"],
    } for row, _sample, audit in samples]
    manifest = {
        "format": "kaggriculture-midgame-mmap", "container_version": 1,
        "validation_status": "accepted", "schema": schema,
        "schema_sha256": _json_hash(schema), "byte_order": "little",
        "counts": {
            "games": len(results), "states": len(samples),
            "events": len(arrays["event_cell"]),
            "train_states": int(np.sum(arrays["split"] == 0)),
            "heldout_states": int(np.sum(arrays["split"] == 1)),
        },
        "class_counts": {
            contract.CLASS_NAMES[index]: counts[index]
            for index in range(len(contract.CLASS_NAMES))
        },
        "dimensions": {
            "causal_context": arrays["causal_context"].shape[1],
            "packed_observation": arrays["packed_observation"].shape[1],
            "event_resources": arrays["event_resources"].shape[1],
            "classes": len(contract.CLASS_NAMES),
        },
        "arrays": array_manifest,
        "causal_context_names": source_manifest["causal_context_names"],
        "event_resource_names": source_manifest["event_resource_names"],
        "rollout": {
            "checkpoint": fingerprints["checkpoint"],
            "checkpoint_sha256": fingerprints["checkpoint_sha256"],
            "manifest": fingerprints["manifest"],
            "manifest_sha256": fingerprints["manifest_sha256"],
            "binary": fingerprints["binary"],
            "binary_sha256": fingerprints["binary_sha256"],
            "checkpoint_training_state_steps":
                fingerprints["checkpoint_training_state_steps"],
            "student_action_steps": list(student_steps),
            "checkpoint_attested_action_steps": list(attested_student_steps),
            "explicit_exploratory_action_steps":
                list(exploratory_student_steps),
            "unattested_exploration": bool(exploratory_student_steps),
            "intermediate_policy": (
                "student_all_day_boundaries" if full_rollout else
                "student_attested_prefix_then_frozen_r1"),
        },
        "split_method": "sha256(seed:opponent_family)[0] < 51",
        "source_family_mapping": contract.OPPONENT_FAMILIES,
        "audit_groups": audit_groups,
        "dagger_policy_agreement": {
            "trace_exact_states": sum(
                row["student_teacher_trace_exact"] for row in audit_groups),
            "trace_exact_rate": sum(
                row["student_teacher_trace_exact"] for row in audit_groups) /
                len(audit_groups),
            "common_prefix_events": sum(
                row["student_teacher_common_prefix"] for row in audit_groups),
            "teacher_events": sum(row["events"] for row in audit_groups),
            "student_events": sum(row["student_events"] for row in audit_groups),
        },
        "validation": {
            "execution_contract": "PASS",
            "scope": "student_state_dagger_day_boundary",
            "state_step": args.step if len(steps) == 1 else None,
            "state_steps": list(steps),
            "day_boundary_step": "PASS",
            "state_source": "live_student_handle_td_clone",
            "label_source": "final_jobs_actions_plus_executable_STOP",
            "clone_isolation": "PASS",
            "resource_replay_exact": "PASS",
            "packed_observation_capacity": "PASS",
            "label_fixed_point": "PASS",
            "stop_terminal_exact": "PASS",
            "controlled_job_projection": "PASS",
            "actor_owned_placement": "PASS",
            "stage_mask_contract": "PASS",
            "rollout_fingerprint": "PASS",
            "student_step_exactly_once": "PASS",
            "student_failures_zero": "PASS",
            "student_fallbacks_zero": "PASS",
            "student_illegal_zero": "PASS",
            "no_identity_in_model_inputs": "PASS",
            "all_states_passed": len(samples),
        },
        "timing": {
            "extract_seconds": elapsed,
            "states_per_second": len(samples) / elapsed,
            "games_per_second": len(results) / elapsed,
            "workers": min(args.workers, len(jobs)),
        },
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--game", type=_parse_game, action="append", required=True,
                        help="repeatable SEED:OPPONENT:SEAT")
    parser.add_argument("--step", type=int, default=312)
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--workers", type=int, default=min(160, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest = build(args)
    print(json.dumps({
        "status": "PASS", "output": str(args.output.resolve()),
        "schema_name": manifest["schema"]["schema_name"],
        "counts": manifest["counts"], "validation": manifest["validation"],
        "timing": manifest["timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
