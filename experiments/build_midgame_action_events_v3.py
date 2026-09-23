#!/usr/bin/env python3
"""Export execution-consistent RELEASE/KEEP + per-cell action events."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.audit_teacher_batch import names_sha256, r1_module
from experiments.build_midgame_live_slots_v2 import _load_jobs
from experiments.train_midgame_student_v1 import (
    ROOT, TOKENIZER_ROOT, _canonical_observation, _json_hash, _save_array, _sha256,
)
from scripts.extract_midgame_bc import observation


CLASS_NAMES = ("STOP", "NONE_OR_KEEP", "RELEASE", "WHEAT", "CARROT",
               "TOMATO", "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP")
PACKED_OBSERVATION_CAPACITY = 3074
OPPONENT_FAMILIES = {
    name: "ahmed-v31-descendant" for name in (
        "thomas_2945", "melon_2749", "demand_preserving", "ahmed_v47",
        "pipe8",
        "market_smart_v24", "ahmed_v56", "thomas_metav4_v13",
        "master_hybrid_2965", "historical_lb_2800_rescue",
        "more_wheat_smarter_sales", "farmer_john_idle_seller_v57",
        "soil_remembers_rain",
    )
} | {"fieldcraft_2887": "fieldcraft", "night_harvest": "night-harvest"}
MODEL_INPUTS = (
    "causal_context", "observation_length", "packed_observation",
    "state_event_offsets", "event_stage", "event_cell", "event_resources",
    "event_legal_mask", "token_category_a", "token_category_b",
    "token_category_c", "token_continuous", "token_count", "token_owner",
    "token_type", "token_x", "token_y",
)
HARVEST, DIG, PLANT, PLACE = 10, 12, 8, 7
SEMANTIC_SMOKE = (
    ROOT / "work/student-v1/v3-semantic-smoke-actor-owned-step288.json")
ReleaseCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
SlotCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)


def validate_day_boundary_pack(observation: dict, packed_length: int) -> None:
    """Validate the fixed-width model ABI used only at hour-zero boundaries."""
    step = int(observation["day"]) * 24 + int(observation["hour"])
    farms = observation["farms"]
    inventories = observation["private"]["inventories"]
    shops = observation["town"]["unlocked_shops"]
    expected_length = 3066 + len(shops)
    if (step % 24 or len(farms) != 2 or
            any(farm.get("hands") or int(farm.get("hires_today", 0))
                for farm in farms) or inventories != [{}] or len(shops) > 8 or
            packed_length != expected_length or
            packed_length > PACKED_OBSERVATION_CAPACITY):
        raise ValueError(
            f"observation violates hour-zero packed ABI: step={step} "
            f"length={packed_length} expected={expected_length}")


def bind(lib):
    lib.td_student_slot_abi_version.argtypes = []
    lib.td_student_slot_abi_version.restype = ctypes.c_int
    lib.td_student_pre_context_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_pre_context_observation.restype = ctypes.c_int
    lib.td_teacher_context_names.argtypes = [ctypes.c_void_p]
    lib.td_teacher_context_names.restype = ctypes.c_char_p
    lib.td_student_slot_resource_names.argtypes = []
    lib.td_student_slot_resource_names.restype = ctypes.c_char_p
    lib.td_student_plan_v3_callback_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ReleaseCallback, SlotCallback, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_plan_v3_callback_observation.restype = ctypes.c_int
    lib.td_student_install_prepared.argtypes = [ctypes.c_void_p]
    lib.td_student_install_prepared.restype = ctypes.c_int
    lib.td_student_candidate_v3_release_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_v3_release_meta.restype = ctypes.c_int
    lib.td_student_candidate_v3_release_resources.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_v3_release_resources.restype = ctypes.c_int
    lib.td_student_candidate_v3_slot_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_v3_slot_meta.restype = ctypes.c_int
    lib.td_student_candidate_v3_slot_resources.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_v3_slot_resources.restype = ctypes.c_int
    lib.td_student_candidate_job_actions_observation.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_job_actions_observation.restype = ctypes.c_int
    lib.td_student_candidate_settings.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_settings.restype = ctypes.c_int


def jobs(lib, handle, packed, index):
    raw = (ctypes.c_int32 * 400)()
    if lib.td_student_candidate_job_actions_observation(
            handle, index, packed, len(packed), raw, len(raw)) != 100:
        raise RuntimeError(lib.td_debug(handle).decode())
    return np.asarray(raw, dtype=np.int32).reshape(100, 4).copy()


def event_exports(lib, handle, release_count, slot_count):
    release_meta = (ctypes.c_int32 * (3 * release_count))()
    release_resource = (ctypes.c_double * (347 * release_count))()
    slot_meta = (ctypes.c_int32 * (4 * slot_count))()
    slot_resource = (ctypes.c_double * (347 * slot_count))()
    if (lib.td_student_candidate_v3_release_meta(
            handle, 0, release_meta, len(release_meta)) != release_count or
            lib.td_student_candidate_v3_release_resources(
                handle, 0, release_resource, len(release_resource)) != release_count or
            lib.td_student_candidate_v3_slot_meta(
                handle, 0, slot_meta, len(slot_meta)) != slot_count or
            lib.td_student_candidate_v3_slot_resources(
                handle, 0, slot_resource, len(slot_resource)) != slot_count):
        raise RuntimeError("v3 event export failed")
    return (
        np.asarray(release_meta, dtype=np.int32).reshape(release_count, 3),
        np.asarray(release_resource, dtype=np.float64).reshape(release_count, 347),
        np.asarray(slot_meta, dtype=np.int32).reshape(slot_count, 4),
        np.asarray(slot_resource, dtype=np.float64).reshape(slot_count, 347),
    )


def placement_map(job_rows):
    result = {}
    for cell, (mask, planted, placed, _count) in enumerate(job_rows):
        if mask & (1 << PLANT):
            if not 0 <= planted < 5:
                raise RuntimeError("invalid final PLANT item")
            result[cell] = int(planted) + 3
        if mask & (1 << PLACE):
            if cell in result or not 9 <= placed < 12:
                raise RuntimeError("invalid final PLACE item")
            result[cell] = int(placed) - 1
    return result


def actor_owned_teacher_jobs(agent, packed):
    """Run R1's native choices through the same actor-owned v3 scaffold."""
    lib = agent.lib

    @ReleaseCallback
    def release(_user, _cell, suggested, mask, _resources, _width):
        return int(suggested) if int(mask) & (1 << int(suggested)) else -1000

    @SlotCallback
    def slot(_user, _cell, suggested, mask, _resources, _width):
        return int(suggested) if int(mask) & (1 << int(suggested)) else -1000

    count = lib.td_student_plan_v3_callback_observation(
        agent.handle, packed, len(packed), release, slot, None, None, 0)
    if count < 0:
        raise RuntimeError(lib.td_debug(agent.handle).decode())
    return jobs(lib, agent.handle, packed, 0)


def project_labels(job_rows, release_rows, slot_cells):
    releases = []
    for cell, _suggested, mask in release_rows:
        ops = int(job_rows[cell, 0])
        label = 2 if ops & ((1 << HARVEST) | (1 << DIG)) else 1
        releases.append(label)
    target = placement_map(job_rows)
    pending = set(target)
    slots = []
    for cell in slot_cells:
        if cell in target:
            label = target[cell]
            pending.remove(cell)
        else:
            label = 1 if pending else 0
        slots.append(label)
        if label == 0:
            break
    return releases, slots, pending


def plan_from_teacher(canonical, packed, teacher_jobs):
    lib = canonical.lib
    teacher_target = placement_map(teacher_jobs)
    pending = set(teacher_target)
    release_events, slot_events = [], []

    @ReleaseCallback
    def release(_user, cell, suggested, mask, resources, width):
        if width != 347:
            return -1000
        ops = int(teacher_jobs[cell, 0])
        label = 2 if ops & ((1 << HARVEST) | (1 << DIG)) else 1
        release_events.append((int(cell), label, int(mask),
                               np.ctypeslib.as_array(resources, (347,)).copy()))
        return label

    @SlotCallback
    def slot(_user, cell, suggested, mask, resources, width):
        if width != 347:
            return -1000
        cell = int(cell)
        if cell in teacher_target:
            label = teacher_target[cell]
            pending.remove(cell)
        else:
            label = 1 if pending else 0
        slot_events.append((cell, label, int(mask),
                            np.ctypeslib.as_array(resources, (347,)).copy()))
        return label

    count = lib.td_student_plan_v3_callback_observation(
        canonical.handle, packed, len(packed), release, slot, None,
        None, 0)
    if count < 0:
        raise RuntimeError(lib.td_debug(canonical.handle).decode())
    if pending or count != len(release_events) + len(slot_events):
        raise RuntimeError(f"teacher placements not representable: pending={sorted(pending)}")
    rm, rr, sm, sr = event_exports(
        lib, canonical.handle, len(release_events), len(slot_events))
    callback_rm = np.asarray(
        [row[:3] for row in release_events], dtype=np.int32).reshape(-1, 3)
    callback_rr = np.asarray([row[3] for row in release_events], dtype=np.float64).reshape(-1, 347)
    callback_sm = np.asarray(
        [[row[0], row[1], row[2], row[1] == 0] for row in slot_events],
        dtype=np.int32).reshape(-1, 4)
    callback_sr = np.asarray([row[3] for row in slot_events], dtype=np.float64).reshape(-1, 347)
    resource_exact = (np.array_equal(callback_rm, rm) and
                      np.array_equal(callback_rr, rr) and
                      np.array_equal(callback_sm, sm) and
                      np.array_equal(callback_sr, sr))
    projected_jobs = jobs(lib, canonical.handle, packed, 0)
    projected_release, projected_slots, projected_pending = project_labels(
        projected_jobs, rm, [int(row[0]) for row in sm])
    fixed_point = (not projected_pending and
                   projected_release == [row[1] for row in release_events] and
                   projected_slots == [row[1] for row in slot_events])
    stop_terminal_exact = (sum(row[1] == 0 for row in slot_events) <= 1 and
                           (not any(row[1] == 0 for row in slot_events) or
                            slot_events[-1][1] == 0))
    events = ([{"stage": 0, "cell": row[0], "label": row[1],
                "mask": row[2], "resources": row[3]} for row in release_events] +
              [{"stage": 1, "cell": row[0], "label": row[1],
                "mask": row[2], "resources": row[3]} for row in slot_events])
    controlled_exact = placement_map(projected_jobs) == placement_map(teacher_jobs)
    for row in release_events:
        cell, label = row[:2]
        teacher_release = bool(int(teacher_jobs[cell, 0]) &
                               ((1 << HARVEST) | (1 << DIG)))
        canonical_release = bool(int(projected_jobs[cell, 0]) &
                                 ((1 << HARVEST) | (1 << DIG)))
        controlled_exact &= teacher_release == canonical_release == (label == 2)
    return (events, projected_jobs, resource_exact, fixed_point,
            stop_terminal_exact, controlled_exact)


def extract_game(job):
    module = r1_module()
    teacher = module.Agent(binary_path=Path(job["binary"]))
    canonical = module.Agent(binary_path=Path(job["binary"]))
    states, steps = [], []
    counts = Counter()
    first_failure = None
    try:
        for agent in (teacher, canonical):
            bind(agent.lib)
            if agent.lib.td_student_slot_abi_version() != 3:
                raise RuntimeError("actor-owned student ABI v3 is required")
        if teacher.config != canonical.config:
            raise RuntimeError("teacher/canonical settings mismatch")
        context_names = canonical.lib.td_teacher_context_names(
            canonical.handle).decode().splitlines()
        resource_names = canonical.lib.td_student_slot_resource_names().decode().splitlines()
        if (len(context_names) != 2233 or len(set(context_names)) != len(context_names) or
                names_sha256(context_names) !=
                "6386bae567618f5c8409075b420dc5dbff3bb616b236bb714aacde151beff3a8" or
                len(resource_names) != 347 or len(set(resource_names)) != 347):
            raise RuntimeError("v3 context/resource schema mismatch")
        with gzip.open(job["trajectory"], "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("type") != "step":
                    continue
                step = int(row["step"])
                steps.append(step)
                current = observation(row, job["seat"])
                expected = row["actions"][job["seat"]]
                if step < 288:
                    teacher.observe_external(current, expected)
                    canonical.observe_external(current, expected)
                    continue
                if step != 288:
                    raise RuntimeError("first handoff step 288 is missing")
                packed = module._pack(current)
                if canonical.external:
                    if canonical.lib.td_activate_external(
                            canonical.handle, packed, len(packed)):
                        raise RuntimeError(canonical.lib.td_debug(
                            canonical.handle).decode())
                    canonical.external = False
                context = (ctypes.c_double * len(context_names))()
                if canonical.lib.td_student_pre_context_observation(
                        canonical.handle, packed, len(packed), context,
                        len(context)) != len(context):
                    raise RuntimeError(canonical.lib.td_debug(
                        canonical.handle).decode())
                canonical_observation = _canonical_observation(current)
                packed_observation = list(module._pack(canonical_observation))
                validate_day_boundary_pack(
                    canonical_observation, len(packed_observation))
                pending = {
                    "step": step, "context": list(context),
                    "observation": canonical_observation,
                    "packed_observation": packed_observation,
                }
                teacher_action = teacher(current)
                if teacher_action != expected:
                    raise RuntimeError(f"teacher action parity step={step}")
                teacher_jobs = jobs(teacher.lib, teacher.handle, packed, -1)
                try:
                    (events, projected_jobs, resource_exact, fixed_point,
                     stop_terminal_exact, controlled_exact) = plan_from_teacher(
                        canonical, packed, teacher_jobs)
                    counts["states"] += 1
                    counts["resource_replay_exact"] += resource_exact
                    counts["label_fixed_point"] += fixed_point
                    counts["stop_terminal_exact"] += stop_terminal_exact
                    counts["controlled_job_projection"] += controlled_exact
                    if not (resource_exact and fixed_point and
                            stop_terminal_exact and controlled_exact):
                        raise RuntimeError(
                            f"first-handoff gate resource={resource_exact} "
                            f"fixed={fixed_point} stop={stop_terminal_exact} "
                            f"controlled={controlled_exact}")
                    pending["events"] = events
                    if events:
                        states.append(pending)
                except Exception as error:
                    first_failure = {"step": step, "error": repr(error)}
                break
        complete = steps == list(range(289)) and first_failure is None
        if not complete and first_failure is None:
            first_failure = {"error": "step0..288 prefix completeness failure"}
        return {
            **{key: job[key] for key in ("seed", "seat", "opponent", "trajectory",
                                          "source_manifest")},
            "opponent_family": OPPONENT_FAMILIES.get(
                job["opponent"], job["opponent"]), "states": states,
            "counts": dict(counts), "failure": first_failure,
            "settings": dict(canonical.config),
            "context_names": context_names, "resource_names": resource_names,
        }
    finally:
        teacher.close()
        canonical.close()


def worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def write_dataset(output, games, source, binary, elapsed, semantic_smoke):
    sys.path.insert(0, str(TOKENIZER_ROOT))
    from kaggrl.tokenizer import ObservationTokenizer
    from experiments.midgame_autofill_student import contains_private_identity

    states = [(game, state) for game in games for state in game["states"]]
    if any(game["settings"] != games[0]["settings"] for game in games):
        raise ValueError("teacher settings changed within the BC corpus")
    tokenizer = ObservationTokenizer()
    encoded = []
    for _game, state in states:
        if contains_private_identity(state["observation"]):
            raise ValueError("identity leaked into model observation")
        encoded.append(tokenizer.encode(state["observation"]))
    n = len(states)
    e = sum(len(state["events"]) for _game, state in states)
    packed_width = PACKED_OBSERVATION_CAPACITY
    arrays = {
        "causal_context": np.asarray([s["context"] for _g, s in states], dtype="<f4"),
        "packed_observation": np.zeros((n, packed_width), dtype="<f8"),
        "observation_length": np.asarray(
            [len(s["packed_observation"]) for _g, s in states], dtype="<u2"),
        "token_continuous": np.zeros((n, 320, 24), dtype="<f4"),
        "token_type": np.zeros((n, 320), dtype="u1"),
        "token_category_a": np.zeros((n, 320), dtype="u1"),
        "token_category_b": np.zeros((n, 320), dtype="u1"),
        "token_category_c": np.zeros((n, 320), dtype="u1"),
        "token_x": np.zeros((n, 320), dtype="u1"),
        "token_y": np.zeros((n, 320), dtype="u1"),
        "token_owner": np.zeros((n, 320), dtype="u1"),
        "token_count": np.asarray([row.num_tokens for row in encoded], dtype="<u2"),
        "state_event_offsets": np.zeros(n + 1, dtype="<i8"),
        "state_step": np.empty(n, dtype="<u2"),
        "event_stage": np.empty(e, dtype="u1"),
        "event_cell": np.empty(e, dtype="u1"),
        "event_resources": np.empty((e, 347), dtype="<f4"),
        "event_legal_mask": np.empty(e, dtype="<u2"),
        "event_label_class": np.empty(e, dtype="u1"),
        "split": np.empty(n, dtype="u1"),
        "group_hash128": np.empty((n, 16), dtype="u1"),
    }
    targets = ("token_type", "token_category_a", "token_category_b",
               "token_category_c", "token_x", "token_y", "token_owner")
    sources = ("token_type", "category_a", "category_b", "category_c",
               "x", "y", "owner")
    audit_groups, cursor = [], 0
    for index, ((game, state), tokenized) in enumerate(zip(states, encoded)):
        packed = state["packed_observation"]
        arrays["packed_observation"][index, :len(packed)] = packed
        count = tokenized.num_tokens
        if count > 320:
            raise RuntimeError("token count exceeds 320")
        arrays["token_continuous"][index, :count] = tokenized.continuous.numpy()
        for target, source_name in zip(targets, sources):
            arrays[target][index, :count] = getattr(tokenized, source_name).numpy()
        arrays["state_event_offsets"][index] = cursor
        arrays["state_step"][index] = state["step"]
        for event in state["events"]:
            arrays["event_stage"][cursor] = event["stage"]
            arrays["event_cell"][cursor] = event["cell"]
            arrays["event_resources"][cursor] = event["resources"]
            arrays["event_legal_mask"][cursor] = event["mask"]
            arrays["event_label_class"][cursor] = event["label"]
            cursor += 1
        digest = hashlib.sha256(
            f'{game["seed"]}:{game["opponent_family"]}'.encode()).digest()[:16]
        arrays["group_hash128"][index] = np.frombuffer(digest, dtype=np.uint8)
        arrays["split"][index] = int(digest[0] < 51)
        audit_groups.append({
            "group_hash128": digest.hex(), "seed": game["seed"],
            "opponent_family": game["opponent_family"], "opponent": game["opponent"],
            "seat": game["seat"], "step": state["step"],
            "trajectory": game["trajectory"], "source_manifest": game["source_manifest"],
        })
    arrays["state_event_offsets"][-1] = cursor
    legal = ((arrays["event_legal_mask"] >> arrays["event_label_class"]) & 1) != 0
    release = arrays["event_stage"] == 0
    placement = ~release
    stage_masks = (np.all((arrays["event_legal_mask"][release] | 0x006) == 0x006) and
                   np.all(np.isin(arrays["event_label_class"][release], (1, 2))) and
                   np.all((arrays["event_legal_mask"][placement] & (1 << 2)) == 0) and
                   np.all(np.isin(arrays["event_label_class"][placement],
                                  (0, 1, 3, 4, 5, 6, 7, 8, 9, 10))))
    terminal_order = all(
        0 not in arrays["event_label_class"][start:stop][:-1]
        for start, stop in zip(arrays["state_event_offsets"][:-1],
                               arrays["state_event_offsets"][1:]))
    if cursor != e or not np.all(legal) or not stage_masks or not terminal_order:
        raise RuntimeError("invalid v3 offsets or illegal label")
    output.mkdir(parents=True, exist_ok=False)
    array_manifest = {name: _save_array(output, name, value)
                      for name, value in arrays.items()}
    schema = {
        "schema_name": "autoregressive-action-event-bc-v3",
        "container": "numpy .npy mmap", "class_names": list(CLASS_NAMES),
        "model_input_arrays": list(MODEL_INPUTS),
        "target_only_arrays": ["event_label_class"],
        "audit_only_arrays": ["state_step", "split", "group_hash128"],
        "identity_policy": "seed/opponent/seat are audit-only and absent from forward",
    }
    counts = Counter(map(int, arrays["event_label_class"]))
    manifest = {
        "format": "kaggriculture-midgame-mmap", "container_version": 1,
        "validation_status": "accepted", "schema": schema,
        "schema_sha256": _json_hash(schema), "byte_order": "little",
        "counts": {"games": len(games), "states": n, "events": e,
                   "train_states": int(np.sum(arrays["split"] == 0)),
                   "heldout_states": int(np.sum(arrays["split"] == 1)),
                   "teacher_step288_action_parity": len(games)},
        "class_counts": {CLASS_NAMES[k]: counts[k] for k in range(11)},
        "dimensions": {"causal_context": 2233,
                       "packed_observation": packed_width,
                       "event_resources": 347, "classes": 11},
        "arrays": array_manifest, "event_resource_names": games[0]["resource_names"],
        "causal_context_names": games[0]["context_names"],
        "teacher_binary": str(binary.resolve()),
        "teacher_binary_sha256": _sha256(binary), "source": source,
        "source_family_mapping": OPPONENT_FAMILIES,
        "model_runtime": {
            "files": {
                relative: _sha256(ROOT / relative)
                for relative in (
                    "agent/main.py", "policy/r1/agent.py", "policy/r1/config.json")
            },
            "tokenizer": {
                "path": str((TOKENIZER_ROOT / "kaggrl/tokenizer.py").resolve()),
                "sha256": _sha256(TOKENIZER_ROOT / "kaggrl/tokenizer.py"),
            },
            "scaffold_settings": games[0]["settings"],
        },
        "split_method": "sha256(seed:source_family)[0] < 51",
        "audit_groups": audit_groups,
        "validation": {
            "execution_contract": "PASS", "scope": "first_handoff_only_step288",
            "label_source": "final_jobs_actions_plus_executable_STOP",
            "label_fixed_point": "PASS", "resource_replay_exact": "PASS",
            "stop_terminal_exact": "PASS", "controlled_job_projection": "PASS",
            "actor_owned_placement": "PASS",
            "execution_semantics": "PASS",
            "stage_mask_contract": "PASS",
            "label_legal_fraction": float(np.mean(legal)),
            "release_prefix_resources": "canonical callback second pass",
            "execution_contract_scope": (
                "RELEASE/HARVEST/DIG, PLANT/PLACE, terminal STOP only"),
            "packed_observation_capacity": "PASS",
            "packed_observation_scope": "hour-zero day boundaries only",
            "post288_commitments": "empty at first handoff by deployment contract",
            "semantic_smoke": semantic_smoke,
            "no_identity_in_model_inputs": "PASS",
        },
        "timing": {"extract_seconds": elapsed,
                   "games_per_second": len(games) / elapsed},
    }
    (output / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(output / "manifest.json.tmp", output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/bc/midgame-v1")
    parser.add_argument("--policy-manifest", default="policy-deployment-bd49e2a13e44.json")
    parser.add_argument("--binding", default="corpus-policy-binding-bd49e2a13e44.json")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(160, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.output.exists() or args.gate_output.exists():
        parser.error("output paths must not exist")
    if _sha256(args.binary) != args.binary_sha256:
        parser.error("binary sha256 mismatch")
    if not SEMANTIC_SMOKE.is_file():
        parser.error("v3 semantic smoke artifact is missing")
    semantic = json.loads(SEMANTIC_SMOKE.read_text())
    if (semantic.get("status") != "PASS" or
            semantic.get("binary_sha256") != args.binary_sha256):
        parser.error("v3 semantic smoke does not attest this binary")
    semantic_audit = {
        "path": str(SEMANTIC_SMOKE.resolve()),
        "sha256": _sha256(SEMANTIC_SMOKE),
        "binary_sha256": semantic["binary_sha256"],
        "checks": semantic.get("checks", []),
    }
    requested_games = args.max_games
    args.max_games = 0
    jobs_available, source = _load_jobs(args)
    args.max_games = requested_games
    buckets = {}
    for job in jobs_available:
        buckets.setdefault((job["opponent"], job["seat"]), []).append(job)
    jobs_to_run = []
    while any(buckets.values()) and (
            not requested_games or len(jobs_to_run) < requested_games):
        for opponent_seat in sorted(buckets):
            if buckets[opponent_seat]:
                jobs_to_run.append(buckets[opponent_seat].pop(0))
                if requested_games and len(jobs_to_run) == requested_games:
                    break
    source["selection_games"] = len(jobs_to_run)
    source["selection_policy"] = "opponent-round-robin"
    for job in jobs_to_run:
        job["binary"] = str(args.binary.resolve())
    worker_init()
    started = time.perf_counter()
    games = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        futures = [pool.submit(extract_game, job) for job in jobs_to_run]
        for future in as_completed(futures):
            games.append(future.result())
    elapsed = time.perf_counter() - started
    games.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"]))
    failures = [{"seed": game["seed"], "opponent": game["opponent"],
                 "seat": game["seat"], **game["failure"]}
                for game in games if game["failure"]]
    totals = Counter()
    for game in games:
        totals.update(game["counts"])
    gate = {
        "status": "PASS" if not failures else "FAIL",
        "binary": str(args.binary.resolve()), "binary_sha256": _sha256(args.binary),
        "games": len(games), "opponents": dict(Counter(
            game["opponent"] for game in games)),
        "seats": dict(Counter(str(game["seat"]) for game in games)),
        "counts": dict(totals), "failures": failures,
        "scope": "first_handoff_only_step288",
        "label_fixed_point": "PASS" if not failures else "FAIL",
        "resource_replay_exact": "PASS" if not failures else "FAIL",
        "stop_terminal_exact": "PASS" if not failures else "FAIL",
        "controlled_job_projection": "PASS" if not failures else "FAIL",
        "actor_owned_placement": "PASS" if not failures else "FAIL",
        "execution_semantics": "PASS",
        "semantic_smoke": semantic_audit,
        "required": ["label_fixed_point", "resource_replay_exact",
                     "stop_terminal_exact", "controlled_job_projection",
                     "actor_owned_placement", "execution_semantics"],
        "elapsed_seconds": elapsed,
    }
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(json.dumps(gate, indent=2) + "\n")
    if failures:
        print(json.dumps(gate))
        raise SystemExit(2)
    manifest = write_dataset(args.output, games, source, args.binary, elapsed,
                             semantic_audit)
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "gate": str(args.gate_output), **manifest["counts"]}))


if __name__ == "__main__":
    main()
