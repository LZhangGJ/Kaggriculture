#!/usr/bin/env python3
"""Compare base-settings and score-winner per-cell scaffolds on held-out games."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.audit_teacher_batch import r1_module
from scripts.extract_midgame_bc import observation


ROOT = Path(__file__).resolve().parents[1]


def bind(lib):
    lib.td_student_candidate_settings.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_settings.restype = ctypes.c_int
    lib.td_student_prepare_prefix_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_prepare_prefix_observation.restype = ctypes.c_int
    lib.td_student_candidate_slot_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_slot_meta.restype = ctypes.c_int
    lib.td_student_candidate_slot_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.td_student_candidate_slot_count.restype = ctypes.c_int
    lib.td_student_candidate_slot_resources.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_slot_resources.restype = ctypes.c_int
    lib.td_student_slot_resource_names.argtypes = []
    lib.td_student_slot_resource_names.restype = ctypes.c_char_p


def slots(lib, handle, index):
    count = lib.td_student_candidate_slot_count(handle, index)
    if count < 0:
        raise RuntimeError("slot count export failed")
    values = (ctypes.c_int32 * (4 * count))()
    if lib.td_student_candidate_slot_meta(handle, index, values, len(values)) != count:
        raise RuntimeError("slot meta export failed")
    return [tuple(map(int, values[4 * i:4 * i + 4])) for i in range(count)]


def resources(lib, handle, index, width):
    count = lib.td_student_candidate_slot_count(handle, index)
    values = (ctypes.c_double * (width * count))()
    if lib.td_student_candidate_slot_resources(
            handle, index, values, len(values)) != count:
        raise RuntimeError("slot resource export failed")
    return np.asarray(values, dtype=np.float64).reshape(count, width)


def plan(lib, handle, packed, settings=None, prefix=()):
    values = None if settings is None else (ctypes.c_double * len(settings))(*settings)
    cells = [row[0] for row in prefix]
    kinds = [row[1] for row in prefix]
    cell_values = (ctypes.c_int32 * len(cells))(*cells) if cells else None
    kind_values = (ctypes.c_int32 * len(kinds))(*kinds) if kinds else None
    count = lib.td_student_prepare_prefix_observation(
        handle, packed, len(packed), cell_values, kind_values, len(cells), values,
        0 if settings is None else len(settings))
    if count < 0:
        raise RuntimeError(lib.td_debug(handle).decode())
    return slots(lib, handle, 0)


def compare(base, winner):
    base_by_cell = {row[0]: row for row in base}
    winner_by_cell = {row[0]: row for row in winner}
    common = sorted(base_by_cell.keys() & winner_by_cell.keys())
    prefix = 0
    for left, right in zip(base, winner):
        if left[0] != right[0]:
            break
        prefix += 1
    return {
        "base_slots": len(base), "winner_slots": len(winner),
        "sequence_equal": [row[0] for row in base] == [row[0] for row in winner],
        "common_prefix": prefix, "common_cells": len(common),
        "cell_union": len(base_by_cell.keys() | winner_by_cell.keys()),
        "label_diff_common": sum(base_by_cell[cell][1] != winner_by_cell[cell][1]
                                 for cell in common),
        "mask_diff_common": sum(base_by_cell[cell][2] != winner_by_cell[cell][2]
                                for cell in common),
    }


def heldout_games(manifest_path: Path, count: int):
    manifest = json.loads(manifest_path.read_text())
    split_spec = manifest["arrays"]["split"]
    split = np.load(manifest_path.parent / split_spec["path"], mmap_mode="r")
    result, seen = [], set()
    for flag, group in zip(split, manifest["audit_groups"]):
        key = (group["group_hash128"], group["trajectory"], group["seat"])
        if int(flag) != 1 or key in seen:
            continue
        seen.add(key)
        result.append(group)
        if len(result) == count:
            return result
    raise ValueError("not enough held-out games")


def audit_game(job, binary: Path, selected_steps):
    module = r1_module()
    agent = module.Agent(binary_path=binary)
    lib = agent.lib
    bind(lib)
    resource_names = lib.td_student_slot_resource_names().decode().splitlines()
    resource_width = len(resource_names)
    output = []
    try:
        with gzip.open(job["trajectory"], "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("type") != "step":
                    continue
                step = int(row["step"])
                current = observation(row, int(job["seat"]))
                expected = row["actions"][int(job["seat"])]
                if step < 288:
                    agent.observe_external(current, expected)
                    continue
                if step in selected_steps:
                    packed = module._pack(current)
                    if agent.external:
                        if lib.td_activate_external(agent.handle, packed, len(packed)):
                            raise RuntimeError(lib.td_debug(agent.handle).decode())
                        agent.external = False
                    count = lib.td_prepare_observation(agent.handle, packed, len(packed))
                    if count <= 0:
                        raise RuntimeError(lib.td_debug(agent.handle).decode())
                    candidates = [json.loads(lib.td_candidate_json(agent.handle, i))
                                  for i in range(count)]
                    normal = [row for row in candidates if not row["diagnostic"]]
                    winner_json = max(normal, key=lambda value: value["score"])
                    winner_index = int(winner_json["index"])
                    winner = slots(lib, agent.handle, winner_index)
                    winner_resources = resources(
                        lib, agent.handle, winner_index, resource_width)
                    settings = (ctypes.c_double * agent.settings_count)()
                    if lib.td_student_candidate_settings(
                            agent.handle, winner_index, settings, len(settings)) != len(settings):
                        raise RuntimeError("winner settings export failed")
                    reproduced = plan(lib, agent.handle, packed, list(settings))
                    if reproduced != winner:
                        raise RuntimeError("winner-settings seam does not reproduce winner trace")
                    reproduced_resources = resources(
                        lib, agent.handle, 0, resource_width)
                    if not np.array_equal(reproduced_resources, winner_resources):
                        raise RuntimeError("winner-settings seam does not reproduce resources")
                    base = plan(lib, agent.handle, packed)
                    base_resources = resources(lib, agent.handle, 0, resource_width)
                    comparison = compare(base, winner)
                    replay_error = None
                    replay_meta_equal = False
                    replay_resource_equal = False
                    replay_max_abs = None
                    replay_first_difference = None
                    try:
                        replay = plan(lib, agent.handle, packed, prefix=winner)
                        replay_resources = resources(
                            lib, agent.handle, 0, resource_width)
                        replay_meta_equal = replay == winner
                        if replay_resources.shape == winner_resources.shape:
                            difference = np.abs(replay_resources - winner_resources)
                            replay_max_abs = float(difference.max(initial=0.0))
                            replay_resource_equal = bool(np.array_equal(
                                replay_resources, winner_resources))
                            unequal = np.argwhere(replay_resources != winner_resources)
                            if unequal.size:
                                slot_index, resource_index = map(int, unequal[0])
                                replay_first_difference = {
                                    "slot": slot_index, "resource_index": resource_index,
                                    "resource_name": resource_names[resource_index],
                                    "base": float(replay_resources[slot_index, resource_index]),
                                    "winner": float(winner_resources[slot_index, resource_index]),
                                }
                        else:
                            replay_first_difference = {
                                "shape_base": list(replay_resources.shape),
                                "shape_winner": list(winner_resources.shape)}
                    except Exception as error:
                        replay_error = repr(error)
                    output.append({
                        "step": step, "winner_id": winner_json["id"], **comparison,
                        "base_cells": [row[0] for row in base],
                        "winner_cells": [row[0] for row in winner],
                        "base_owned_quadrants": [int(value) for value in base_resources[:, 2]],
                        "winner_owned_quadrants": [int(value) for value in winner_resources[:, 2]],
                        "winner_prefix_base_meta_equal": replay_meta_equal,
                        "winner_prefix_base_resource_equal": replay_resource_equal,
                        "winner_prefix_base_max_abs_resource_error": replay_max_abs,
                        "winner_prefix_base_first_difference": replay_first_difference,
                        "winner_prefix_base_error": replay_error,
                    })
                actual = agent(current)
                if actual != expected:
                    raise RuntimeError(f"trajectory parity mismatch at step {step}")
        return output
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT /
                        "work/student-v1/live-slot-v2-batch2k-merged/manifest.json")
    parser.add_argument("--binary", type=Path, default=ROOT /
                        "work/agent-student-closed-loop-v1.so")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--steps", default="288,384,480,576,672")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected_steps = {int(value) for value in args.steps.split(",")}
    games = heldout_games(args.manifest.resolve(), args.games)
    states = []
    for game in games:
        for row in audit_game(game, args.binary.resolve(), selected_steps):
            states.append({"seed": game["seed"], "opponent": game["opponent"],
                           "family": game["opponent_family"], "seat": game["seat"], **row})
    total_common = sum(row["common_cells"] for row in states)
    meta_equal = sum(row["winner_prefix_base_meta_equal"] for row in states)
    resource_equal = sum(row["winner_prefix_base_resource_equal"] for row in states)
    execution_contract = "PASS" if meta_equal == resource_equal == len(states) else "FAIL"
    result = {
        "status": execution_contract, "audit_status": "PASS",
        "execution_contract": execution_contract,
        "split": "heldout whole (seed, opponent_family) groups",
        "games": len(games), "states": len(states),
        "sequence_equal_states": sum(row["sequence_equal"] for row in states),
        "label_diff_common": sum(row["label_diff_common"] for row in states),
        "mask_diff_common": sum(row["mask_diff_common"] for row in states),
        "common_cells": total_common,
        "label_diff_rate_common": (sum(row["label_diff_common"] for row in states) /
                                   total_common if total_common else None),
        "mask_diff_rate_common": (sum(row["mask_diff_common"] for row in states) /
                                  total_common if total_common else None),
        "winner_prefix_base_meta_equal_states": meta_equal,
        "winner_prefix_base_resource_equal_states": resource_equal,
        "winner_prefix_base_replay_errors": sum(
            row["winner_prefix_base_error"] is not None for row in states),
        "winner_prefix_base_max_abs_resource_error": max(
            (row["winner_prefix_base_max_abs_resource_error"] or 0.0)
            for row in states),
        "states_detail": states,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
