#!/usr/bin/env python3
"""Executable check for v3 release/NONE/STOP and two-to-three-land semantics."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
from pathlib import Path

from experiments.audit_teacher_batch import r1_module
from scripts.extract_midgame_bc import observation


ROOT = Path(__file__).resolve().parents[1]
HARVEST, PLANT = 10, 8
ReleaseCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
SlotCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)


def bind(lib):
    lib.td_student_plan_v3_callback_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ReleaseCallback, SlotCallback, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_plan_v3_callback_observation.restype = ctypes.c_int
    lib.td_student_candidate_job_actions_observation.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_job_actions_observation.restype = ctypes.c_int
    lib.td_student_candidate_plan_audit.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    lib.td_student_candidate_plan_audit.restype = ctypes.c_int


def restore(agent, module, path, seat, step):
    current = None
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("type") != "step":
                continue
            current = observation(row, seat)
            if int(row["step"]) == step:
                break
            agent.observe_external(current, row["actions"][seat])
    if current is None or agent.last != step - 1:
        raise RuntimeError("trajectory does not contain a consecutive target state")
    packed = module._pack(current)
    if agent.lib.td_activate_external(agent.handle, packed, len(packed)):
        raise RuntimeError(agent.lib.td_debug(agent.handle).decode())
    agent.external = False
    return current, packed


def is_empty(tile):
    return tile is None or tile in ("EMPTY", "SOIL") or (
        isinstance(tile, dict) and tile.get("kind", "EMPTY") in ("EMPTY", "SOIL"))


def is_locked(tile):
    return tile == "LOCKED" or (
        isinstance(tile, dict) and tile.get("kind") == "LOCKED")


def run_plan(agent, packed, mode):
    releases, slots = [], []

    @ReleaseCallback
    def release(_user, cell, suggested, mask, resources, width):
        assert width == 347 and resources
        choice = 2 if mask & (1 << 2) else 1
        releases.append((int(cell), int(choice), int(mask)))
        return choice

    @SlotCallback
    def slot(_user, cell, suggested, mask, resources, width):
        assert width == 347 and resources
        choice = 0 if mode == "stop" else 1
        slots.append((int(cell), int(choice), int(mask)))
        return choice

    count = agent.lib.td_student_plan_v3_callback_observation(
        agent.handle, packed, len(packed), release, slot, None, None, 0)
    if count < 0:
        raise RuntimeError(agent.lib.td_debug(agent.handle).decode())
    raw = (ctypes.c_int32 * 400)()
    written = agent.lib.td_student_candidate_job_actions_observation(
        agent.handle, 0, packed, len(packed), raw, len(raw))
    if written != 100:
        raise RuntimeError(agent.lib.td_debug(agent.handle).decode())
    jobs = [tuple(map(int, raw[4 * cell:4 * cell + 4])) for cell in range(100)]
    targets = (ctypes.c_int32 * 100)()
    releases_meta = (ctypes.c_int32 * 100)()
    plan_meta = (ctypes.c_int32 * 5)()
    if agent.lib.td_student_candidate_plan_audit(
            agent.handle, 0, targets, releases_meta, plan_meta):
        raise RuntimeError("plan audit export failed")
    forced = [cell for cell, choice, _mask in releases if choice == 2]
    if not forced:
        raise AssertionError("fixture has no legally releasable finite crop")
    for cell in forced:
        opmask = jobs[cell][0]
        assert opmask & (1 << HARVEST), (
            mode, cell, jobs[cell], int(targets[cell]), releases, slots)
        assert not opmask & (1 << PLANT), (mode, cell, jobs[cell])
    return {"release_events": releases, "slot_events": slots,
            "forced_release_cells": forced,
            "forced_release_jobs": {str(cell): jobs[cell] for cell in forced},
            "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--step", type=int, default=288)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = r1_module()
    agent = module.Agent(binary_path=args.binary.resolve())
    try:
        bind(agent.lib)
        current, packed = restore(
            agent, module, args.trajectory.resolve(), args.seat, args.step)
        none = run_plan(agent, packed, "none")
        stop = run_plan(agent, packed, "stop")
        empty_none = next((cell for cell, _choice, _mask in none["slot_events"]
                           if is_empty(current["farms"][args.seat]["tiles"]
                                       [cell // 10][cell % 10])), None)
        if empty_none is None:
            raise AssertionError("fixture has no EMPTY placement event")
        assert not none["jobs"][empty_none][0] & (1 << PLANT), (
            empty_none, none["jobs"][empty_none])
        farm = current["farms"][args.seat]
        if len(farm["unlocked_quadrants"]) != 2:
            raise AssertionError("semantic fixture must start with exactly two lands")
        third_land_slots = [
            cell for cell, _choice, _mask in none["slot_events"]
            if is_locked(farm["tiles"][cell // 10][cell % 10])]
        if not third_land_slots:
            raise AssertionError("two-land plan never exposed a third-land slot")
        result = {
            "status": "PASS", "step": args.step,
            "binary": str(args.binary.resolve()),
            "binary_sha256": hashlib.sha256(
                args.binary.resolve().read_bytes()).hexdigest(),
            "checks": ["release_to_none_harvest_without_plant",
                       "release_to_stop_harvest_without_plant",
                       "empty_to_none_without_plant",
                       "two_land_plan_exposes_third_land_slots"],
            "empty_none_cell": empty_none,
            "third_land_slots": third_land_slots,
            "none": {key: value for key, value in none.items() if key != "jobs"},
            "stop": {key: value for key, value in stop.items() if key != "jobs"},
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
