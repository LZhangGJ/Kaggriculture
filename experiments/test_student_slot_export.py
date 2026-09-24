#!/usr/bin/env python3
"""Small ABI check for the autoregressive per-cell student trace."""
import argparse
import ctypes
import json
import math

from audit_teacher_batch import bind, restore, r1_module


KINDS = (-1, 0, 1, 2, 3, 4, 9, 10, 11)


def bind_slots(lib):
    lib.td_student_slot_abi_version.argtypes = []
    lib.td_student_slot_abi_version.restype = ctypes.c_int
    lib.td_student_slot_contract_json.argtypes = []
    lib.td_student_slot_contract_json.restype = ctypes.c_char_p
    lib.td_student_slot_resource_names.argtypes = []
    lib.td_student_slot_resource_names.restype = ctypes.c_char_p
    lib.td_student_candidate_slot_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.td_student_candidate_slot_count.restype = ctypes.c_int
    lib.td_student_candidate_slot_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_slot_meta.restype = ctypes.c_int
    lib.td_student_candidate_slot_resources.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_candidate_slot_resources.restype = ctypes.c_int


def snapshot(lib, handle, candidate_count, contract):
    result = []
    for candidate in range(candidate_count):
        count = lib.td_student_candidate_slot_count(handle, candidate)
        if count <= 0:
            raise RuntimeError(f"candidate {candidate}: no per-cell decisions ({count})")
        meta = (ctypes.c_int32 * (count * contract["meta_width"]))()
        resources = (ctypes.c_double * (count * contract["resource_width"]))()
        if lib.td_student_candidate_slot_meta(handle, candidate, meta, len(meta)) != count:
            raise RuntimeError(f"candidate {candidate}: meta export failed")
        if lib.td_student_candidate_slot_resources(
                handle, candidate, resources, len(resources)) != count:
            raise RuntimeError(f"candidate {candidate}: resource export failed")
        rows = [tuple(meta[i * 4:(i + 1) * 4]) for i in range(count)]
        if len({row[0] for row in rows}) != count:
            raise RuntimeError(f"candidate {candidate}: duplicate cell")
        for slot, (cell, label, mask, terminal) in enumerate(rows):
            if not 0 <= cell < 100 or label not in KINDS or mask & ~0x1FF or not mask & 1:
                raise RuntimeError(f"candidate {candidate}, slot {slot}: invalid meta")
            if not mask & (1 << KINDS.index(label)):
                raise RuntimeError(f"candidate {candidate}, slot {slot}: illegal label")
            if terminal not in (0, 1) or (terminal and (slot != count - 1 or label != -1)):
                raise RuntimeError(f"candidate {candidate}, slot {slot}: invalid terminal")
        values = tuple(resources)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"candidate {candidate}: non-finite resource")
        width = contract["resource_width"]
        for slot in range(count):
            row = values[slot * width:(slot + 1) * width]
            if row[3] != slot or row[4] < slot + 1:
                raise RuntimeError(f"candidate {candidate}, slot {slot}: invalid sequence index")
        result.append((tuple(meta), values))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--step", type=int, default=288)
    args = parser.parse_args()
    module = r1_module()
    agent, *_rest, packed_values = restore(
        module, args.binary, args.trajectory, args.step)
    try:
        lib = agent.lib
        bind(lib)
        bind_slots(lib)
        contract = json.loads(lib.td_student_slot_contract_json())
        names = lib.td_student_slot_resource_names().decode().splitlines()
        if (lib.td_student_slot_abi_version() != 2 or contract["abi_version"] != 2 or
                tuple(contract["class_kinds"]) != KINDS or
                contract["label_semantics"] != "greedy_proposal_before_preview" or
                len(names) != contract["resource_width"] or len(set(names)) != len(names)):
            raise RuntimeError("student slot ABI/schema mismatch")
        packed = (ctypes.c_double * len(packed_values))(*packed_values)
        count = lib.td_teacher_prepare_features_observation(agent.handle, packed, len(packed))
        if count <= 0:
            raise RuntimeError(lib.td_debug(agent.handle).decode())
        first = snapshot(lib, agent.handle, count, contract)
        repeat_count = lib.td_teacher_prepare_features_observation(
            agent.handle, packed, len(packed))
        second = snapshot(lib, agent.handle, repeat_count, contract)
        if repeat_count != count or second != first:
            raise RuntimeError("per-cell export is not deterministic")
        slots = sum(len(meta) // contract["meta_width"] for meta, _ in first)
        print(json.dumps({"status": "PASS", "candidates": count, "slots": slots,
                          "resource_width": contract["resource_width"]}))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
