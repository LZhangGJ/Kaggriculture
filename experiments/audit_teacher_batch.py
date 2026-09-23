#!/usr/bin/env python3
"""Replay saved warm histories and export diagnostic multi-scenario teacher rows."""
import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import struct
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def r1_module():
    spec = importlib.util.spec_from_file_location("teacher_r1_agent", ROOT / "policy/r1/agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore(module, binary, path, step):
    agent = module.Agent(binary_path=binary)
    meta = handoff = None
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("type") == "meta":
                meta = row
                continue
            if row.get("type") != "step":
                continue
            seat = int(meta["seat"])
            observation = {**row["public"], "private": row["private"][seat], "player": seat}
            if row["step"] == step:
                handoff = observation
                break
            agent.observe_external(observation, row["actions"][seat])
    if not meta or handoff is None or agent.last != step - 1:
        agent.close()
        raise ValueError(f"{path}: trajectory does not contain a consecutive step {step}")
    packed = module._pack(handoff)
    # The teacher ABI consumes this boundary frame using the same observation
    # update prefix as SearchController::act; do not pre-observe it here.
    agent.external = False
    return agent, meta, handoff, list(packed)


def bind(lib):
    lib.td_teacher_abi_version.argtypes = []
    lib.td_teacher_abi_version.restype = ctypes.c_int
    lib.td_teacher_row_width.argtypes = []
    lib.td_teacher_row_width.restype = ctypes.c_size_t
    lib.td_teacher_contract_json.argtypes = []
    lib.td_teacher_contract_json.restype = ctypes.c_char_p
    lib.td_teacher_scenarios_json.argtypes = []
    lib.td_teacher_scenarios_json.restype = ctypes.c_char_p
    lib.td_teacher_prepare_features_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_teacher_prepare_features_observation.restype = ctypes.c_int
    lib.td_teacher_candidate_feature_names.argtypes = [ctypes.c_void_p]
    lib.td_teacher_candidate_feature_names.restype = ctypes.c_char_p
    lib.td_teacher_context_names.argtypes = [ctypes.c_void_p]
    lib.td_teacher_context_names.restype = ctypes.c_char_p
    lib.td_teacher_context.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_teacher_context.restype = ctypes.c_int
    lib.td_teacher_candidate_key.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_teacher_candidate_key.restype = ctypes.c_int
    lib.td_teacher_last_timings.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_teacher_last_timings.restype = ctypes.c_int
    lib.td_teacher_batch_many.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)]
    lib.td_teacher_batch_many.restype = ctypes.c_int
    lib.td_teacher_candidate_targets.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_teacher_candidate_targets.restype = ctypes.c_int
    lib.td_candidate_features.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_candidate_features.restype = ctypes.c_int
    lib.td_candidate_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.td_candidate_id.restype = ctypes.c_int


def names_from(lib, symbol, handle):
    raw = getattr(lib, symbol)(handle)
    if not raw:
        raise RuntimeError(f"{symbol} failed")
    return raw.decode().splitlines()


def names_sha256(names):
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def candidate_features(lib, handle, index, width):
    output = (ctypes.c_double * width)()
    size = lib.td_candidate_features(handle, index, output, width)
    if size != width:
        raise RuntimeError(f"candidate {index} feature export returned {size}, expected {width}")
    return list(output)


def candidate_key(lib, handle, index):
    size = lib.td_teacher_candidate_key(handle, index, None, 0)
    if size <= 0:
        raise RuntimeError(f"candidate {index} key size returned {size}")
    output = (ctypes.c_int32 * size)()
    written = lib.td_teacher_candidate_key(handle, index, output, size)
    if written != size:
        raise RuntimeError(f"candidate {index} key export returned {written}, expected {size}")
    return list(output)


def key_hash128(key):
    payload = struct.pack(f"<{len(key)}i", *key)
    return hashlib.sha256(payload).hexdigest()[:32]


def causal_context(lib, handle, width):
    output = (ctypes.c_double * width)()
    size = lib.td_teacher_context(handle, output, width)
    if size != width:
        raise RuntimeError(f"causal context export returned {size}, expected {width}")
    return list(output)


def feature_snapshot(lib, handle, count, width):
    return [{
        "prepared_index": index,
        "candidate_id": lib.td_candidate_id(handle, index),
        "proposal_key": candidate_key(lib, handle, index),
        "features": candidate_features(lib, handle, index, width),
    } for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--step", type=int, default=288)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--check-repeat-batch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step != 288 or args.top_k < 1 or args.output.exists():
        parser.error("v2 requires step=288, positive top-k, and a new output path")
    if len(args.trajectory) != 1:
        parser.error("one warm state per process; use process-level parallelism across states")
    module = r1_module()
    restored = [restore(module, args.binary.resolve(), path.resolve(), args.step)
                for path in args.trajectory]
    try:
        agent, meta, observation, values = restored[0]
        lib = agent.lib
        bind(lib)
        contract = json.loads(lib.td_teacher_contract_json())
        if lib.td_teacher_abi_version() != 2 or contract.get("abi_version") != 2:
            raise RuntimeError("teacher ABI version mismatch")
        width = lib.td_teacher_row_width()
        if width != len(contract["row_fields"]):
            raise RuntimeError("teacher row schema/width mismatch")
        scenarios = json.loads(lib.td_teacher_scenarios_json())
        if len(scenarios) != 4:
            raise RuntimeError("teacher scenario count mismatch")
        packed = (ctypes.c_double * len(values))(*values)

        feature_started = time.perf_counter()
        normal_count = lib.td_teacher_prepare_features_observation(
            agent.handle, packed, len(packed))
        feature_seconds = time.perf_counter() - feature_started
        if normal_count <= 0:
            raise RuntimeError(lib.td_debug(agent.handle).decode())
        feature_names = names_from(
            lib, "td_teacher_candidate_feature_names", agent.handle)
        context_names = names_from(lib, "td_teacher_context_names", agent.handle)
        feature_width = len(feature_names)
        context_width = len(context_names)
        if (feature_width != contract["candidate_feature_width"] or
                names_sha256(feature_names) != contract["candidate_feature_names_sha256"]):
            raise RuntimeError("candidate feature schema hash/width mismatch")
        if (context_width != contract["causal_context_width"] or
                names_sha256(context_names) != contract["causal_context_names_sha256"]):
            raise RuntimeError("causal context schema hash/width mismatch")
        first_snapshot = feature_snapshot(
            lib, agent.handle, normal_count, feature_width)
        first_context = causal_context(lib, agent.handle, context_width)

        repeat_started = time.perf_counter()
        repeat_count = lib.td_teacher_prepare_features_observation(
            agent.handle, packed, len(packed))
        feature_repeat_seconds = time.perf_counter() - repeat_started
        second_snapshot = feature_snapshot(
            lib, agent.handle, repeat_count, feature_width)
        second_context = causal_context(lib, agent.handle, context_width)
        if (repeat_count != normal_count or second_snapshot != first_snapshot or
                second_context != first_context):
            raise RuntimeError("feature/context export is not deterministic")
        canonical_normals = [row for row in first_snapshot if row["candidate_id"] == 0]
        if len(canonical_normals) != 1:
            raise RuntimeError("feature-only candidate set lacks one canonical id0")
        canonical_normal_hash = key_hash128(canonical_normals[0]["proposal_key"])

        handles = (ctypes.c_void_p * 1)(agent.handle)
        packed_offsets = (ctypes.c_size_t * 2)(0, len(values))
        stride = (2 * args.top_k + 1) * width

        def run_batch():
            output = (ctypes.c_double * stride)()
            counts = (ctypes.c_int32 * 1)()
            started = time.perf_counter()
            status = lib.td_teacher_batch_many(
                handles, 1, packed, packed_offsets, args.top_k,
                output, len(output), counts)
            elapsed = time.perf_counter() - started
            if status < 0:
                raise RuntimeError((status, list(counts), lib.td_debug(agent.handle).decode()))
            timing = (ctypes.c_double * 2)()
            if lib.td_teacher_last_timings(timing, len(timing)) != 2:
                raise RuntimeError("teacher timing export failed")
            return counts[0], list(output[:counts[0] * width]), elapsed, list(timing)

        count, raw_rows, elapsed, teacher_timing = run_batch()
        repeat_batch_seconds = None
        if args.check_repeat_batch:
            repeated_count, repeated_rows, repeat_batch_seconds, _ = run_batch()
            if repeated_count != count or repeated_rows != raw_rows:
                raise RuntimeError("teacher batch is not deterministic")

        if (names_from(lib, "td_teacher_candidate_feature_names", agent.handle) !=
                feature_names or causal_context(lib, agent.handle, context_width) != first_context):
            raise RuntimeError("full teacher changed feature/context schema or state")
        candidates = []
        for row in range(count):
            raw = raw_rows[row * width:(row + 1) * width]
            index, candidate_id, family = map(int, raw[:3])
            is_outer = bool(raw[3])
            if family not in (0, 1) or is_outer != (family == 1):
                raise RuntimeError("diagnostic candidate leaked into teacher selection")
            if lib.td_candidate_id(agent.handle, index) != candidate_id:
                raise RuntimeError("candidate id mismatch")
            targets = (ctypes.c_int32 * 200)()
            target_count = lib.td_teacher_candidate_targets(
                agent.handle, index, targets, len(targets))
            if target_count < 0:
                raise RuntimeError(f"target export failed: {target_count}")
            key = candidate_key(lib, agent.handle, index)
            candidates.append({
                "prepared_index": index,
                "candidate_id": candidate_id,
                "candidate_family": "outer" if family else "normal",
                "is_outer": is_outer,
                "short_score": raw[4],
                "canonical_anchor_prepared_index": int(raw[5]),
                "selection_reference_prepared_index": int(raw[6]),
                "scenario_q": raw[7:11],
                "paired_advantage": raw[11:15],
                "selection_paired_advantage": raw[15:19],
                "scenario_valid": [True] * 4,
                "proposal_key_length": len(key),
                "proposal_key_hash128": key_hash128(key),
                "features": candidate_features(lib, agent.handle, index, feature_width),
                "target": [[targets[2 * i], targets[2 * i + 1]]
                           for i in range(target_count)],
            })
        by_index = {row["prepared_index"]: row for row in candidates}
        anchors = {row["canonical_anchor_prepared_index"] for row in candidates}
        references = {row["selection_reference_prepared_index"] for row in candidates}
        if len(anchors) != 1 or len(references) != 1:
            raise RuntimeError("teacher rows disagree on reference candidates")
        anchor_index, reference_index = anchors.pop(), references.pop()
        if (anchor_index not in by_index or reference_index not in by_index or
                by_index[anchor_index]["candidate_family"] != "normal" or
                by_index[anchor_index]["candidate_id"] != 0 or
                by_index[reference_index]["candidate_family"] != "normal"):
            raise RuntimeError("invalid canonical/selection reference pairing")
        anchor_q = by_index[anchor_index]["scenario_q"]
        reference_q = by_index[reference_index]["scenario_q"]
        for candidate in candidates:
            for scenario, q in enumerate(candidate["scenario_q"]):
                tolerance = 1e-9 * max(1.0, abs(q))
                if abs(candidate["paired_advantage"][scenario] -
                       (q - anchor_q[scenario])) > tolerance:
                    raise RuntimeError("canonical paired advantage mismatch")
                if abs(candidate["selection_paired_advantage"][scenario] -
                       (q - reference_q[scenario])) > tolerance:
                    raise RuntimeError("selection paired advantage mismatch")
        if by_index[anchor_index]["proposal_key_hash128"] != canonical_normal_hash:
            raise RuntimeError("canonical anchor changed between feature-only and teacher prepare")

        normal_candidates = [{
            "prepared_index": row["prepared_index"],
            "candidate_id": row["candidate_id"],
            "proposal_key_length": len(row["proposal_key"]),
            "proposal_key_hash128": key_hash128(row["proposal_key"]),
            "features": row["features"],
        } for row in first_snapshot]
        clock = agent.clock_debug()
        rows = [{
            "step": args.step,
            "source": str(args.trajectory[0].resolve()),
            "audit_group": {
                "seed": meta.get("seed"), "seat": meta.get("seat"),
                "policy_label": meta.get("label"), "opponent": meta.get("bot"),
            },
            "packed_observation": {
                "encoding": "little_endian_float64_values_from_observation_codec",
                "value_count": len(values),
                "sha256": hashlib.sha256(
                    struct.pack(f"<{len(values)}d", *values)).hexdigest(),
            },
            "public_stock_history_restored": True,
            "stock_interval": {
                "lower": clock["rival_pending_lower"],
                "upper": clock["rival_pending_upper"],
                "lower_sum": sum(clock["rival_pending_lower"]),
                "upper_sum": sum(clock["rival_pending_upper"]),
            },
            "causal_context": first_context,
            "online_normal_candidates": normal_candidates,
            "canonical_anchor_prepared_index": anchor_index,
            "canonical_anchor_key_hash128": canonical_normal_hash,
            "selection_reference_prepared_index": reference_index,
            "candidates": candidates,
        }]
        result = {
            "format": "kaggriculture-candidate-critic-export-v2",
            "diagnostic": True, "training_ready": False,
            "context_complete": False,
            "incomplete_reasons": [
                "full canonical public frame remains in the hashed source trajectory",
                "post-handoff commitment restoration is intentionally unsupported",
                "executor pending/queue snapshot is not exported",
            ],
            "binary": str(args.binary.resolve()),
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "contract": contract,
            "schemas": {
                "candidate_feature_names": feature_names,
                "candidate_feature_names_sha256": names_sha256(feature_names),
                "causal_context_names": context_names,
                "causal_context_names_sha256": names_sha256(context_names),
            },
            "scenarios": scenarios,
            "scenario_semantics": "uncalibrated feasible support; no probability weights",
            "paired_advantage_reference": "canonical normal candidate id0",
            "selection_paired_advantage_role": "diagnostic only; reference may vary with top-k",
            "short_score_role": "audit/teacher selection only; forbidden as model input",
            "top_k_per_family": args.top_k,
            "timing_seconds": {
                "feature_only_prepare": feature_seconds,
                "feature_only_repeat": feature_repeat_seconds,
                "short_score_prepare": teacher_timing[0],
                "scenario_scoring": teacher_timing[1],
                "teacher_total": elapsed,
                "repeat_teacher_total": repeat_batch_seconds,
            },
            "self_checks": {
                "shape_and_schema_hash": "PASS",
                "feature_context_repeat_determinism": "PASS",
                "teacher_batch_repeat_determinism": (
                    "PASS" if args.check_repeat_batch else "NOT_RUN"),
                "canonical_and_selection_reference_pairing": "PASS",
            },
            "states": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "states": len(rows),
                          "rows": count, "elapsed_seconds": elapsed,
                          "output": str(args.output)}))
    finally:
        for agent, *_ in restored:
            agent.close()


if __name__ == "__main__":
    main()
