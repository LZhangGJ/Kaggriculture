#!/usr/bin/env python3
"""Convert frozen dynamic-policy trajectories to live slot-ABI-v2 mmap shards."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import os
import struct
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.train_midgame_autofill_v1 import (
    CLASS_KINDS, MODEL_INPUTS, _bind_slots,
)
from experiments.train_midgame_student_v1 import (
    ROOT, TOKENIZER_ROOT, _canonical_observation, _json_hash, _save_array,
    _sha256,
)
from scripts.extract_midgame_bc import observation


def _canonical_manifest_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items()
               if key not in ("canonical_sha256", "digest_semantics")}
    return _json_hash(payload)


def _load_jobs(args) -> tuple[list[dict], dict]:
    corpus = args.corpus.resolve()
    policy_path = (corpus / args.policy_manifest).resolve()
    binding_path = (corpus / args.binding).resolve()
    policy = json.loads(policy_path.read_text())
    binding = json.loads(binding_path.read_text())
    canonical = policy.get("canonical_sha256")
    if (policy.get("schema") != "kaggriculture-policy-deployment-v1" or
            canonical != _canonical_manifest_hash(policy)):
        raise ValueError("invalid canonical deployment manifest")
    if (binding.get("schema") != "kaggriculture-corpus-policy-binding-v1" or
            binding.get("policy_manifest_canonical_sha256") != canonical or
            binding.get("policy_manifest_file_sha256") != _sha256(policy_path)):
        raise ValueError("invalid corpus policy binding")
    for relative, expected in policy["files"].items():
        path = (ROOT / relative).resolve()
        if _sha256(path) != expected:
            raise ValueError(f"frozen policy file digest mismatch: {relative}")

    legacy = set(binding["corpora"])
    names = args.manifest or sorted(path.name for path in corpus.glob("train-*.json"))
    groups = []
    seen_paths = set()
    manifest_audit = []
    for name in names:
        path = (corpus / name).resolve()
        result = json.loads(path.read_text())
        embedded_canonical = result.get("policy_manifest_canonical_sha256")
        embedded_legacy_name = result.get("policy_manifest_sha256")
        if (embedded_canonical is not None and embedded_legacy_name is not None and
                embedded_canonical != embedded_legacy_name):
            raise ValueError(f"{name}: conflicting embedded policy hashes")
        embedded = embedded_canonical or embedded_legacy_name
        legacy_attested = name in legacy
        if embedded != canonical and not (embedded is None and legacy_attested):
            raise ValueError(f"{name}: neither canonical hash nor legacy sidecar binding")
        if result.get("diagnostic") not in (None, False):
            raise ValueError(f"{name}: diagnostic corpus is forbidden")
        jobs = []
        for row in result.get("rows", []):
            if row.get("error") is not None:
                raise ValueError(f"{name}: errored result row")
            opponent = row.get("opponent", row.get("bot"))
            trajectory = Path(row["trajectory"])
            if not trajectory.is_absolute():
                trajectory = (ROOT / trajectory).resolve()
            else:
                trajectory = trajectory.resolve()
            if trajectory in seen_paths:
                raise ValueError(f"{name}: duplicate trajectory {trajectory}")
            seen_paths.add(trajectory)
            jobs.append({
                "trajectory": str(trajectory), "seed": int(row["seed"]),
                "seat": int(row["seat"]), "opponent": str(opponent),
                "trajectory_format": result["trajectory_format"],
                "policy_sha256": canonical, "legacy_attested": legacy_attested,
                "source_manifest": str(path),
                "source_manifest_sha256": _sha256(path),
            })
        jobs.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"],
                                   row["trajectory"]))
        groups.append(jobs)
        manifest_audit.append({
            "path": str(path), "sha256": _sha256(path), "rows": len(jobs),
            "binding": "embedded-canonical" if embedded == canonical else "legacy-sidecar",
        })

    # Round-robin keeps small pilot shards diverse across requested manifests.
    ordered = []
    for index in range(max(map(len, groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    selected = ordered[args.start_game:]
    if args.max_games:
        selected = selected[:args.max_games]
    if not selected:
        raise ValueError("no corpus games selected")
    missing = [job["trajectory"] for job in selected
               if not Path(job["trajectory"]).is_file()]
    if missing:
        raise ValueError(f"selected trajectory is missing: {missing[0]}")
    return selected, {
        "policy_manifest": str(policy_path), "policy_manifest_sha256": _sha256(policy_path),
        "policy_manifest_canonical_sha256": canonical,
        "binding": str(binding_path), "binding_sha256": _sha256(binding_path),
        "manifests": manifest_audit, "corpus_games": len(ordered),
        "selection_start": args.start_game, "selection_games": len(selected),
    }


def _bind_live(lib) -> None:
    lib.td_student_pre_context_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_pre_context_observation.restype = ctypes.c_int
    lib.td_student_live_meta.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_meta.restype = ctypes.c_int
    lib.td_student_live_slot_count.argtypes = [ctypes.c_void_p]
    lib.td_student_live_slot_count.restype = ctypes.c_int
    lib.td_student_live_slot_meta.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_slot_meta.restype = ctypes.c_int
    lib.td_student_live_slot_resources.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_live_slot_resources.restype = ctypes.c_int
    lib.td_teacher_context_names.argtypes = [ctypes.c_void_p]
    lib.td_teacher_context_names.restype = ctypes.c_char_p
    lib.td_student_prepare_prefix_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_student_prepare_prefix_observation.restype = ctypes.c_int


def _live_meta(lib, handle) -> tuple[int, ...]:
    values = (ctypes.c_int32 * 6)()
    if lib.td_student_live_meta(handle, values, len(values)) != len(values):
        raise RuntimeError("live plan metadata export failed")
    return tuple(map(int, values))


def _slot_rows(lib, handle, contract: dict) -> list[dict]:
    count = lib.td_student_live_slot_count(handle)
    if count < 0:
        raise RuntimeError("live slot count export failed")
    if count == 0:
        return []
    meta = (ctypes.c_int32 * (count * contract["meta_width"]))()
    resources = (ctypes.c_double * (count * contract["resource_width"]))()
    if lib.td_student_live_slot_meta(handle, meta, len(meta)) != count:
        raise RuntimeError("live slot meta export failed")
    if lib.td_student_live_slot_resources(handle, resources, len(resources)) != count:
        raise RuntimeError("live slot resources export failed")
    result = []
    seen = set()
    width = contract["resource_width"]
    for index in range(count):
        cell, label, mask, terminal = map(int, meta[index * 4:(index + 1) * 4])
        resource = list(resources[index * width:(index + 1) * width])
        if (not 0 <= cell < 100 or cell in seen or label not in CLASS_KINDS or
                mask & ~0x1FF or not mask & 1 or
                not mask & (1 << CLASS_KINDS.index(label)) or terminal not in (0, 1) or
                (terminal and (index != count - 1 or label != -1)) or
                not np.all(np.isfinite(resource)) or resource[3] != index or
                resource[4] < index + 1):
            raise RuntimeError(f"invalid live slot {index}")
        seen.add(cell)
        result.append({
            "cell": cell, "label_kind": label,
            "label_class": CLASS_KINDS.index(label), "legal_mask": mask,
            "terminal": terminal, "resources": resource,
        })
    return result


def _base_slot_rows(lib, handle, contract: dict, packed) -> list[dict]:
    """Export one fixed-base plan without changing the live R1 trajectory."""
    count = lib.td_student_prepare_prefix_observation(
        handle, packed, len(packed), None, None, 0, None, 0)
    if count < 0:
        raise RuntimeError(lib.td_debug(handle).decode())
    if count == 0:
        return []
    meta = (ctypes.c_int32 * (count * contract["meta_width"]))()
    resources = (ctypes.c_double * (count * contract["resource_width"]))()
    if lib.td_student_candidate_slot_meta(handle, 0, meta, len(meta)) != count:
        raise RuntimeError("base slot meta export failed")
    if lib.td_student_candidate_slot_resources(
            handle, 0, resources, len(resources)) != count:
        raise RuntimeError("base slot resource export failed")
    result, seen = [], set()
    width = contract["resource_width"]
    for index in range(count):
        cell, label, mask, terminal = map(int, meta[index * 4:(index + 1) * 4])
        resource = list(resources[index * width:(index + 1) * width])
        if (not 0 <= cell < 100 or cell in seen or label not in CLASS_KINDS or
                mask & ~0x1FF or not mask & 1 or
                not mask & (1 << CLASS_KINDS.index(label)) or terminal not in (0, 1) or
                (terminal and (index != count - 1 or label != -1)) or
                not np.all(np.isfinite(resource)) or resource[3] != index or
                resource[4] < index + 1):
            raise RuntimeError(f"invalid base slot {index}")
        seen.add(cell)
        result.append({
            "cell": cell, "label_kind": label,
            "label_class": CLASS_KINDS.index(label), "legal_mask": mask,
            "terminal": terminal, "resources": resource,
        })
    return result


def _extract_game(job: dict) -> dict:
    from experiments.audit_teacher_batch import names_sha256, r1_module

    path = Path(job["trajectory"])
    module = r1_module()
    agent = module.Agent(binary_path=Path(job["binary"]))
    meta = None
    terminal = 0
    steps = []
    states = []
    action_frames = 0
    try:
        lib = agent.lib
        _bind_slots(lib)
        _bind_live(lib)
        contract = json.loads(lib.td_student_slot_contract_json())
        resource_names = lib.td_student_slot_resource_names().decode().splitlines()
        context_names = lib.td_teacher_context_names(agent.handle).decode().splitlines()
        if (lib.td_student_slot_abi_version() != 2 or contract["abi_version"] != 2 or
                tuple(contract["class_kinds"]) != CLASS_KINDS or
                contract["label_semantics"] != "greedy_proposal_before_preview" or
                contract["prefix_semantics"] != "before_current_slot" or
                len(resource_names) != contract["resource_width"] or
                len(set(resource_names)) != len(resource_names) or
                names_sha256(context_names) !=
                "6386bae567618f5c8409075b420dc5dbff3bb616b236bb714aacde151beff3a8"):
            raise RuntimeError("live slot/context ABI mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("type") == "meta":
                    if meta is not None:
                        raise ValueError("multiple trajectory metadata rows")
                    meta = row
                    if (meta.get("format") != job["trajectory_format"] or
                            int(meta.get("seed", -1)) != job["seed"] or
                            int(meta.get("seat", -1)) != job["seat"] or
                            meta.get("bot") != job["opponent"] or
                            meta.get("label") != "candidate" or
                            meta.get("config_override") is not None or
                            str(meta.get("handoff_selector", "0")) != "0"):
                        raise ValueError("trajectory metadata mismatch")
                    embedded_canonical = meta.get("policy_manifest_canonical_sha256")
                    embedded_legacy_name = meta.get("policy_manifest_sha256")
                    if (embedded_canonical is not None and
                            embedded_legacy_name is not None and
                            embedded_canonical != embedded_legacy_name):
                        raise ValueError("trajectory has conflicting policy hashes")
                    embedded = embedded_canonical or embedded_legacy_name
                    if embedded not in (None, job["policy_sha256"]):
                        raise ValueError("trajectory policy digest mismatch")
                    if embedded is None and not job["legacy_attested"]:
                        raise ValueError("trajectory lacks canonical policy binding")
                    continue
                if row.get("type") == "terminal":
                    terminal += 1
                    continue
                if row.get("type") != "step" or meta is None:
                    if row.get("type") == "error":
                        raise ValueError("errored trajectory")
                    raise ValueError("unexpected trajectory row")
                step = int(row["step"])
                steps.append(step)
                seat = job["seat"]
                current = observation(row, seat)
                expected = row["actions"][seat]
                if step < 288:
                    agent.observe_external(current, expected)
                    continue

                pending = None
                if step % 24 == 0:
                    packed = module._pack(current)
                    if agent.external:
                        if lib.td_activate_external(agent.handle, packed, len(packed)):
                            raise RuntimeError(lib.td_debug(agent.handle).decode())
                        agent.external = False
                    context = (ctypes.c_double * len(context_names))()
                    got = lib.td_student_pre_context_observation(
                        agent.handle, packed, len(packed), context, len(context))
                    if got != len(context):
                        raise RuntimeError(lib.td_debug(agent.handle).decode())
                    pending = {
                        "step": step, "context": list(context),
                        "observation": _canonical_observation(current),
                        "packed_observation": list(module._pack(
                            _canonical_observation(current))),
                        "before": _live_meta(lib, agent.handle),
                    }
                    if job["scaffold"] == "base":
                        pending["slots"] = _base_slot_rows(
                            lib, agent.handle, contract, packed)

                actual = agent(current)
                action_frames += 1
                if actual != expected:
                    raise RuntimeError(
                        f"action parity mismatch step={step}: expected={expected!r} actual={actual!r}")
                if pending is not None:
                    after = _live_meta(lib, agent.handle)
                    if (after[0] != step or after[1] != pending["before"][1] + 1 or
                            after[2] != step // 24):
                        raise RuntimeError(
                            f"day-boundary choose metadata mismatch: {pending['before']} -> {after}")
                    slots = (pending["slots"] if job["scaffold"] == "base" else
                             _slot_rows(lib, agent.handle, contract))
                    if slots:
                        pending.update({"after": after, "slots": slots})
                        states.append(pending)
        if meta is None or steps != list(range(719)) or terminal != 1:
            raise ValueError("trajectory must contain steps 0..718 and one terminal")
        family = meta.get("source_family") or meta.get("bot")
        return {
            "seed": job["seed"], "seat": job["seat"],
            "opponent": job["opponent"], "opponent_family": str(family),
            "trajectory": str(path), "source_manifest": job["source_manifest"],
            "slot_contract": contract, "resource_names": resource_names,
            "context_names": context_names, "states": states,
            "action_frames": action_frames, "day_boundaries": 18,
        }
    finally:
        agent.close()


def _worker_init() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def build(args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output}")
    if _sha256(args.binary) != args.binary_sha256:
        raise ValueError("live slot teacher binary digest mismatch")
    jobs, source_audit = _load_jobs(args)
    for job in jobs:
        job["binary"] = str(args.binary.resolve())
        job["scaffold"] = args.scaffold
    _worker_init()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    games = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as pool:
        futures = {pool.submit(_extract_game, job): job for job in jobs}
        for future in as_completed(futures):
            games.append(future.result())
    extract_seconds = time.perf_counter() - started
    games.sort(key=lambda row: (row["seed"], row["opponent_family"], row["seat"],
                                row["trajectory"]))
    first = games[0]
    for game in games:
        if (game["slot_contract"] != first["slot_contract"] or
                game["resource_names"] != first["resource_names"] or
                game["context_names"] != first["context_names"]):
            raise RuntimeError("slot/context schema drift")
    states = [(game, state) for game in games for state in game["states"]]
    if not states:
        raise RuntimeError("no non-empty live slot states")

    labels_by_input = defaultdict(set)
    for _, state in states:
        state_bytes = struct.pack(f'<{len(state["context"])}d', *state["context"])
        observation_bytes = struct.pack(
            f'<{len(state["packed_observation"])}d', *state["packed_observation"])
        for slot in state["slots"]:
            payload = (state_bytes + observation_bytes +
                       struct.pack("<iH", slot["cell"], slot["legal_mask"]) +
                       struct.pack(f'<{len(slot["resources"])}d', *slot["resources"]))
            labels_by_input[hashlib.sha256(payload).digest()].add(slot["label_class"])
    conflicts = sum(len(labels) > 1 for labels in labels_by_input.values())
    if conflicts:
        raise RuntimeError(f"{conflicts} identical live slot inputs have conflicting labels")

    sys.path.insert(0, str(TOKENIZER_ROOT))
    from kaggrl.tokenizer import ObservationTokenizer
    from experiments.midgame_autofill_student import contains_private_identity

    tokenizer = ObservationTokenizer()
    encoded = []
    for _, state in states:
        if contains_private_identity(state["observation"]):
            raise ValueError("forbidden identity in canonical observation")
        encoded.append(tokenizer.encode(state["observation"]))
    max_tokens = 320
    if max(row.num_tokens for row in encoded) > max_tokens:
        raise ValueError("token count exceeds fixed 320-token contract")
    state_count = len(states)
    slot_count = sum(len(state["slots"]) for _, state in states)
    context_width = len(first["context_names"])
    resource_width = len(first["resource_names"])
    packed_width = max(len(state["packed_observation"]) for _, state in states)
    arrays = {
        "causal_context": np.asarray([state["context"] for _, state in states], dtype="<f4"),
        "packed_observation": np.zeros((state_count, packed_width), dtype="<f8"),
        "observation_length": np.asarray(
            [len(state["packed_observation"]) for _, state in states], dtype="<u2"),
        "token_continuous": np.zeros((state_count, max_tokens, 24), dtype="<f4"),
        "token_type": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_a": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_b": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_category_c": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_x": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_y": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_owner": np.zeros((state_count, max_tokens), dtype="u1"),
        "token_count": np.asarray([row.num_tokens for row in encoded], dtype="<u2"),
        "state_slot_offsets": np.zeros(state_count + 1, dtype="<u4"),
        "state_step": np.empty(state_count, dtype="<u2"),
        "slot_cell": np.empty(slot_count, dtype="u1"),
        "slot_label_kind": np.empty(slot_count, dtype="i1"),
        "slot_label_class": np.empty(slot_count, dtype="u1"),
        "slot_legal_mask": np.empty(slot_count, dtype="<u2"),
        "slot_terminal": np.empty(slot_count, dtype="u1"),
        "slot_resources": np.empty((slot_count, resource_width), dtype="<f4"),
        "split": np.empty(state_count, dtype="u1"),
        "group_hash128": np.empty((state_count, 16), dtype="u1"),
    }
    categorical = ("token_type", "token_category_a", "token_category_b",
                   "token_category_c", "token_x", "token_y", "token_owner")
    encoded_fields = ("token_type", "category_a", "category_b", "category_c",
                      "x", "y", "owner")
    audit_groups = []
    cursor = 0
    for index, ((game, state), tokenized) in enumerate(zip(states, encoded)):
        packed = state["packed_observation"]
        arrays["packed_observation"][index, :len(packed)] = packed
        count = tokenized.num_tokens
        arrays["token_continuous"][index, :count] = tokenized.continuous.numpy()
        for target, source in zip(categorical, encoded_fields):
            arrays[target][index, :count] = getattr(tokenized, source).numpy()
        arrays["state_slot_offsets"][index] = cursor
        arrays["state_step"][index] = state["step"]
        for slot in state["slots"]:
            arrays["slot_cell"][cursor] = slot["cell"]
            arrays["slot_label_kind"][cursor] = slot["label_kind"]
            arrays["slot_label_class"][cursor] = slot["label_class"]
            arrays["slot_legal_mask"][cursor] = slot["legal_mask"]
            arrays["slot_terminal"][cursor] = slot["terminal"]
            arrays["slot_resources"][cursor] = slot["resources"]
            cursor += 1
        group = f'{game["seed"]}:{game["opponent_family"]}'
        digest = hashlib.sha256(group.encode()).digest()[:16]
        arrays["group_hash128"][index] = np.frombuffer(digest, dtype=np.uint8)
        arrays["split"][index] = int(digest[0] < 51)
        audit_groups.append({
            "group_hash128": digest.hex(), "seed": game["seed"],
            "opponent_family": game["opponent_family"], "opponent": game["opponent"],
            "seat": game["seat"], "step": state["step"],
            "trajectory": game["trajectory"], "source_manifest": game["source_manifest"],
        })
    arrays["state_slot_offsets"][-1] = cursor
    if (cursor != slot_count or not np.any(arrays["split"] == 0) or
            not np.any(arrays["split"] == 1)):
        raise RuntimeError("invalid live offsets or empty split")
    legal = ((arrays["slot_legal_mask"] >> arrays["slot_label_class"]) & 1).astype(bool)
    if not np.all(legal):
        raise RuntimeError("illegal live teacher label")
    array_manifest = {name: _save_array(args.output, name, value)
                      for name, value in arrays.items()}
    schema = {
        "schema_name": "autoregressive-slot-bc-v2", "container": "numpy .npy mmap",
        "class_kinds": list(CLASS_KINDS), "model_input_arrays": sorted(MODEL_INPUTS),
        "target_only_arrays": ["slot_label_kind", "slot_label_class", "slot_terminal"],
        "audit_only_arrays": ["state_step", "split", "group_hash128"],
        "teacher_plan": ("fixed base-settings greedy plan on real step288+ history"
                         if args.scaffold == "base" else
                         "actual live SearchController winner after real step288+ history"),
        "label_semantics": "greedy_proposal_before_preview",
        "candidate_ranking": "absent from dataset construction, model, and loss",
        "identity_policy": "seed/opponent/seat are audit-only and absent from forward",
    }
    class_counts = Counter(map(int, arrays["slot_label_kind"]))
    manifest = {
        "format": "kaggriculture-midgame-mmap", "container_version": 1,
        "validation_status": "accepted", "schema": schema,
        "schema_sha256": _json_hash(schema), "byte_order": "little",
        "constants": {"terminal_action_exclusive": 719, "turns_per_day": 24,
                      "max_tokens": max_tokens},
        "counts": {"games": len(games), "states": state_count, "slots": slot_count,
                   "train_states": int(np.sum(arrays["split"] == 0)),
                   "heldout_states": int(np.sum(arrays["split"] == 1)),
                   "action_frames": sum(game["action_frames"] for game in games),
                   "day_boundaries": sum(game["day_boundaries"] for game in games)},
        "class_counts_by_kind": {str(kind): class_counts[kind] for kind in CLASS_KINDS},
        "dimensions": {"causal_context": context_width,
                       "packed_observation": packed_width,
                       "slot_resources": resource_width},
        "arrays": array_manifest, "slot_contract": first["slot_contract"],
        "slot_resource_names": first["resource_names"],
        "causal_context_names": first["context_names"],
        "teacher_binary": str(args.binary.resolve()),
        "teacher_binary_sha256": args.binary_sha256,
        "execution_scaffold": args.scaffold,
        "source": source_audit,
        "split_method": "sha256(seed:opponent_family)[0] < 51; whole family-seed group",
        "audit_groups": audit_groups,
        "validation": {
            "action_parity_frames": sum(game["action_frames"] for game in games),
            "action_parity_mismatches": 0, "actual_choose_coverage": 1.0,
            "identical_input_label_conflicts": conflicts,
            "label_legal_fraction": float(np.mean(legal)),
            "prefix_semantics": "before_current_slot",
            "trajectory_steps_0_718": "PASS", "one_terminal_per_trajectory": "PASS",
            "post288_commitments": "restored by driving the same live R1 from step288",
            "slot_scaffold": ("fixed base settings used by online callback"
                              if args.scaffold == "base" else
                              "live score-winner settings"),
            "no_identity_in_model_inputs": "PASS",
        },
        "timing": {"extract_seconds": extract_seconds,
                   "games_per_second": len(games) / extract_seconds},
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      **manifest["counts"], **manifest["validation"],
                      **manifest["timing"], "class_counts_by_kind":
                      manifest["class_counts_by_kind"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/bc/midgame-v1")
    parser.add_argument("--policy-manifest", default="policy-deployment-bd49e2a13e44.json")
    parser.add_argument("--binding", default="corpus-policy-binding-bd49e2a13e44.json")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(160, os.cpu_count() or 1))
    parser.add_argument("--scaffold", choices=("live-winner", "base"),
                        default="live-winner")
    args = parser.parse_args()
    if args.start_game < 0 or args.max_games < 0 or args.workers < 1:
        parser.error("start/max games and workers must be non-negative")
    build(args)


if __name__ == "__main__":
    main()
