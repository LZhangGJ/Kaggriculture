#!/usr/bin/env python3
"""Build/train the real step-288 autoregressive per-cell BC student.

Exactly one trace is retained per state: the normal proposal selected by the
production SearchController score.  Proposal keys are audit-only; neither a
candidate id nor a score is a model input.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# This exporter/trainer is CPU-only.  The npu-torch environment registers
# torch_npu as an optional backend, but transient services do not inherit the
# Ascend runtime library path and would otherwise fail while importing torch.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np

from experiments.train_midgame_student_v1 import (
    ROOT, TOKENIZER_ROOT, _canonical_observation, _json_hash, _load_shard,
    _save_array, _sha256, _source_jobs, _validate_trajectory,
)


CLASS_KINDS = (-1, 0, 1, 2, 3, 4, 9, 10, 11)
MODEL_INPUTS = {
    "causal_context", "packed_observation", "observation_length",
    "token_continuous", "token_type", "token_category_a", "token_category_b",
    "token_category_c", "token_x", "token_y", "token_owner", "token_count",
    "state_slot_offsets", "slot_cell", "slot_resources", "slot_legal_mask",
}


def _winner_keys(audit_shard: Path) -> tuple[dict[str, str], str]:
    manifest, arrays = _load_shard(audit_shard)
    if manifest["schema"]["label"] != "argmax short_score among normal candidates":
        raise ValueError("winner audit is not the production normal-score selection")
    labels = np.asarray(arrays["state_label_global"], dtype=np.int64)
    hashes = np.asarray(arrays["candidate_key_hash128"], dtype=np.uint8)
    groups = manifest["audit_groups"]
    if len(groups) != len(labels):
        raise ValueError("winner audit group count mismatch")
    result = {}
    for group, label in zip(groups, labels):
        trajectory = str(Path(group["trajectory"]).resolve())
        if trajectory in result:
            raise ValueError("duplicate winner trajectory")
        result[trajectory] = bytes(hashes[label]).hex()
    return result, _sha256(audit_shard / "manifest.json")


def _bind_slots(lib) -> None:
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


def _extract_state(job: dict) -> dict:
    experiments = ROOT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    from audit_teacher_batch import (
        bind, candidate_key, causal_context, key_hash128, names_from,
        names_sha256, r1_module, restore,
    )

    path = Path(job["trajectory"])
    meta = _validate_trajectory(path, job)
    module = r1_module()
    agent, restored_meta, observation, values = restore(
        module, Path(job["binary"]), path, 288)
    try:
        if restored_meta != meta:
            raise RuntimeError("trajectory metadata changed during restore")
        lib = agent.lib
        bind(lib)
        _bind_slots(lib)
        contract = json.loads(lib.td_student_slot_contract_json())
        resource_names = lib.td_student_slot_resource_names().decode().splitlines()
        if (lib.td_student_slot_abi_version() != 2 or contract["abi_version"] != 2 or
                tuple(contract["class_kinds"]) != CLASS_KINDS or
                contract["meta_fields"] != ["cell", "proposed_kind", "legal_mask_9bit", "terminal"] or
                contract["prefix_semantics"] != "before_current_slot" or
                contract["label_semantics"] != "greedy_proposal_before_preview" or
                len(resource_names) != contract["resource_width"] or
                len(set(resource_names)) != len(resource_names)):
            raise RuntimeError("student slot ABI/schema mismatch")
        teacher_contract = json.loads(lib.td_teacher_contract_json())
        packed = (ctypes.c_double * len(values))(*values)
        count = lib.td_teacher_prepare_features_observation(agent.handle, packed, len(packed))
        if count <= 0:
            raise RuntimeError(lib.td_debug(agent.handle).decode())
        matches = [index for index in range(count)
                   if key_hash128(candidate_key(lib, agent.handle, index)) == job["winner_key"]]
        if len(matches) != 1:
            raise RuntimeError(f"production winner key matched {len(matches)} slot traces")
        candidate = matches[0]
        slot_count = lib.td_student_candidate_slot_count(agent.handle, candidate)
        if slot_count <= 0:
            raise RuntimeError("selected winner has no slot trace")
        meta_values = (ctypes.c_int32 * (slot_count * contract["meta_width"]))()
        resources = (ctypes.c_double * (slot_count * contract["resource_width"]))()
        if lib.td_student_candidate_slot_meta(
                agent.handle, candidate, meta_values, len(meta_values)) != slot_count:
            raise RuntimeError("slot meta export failed")
        if lib.td_student_candidate_slot_resources(
                agent.handle, candidate, resources, len(resources)) != slot_count:
            raise RuntimeError("slot resource export failed")
        rows = []
        width = contract["resource_width"]
        seen_cells = set()
        for slot in range(slot_count):
            cell, label, mask, terminal = map(
                int, meta_values[slot * 4:(slot + 1) * 4])
            if (not 0 <= cell < 100 or cell in seen_cells or label not in CLASS_KINDS or
                    mask & ~0x1FF or not mask & 1 or
                    not mask & (1 << CLASS_KINDS.index(label)) or terminal not in (0, 1) or
                    (terminal and (slot != slot_count - 1 or label != -1))):
                raise RuntimeError(f"invalid selected slot {slot}")
            seen_cells.add(cell)
            resource = list(resources[slot * width:(slot + 1) * width])
            if not all(np.isfinite(resource)) or resource[3] != slot or resource[4] < slot + 1:
                raise RuntimeError(f"invalid selected slot resources {slot}")
            rows.append({"cell": cell, "label_kind": label,
                         "label_class": CLASS_KINDS.index(label), "legal_mask": mask,
                         "terminal": terminal, "resources": resource})
        feature_names = names_from(lib, "td_teacher_candidate_feature_names", agent.handle)
        context_names = names_from(lib, "td_teacher_context_names", agent.handle)
        if (names_sha256(feature_names) != teacher_contract["candidate_feature_names_sha256"] or
                names_sha256(context_names) != teacher_contract["causal_context_names_sha256"]):
            raise RuntimeError("teacher feature/context schema mismatch")
        context = causal_context(lib, agent.handle, len(context_names))
        canonical = _canonical_observation(observation)
        return {
            "seed": int(meta["seed"]), "seat": int(meta["seat"]),
            "opponent": str(meta["bot"]), "trajectory": str(path.resolve()),
            "winner_key": job["winner_key"], "matched_candidate_index": candidate,
            "slot_contract": contract, "resource_names": resource_names,
            "teacher_contract": teacher_contract, "context_names": context_names,
            "context": context, "observation": canonical,
            "packed_observation": [float(value) for value in module._pack(canonical)],
            "slots": rows,
        }
    finally:
        agent.close()


def build_shard(args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if _sha256(args.binary) != args.binary_sha256:
        raise ValueError("slot teacher binary digest mismatch")
    jobs, allowlist = _source_jobs(args.allowlist, args.binary, args.max_states)
    winner_keys, winner_manifest_sha256 = _winner_keys(args.winner_audit_shard)
    for job in jobs:
        trajectory = str(Path(job["trajectory"]).resolve())
        if trajectory not in winner_keys:
            raise ValueError(f"winner audit lacks {trajectory}")
        job["winner_key"] = winner_keys[trajectory]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = time.perf_counter()
    states = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_extract_state, job) for job in jobs]
        for future in as_completed(futures):
            states.append(future.result())
    states.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"]))
    extract_seconds = time.perf_counter() - started
    first = states[0]
    for state in states:
        if (state["slot_contract"] != first["slot_contract"] or
                state["resource_names"] != first["resource_names"] or
                state["teacher_contract"] != first["teacher_contract"] or
                state["context_names"] != first["context_names"]):
            raise RuntimeError("slot/context schema drift across states")

    # Exact-input duplicate labels must be deterministic after retaining only
    # the production winner.  This catches accidental multi-candidate mixing.
    labels_by_input = defaultdict(set)
    for state in states:
        state_bytes = struct.pack(f'<{len(state["context"])}d', *state["context"])
        for slot in state["slots"]:
            payload = (state_bytes + struct.pack("<i", slot["cell"]) +
                       struct.pack(f'<{len(slot["resources"])}d', *slot["resources"]))
            labels_by_input[hashlib.sha256(payload).digest()].add(slot["label_class"])
    ambiguous = sum(len(labels) > 1 for labels in labels_by_input.values())
    if ambiguous:
        raise RuntimeError(f"{ambiguous} identical per-slot inputs have conflicting labels")

    sys.path.insert(0, str(TOKENIZER_ROOT))
    from kaggrl.tokenizer import ObservationTokenizer
    from experiments.midgame_autofill_student import contains_private_identity

    tokenizer = ObservationTokenizer()
    encoded = []
    for state in states:
        if contains_private_identity(state["observation"]):
            raise ValueError("forbidden identity in canonical observation")
        encoded.append(tokenizer.encode(state["observation"]))
    max_tokens = 320
    if max(row.num_tokens for row in encoded) > max_tokens:
        raise ValueError("token count exceeds fixed 320-token contract")
    state_count = len(states)
    slot_count = sum(len(state["slots"]) for state in states)
    context_width = len(first["context_names"])
    resource_width = len(first["resource_names"])
    packed_width = max(len(state["packed_observation"]) for state in states)
    arrays = {
        "causal_context": np.asarray([state["context"] for state in states], dtype="<f4"),
        "packed_observation": np.zeros((state_count, packed_width), dtype="<f8"),
        "observation_length": np.asarray(
            [len(state["packed_observation"]) for state in states], dtype="<u2"),
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
        "slot_cell": np.empty(slot_count, dtype="u1"),
        "slot_label_kind": np.empty(slot_count, dtype="i1"),
        "slot_label_class": np.empty(slot_count, dtype="u1"),
        "slot_legal_mask": np.empty(slot_count, dtype="<u2"),
        "slot_terminal": np.empty(slot_count, dtype="u1"),
        "slot_resources": np.empty((slot_count, resource_width), dtype="<f4"),
        "split": np.empty(state_count, dtype="u1"),
        "group_hash128": np.empty((state_count, 16), dtype="u1"),
        "winner_key_hash128": np.empty((state_count, 16), dtype="u1"),
    }
    categorical = ("token_type", "token_category_a", "token_category_b",
                   "token_category_c", "token_x", "token_y", "token_owner")
    encoded_fields = ("token_type", "category_a", "category_b", "category_c",
                      "x", "y", "owner")
    audit_groups = []
    cursor = 0
    for index, (state, tokenized) in enumerate(zip(states, encoded)):
        exact = state["packed_observation"]
        arrays["packed_observation"][index, :len(exact)] = exact
        count = tokenized.num_tokens
        arrays["token_continuous"][index, :count] = tokenized.continuous.numpy()
        for target_name, source_name in zip(categorical, encoded_fields):
            arrays[target_name][index, :count] = getattr(tokenized, source_name).numpy()
        arrays["state_slot_offsets"][index] = cursor
        for slot in state["slots"]:
            arrays["slot_cell"][cursor] = slot["cell"]
            arrays["slot_label_kind"][cursor] = slot["label_kind"]
            arrays["slot_label_class"][cursor] = slot["label_class"]
            arrays["slot_legal_mask"][cursor] = slot["legal_mask"]
            arrays["slot_terminal"][cursor] = slot["terminal"]
            arrays["slot_resources"][cursor] = slot["resources"]
            cursor += 1
        group = f'{state["seed"]}:{state["opponent"]}'
        group_digest = hashlib.sha256(group.encode()).digest()[:16]
        arrays["group_hash128"][index] = np.frombuffer(group_digest, dtype=np.uint8)
        arrays["split"][index] = int(group_digest[0] < 51)
        arrays["winner_key_hash128"][index] = np.frombuffer(
            bytes.fromhex(state["winner_key"]), dtype=np.uint8)
        audit_groups.append({
            "group_hash128": group_digest.hex(), "seed": state["seed"],
            "opponent_family": state["opponent"], "seat": state["seat"],
            "trajectory": state["trajectory"],
            "winner_key_hash128": state["winner_key"],
            "matched_candidate_index": state["matched_candidate_index"],
        })
    arrays["state_slot_offsets"][-1] = cursor
    if (cursor != slot_count or not np.any(arrays["split"] == 0) or
            not np.any(arrays["split"] == 1)):
        raise RuntimeError("invalid slot offsets or empty split")
    legal = ((arrays["slot_legal_mask"] >> arrays["slot_label_class"]) & 1).astype(bool)
    if not np.all(legal):
        raise RuntimeError("illegal teacher label")

    array_manifest = {name: _save_array(args.output, name, value)
                      for name, value in arrays.items()}
    schema = {
        "schema_name": "autoregressive-slot-bc-v2",
        "container": "numpy .npy mmap",
        "class_kinds": list(CLASS_KINDS),
        "model_input_arrays": sorted(MODEL_INPUTS),
        "target_only_arrays": ["slot_label_kind", "slot_label_class", "slot_terminal"],
        "audit_only_arrays": ["split", "group_hash128", "winner_key_hash128"],
        "teacher_plan": "one production SearchController normal-score winner per state",
        "label_semantics": "greedy_proposal_before_preview",
        "candidate_ranking": "absent from model and loss",
        "future_suffix_label": "not used",
        "identity_policy": "seed/opponent/seat are audit-only and absent from forward",
    }
    manifest = {
        "format": "kaggriculture-midgame-mmap", "container_version": 1,
        "validation_status": "accepted", "schema": schema,
        "schema_sha256": _json_hash(schema), "byte_order": "little",
        "constants": {"step": 288, "terminal_action_exclusive": 719,
                      "turns_per_day": 24, "max_tokens": max_tokens},
        "counts": {"states": state_count, "slots": slot_count,
                   "train_states": int(np.sum(arrays["split"] == 0)),
                   "heldout_states": int(np.sum(arrays["split"] == 1))},
        "dimensions": {"causal_context": context_width,
                       "packed_observation": packed_width,
                       "slot_resources": resource_width},
        "arrays": array_manifest,
        "slot_contract": first["slot_contract"],
        "slot_resource_names": first["resource_names"],
        "causal_context_names": first["context_names"],
        "teacher_binary": str(args.binary.resolve()),
        "teacher_binary_sha256": args.binary_sha256,
        "winner_audit_manifest": str((args.winner_audit_shard / "manifest.json").resolve()),
        "winner_audit_manifest_sha256": winner_manifest_sha256,
        "source_allowlist": str(args.allowlist.resolve()),
        "source_allowlist_sha256": _sha256(args.allowlist),
        "source_allowlist_payload_sha256": _json_hash(allowlist),
        "split_method": "sha256(seed:opponent_family)[0] < 51; whole group",
        "audit_groups": audit_groups,
        "validation": {
            "winner_key_matches": state_count, "winner_key_total": state_count,
            "winner_key_coverage": 1.0, "identical_input_label_conflicts": ambiguous,
            "label_legal_fraction": float(np.mean(legal)),
            "prefix_semantics": "before_current_slot", "trajectory_steps_0_718": "PASS",
            "one_terminal_per_trajectory": "PASS", "no_identity_in_model_inputs": "PASS",
        },
        "timing": {"extract_seconds": extract_seconds},
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      **manifest["counts"], **manifest["validation"],
                      **manifest["timing"]}))


def _load_slot_shard(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("validation_status") != "accepted":
        raise ValueError("only accepted slot shards may be trained")
    arrays = {}
    for name, spec in manifest["arrays"].items():
        path = directory / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"{name}: array digest mismatch")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
            raise ValueError(f"{name}: dtype/shape mismatch")
        arrays[name] = value
    if (manifest["schema"]["schema_name"] != "autoregressive-slot-bc-v2" or
            set(manifest["schema"]["model_input_arrays"]) != MODEL_INPUTS):
        raise ValueError("not an autoregressive slot shard")
    offsets = arrays["state_slot_offsets"]
    if offsets[0] != 0 or offsets[-1] != len(arrays["slot_cell"]):
        raise ValueError("invalid state-slot offsets")
    labels = arrays["slot_label_class"]
    if not np.all((arrays["slot_legal_mask"] >> labels) & 1):
        raise ValueError("illegal stored teacher label")
    return manifest, arrays


def build_model(context_width: int, observation_width: int, resource_width: int):
    import torch

    class AutofillStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = torch.nn.Linear(context_width, 16)
            self.observation = torch.nn.Linear(observation_width, 16)
            self.token_embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(size, 4) for size in (7, 32, 32, 32, 64, 64, 4))
            self.token = torch.nn.Linear(24 + 7 * 4, 16)
            self.begin = torch.nn.Linear(48, 64)
            self.resource = torch.nn.Linear(resource_width, 64)
            self.cell = torch.nn.Embedding(100, 8)
            self.previous = torch.nn.Embedding(10, 8)  # nine classes + BOS
            self.gru = torch.nn.GRUCell(80, 64)
            self.head = torch.nn.Linear(64, 9)

        def initial_hidden(self, context, observation, token_continuous,
                           token_categories, token_count):
            positions = torch.arange(token_continuous.shape[1], device=context.device)[None]
            mask = positions < token_count[:, None]
            denominator = token_count.clamp_min(1).float()[:, None]
            continuous = (token_continuous * mask[:, :, None]).sum(1) / denominator
            embedded = [(table(values) * mask[:, :, None]).sum(1) / denominator
                        for table, values in zip(self.token_embeddings, token_categories)]
            token_state = torch.cat((continuous, *embedded), dim=1)
            state = torch.cat((torch.relu(self.context(context)),
                               torch.relu(self.observation(observation)),
                               torch.relu(self.token(token_state))), dim=1)
            return torch.tanh(self.begin(state))

        def step(self, hidden, resources, cells, previous, legal):
            inputs = torch.cat((torch.relu(self.resource(resources)), self.cell(cells),
                                self.previous(previous)), dim=1)
            hidden = self.gru(inputs, hidden)
            logits = self.head(hidden).masked_fill(~legal, -1e9)
            return logits, hidden

    return AutofillStudent()


def train(args) -> None:
    import torch
    import torch.nn.functional as functional

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401 - registers the private-use backend
        if not torch.npu.is_available():
            raise RuntimeError("requested NPU device is unavailable")
        torch.npu.set_device(args.device)
    elif args.device != "cpu":
        raise ValueError("device must be cpu or npu[:index]")
    device = torch.device(args.device)

    manifest, arrays = _load_slot_shard(args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    progress_events = []

    def write_progress(status: str, **extra) -> None:
        value = {
            "status": status,
            "training_task": "autoregressive_slot_actor",
            "schema_name": manifest["schema"]["schema_name"],
            "schema_sha256": manifest["schema_sha256"],
            "shard_manifest_sha256": _sha256(args.shard / "manifest.json"),
            "slot_contract": {
                "abi_version": manifest["slot_contract"]["abi_version"],
                "label_semantics": manifest["slot_contract"]["label_semantics"],
                "prefix_semantics": manifest["slot_contract"]["prefix_semantics"],
            },
            "updated_at_unix": time.time(),
            "states": int(manifest["counts"]["states"]),
            "slots": int(manifest["counts"]["slots"]),
            "epochs": args.epochs,
            "events": progress_events,
            **extra,
        }
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, progress_path)

    torch.manual_seed(args.seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    split = np.asarray(arrays["split"])
    train_states = np.flatnonzero(split == 0)
    heldout_states = np.flatnonzero(split == 1)
    offsets_np = np.asarray(arrays["state_slot_offsets"], dtype=np.int64)
    slot_state_np = np.repeat(np.arange(len(split)), np.diff(offsets_np))
    train_slots = np.isin(slot_state_np, train_states)

    def standardize(value, rows):
        sample = np.asarray(value[rows], dtype=np.float32)
        mean = sample.mean(0, dtype=np.float64).astype(np.float32)
        std = sample.std(0, dtype=np.float64).astype(np.float32)
        std[std < 1e-5] = 1.0
        return mean, std

    context_mean, context_std = standardize(arrays["causal_context"], train_states)
    observation_mean, observation_std = standardize(arrays["packed_observation"], train_states)
    resource_mean, resource_std = standardize(arrays["slot_resources"], train_slots)
    def tensor(value, dtype=None):
        return torch.as_tensor(
            np.asarray(value).copy(), dtype=dtype, device=device)

    legal_np = ((np.asarray(arrays["slot_legal_mask"], dtype=np.int64)[:, None] >>
                 np.arange(len(CLASS_KINDS), dtype=np.int64)) & 1).astype(bool)
    inputs = {
        "context": (tensor(arrays["causal_context"], torch.float32) - tensor(context_mean)) / tensor(context_std),
        "observation": (tensor(arrays["packed_observation"], torch.float32) - tensor(observation_mean)) / tensor(observation_std),
        "token_continuous": tensor(arrays["token_continuous"], torch.float32),
        "token_categories": [tensor(arrays[name], torch.long) for name in
                             ("token_type", "token_category_a", "token_category_b",
                              "token_category_c", "token_x", "token_y", "token_owner")],
        "token_count": tensor(arrays["token_count"], torch.long),
        "resources": (tensor(arrays["slot_resources"], torch.float32) - tensor(resource_mean)) / tensor(resource_std),
        "cell": tensor(arrays["slot_cell"], torch.long),
        "label": tensor(arrays["slot_label_class"], torch.long),
        "legal": tensor(legal_np, torch.bool),
    }
    dimensions = manifest["dimensions"]
    make_model = lambda: build_model(dimensions["causal_context"],
                                     dimensions["packed_observation"],
                                     dimensions["slot_resources"]).to(device)

    def forward(model, states):
        states_tensor = tensor(states, torch.long)
        hidden = model.initial_hidden(
            inputs["context"][states_tensor], inputs["observation"][states_tensor],
            inputs["token_continuous"][states_tensor],
            [value[states_tensor] for value in inputs["token_categories"]],
            inputs["token_count"][states_tensor])
        losses = []
        predicted = []
        expected = []
        legal_rows = []
        previous = torch.full(
            (len(states),), 9, dtype=torch.long, device=device)
        max_slots = max(int(offsets_np[state + 1] - offsets_np[state])
                        for state in states)
        for slot in range(max_slots):
            active_at = [row for row, state in enumerate(states)
                         if slot < int(offsets_np[state + 1] - offsets_np[state])]
            if not active_at:
                continue
            active = tensor(active_at, torch.long)
            global_rows = tensor(
                [int(offsets_np[state]) + slot for state in states
                 if slot < int(offsets_np[state + 1] - offsets_np[state])],
                torch.long)
            legal = inputs["legal"][global_rows]
            logits, next_hidden = model.step(
                hidden[active], inputs["resources"][global_rows],
                inputs["cell"][global_rows], previous[active], legal)
            labels = inputs["label"][global_rows]
            losses.append(functional.cross_entropy(logits, labels, reduction="none"))
            predicted.append(logits.argmax(1))
            expected.append(labels)
            legal_rows.append(legal)
            hidden = hidden.index_copy(0, active, next_hidden)
            previous = previous.index_copy(0, active, labels)
        return (torch.cat(losses), torch.cat(predicted), torch.cat(expected),
                torch.cat(legal_rows))

    def batches(states, size, shuffle=False):
        values = np.asarray(states, dtype=np.int64)
        if shuffle:
            values = values[rng.permutation(len(values))]
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def metrics(model, states):
        was_training = model.training
        model.eval()
        total_loss = total = correct = illegal = 0
        support = np.zeros(len(CLASS_KINDS), dtype=np.int64)
        hits = np.zeros(len(CLASS_KINDS), dtype=np.int64)
        with torch.no_grad():
            for batch in batches(states, args.eval_batch_states):
                losses, predicted, labels, legal = forward(model, batch)
                total_loss += float(losses.sum().item())
                total += len(labels)
                correct += int((predicted == labels).sum().item())
                rows = torch.arange(len(predicted), device=device)
                illegal += int((~legal[rows, predicted]).sum().item())
                for class_index in range(len(CLASS_KINDS)):
                    selected = labels == class_index
                    support[class_index] += int(selected.sum().item())
                    hits[class_index] += int(
                        (predicted[selected] == class_index).sum().item())
        model.train(was_training)
        recalls = {str(kind): {
            "support": int(support[class_index]),
            "recall": (float(hits[class_index] / support[class_index])
                       if support[class_index] else None),
        } for class_index, kind in enumerate(CLASS_KINDS)}
        return total_loss / total, correct / total, recalls, illegal

    probe = make_model()
    parameters = sum(value.numel() for value in probe.parameters())
    if not 80_000 <= parameters <= 180_000:
        raise RuntimeError(f"model size {parameters} outside first-version budget")
    overfit_states = train_states[:min(8, len(train_states))]
    overfit = make_model()
    optimizer = torch.optim.Adam(overfit.parameters(), lr=0.02)
    first_overfit = None
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        losses, *_ = forward(overfit, overfit_states)
        loss = losses.mean()
        if first_overfit is None:
            first_overfit = float(loss.detach().item())
        loss.backward()
        optimizer.step()
    overfit_loss, overfit_accuracy, _, overfit_illegal = metrics(overfit, overfit_states)
    if not float(overfit_loss) < first_overfit or overfit_accuracy < 0.999 or overfit_illegal:
        raise RuntimeError("real one-batch autoregressive overfit failed")
    print(json.dumps({"event": "overfit_check", "status": "PASS",
                      "loss": float(overfit_loss),
                      "masked_accuracy": overfit_accuracy}), flush=True)
    write_progress("running", phase="training", parameters=parameters,
                   overfit_check={"loss": float(overfit_loss),
                                  "masked_accuracy": overfit_accuracy})

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    initial_train = metrics(model, train_states)
    initial_heldout = metrics(model, heldout_states)
    started = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(args.epochs):
        epoch_loss = epoch_slots = 0
        for batch in batches(train_states, args.batch_states, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            losses, *_ = forward(model, batch)
            loss = losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            epoch_loss += float(loss.detach().item()) * len(losses)
            epoch_slots += len(losses)
        print(json.dumps({"event": "epoch", "epoch": epoch + 1,
                          "epochs": args.epochs,
                          "train_loss": epoch_loss / epoch_slots,
                          "optimizer_steps": optimizer_steps}), flush=True)
        progress_events.append({"epoch": epoch + 1,
                                "train_loss": epoch_loss / epoch_slots,
                                "optimizer_steps": optimizer_steps})
        write_progress("running", phase="training", parameters=parameters,
                       completed_epochs=epoch + 1)
    elapsed = time.perf_counter() - started
    write_progress("running", phase="final_evaluation", parameters=parameters,
                   completed_epochs=args.epochs)
    final_train = metrics(model, train_states)
    final_heldout = metrics(model, heldout_states)
    if not float(final_train[0]) < float(initial_train[0]):
        raise RuntimeError("autoregressive training loss did not decrease")
    if final_train[3] or final_heldout[3]:
        raise RuntimeError("masked model emitted an illegal argmax")

    torch.save({
        "model": {name: value.detach().cpu()
                  for name, value in model.state_dict().items()},
        "model_dimensions": dimensions,
        "class_kinds": CLASS_KINDS,
        "normalization": {"context_mean": context_mean, "context_std": context_std,
                          "observation_mean": observation_mean,
                          "observation_std": observation_std,
                          "resource_mean": resource_mean, "resource_std": resource_std},
        "shard_manifest_sha256": _sha256(args.shard / "manifest.json"),
        "model_inputs": sorted(MODEL_INPUTS),
    }, args.output)

    def report(values):
        loss, accuracy, recalls, illegal = values
        return {"loss": float(loss), "masked_accuracy": accuracy,
                "class_recall": recalls, "illegal_argmax": illegal}

    metrics_value = {
        "status": "PASS", "training_task": "autoregressive_slot_actor",
        "schema_name": manifest["schema"]["schema_name"],
        "schema_sha256": manifest["schema_sha256"],
        "shard_manifest_sha256": _sha256(args.shard / "manifest.json"),
        "slot_contract": {"abi_version": manifest["slot_contract"]["abi_version"],
                          "label_semantics": manifest["slot_contract"]["label_semantics"],
                          "prefix_semantics": manifest["slot_contract"]["prefix_semantics"]},
        "device": str(device), "states": len(split),
        "slots": len(arrays["slot_cell"]), "train_states": len(train_states),
        "heldout_states": len(heldout_states), "parameters": parameters,
        "winner_key_coverage": manifest["validation"].get("winner_key_coverage"),
        "identical_input_label_conflicts": manifest["validation"].get(
            "identical_input_label_conflicts",
            manifest["validation"].get("source_identical_input_label_conflicts", 0)),
        "teacher_label_legal_fraction": manifest["validation"].get(
            "label_legal_fraction",
            1.0 if manifest["validation"].get("label_legality") == "PASS" else None),
        "one_batch_overfit": {"states": len(overfit_states), "initial_loss": first_overfit,
                              "final_loss": float(overfit_loss),
                              "masked_accuracy": overfit_accuracy,
                              "illegal_argmax": overfit_illegal},
        "train": {"initial": report(initial_train), "final": report(final_train)},
        "heldout": {"initial": report(initial_heldout), "final": report(final_heldout)},
        "timing": {"epochs": args.epochs, "optimizer_steps": optimizer_steps,
                   "batch_states": args.batch_states,
                   "eval_batch_states": args.eval_batch_states, "seconds": elapsed,
                   "slot_epochs_per_second": args.epochs * int(np.sum(train_slots)) / elapsed},
        "checkpoint": str(args.output),
        "limitations": [
            "BC target is the current production-score winner, not a future-suffix oracle",
            "metrics use teacher-forced prefix resources; C++ closed-loop student integration is not done",
            f"{len(split)}-state smoke validates the pipeline, not deployment quality",
        ],
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics_value, indent=2) + "\n")
    write_progress("complete", phase="complete", parameters=parameters,
                   completed_epochs=args.epochs,
                   metrics_file=metrics_path.name,
                   heldout=metrics_value["heldout"]["final"])
    print(json.dumps(metrics_value))


def check(args) -> None:
    manifest, arrays = _load_slot_shard(args.shard)
    print(json.dumps({"status": "PASS", **manifest["counts"],
                      **manifest["validation"],
                      "schema_sha256": manifest["schema_sha256"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--allowlist", type=Path, required=True)
    build.add_argument("--binary", type=Path, required=True)
    build.add_argument("--binary-sha256", required=True)
    build.add_argument("--winner-audit-shard", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--max-states", type=int, default=48)
    build.add_argument("--workers", type=int, default=min(160, os.cpu_count() or 1))
    build.set_defaults(function=build_shard)
    fit = commands.add_parser("train")
    fit.add_argument("--shard", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=150)
    fit.add_argument("--overfit-steps", type=int, default=250)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--batch-states", type=int, default=1024)
    fit.add_argument("--eval-batch-states", type=int, default=2048)
    fit.add_argument("--seed", type=int, default=20260922)
    fit.set_defaults(function=train)
    verify = commands.add_parser("check")
    verify.add_argument("--shard", type=Path, required=True)
    verify.set_defaults(function=check)
    args = parser.parse_args()
    if args.command == "build" and (args.max_states < 2 or args.workers < 1):
        parser.error("positive workers and at least two states required")
    if args.command == "train" and (args.batch_states < 1 or args.eval_batch_states < 1):
        parser.error("training batch sizes must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
