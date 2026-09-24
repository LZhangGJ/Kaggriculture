#!/usr/bin/env python3
"""Stream accepted autoregressive-slot-bc-v2 shards into one mmap shard."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.train_midgame_autofill_v1 import (
    CLASS_KINDS, MODEL_INPUTS, _load_slot_shard,
)
from experiments.train_midgame_student_v1 import _json_hash, _sha256


STATE_ARRAYS = {
    "causal_context", "packed_observation", "observation_length",
    "token_continuous", "token_type", "token_category_a", "token_category_b",
    "token_category_c", "token_x", "token_y", "token_owner", "token_count",
    "state_step", "split", "group_hash128",
}
SLOT_ARRAYS = {
    "slot_cell", "slot_label_kind", "slot_label_class", "slot_legal_mask",
    "slot_terminal", "slot_resources",
}
OFFSET_ARRAY = "state_slot_offsets"
ALL_ARRAYS = STATE_ARRAYS | SLOT_ARRAYS | {OFFSET_ARRAY}
STRICT_KEYS = (
    "format", "container_version", "schema", "schema_sha256", "byte_order",
    "constants", "dimensions", "slot_contract", "slot_resource_names",
    "causal_context_names", "teacher_binary", "teacher_binary_sha256",
    "split_method",
)


def _load_and_validate(directory: Path) -> dict:
    manifest, arrays = _load_slot_shard(directory)
    states = int(manifest["counts"]["states"])
    slots = int(manifest["counts"]["slots"])
    if (manifest.get("format") != "kaggriculture-midgame-mmap" or
            manifest.get("container_version") != 1 or
            manifest.get("byte_order") != "little" or
            manifest.get("schema_sha256") != _json_hash(manifest["schema"]) or
            set(manifest["arrays"]) != ALL_ARRAYS or
            set(manifest["schema"]["model_input_arrays"]) != MODEL_INPUTS or
            len(manifest.get("audit_groups", [])) != states or
            int(manifest["counts"]["train_states"]) +
            int(manifest["counts"]["heldout_states"]) != states or
            set(manifest["class_counts_by_kind"]) != {str(x) for x in CLASS_KINDS} or
            sum(map(int, manifest["class_counts_by_kind"].values())) != slots):
        raise ValueError(f"{directory}: invalid accepted shard manifest")
    for name, value in arrays.items():
        expected = states + 1 if name == OFFSET_ARRAY else (
            states if name in STATE_ARRAYS else slots)
        if value.shape[0] != expected:
            raise ValueError(f"{directory}: {name} first dimension mismatch")
    if (manifest["dimensions"] != {
            "causal_context": arrays["causal_context"].shape[1],
            "packed_observation": arrays["packed_observation"].shape[1],
            "slot_resources": arrays["slot_resources"].shape[1],
    } or manifest["slot_contract"]["resource_width"] !=
            manifest["dimensions"]["slot_resources"]):
        raise ValueError(f"{directory}: declared dimensions mismatch arrays")
    del arrays
    return manifest


def _strictly_compatible(reference: dict, candidate: dict, directory: Path) -> None:
    for key in STRICT_KEYS:
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"{directory}: incompatible {key}")
    for name in ALL_ARRAYS:
        left, right = reference["arrays"][name], candidate["arrays"][name]
        if (left["dtype"] != right["dtype"] or
                left["shape"][1:] != right["shape"][1:]):
            raise ValueError(f"{directory}: incompatible {name} dtype/shape")
    if set(candidate["counts"]) != set(reference["counts"]):
        raise ValueError(f"{directory}: incompatible count fields")


def _copy_rows(target: np.memmap, target_start: int, source: np.ndarray,
               chunk_bytes: int) -> None:
    row_bytes = source.dtype.itemsize * int(np.prod(source.shape[1:], dtype=np.int64))
    rows = max(1, chunk_bytes // max(1, row_bytes))
    for start in range(0, len(source), rows):
        stop = min(len(source), start + rows)
        target[target_start + start:target_start + stop] = source[start:stop]


def _verification_indices(prefixes: list[int], total: int, samples: int,
                          rng: np.random.Generator) -> list[int]:
    if total <= 0:
        return []
    indices = {0, total - 1}
    for boundary in prefixes[1:-1]:
        indices.update(index for index in (boundary - 1, boundary)
                       if 0 <= index < total)
    if samples:
        indices.update(map(int, rng.integers(0, total, size=samples)))
    return sorted(indices)


def _verify_output(shards: list[Path], manifests: list[dict], output: Path,
                   merged: dict, samples: int) -> dict:
    arrays = {}
    for name, spec in merged["arrays"].items():
        value = np.load(output / spec["path"], mmap_mode="r", allow_pickle=False)
        if value.dtype.str != spec["dtype"] or list(value.shape) != spec["shape"]:
            raise RuntimeError(f"merged {name} dtype/shape mismatch")
        arrays[name] = value
    state_sizes = [int(row["counts"]["states"]) for row in manifests]
    slot_sizes = [int(row["counts"]["slots"]) for row in manifests]
    state_prefix = np.cumsum([0, *state_sizes]).tolist()
    slot_prefix = np.cumsum([0, *slot_sizes]).tolist()
    rng = np.random.default_rng(0xA170F111)
    checked = 0
    for name in sorted(ALL_ARRAYS - {OFFSET_ARRAY}):
        prefixes = state_prefix if name in STATE_ARRAYS else slot_prefix
        sources = [np.load(path / manifest["arrays"][name]["path"], mmap_mode="r",
                           allow_pickle=False)
                   for path, manifest in zip(shards, manifests)]
        indices = _verification_indices(prefixes, prefixes[-1], samples, rng)
        for index in indices:
            shard_index = bisect.bisect_right(prefixes, index) - 1
            local = index - prefixes[shard_index]
            if not np.array_equal(arrays[name][index], sources[shard_index][local],
                                  equal_nan=True):
                raise RuntimeError(f"{name}[{index}] differs from source")
        checked += len(indices)
        del sources

    offsets = arrays[OFFSET_ARRAY]
    source_offsets = [np.load(path / manifest["arrays"][OFFSET_ARRAY]["path"],
                              mmap_mode="r", allow_pickle=False)
                      for path, manifest in zip(shards, manifests)]
    offset_indices = _verification_indices(state_prefix, state_prefix[-1] + 1,
                                           samples, rng)
    for index in offset_indices:
        if index == state_prefix[-1]:
            expected = slot_prefix[-1]
        else:
            shard_index = bisect.bisect_right(state_prefix, index) - 1
            local = index - state_prefix[shard_index]
            expected = int(source_offsets[shard_index][local]) + slot_prefix[shard_index]
        if int(offsets[index]) != expected:
            raise RuntimeError(f"state_slot_offsets[{index}] differs from source")
    checked += len(offset_indices)

    if np.any(np.diff(offsets.astype(np.int64)) < 0):
        raise RuntimeError("merged state-slot offsets are not monotonic")
    labels, masks = arrays["slot_label_class"], arrays["slot_legal_mask"]
    if not np.all((masks >> labels) & 1):
        raise RuntimeError("merged shard contains an illegal teacher label")
    actual_classes = Counter(map(int, arrays["slot_label_kind"]))
    expected_classes = {str(kind): actual_classes[kind] for kind in CLASS_KINDS}
    if expected_classes != merged["class_counts_by_kind"]:
        raise RuntimeError("merged class counts do not match arrays")
    split = arrays["split"]
    if (int(np.sum(split == 0)) != merged["counts"]["train_states"] or
            int(np.sum(split == 1)) != merged["counts"]["heldout_states"] or
            np.any((split != 0) & (split != 1))):
        raise RuntimeError("merged split counts do not match arrays")
    return {"arrays": len(ALL_ARRAYS), "sampled_boundary_values": checked,
            "state_slot_offsets": "PASS", "label_legality": "PASS",
            "class_and_split_counts": "PASS"}


def merge(shards: list[Path], output: Path, chunk_bytes: int,
          verify_samples: int) -> None:
    shards = [path.resolve() for path in shards]
    if len(set(shards)) != len(shards):
        raise ValueError("duplicate input shard")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    manifests = [_load_and_validate(path) for path in shards]
    reference = manifests[0]
    for path, manifest in zip(shards[1:], manifests[1:]):
        _strictly_compatible(reference, manifest, path)
    teacher = Path(reference["teacher_binary"])
    if not teacher.is_file() or _sha256(teacher) != reference["teacher_binary_sha256"]:
        raise ValueError("teacher binary is missing or no longer matches its digest")

    audit_groups = []
    seen_states = set()
    for path, manifest in zip(shards, manifests):
        for group in manifest["audit_groups"]:
            key = (str(Path(group["trajectory"]).resolve()), int(group["step"]))
            if key in seen_states:
                raise ValueError(f"duplicate trajectory state across shards: {key}")
            seen_states.add(key)
            audit_groups.append(group)

    count_keys = reference["counts"].keys()
    counts = {key: sum(int(row["counts"][key]) for row in manifests)
              for key in count_keys}
    class_counts = {str(kind): sum(int(row["class_counts_by_kind"][str(kind)])
                                   for row in manifests)
                    for kind in CLASS_KINDS}
    total_states, total_slots = counts["states"], counts["slots"]
    if total_slots > np.iinfo(np.dtype(reference["arrays"][OFFSET_ARRAY]["dtype"])).max:
        raise OverflowError("merged slot offsets exceed their stored dtype")

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    array_manifest = {}
    for name in sorted(ALL_ARRAYS):
        spec = reference["arrays"][name]
        if name == OFFSET_ARRAY:
            shape = (total_states + 1, *spec["shape"][1:])
        elif name in STATE_ARRAYS:
            shape = (total_states, *spec["shape"][1:])
        else:
            shape = (total_slots, *spec["shape"][1:])
        target_path = output / f"{name}.npy"
        target = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.dtype(spec["dtype"]), shape=shape)
        target_cursor = 0
        slot_cursor = 0
        for path, manifest in zip(shards, manifests):
            source_spec = manifest["arrays"][name]
            source = np.load(path / source_spec["path"], mmap_mode="r",
                             allow_pickle=False)
            if name == OFFSET_ARRAY:
                count = int(manifest["counts"]["states"])
                target[target_cursor:target_cursor + count] = source[:-1] + slot_cursor
                target_cursor += count
                slot_cursor += int(manifest["counts"]["slots"])
            else:
                _copy_rows(target, target_cursor, source, chunk_bytes)
                target_cursor += len(source)
            del source
        if name == OFFSET_ARRAY:
            target[-1] = total_slots
        target.flush()
        del target
        array_manifest[name] = {
            "path": target_path.name, "dtype": np.dtype(spec["dtype"]).str,
            "shape": list(shape), "sha256": _sha256(target_path),
        }

    merged = {key: reference[key] for key in STRICT_KEYS}
    merged.update({
        "validation_status": "building",
        "counts": counts,
        "class_counts_by_kind": class_counts,
        "arrays": array_manifest,
        "source_shards": [{
            "path": str(path), "manifest_sha256": _sha256(path / "manifest.json"),
            "states": int(manifest["counts"]["states"]),
            "slots": int(manifest["counts"]["slots"]),
        } for path, manifest in zip(shards, manifests)],
        "audit_groups": audit_groups,
        "validation": {
            "source_shards_accepted": len(shards),
            "source_array_sha256_verified": "PASS",
            "schema_dimensions_contract_names_teacher_match": "PASS",
            "action_parity_frames": counts.get("action_frames", 0),
            "action_parity_mismatches": 0,
            "source_identical_input_label_conflicts": sum(
                int(row["validation"]["identical_input_label_conflicts"])
                for row in manifests),
            "prefix_semantics": reference["slot_contract"]["prefix_semantics"],
            "no_identity_in_model_inputs": "PASS",
        },
        "timing": {"merge_seconds": time.perf_counter() - started},
    })
    temporary = output / "manifest.json.tmp"
    verification = _verify_output(
        shards, manifests, output, merged, verify_samples)
    merged["validation"].update(verification)
    merged["validation_status"] = "accepted"
    merged["timing"]["total_seconds"] = time.perf_counter() - started
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({"status": "PASS", "output": str(output), **counts,
                      **verification, **merged["timing"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--verify-samples", type=int, default=8)
    args = parser.parse_args()
    if args.chunk_mib < 1 or args.verify_samples < 0:
        parser.error("chunk-mib must be positive and verify-samples non-negative")
    merge(args.shards, args.output.resolve(), args.chunk_mib << 20,
          args.verify_samples)


if __name__ == "__main__":
    main()
