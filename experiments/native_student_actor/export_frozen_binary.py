#!/usr/bin/env python3
"""Export a v3 actor checkpoint as the fixed little-endian C++ weight blob."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import EVENT_CLASSES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    ROOT / "work/student-v1/action-event-v3-actor-owned-dagger-r3-scale3-e20.pt")
MAGIC = b"KAGSV3A\0"
VERSION = 1
ENDIAN_TAG = 0x01020304
SCALAR_F32 = 1
HEADER = struct.Struct("<8s23IQ32s32s32s32s")
NORMALIZATION_ORDER = (
    "context_mean", "context_std", "observation_mean", "observation_std",
    "observation_length_mean", "observation_length_std",
    "resource_mean", "resource_std",
)
PARAMETER_ORDER = (
    "context.weight", "context.bias", "observation.weight", "observation.bias",
    "observation_length.weight", "observation_length.bias",
    "token_embeddings.0.weight", "token_embeddings.1.weight",
    "token_embeddings.2.weight", "token_embeddings.3.weight",
    "token_embeddings.4.weight", "token_embeddings.5.weight",
    "token_embeddings.6.weight", "token.weight", "token.bias",
    "begin.weight", "begin.bias", "resource.weight", "resource.bias",
    "cell.weight", "stage.weight", "previous.weight", "gru.weight_ih",
    "gru.weight_hh", "gru.bias_ih", "gru.bias_hh", "head.weight", "head.bias",
)
CONTRACT_TEXT = "\n".join((
    "kaggriculture.action-event-v3.actor-a.le-f32.v1",
    "normalization:" + ",".join(NORMALIZATION_ORDER),
    "state_dict:" + ",".join(PARAMETER_ORDER),
    "gru_gate_order:r,z,n",
    "payload:raw-c-contiguous-f32",
))
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_TEXT.encode("ascii")).digest()


def file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def f32_bytes(value) -> bytes:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"expected finite float32 tensor, got {array.dtype}")
    return array.astype("<f4", copy=False).tobytes(order="C")


def expected_shapes(d: dict[str, int]) -> dict[str, tuple[int, ...]]:
    p, s, e, h = d["projection"], d["scalar"], d["embedding"], d["hidden"]
    rh, ee = d["resource_hidden"], d["event_embedding"]
    shapes = {
        "context.weight": (p, d["context"]), "context.bias": (p,),
        "observation.weight": (p, d["observation"]), "observation.bias": (p,),
        "observation_length.weight": (s, 1), "observation_length.bias": (s,),
        "token.weight": (p, 24 + 7 * e), "token.bias": (p,),
        "begin.weight": (h, 3 * p + s), "begin.bias": (h,),
        "resource.weight": (rh, d["resource"]), "resource.bias": (rh,),
        "cell.weight": (100, ee), "stage.weight": (2, ee),
        "previous.weight": (len(EVENT_CLASSES) + 1, ee),
        "gru.weight_ih": (3 * h, rh + 3 * ee), "gru.weight_hh": (3 * h, h),
        "gru.bias_ih": (3 * h,), "gru.bias_hh": (3 * h,),
        "head.weight": (len(EVENT_CLASSES), h), "head.bias": (len(EVENT_CLASSES),),
    }
    for index, size in enumerate((7, 32, 32, 32, 64, 64, 4)):
        shapes[f"token_embeddings.{index}.weight"] = (size, e)
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", {})
    normalization = checkpoint.get("normalization", {})
    dimensions = checkpoint.get("model_dimensions", {})
    scale = int(checkpoint.get("model_scale", 0))
    dims = {
        "context": int(dimensions.get("causal_context", 0)),
        "observation": int(dimensions.get("packed_observation", 0)),
        "resource": int(dimensions.get("event_resources", 0)),
        "projection": 16 * scale, "scalar": 4 * scale,
        "embedding": 4 * scale, "hidden": 64 * scale,
        "resource_hidden": 64 * scale, "event_embedding": 8 * scale,
    }
    shapes = expected_shapes(dims)
    normalization_shapes = {
        "context_mean": (dims["context"],), "context_std": (dims["context"],),
        "observation_mean": (dims["observation"],),
        "observation_std": (dims["observation"],),
        "observation_length_mean": (), "observation_length_std": (),
        "resource_mean": (dims["resource"],), "resource_std": (dims["resource"],),
    }
    if (not 1 <= scale <= 4 or tuple(checkpoint.get("event_classes", ())) !=
            EVENT_CLASSES or tuple(state) != PARAMETER_ORDER or
            tuple(normalization) != NORMALIZATION_ORDER):
        raise ValueError("checkpoint does not match the fixed v3 actor contract")
    for name, shape in shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"{name}: {tuple(state[name].shape)} != {shape}")
    for name, shape in normalization_shapes.items():
        if np.asarray(normalization[name]).shape != shape:
            raise ValueError(f"{name}: invalid normalization shape")
    if any(np.any(np.asarray(normalization[name]) <= 0)
           for name in ("context_std", "observation_std",
                        "observation_length_std", "resource_std")):
        raise ValueError("normalization std must be positive")

    payload = b"".join(f32_bytes(normalization[name])
                       for name in NORMALIZATION_ORDER)
    payload += b"".join(f32_bytes(state[name]) for name in PARAMETER_ORDER)
    payload_sha = hashlib.sha256(payload).digest()
    checkpoint_sha = file_sha256(args.checkpoint)
    try:
        shard_sha = bytes.fromhex(checkpoint["shard_manifest_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid checkpoint shard manifest hash") from error
    if len(shard_sha) != 32:
        raise ValueError("invalid checkpoint shard manifest hash length")

    header = HEADER.pack(
        MAGIC, VERSION, HEADER.size, ENDIAN_TAG, SCALAR_F32,
        len(NORMALIZATION_ORDER), len(PARAMETER_ORDER), len(EVENT_CLASSES),
        320, 24, 7, dims["context"], dims["observation"], dims["resource"],
        scale, dims["projection"], dims["scalar"], dims["embedding"],
        dims["hidden"], dims["resource_hidden"], dims["event_embedding"],
        100, 2, len(EVENT_CLASSES) + 1, len(payload),
        checkpoint_sha, shard_sha, CONTRACT_SHA256, payload_sha)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        target.write(header)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, args.output)

    written = args.output.read_bytes()
    unpacked = HEADER.unpack_from(written)
    if (len(written) != HEADER.size + len(payload) or unpacked[0] != MAGIC or
            unpacked[2] != HEADER.size or unpacked[-1] != payload_sha or
            hashlib.sha256(written[HEADER.size:]).digest() != payload_sha):
        raise RuntimeError("written actor binary failed self-verification")
    print(json.dumps({
        "status": "PASS", "output": str(args.output), "bytes": len(written),
        "header_bytes": HEADER.size, "payload_bytes": len(payload),
        "checkpoint_sha256": checkpoint_sha.hex(),
        "shard_manifest_sha256": shard_sha.hex(),
        "contract_sha256": CONTRACT_SHA256.hex(),
        "payload_sha256": payload_sha.hex(),
        "file_sha256": hashlib.sha256(written).hexdigest(),
        "normalization_tensors": len(NORMALIZATION_ORDER),
        "parameter_tensors": len(PARAMETER_ORDER), "dimensions": dims,
    }))


if __name__ == "__main__":
    main()
