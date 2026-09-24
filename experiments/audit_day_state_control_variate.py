#!/usr/bin/env python3
"""Read-only diagnostic for the seed-cross-fitted PPO day-state baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.day_state_control_variate import (
    actionable_day_counts,
    crossfit_global_hgb,
    extract_day_state_features,
)
from experiments.train_student_action_event_rl_v3 import (
    NATIVE_ROLLOUT_FORMAT,
    NATIVE_ROLLOUT_METADATA,
    ROLLOUT_SCHEMA,
)


REQUIRED_ARRAYS = (
    "seed", "own_cash", "rival_cash", "day_session_index", "day_step",
    "observation", "observation_length", "event_day_index",
    "event_legal_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_required(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if NATIVE_ROLLOUT_METADATA not in archive.files:
                raise RuntimeError("native rollout metadata is missing")
            encoded = archive[NATIVE_ROLLOUT_METADATA]
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise RuntimeError("invalid native rollout metadata encoding")
            metadata = json.loads(encoded.tobytes().decode("utf-8"))
            missing = set(REQUIRED_ARRAYS) - set(archive.files)
            if missing:
                raise RuntimeError(f"native rollout arrays missing: {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS}
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            UnicodeDecodeError) as error:
        raise RuntimeError(f"invalid native rollout: {error}") from error
    if (not isinstance(metadata, dict) or
            metadata.get("storage_format") != NATIVE_ROLLOUT_FORMAT or
            metadata.get("schema") != ROLLOUT_SCHEMA or
            metadata.get("action_unit") != "day_bundle" or
            metadata.get("rollout_engine") != "pure_cpp_job_batch"):
        raise RuntimeError("rollout is not a native v3 day-bundle archive")
    return metadata, arrays


def audit(args) -> dict:
    if not args.rollout.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("rollout/checkpoint is missing")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    metadata, arrays = _load_required(args.rollout)
    checkpoint_sha256 = _sha256(args.checkpoint)
    if metadata.get("policy_version") != checkpoint_sha256:
        raise RuntimeError("rollout/checkpoint behavior-policy fingerprint mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("behavior checkpoint has no normalization contract")

    margin = (np.asarray(arrays["own_cash"], dtype=np.float64) -
              np.asarray(arrays["rival_cash"], dtype=np.float64))
    reward_contract = metadata.get("reward", {})
    if reward_contract.get("formula") != (
            "sign(margin)+weight*tanh(margin/scale)"):
        raise RuntimeError("unknown terminal reward contract")
    weight = float(reward_contract.get("margin_weight"))
    scale = float(reward_contract.get("margin_scale"))
    if not np.isfinite(weight) or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid terminal reward parameters")
    if not math.isclose(weight, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "day-state diagnostic reward support requires margin_weight=0.1")
    rewards = np.sign(margin) + weight * np.tanh(margin / scale)

    extraction_started = time.perf_counter()
    features = extract_day_state_features(
        arrays["observation"], arrays["observation_length"],
        arrays["day_step"], normalization, chunk_size=args.chunk_size)
    extraction_seconds = time.perf_counter() - extraction_started
    counts = actionable_day_counts(
        len(features), arrays["event_day_index"], arrays["event_legal_mask"])
    result = crossfit_global_hgb(
        features, arrays["day_session_index"], arrays["day_step"],
        arrays["seed"], rewards, actionable_counts=counts,
        n_splits=6, max_workers=args.workers)
    metrics = {
        "status": "PASS",
        "scope": "read_only_historical_rollout_diagnostic",
        "rollout": str(args.rollout.resolve()),
        "rollout_sha256": _sha256(args.rollout),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "extraction_seconds": extraction_seconds,
        **result.metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    metrics = audit(args)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
