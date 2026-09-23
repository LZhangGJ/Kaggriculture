#!/usr/bin/env python3
"""Feed native JobBatch arrays through the existing PPO replay/update path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native_student_actor.native_job_batch import ppo_games
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_student_action_event_rl_v3 import (
    _batch_terms,
    _day_bundle_objective,
    _stratified_loo_advantages,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (ROOT / "work/student-v1/"
              "v3-ppo-native-strong-r3-v9-daybundle-stratified-1536g.pt")
WEIGHTS = ROOT / "work/student-v1/native-v9-daybundle-stratified.bin"
ARRAYS = ROOT / "work/native-student-rollout/arbitrary-job-new-seed-b128-arrays.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=ARRAYS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--logprob-tolerance", type=float, default=2e-5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-job-v9-ppo-update-b32.json"))
    args = parser.parse_args()
    if (args.games < 2 or args.threads < 1 or args.output.exists() or
            not args.arrays.is_file() or not args.checkpoint.is_file() or
            not args.weights.is_file()):
        parser.error("invalid inputs or existing output")
    checkpoint_sha = sha256(args.checkpoint)
    weight_bytes = args.weights.read_bytes()
    if (weight_bytes[:8] != b"KAGSV3A\0" or len(weight_bytes) < 236 or
            weight_bytes[108:140].hex() != checkpoint_sha):
        raise RuntimeError("frozen behavior weights do not bind to checkpoint")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arrays_started = time.perf_counter()
    with np.load(args.arrays) as archive:
        arrays = {name: archive[name] for name in archive.files}
    games = ppo_games(arrays)[:args.games]
    arrays_seconds = time.perf_counter() - arrays_started
    if len(games) != args.games:
        raise RuntimeError("not enough native games")
    binary = ROOT / "work/agent-student-actor-owned-v3.so"
    manifest_sha = str(payload["shard_manifest_sha256"])
    identity = {
        "policy_sha256": checkpoint_sha,
        "binary_sha256": sha256(binary),
        "manifest_sha256": manifest_sha,
        "opponent_backend": "native_cpp",
        "elapsed_seconds": 0.0,
    }
    for game in games:
        game.update(identity)

    torch.set_num_threads(args.threads)
    model = build_model(
        payload["model_dimensions"]["causal_context"],
        payload["model_dimensions"]["packed_observation"],
        payload["model_dimensions"]["event_resources"],
        payload["model_scale"])
    model.load_state_dict(payload["model"])
    model.train()
    replay_started = time.perf_counter()
    with torch.no_grad():
        (new, old, entropy_values, actionable, _event_games,
         event_days, day_games) = _batch_terms(
             model, games, torch.device("cpu"), 1.0)
    error = (new - old).abs()
    max_error = float(error.max())
    if max_error > args.logprob_tolerance:
        raise RuntimeError(
            f"native old-logprob replay mismatch {max_error} > "
            f"{args.logprob_tolerance}")
    replay_seconds = time.perf_counter() - replay_started

    rewards = np.asarray([game["reward"] for game in games], dtype=np.float32)
    advantages_np = _stratified_loo_advantages(games, rewards)
    advantages = torch.from_numpy(advantages_np)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    optimizer_restored = "rl_optimizer" in payload
    if optimizer_restored:
        optimizer.load_state_dict(payload["rl_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    update_started = time.perf_counter()
    (new, old, entropy_values, actionable, _event_games,
     event_days, day_games) = _batch_terms(
         model, games, torch.device("cpu"), 1.0)
    policy_loss, entropy, bundle = _day_bundle_objective(
        new, old, entropy_values, actionable, event_days, day_games,
        advantages, args.clip_ratio)
    loss = policy_loss - args.entropy_coef * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    update_seconds = time.perf_counter() - update_started
    delta_squared = 0.0
    for name, value in model.state_dict().items():
        delta_squared += float(((value.detach() - before[name]).double() ** 2).sum())
    parameter_delta = math.sqrt(delta_squared)
    if (not math.isfinite(float(loss)) or not math.isfinite(gradient_norm) or
            not math.isfinite(parameter_delta) or parameter_delta <= 0):
        raise RuntimeError("native PPO update was non-finite or a no-op")

    opponents = {
        "thomas_module": next((ROOT / "experiments/native_opponents/"
                               "thomas_2945_cpp/build").glob("thomas_2945_cpp_native*.so")),
        "thomas_asset": ROOT / "experiments/native_opponents/"
                         "thomas_2945_cpp/thomas_2945.assets.bin",
        "meta_module": next((ROOT / "experiments/native_opponents/"
                             "metav4_2965/build").glob("metav4_2965_native*.so")),
        "meta_asset": ROOT / "experiments/native_opponents/"
                      "metav4_2965/metav4_2965.assets.bin",
    }
    result = {
        "status": "PASS",
        "scope": "native-job-arrays-existing-ppo-replay-and-update",
        "games": args.games,
        "days": len(day_games),
        "events": len(new),
        "actionable_events": int(actionable.sum()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "frozen_weights": str(args.weights.resolve()),
        "frozen_weights_sha256": sha256(args.weights),
        "frozen_header_checkpoint_match": True,
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": sha256(args.arrays),
        **identity,
        "native_opponent_artifacts": {
            name: sha256(path) for name, path in opponents.items()
        },
        "old_logprob_replay_max_abs_error": max_error,
        "old_logprob_replay_mean_abs_error": float(error.mean()),
        "old_logprob_tolerance": args.logprob_tolerance,
        "optimizer_state_restored": optimizer_restored,
        "advantage_baseline": "opponent_seat_stratified_leave_one_out",
        "advantage_scaling": "terminal advantage once per day bundle",
        "policy_loss": float(policy_loss.detach()),
        "entropy": float(entropy.detach()),
        "loss": float(loss.detach()),
        "gradient_norm_before_clip": gradient_norm,
        "parameter_delta_l2": parameter_delta,
        "timing": {
            "array_load_and_schema_seconds": arrays_seconds,
            "old_logprob_replay_seconds": replay_seconds,
            "update_seconds": update_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
