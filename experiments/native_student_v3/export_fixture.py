#!/usr/bin/env python3
"""Export real R3 rollout events for the dependency-free C++ parity runner."""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import EVENT_CLASSES, build_model


ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"KSV3FX1\0"
HEADER = struct.Struct("<8s12IQ")
DEFAULT_CHECKPOINT = (
    ROOT / "work/student-v1/action-event-v3-actor-owned-dagger-r3-scale3-e20.pt")
DEFAULT_ROLLOUT = (
    ROOT / "work/student-v1/v3-ppo-native-strong-r3-192g-v0.rollout.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32(value) -> bytes:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("fixture contains non-finite float32")
    return array.astype("<f4", copy=False).tobytes(order="C")


def u32(value) -> bytes:
    return np.asarray(value, dtype="<u4").tobytes(order="C")


def counter_word(seed: int, counter: int) -> int:
    mask = (1 << 64) - 1
    value = (seed + counter * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def counter_action(logits: np.ndarray, legal_mask: int, seed: int,
                   counter: int) -> int:
    legal = [index for index in range(len(logits))
             if legal_mask & (1 << index)]
    scaled = np.asarray(logits, dtype=np.float32)
    maximum = np.float32(max(scaled[index] for index in legal))
    probabilities = np.zeros(len(logits), dtype=np.float32)
    total = np.float32(0)
    for index in legal:
        probabilities[index] = np.exp(
            np.float32(scaled[index] - maximum), dtype=np.float32)
        total = np.float32(total + probabilities[index])
    for index in legal:
        probabilities[index] = np.float32(probabilities[index] / total)
    top = counter_word(seed, counter) >> 40
    target = np.float32((np.float32(top) + np.float32(0.5)) *
                        np.float32(1.0 / 16777216.0))
    cumulative = np.float32(0)
    for index in legal:
        cumulative = np.float32(cumulative + probabilities[index])
        if target < cumulative:
            return index
    return legal[-1]


def state_tensors(day: dict) -> tuple:
    state = day["state"]
    return (
        torch.from_numpy(state["context"]),
        torch.from_numpy(state["observation"]),
        torch.from_numpy(state["observation_length"]),
        torch.from_numpy(state["token_continuous"]),
        [torch.from_numpy(value) for value in state["token_categories"]],
        torch.from_numpy(state["token_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--rng-seed", type=lambda value: int(value, 0),
                        default=0x123456789ABCDEF0)
    args = parser.parse_args()
    if args.events <= 0 or not 0 <= args.rng_seed < 1 << 64:
        parser.error("invalid event count or RNG seed")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rollout = torch.load(args.rollout, map_location="cpu", weights_only=False)
    dimensions = checkpoint["model_dimensions"]
    if (tuple(checkpoint["event_classes"]) != EVENT_CLASSES or
            rollout.get("policy_version") != checkpoint_sha):
        raise ValueError("rollout does not belong to the checkpoint")
    model = build_model(dimensions["causal_context"],
                        dimensions["packed_observation"],
                        dimensions["event_resources"], checkpoint["model_scale"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_num_threads(1)

    first_day = rollout["games"][0]["days"][0]
    first_state = state_tensors(first_day)
    rows: list[dict] = []
    with torch.inference_mode():
        expected_initial = model.initial_hidden(*first_state)
        for game in rollout["games"]:
            for day in game["days"]:
                hidden = model.initial_hidden(*state_tensors(day))
                for event in day["events"]:
                    logits, next_hidden = model.step(
                        hidden, torch.from_numpy(event["resources"]),
                        torch.tensor([event["cell"]]),
                        torch.tensor([event["stage"]]),
                        torch.tensor([event["previous"]]),
                        torch.from_numpy(event["legal"]))
                    distribution = torch.distributions.Categorical(logits=logits)
                    log_probability = float(distribution.log_prob(
                        torch.tensor([event["action"]])).item())
                    entropy = float(distribution.entropy().item())
                    if (log_probability != event["old_logprob"] or
                            entropy != event["old_entropy"]):
                        raise ValueError("rollout log-probability drift")
                    legal_mask = int(event["legal_mask"])
                    logits_numpy = logits[0].numpy().copy()
                    rows.append({
                        "hidden": hidden[0].numpy().copy(),
                        "resources": event["resources"][0],
                        "cell": event["cell"], "stage": event["stage"],
                        "previous": event["previous"], "legal_mask": legal_mask,
                        "action": event["action"],
                        "greedy": int(logits.argmax(1).item()),
                        "logits": logits_numpy,
                        "next_hidden": next_hidden[0].numpy().copy(),
                        "log_probability": log_probability, "entropy": entropy,
                        "counter_action": counter_action(
                            logits_numpy, legal_mask, args.rng_seed, len(rows)),
                    })
                    hidden = next_hidden
                    if len(rows) == args.events:
                        break
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
    if len(rows) != args.events:
        raise ValueError(f"rollout supplied only {len(rows)} events")

    state = first_day["state"]
    token_capacity, token_width = state["token_continuous"].shape[1:]
    context_width = dimensions["causal_context"]
    observation_width = dimensions["packed_observation"]
    resource_width = dimensions["event_resources"]
    hidden_width = expected_initial.shape[1]
    header = HEADER.pack(
        MAGIC, 1, HEADER.size, len(rows), token_capacity, token_width,
        len(state["token_categories"]), context_width, observation_width,
        resource_width, hidden_width, len(EVENT_CLASSES), 1, args.rng_seed)
    payload = b"".join((
        f32(state["context"]), f32(state["observation"]),
        f32(state["observation_length"]), f32(state["token_continuous"]),
        u32(np.stack(state["token_categories"])), u32(state["token_count"]),
        f32(expected_initial),
        f32(np.stack([row["hidden"] for row in rows])),
        f32(np.stack([row["resources"] for row in rows])),
        u32([row["cell"] for row in rows]),
        u32([row["stage"] for row in rows]),
        u32([row["previous"] for row in rows]),
        u32([row["legal_mask"] for row in rows]),
        u32([row["action"] for row in rows]),
        u32([row["greedy"] for row in rows]),
        f32(np.stack([row["logits"] for row in rows])),
        f32(np.stack([row["next_hidden"] for row in rows])),
        f32([row["log_probability"] for row in rows]),
        f32([row["entropy"] for row in rows]),
        u32([row["counter_action"] for row in rows]),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        target.write(header)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, args.output)
    print({
        "status": "PASS", "events": len(rows), "bytes": args.output.stat().st_size,
        "checkpoint_sha256": checkpoint_sha, "fixture_sha256": sha256(args.output),
    })


if __name__ == "__main__":
    main()
