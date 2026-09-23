#!/usr/bin/env python3
"""Export the v3 actor's two inference methods plus real-rollout parity data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

from experiments.train_midgame_autofill_v3 import EVENT_CLASSES, build_model


ROOT = Path(__file__).resolve().parents[2]
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


class Fixture(torch.nn.Module):
    """TorchScript container readable from C++ without another file format."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)

    def forward(self):
        return self.context


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
    parser.add_argument("--fixture-output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=256)
    args = parser.parse_args()
    if args.events <= 0:
        parser.error("--events must be positive")
    for path in (args.checkpoint, args.rollout):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.fixture_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rollout = torch.load(args.rollout, map_location="cpu", weights_only=False)
    dimensions = checkpoint.get("model_dimensions", {})
    if (tuple(checkpoint.get("event_classes", ())) != EVENT_CLASSES or
            dimensions != {"causal_context": 2233, "packed_observation": 3074,
                           "event_resources": 347} or
            rollout.get("policy_version") != checkpoint_sha):
        raise ValueError("checkpoint/rollout is not the captured full-day v3 parent")

    model = build_model(dimensions["causal_context"],
                        dimensions["packed_observation"],
                        dimensions["event_resources"], checkpoint["model_scale"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    first_day = rollout["games"][0]["days"][0]
    example_state = state_tensors(first_day)
    with torch.inference_mode():
        example_hidden = model.initial_hidden(*example_state)
        first_event = first_day["events"][0]
        example_step = (
            example_hidden, torch.from_numpy(first_event["resources"]),
            torch.tensor([first_event["cell"]]),
            torch.tensor([first_event["stage"]]),
            torch.tensor([first_event["previous"]]),
            torch.from_numpy(first_event["legal"]),
        )
        scripted = torch.jit.trace_module(
            model, {"initial_hidden": example_state, "step": example_step},
            check_trace=True, strict=True)
        scripted = torch.jit.freeze(scripted)
        scripted.save(str(args.output))

        hidden_rows, resource_rows, cells, stages, previous = [], [], [], [], []
        legal_rows, logits_rows, next_hidden_rows, actions = [], [], [], []
        initial_reference = model.initial_hidden(*example_state)
        for game in rollout["games"]:
            for day in game["days"]:
                hidden = model.initial_hidden(*state_tensors(day))
                for event in day["events"]:
                    resources = torch.from_numpy(event["resources"])
                    cell = torch.tensor([event["cell"]])
                    stage = torch.tensor([event["stage"]])
                    prior = torch.tensor([event["previous"]])
                    legal = torch.from_numpy(event["legal"])
                    logits, next_hidden = model.step(
                        hidden, resources, cell, stage, prior, legal)
                    hidden_rows.append(hidden.clone())
                    resource_rows.append(resources)
                    cells.append(cell)
                    stages.append(stage)
                    previous.append(prior)
                    legal_rows.append(legal)
                    logits_rows.append(logits)
                    next_hidden_rows.append(next_hidden)
                    actions.append(logits.argmax(1))
                    hidden = next_hidden
                    if len(hidden_rows) == args.events:
                        break
                if len(hidden_rows) == args.events:
                    break
            if len(hidden_rows) == args.events:
                break
        if len(hidden_rows) != args.events:
            raise ValueError(f"rollout only supplied {len(hidden_rows)} events")

    fixture_tensors = {
        "context": example_state[0], "observation": example_state[1],
        "observation_length": example_state[2],
        "token_continuous": example_state[3], "token_count": example_state[5],
        "expected_initial_hidden": initial_reference,
        "hidden": torch.cat(hidden_rows), "resources": torch.cat(resource_rows),
        "cells": torch.cat(cells), "stages": torch.cat(stages),
        "previous": torch.cat(previous), "legal": torch.cat(legal_rows),
        "expected_logits": torch.cat(logits_rows),
        "expected_next_hidden": torch.cat(next_hidden_rows),
        "expected_actions": torch.cat(actions),
    }
    fixture_tensors.update({
        f"token_category_{index}": value
        for index, value in enumerate(example_state[4])
    })
    torch.jit.script(Fixture(fixture_tensors)).save(str(args.fixture_output))

    loaded = torch.jit.load(str(args.output))
    with torch.inference_mode():
        exported_hidden = loaded.initial_hidden(*example_state)
        exported_logits, exported_next = loaded.step(*example_step)
        python_max_abs = max(
            float((exported_hidden - example_hidden).abs().max()),
            float((exported_logits - model.step(*example_step)[0]).abs().max()),
            float((exported_next - model.step(*example_step)[1]).abs().max()),
        )
    if python_max_abs != 0.0:
        raise RuntimeError(f"Python TorchScript parity failed: {python_max_abs}")
    print(json.dumps({
        "status": "PASS", "checkpoint_sha256": checkpoint_sha,
        "rollout_policy_version": rollout["policy_version"],
        "events": len(hidden_rows), "python_torchscript_max_abs": python_max_abs,
        "model": str(args.output), "model_sha256": sha256(args.output),
        "fixture": str(args.fixture_output),
        "fixture_sha256": sha256(args.fixture_output),
    }))


if __name__ == "__main__":
    main()
