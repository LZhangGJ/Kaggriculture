#!/usr/bin/env python3
"""Warm-start the isolated crop-DIG / animal-RETIRE actor from a v3 checkpoint."""

import argparse
import copy
import hashlib
import math
from pathlib import Path

import torch

from experiments.student_v3_runtime_model import EVENT_CLASSES, EVENT_CLASSES_V4, LOCAL_FEATURE_WIDTH_V4
from experiments.train_midgame_autofill_v3 import build_model


def migrate(source: dict, new_action_logit_gap: float = 4.0) -> dict:
    if not math.isfinite(new_action_logit_gap) or new_action_logit_gap < 0:
        raise ValueError("new-action logit gap must be finite and nonnegative")
    if (tuple(source["event_classes"]) != EVENT_CLASSES or
            source["model_dimensions"] != {
                "causal_context": 2233, "packed_observation": 3145,
                "event_resources": 374} or
            source.get("shop_action_head_semantics") == 1):
        raise ValueError("source is not the 374D economic v3 actor")
    dims = source["model_dimensions"]
    model = build_model(dims["causal_context"], dims["packed_observation"],
                        dims["event_resources"] + LOCAL_FEATURE_WIDTH_V4,
                        source["model_scale"], event_classes=EVENT_CLASSES_V4)
    old = source["model"]
    state = model.state_dict()
    for name, value in state.items():
        if name == "resource.weight":
            value.zero_()
            value[:, :374].copy_(old[name])
        elif name == "previous.weight":
            value[:len(EVENT_CLASSES)].copy_(old[name][:-1])
            value[len(EVENT_CLASSES):len(EVENT_CLASSES_V4)].copy_(old[name][2])
            value[len(EVENT_CLASSES_V4)].copy_(old[name][-1])
        elif name == "head.weight":
            value[:len(EVENT_CLASSES)].copy_(old[name])
            value[len(EVENT_CLASSES_V4)-2:].copy_(old[name][1])
        elif name == "head.bias":
            value[:len(EVENT_CLASSES)].copy_(old[name])
            value[len(EVENT_CLASSES_V4)-2:] = old[name][1] - new_action_logit_gap
        else:
            value.copy_(old[name])
    result = copy.deepcopy(source)
    result["model"] = state
    result["event_classes"] = EVENT_CLASSES_V4
    result["model_dimensions"] = {**dims, "event_resources": 374 + LOCAL_FEATURE_WIDTH_V4}
    result["student_lifecycle_semantics"] = 4
    norms = result["normalization"]
    norms["resource_mean"] = torch.cat((torch.as_tensor(norms["resource_mean"]),
                                         torch.zeros(LOCAL_FEATURE_WIDTH_V4)))
    norms["resource_std"] = torch.cat((torch.as_tensor(norms["resource_std"]),
                                        torch.ones(LOCAL_FEATURE_WIDTH_V4)))
    names = list(state)
    for index, name in enumerate(names):
        if name not in ("resource.weight", "previous.weight", "head.weight", "head.bias"):
            continue
        entry = result["rl_optimizer"]["state"][index]
        for key in ("exp_avg", "exp_avg_sq"):
            previous = entry[key]
            expanded = torch.zeros_like(state[name])
            if name == "resource.weight":
                expanded[:, :374].copy_(previous)
            elif name == "previous.weight":
                expanded[:len(EVENT_CLASSES)].copy_(previous[:-1])
                expanded[len(EVENT_CLASSES_V4)].copy_(previous[-1])
            else:
                expanded[:len(EVENT_CLASSES)].copy_(previous)
            entry[key] = expanded
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-action-logit-gap", type=float, default=4.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = migrate(source, args.new_action_logit_gap)
    result["lifecycle_migration"] = {
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "old_action_parity_when_new_actions_masked": True,
        "new_action_logit_bias_relative_to_keep": -args.new_action_logit_gap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
