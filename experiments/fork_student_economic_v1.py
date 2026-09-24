#!/usr/bin/env python3
"""Fork an RL actor with zero-initialized economic inputs; never alter source."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.student_economic_features_v1 import FEATURE_WIDTH, PREFIX_WIDTH
from experiments.train_midgame_autofill_v3 import build_model


def _extend(value, width):
    return torch.cat((value, torch.zeros((*value.shape[:-1], width),
                                           dtype=value.dtype)), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dims = source["model_dimensions"]
    if dims != {"causal_context": 2233, "packed_observation": 3074,
                "event_resources": 347}:
        raise ValueError("unexpected source actor dimensions")
    old = build_model(2233, 3074, 347, source["model_scale"])
    old.load_state_dict(source["model"])
    new = build_model(2233, 3074 + FEATURE_WIDTH,
                      347 + PREFIX_WIDTH, source["model_scale"])
    state = {name: (_extend(value, FEATURE_WIDTH)
                    if name == "observation.weight" else
                    _extend(value, PREFIX_WIDTH)
                    if name == "resource.weight" else value.clone())
             for name, value in source["model"].items()}
    new.load_state_dict(state)

    # The two new columns of each AdamW moment are zero, while every old
    # moment and step is preserved. The initial policy is therefore unchanged.
    output = copy.copy(source)
    output["model"] = state
    output["model_dimensions"] = {**dims,
        "packed_observation": 3074 + FEATURE_WIDTH,
        "event_resources": 347 + PREFIX_WIDTH}
    output["normalization"] = dict(source["normalization"])
    for name, width in (("observation", FEATURE_WIDTH),
                        ("resource", PREFIX_WIDTH)):
        for suffix, fill in (("mean", 0.0), ("std", 1.0)):
            key = f"{name}_{suffix}"
            old_values = np.asarray(output["normalization"][key])
            output["normalization"][key] = np.concatenate((
                old_values, np.full(width, fill, dtype=old_values.dtype)))
    output["rl_optimizer"] = copy.deepcopy(source["rl_optimizer"])
    parameters = list(old.named_parameters())
    group = output["rl_optimizer"]["param_groups"][0]["params"]
    if len(parameters) != len(group):
        raise ValueError("optimizer/model parameter count mismatch")
    for (name, parameter), key in zip(parameters, group):
        moments = output["rl_optimizer"]["state"][key]
        if tuple(moments["exp_avg"].shape) != tuple(parameter.shape):
            raise ValueError(f"optimizer state drift: {name}")
        width = FEATURE_WIDTH if name == "observation.weight" else (
            PREFIX_WIDTH if name == "resource.weight" else 0)
        if width:
            for moment in ("exp_avg", "exp_avg_sq"):
                moments[moment] = _extend(moments[moment], width)
    output["rl"] = {**source["rl"], "deployment_eligible": False}
    output["economic_fork"] = {
        "schema": "public-economic-input-v1-diagnostic-only",
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "shop_and_rival_features": FEATURE_WIDTH,
        "per_slot_prefix_features": PREFIX_WIDTH,
        "native_rollout_ready": False,
    }

    # One check is sufficient: changing *only* the newly appended inputs
    # must leave both initial state and per-slot logits unchanged at the fork.
    old.eval(); new.eval()
    with torch.no_grad():
        context = torch.zeros((2, 2233))
        observation = torch.zeros((2, 3074))
        length = torch.zeros(2)
        continuous = torch.zeros((2, 1, 24))
        categories = [torch.zeros((2, 1), dtype=torch.long) for _ in range(7)]
        count = torch.ones(2, dtype=torch.long)
        a = old.initial_hidden(context, observation, length, continuous,
                               categories, count)
        b = new.initial_hidden(context, torch.cat((observation,
            torch.randn(2, FEATURE_WIDTH)), 1), length, continuous,
            categories, count)
        legal = torch.ones((2, 11), dtype=torch.bool)
        old_logits, _ = old.step(a, torch.zeros((2, 347)),
                                 torch.zeros(2, dtype=torch.long),
                                 torch.ones(2, dtype=torch.long),
                                 torch.zeros(2, dtype=torch.long), legal)
        new_logits, _ = new.step(b, torch.cat((torch.zeros((2, 347)),
            torch.randn(2, PREFIX_WIDTH)), 1),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long), legal)
        maximum = max(float((a - b).abs().max()),
                      float((old_logits - new_logits).abs().max()))
        if maximum > 1e-5:
            raise RuntimeError(f"initial policy drift: {maximum}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({"output": str(args.output), "source": str(args.checkpoint),
           "new_observation": output["model_dimensions"]["packed_observation"],
           "new_resources": output["model_dimensions"]["event_resources"],
           "initial_max_abs_drift": maximum,
           "native_rollout_ready": False})


if __name__ == "__main__":
    main()
