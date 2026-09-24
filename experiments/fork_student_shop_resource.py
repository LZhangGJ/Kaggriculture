#!/usr/bin/env python3
"""Append nine per-slot known shop-demand inputs without changing the policy."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from experiments.fork_student_economic_v1 import _extend
from experiments.train_midgame_autofill_v3 import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shop-action-head", action="store_true",
                        help="isolated state-conditioned per-product logit gate")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dims = source["model_dimensions"]
    if (dims != {"causal_context": 2233, "packed_observation": 3145,
                 "event_resources": 374} or
            source.get("economic_features_semantics", 1) not in (1, 2) or
            source.get("shop_resource_semantics")):
        raise ValueError("expected an isolated 374D economic actor")
    output = copy.deepcopy(source)
    output["model_dimensions"]["event_resources"] = 383
    output["model"]["resource.weight"] = _extend(
        output["model"]["resource.weight"], 9)
    if args.shop_action_head:
        shop_model = build_model(2233, 3145, 383, source["model_scale"],
                                 shop_action_head=True)
        output["model"]["shop_gate.weight"] = shop_model.state_dict()["shop_gate.weight"]
        output["model"]["shop_gate.bias"] = shop_model.state_dict()["shop_gate.bias"]
    normalized = output["normalization"]
    normalized["resource_mean"] = np.concatenate((
        normalized["resource_mean"], np.zeros(9, dtype=np.float32)))
    normalized["resource_std"] = np.concatenate((
        normalized["resource_std"], np.full(9, 1 / 32, dtype=np.float32)))
    old = build_model(2233, 3145, 374, source["model_scale"])
    ids = output["rl_optimizer"]["param_groups"][0]["params"]
    key = dict((name, index) for (name, _), index in
               zip(old.named_parameters(), ids))["resource.weight"]
    for moment in ("exp_avg", "exp_avg_sq"):
        output["rl_optimizer"]["state"][key][moment] = _extend(
            output["rl_optimizer"]["state"][key][moment], 9)
    if args.shop_action_head:
        ids.extend((max(ids) + 1, max(ids) + 2))
    # One invariant check covers the added affine input path.
    before = source["model"]["resource.weight"]
    after = output["model"]["resource.weight"]
    sample = torch.randn(4, 374)
    if not torch.equal(sample @ before.T,
                       torch.cat((sample, torch.randn(4, 9)), 1) @ after.T):
        raise RuntimeError("per-slot shop fork changed the source policy")
    if args.shop_action_head:
        old = build_model(2233, 3145, 374, source["model_scale"])
        old.load_state_dict(source["model"])
        shop_model.load_state_dict(output["model"])
        hidden = torch.randn(4, 64 * source["model_scale"])
        resource = torch.randn(4, 374)
        cells = torch.arange(4)
        stages = torch.ones(4, dtype=torch.long)
        previous = torch.zeros(4, dtype=torch.long)
        legal = torch.ones(4, 11, dtype=torch.bool)
        with torch.no_grad():
            baseline = old.step(hidden, resource, cells, stages, previous, legal)
            forked = shop_model.step(hidden, torch.cat((resource, torch.randn(4, 9)), 1),
                                     cells, stages, previous, legal)
        if not all(torch.equal(a, b) for a, b in zip(baseline, forked)):
            raise RuntimeError("shop action head changed the source policy at fork")
    output["shop_resource_semantics"] = 1
    if args.shop_action_head:
        output["shop_action_head_semantics"] = 1
    output["rl"]["deployment_eligible"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({"output": str(args.output), "event_resources": 383,
           "shop_action_head": args.shop_action_head,
           "function_preserving_at_fork": True})


if __name__ == "__main__":
    main()
