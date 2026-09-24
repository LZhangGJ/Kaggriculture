#!/usr/bin/env python3
"""Function-preserving gain for the appended economic inputs only."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import build_model


EXTRAS = (("observation", 71), ("resource", 27))


def fork(source: dict, gain: float) -> dict:
    if not math.isfinite(gain) or gain <= 1:
        raise ValueError("gain must be finite and greater than one")
    dims = source["model_dimensions"]
    if (dims["packed_observation"], dims["event_resources"]) != (3145, 374):
        raise ValueError("expected the existing economic actor")
    if source.get("economic_gain"):
        raise ValueError("refusing to stack economic gains")
    output = copy.deepcopy(source)
    model = build_model(dims["causal_context"], dims["packed_observation"],
                        dims["event_resources"], source["model_scale"])
    ids = output["rl_optimizer"]["param_groups"][0]["params"]
    keys = dict((name, key) for (name, _), key in zip(model.named_parameters(), ids))
    for name, width in EXTRAS:
        weight = output["model"][f"{name}.weight"]
        weight[:, -width:] /= gain
        std = np.asarray(output["normalization"][f"{name}_std"]).copy()
        std[-width:] /= gain
        output["normalization"][f"{name}_std"] = std
        moments = output["rl_optimizer"]["state"][keys[f"{name}.weight"]]
        moments["exp_avg"][:, -width:] *= gain
        moments["exp_avg_sq"][:, -width:] *= gain * gain
    output["economic_gain"] = {"gain": gain, "source": "economic-v1",
                               "function_preserving_at_fork": True}
    output["rl"]["deployment_eligible"] = False
    return output


def demo() -> None:
    raw = torch.tensor([[1., 2., 3.]])
    old = torch.tensor([[4., 5., 6.]])
    gain = 32.
    scaled = old.clone(); scaled[:, -2:] /= gain
    std = torch.ones(3); std[-2:] /= gain
    assert torch.allclose(raw @ old.T, (raw / std) @ scaled.T)


def main() -> None:
    demo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=32.)
    parser.add_argument("--mature-stored", action="store_true")
    parser.add_argument("--explicit-shop-tokens", action="store_true")
    parser.add_argument("--shop-token-gain", type=float, default=1.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = fork(source, args.gain)
    if args.mature_stored:
        output["economic_features_semantics"] = 2
        output["economic_gain"]["function_preserving_at_fork"] = False
    if args.explicit_shop_tokens:
        if not math.isfinite(args.shop_token_gain) or args.shop_token_gain <= 0:
            parser.error("shop token gain must be finite and positive")
        model = build_model(2233, 3145, 374, output["model_scale"])
        ids = output["rl_optimizer"]["param_groups"][0]["params"]
        key = dict((name, index) for (name, _), index in
                   zip(model.named_parameters(), ids))["token.weight"]
        output["model"]["token.weight"][:, 8:16] = 0
        for name in ("exp_avg", "exp_avg_sq"):
            output["rl_optimizer"]["state"][key][name][:, 8:16] = 0
        output["shop_token_semantics"] = 2
        output["shop_token_gain"] = args.shop_token_gain
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({"output": str(args.output), "gain": args.gain})


if __name__ == "__main__":
    main()
