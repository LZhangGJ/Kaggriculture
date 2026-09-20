#!/usr/bin/env python3
"""Small regression check for experiment-only R1 config overrides."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "policy/r1/agent.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    from kaggle_environments import make

    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    original = load("r1_default_check")
    explicit = original.Agent(config=json.loads((ROOT / "policy/r1/config.json").read_text()))
    default = original.Agent()
    env = make("kaggriculture", configuration={"seed": 2609700000}, debug=True)
    observation = json.loads(json.dumps(env.reset()[0].observation))
    observation["player"] = 0
    assert default.config == explicit.config
    assert default(observation) == explicit(observation)
    default.close(); explicit.close()

    os.environ["R1_CONFIG_OVERRIDES"] = '{"work_price":2}'
    work = load("r1_work_check").Agent()
    assert work.config["work_price"] == 2 and work.config["delay_sale"] == 0
    work.close()
    os.environ["R1_CONFIG_OVERRIDES"] = '{"delay_sale":1}'
    delay = load("r1_delay_check").Agent()
    assert delay.config["work_price"] == 4 and delay.config["delay_sale"] == 1
    delay.close()
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    print("PASS: default action unchanged; independent overrides accepted")


if __name__ == "__main__":
    main()
