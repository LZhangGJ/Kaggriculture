#!/usr/bin/env python3
"""Regression checks for the opt-in warm-handoff selector hook."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_agent.src.route_switch_features import RouteSwitchHistory, route_switch_feature_names


def load_main():
    spec = importlib.util.spec_from_file_location("handoff_selector_test_main", ROOT / "agent/main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selector_file(directory, choice, candidate_diff=False):
    order = ["competitive_sale", "crop_succession"]
    names = ([f"{name}_minus_default_{index:03d}" for name in order for index in range(356)]
             if candidate_diff else route_switch_feature_names())
    payload = {
        "schema": "r1-handoff-selector-v1",
        "feature_schema": "r1_candidate_diff_v1" if candidate_diff else "semantic_route_switch_v1",
        "feature_names": names,
        "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "eligible": True,
        "classes": [choice],
        "tree": {"class_kind": "handoff_config", "classes": [choice],
                 "left": [-1], "right": [-1], "feature": [-2], "threshold": [-2.0],
                 "value": [[1.0]]},
    }
    if candidate_diff:
        payload.update(feature_mode="candidate_diff", feature_order=order)
    path = Path(directory) / f"{'diff-' if candidate_diff else ''}{choice}.json"
    path.write_text(json.dumps(payload))
    return path


def observation(step):
    farm = {"money": 1000, "hands": [], "unlocked_quadrants": ["NW", "NE"],
            "hires_today": 0, "tiles": [[None] * 10 for _ in range(10)]}
    return {"step": step, "day": step // 24, "hour": step % 24, "player": 0,
            "farms": [farm, dict(farm)], "private": {"shed": {}, "seeds": {},
            "inventories": [{}]}, "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []}}


class Replay:
    def __init__(self):
        controller = SimpleNamespace(history=RouteSwitchHistory(), opening="G001", current="G001",
                                     switched=False,
                                     route_by_family={"G001": "route"})
        self.controller = controller
        self.expanded_agent = SimpleNamespace(action_tapes={"route": []})

    def __call__(self, obs, configuration=None):
        self.controller.history.update(obs)
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def close(self):
        pass


class Dynamic:
    def __init__(self):
        self.external = False
        self.prepare_calls = 0
        self.install_calls = 0
        self.installed_day = None

    def observe_external(self, observation, action):
        self.external = True

    def prepare_candidates(self, obs, with_features=False):
        self.prepare_calls += 1
        self.prepared_day = obs["day"]
        rows = [{"index": 3, "diagnostic": "", "score": 1.0},
                {"index": 4, "diagnostic": "", "score": 2.0},
                {"index": 7, "diagnostic": "competitive_sale", "score": 0.0},
                {"index": 8, "diagnostic": "crop_succession", "score": 0.0}]
        if with_features:
            for index, row in enumerate(rows):
                row["features"] = [float(index)] * 356
        return rows

    def install_candidate(self, index):
        self.install_calls += 1
        self.installed_day = self.prepared_day

    def __call__(self, obs, configuration=None):
        diagnostic = self.installed_day == obs["day"]
        self.external = False
        return {"farmer": ["PASS"], "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]] if diagnostic else []}

    def close(self):
        pass

    def debug(self):
        return {}


def agent(module, selector):
    return module.ReplayThenDynamicAgent(Replay(), Dynamic(), handoff_step=24, selector=selector)


def official_actions(module, selector_path):
    from kaggle_environments import make
    previous = os.environ.get("REPLAY_HANDOFF_SELECTOR")
    try:
        if selector_path is None:
            os.environ.pop("REPLAY_HANDOFF_SELECTOR", None)
        else:
            os.environ["REPLAY_HANDOFF_SELECTOR"] = str(selector_path)
        policy = module.create_agent(0)
        env = make("kaggriculture", configuration={"seed": 2609400000}, debug=True)
        state = env.reset()
        actions = []
        for _ in range(270):
            obs = json.loads(json.dumps(state[0].observation))
            obs["player"] = 0
            action = policy(obs, env.configuration)
            actions.append(action)
            state = env.step([action, ["PASS"]])
        return actions, policy.debug()
    finally:
        if "policy" in locals():
            policy.close()
        if previous is None:
            os.environ.pop("REPLAY_HANDOFF_SELECTOR", None)
        else:
            os.environ["REPLAY_HANDOFF_SELECTOR"] = previous


def main():
    module = load_main()
    with tempfile.TemporaryDirectory() as directory:
        default_path = selector_file(directory, "default")
        sale_path = selector_file(directory, "competitive_sale")
        diff_default_path = selector_file(directory, "default", candidate_diff=True)
        previous = os.environ.get("REPLAY_HANDOFF_SELECTOR")
        try:
            os.environ["REPLAY_HANDOFF_SELECTOR"] = "0"
            assert module.handoff_selector() is None
            os.environ["REPLAY_HANDOFF_SELECTOR"] = str(default_path)
            default_tree = module.handoff_selector()
        finally:
            if previous is None:
                os.environ.pop("REPLAY_HANDOFF_SELECTOR", None)
            else:
                os.environ["REPLAY_HANDOFF_SELECTOR"] = previous

        off = agent(module, None)
        default = agent(module, default_tree)
        off_actions = [off(observation(step)) for step in range(50)]
        default_actions = [default(observation(step)) for step in range(50)]
        assert off_actions == default_actions
        assert default.dynamic.prepare_calls == default.dynamic.install_calls == 0
        assert off.debug()["selector_skip_reason"] == "off"
        assert default.debug()["selector_class"] == "default"

        diff_default = agent(module, module.handoff_selector(str(diff_default_path)))
        diff_actions = [diff_default(observation(step)) for step in range(50)]
        assert off_actions == diff_actions
        assert diff_default.dynamic.prepare_calls == 1
        assert diff_default.dynamic.install_calls == 0

        real_off, _ = official_actions(module, None)
        real_default, real_debug = official_actions(module, diff_default_path)
        assert real_off == real_default
        assert real_debug["selector_class"] == "default"
        assert real_debug["selector_installed_index"] is None

        cold = agent(module, module.handoff_selector(str(sale_path)))
        cold(observation(24))
        assert cold.dynamic.prepare_calls == cold.dynamic.install_calls == 0
        assert cold.debug()["selector_skip_reason"] == "external_false"

        non_boundary = module.ReplayThenDynamicAgent(
            Replay(), Dynamic(), handoff_step=25,
            selector=module.handoff_selector(str(sale_path)))
        for step in range(26):
            non_boundary(observation(step))
        assert non_boundary.debug()["selector_skip_reason"] == "not_day_boundary"

        diagnostic = agent(module, module.handoff_selector(str(sale_path)))
        for step in range(24):
            diagnostic(observation(step))
        first = diagnostic(observation(24))
        same_day = diagnostic(observation(25))
        next_day = diagnostic(observation(48))
        assert diagnostic.dynamic.prepare_calls == diagnostic.dynamic.install_calls == 1
        assert first["market"] and same_day["market"] and not next_day["market"]
        audit = diagnostic.debug()
        assert audit["selector_class"] == "competitive_sale"
        assert audit["selector_step"] == 24 and audit["selector_installed_index"] == 7
        assert audit["selector_eligible"] is True and audit["selector_skip_reason"] is None

        valid = json.loads(diff_default_path.read_text())
        broken = []
        bad_hash = {**valid, "feature_names_sha256": "0" * 64}
        broken.append(bad_hash)
        bad_order = {**valid, "feature_order": list(reversed(valid["feature_order"]))}
        broken.append(bad_order)
        bad_dim = {**valid, "feature_names": valid["feature_names"][:-1]}
        bad_dim["feature_names_sha256"] = hashlib.sha256(
            "\n".join(bad_dim["feature_names"]).encode()).hexdigest()
        broken.append(bad_dim)
        for index, payload in enumerate(broken):
            bad = Path(directory) / f"bad-{index}.json"
            bad.write_text(json.dumps(payload))
            try:
                module.handoff_selector(str(bad))
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid candidate-diff selector {index} was accepted")
    print(json.dumps({"status": "PASS", "off_default_frames": 50,
                      "official_candidate_diff_default_frames": 270,
                      "cold_prepare_calls": 0, "diagnostic_install_calls": 1}))


if __name__ == "__main__":
    main()
