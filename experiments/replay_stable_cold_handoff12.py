"""Diagnostic: replay route tree, then the frozen reliable dynamic policy."""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["REPLAY_FORCED_OPENING"] = "G001"
base_spec = importlib.util.spec_from_file_location("replay_stable_route", HERE.parent / "agent/main.py")
base = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base)
stable_path = HERE.parent / "policy/r1/agent.py"
stable_spec = importlib.util.spec_from_file_location("replay_stable_dynamic", stable_path)
stable = importlib.util.module_from_spec(stable_spec)
stable_spec.loader.exec_module(stable)


class Agent:
    def __init__(self):
        deployment = base.replay_deployment()
        self.replay = base.create_replay_agent(deployment, "stable_cold_handoff")
        self.dynamic = stable.Agent(binary_path=stable_path.with_name("agent.so"))

    def __call__(self, observation, configuration=None):
        if int(observation.get("step", 0)) >= 288:
            return self.dynamic(observation, configuration)
        return self.replay(observation, configuration)

    def close(self):
        self.dynamic.close()


def create_agent():
    return Agent()
