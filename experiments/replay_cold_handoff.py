"""Current replay deployment followed by a cold, unobserved R1 handoff."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cold_handoff_base", ROOT / "agent/main.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def create_agent():
    agent = base.create_agent()
    agent.dynamic.observe_external = lambda observation, action: None
    return agent
