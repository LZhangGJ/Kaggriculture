"""AFS R1 with the validated workflow-integrity repair enabled."""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_afs_r1_workflow_codec", ROOT / "policy/agent.py")
_codec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_codec)


def create_agent(binary=None):
    config = json.loads((ROOT / "policy/config.json").read_text())
    path = Path(binary).resolve() if binary else ROOT / "policy/agent.so"
    return _codec.Agent(config=config, binary_path=path)


_instances = {}


def agent(observation, configuration=None):
    seat = int(observation["player"])
    if seat not in (0, 1):
        raise ValueError("Invalid player")
    if seat not in _instances:
        _instances[seat] = create_agent()
    return _instances[seat](observation, configuration)


def reset():
    for instance in _instances.values():
        instance.close()
    _instances.clear()
