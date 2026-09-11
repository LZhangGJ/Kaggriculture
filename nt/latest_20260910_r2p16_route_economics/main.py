"""R2P16 route-economics candidate; build mode 3 before loading."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r2p16_route_native', ROOT / 'policy/agent.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

def create_agent(binary=None):
    return runtime.Agent(binary_path=binary or ROOT / 'build/revision5/route3.so')

_agents = {}
def agent(observation, configuration=None):
    seat = observation['player'] if isinstance(observation, dict) else observation.player
    if seat not in _agents:
        _agents[seat] = create_agent()
    return _agents[seat](observation)
