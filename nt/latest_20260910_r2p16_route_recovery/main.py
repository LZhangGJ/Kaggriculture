"""R2P16 procurement recovery candidate; reconcile actual fills and replan remaining work."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r2p16_route_native', ROOT / 'policy/agent.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

def create_agent(binary=None):
    return runtime.Agent(binary_path=binary or ROOT / 'build/revision3/route3.so')

_agents = {}
def agent(observation, configuration=None):
    seat = observation['player'] if isinstance(observation, dict) else observation.player
    if seat not in _agents:
        _agents[seat] = create_agent()
    return _agents[seat](observation)
