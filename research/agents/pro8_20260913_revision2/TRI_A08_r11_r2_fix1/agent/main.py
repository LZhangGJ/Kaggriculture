"""TRI_A08_r11_r2_fix1: collection-work-consistent animal service valuation (not accepted). Explicit binary/config; no replay, seed or agent ID input."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_tri_a08_r11_r2_fix1_codec', ROOT/'policy/agent.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)

def create_agent(binary_path=None):
    config = json.loads((ROOT/'policy/config.json').read_text(encoding='utf-8'))
    return codec.Agent(config=config, binary_path=binary_path or ROOT/'policy/tri_a08_r11_r2_fix1.so')

_seats = {}
def agent(observation, configuration=None):
    seat = int(codec._get(observation, 'player', 0))
    if seat not in _seats:
        _seats[seat] = create_agent()
    return _seats[seat](observation, configuration)

def reset():
    for instance in _seats.values():
        instance.close()
    _seats.clear()
