"""P16 JointAFS R1 structural-candidate policy. Only current observation enters policy."""
from pathlib import Path
import importlib.util
import json
ROOT=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_p16_t3_takeover_codec',ROOT/'policy/agent.py')
_codec=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_codec)
def create_agent(binary=None):
    cfg=json.loads((ROOT/'policy/config.json').read_text())
    if set(cfg)!=set(_codec._ORDER):
        raise ValueError('The complete frozen 38-setting configuration is required')
    return _codec.Agent(config=cfg,binary_path=Path(binary).resolve() if binary else ROOT/'policy/joint.so')
_instances={}
def agent(observation,configuration=None):
    seat=int(observation['player'])
    if seat not in (0,1):raise ValueError('Invalid player')
    if seat not in _instances:_instances[seat]=create_agent()
    return _instances[seat](observation,configuration)

def reset():
    for instance in _instances.values():
        instance.close()
    _instances.clear()
