"""Test-only root-entry adapter for the documented collection-labor ablation."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/'candidate' if (ROOT/'candidate').exists() else ROOT
spec=importlib.util.spec_from_file_location('ablation_runtime',PROD/'main.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
_seats={}
def agent(obs,configuration=None):
 seat=int(obs['player'])
 if seat not in _seats:_seats[seat]=module.create_agent(ROOT/'probes/ablation.so')
 return _seats[seat](obs,configuration)
def reset():
 for a in _seats.values():a.close()
 _seats.clear()
