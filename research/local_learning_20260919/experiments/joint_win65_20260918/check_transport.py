"""The inference process must be able to unpickle observations without referee imports."""
import json
import pickle
from pathlib import Path
import subprocess
import sys
from evaluate import observation_message

sys.path.insert(0, 'F:/Kaggriculture/experiments/local_teacher_bc_20260917/frozen/pool/referee')
from cpu_runtime import AttrDict

obs = AttrDict(step=0, private=AttrDict(seeds={'WHEAT': 17}),
               farms=[{'hands': [{'location': [1, 2]}]}], market={'price': 1.25})
message = observation_message(obs)
assert message['obs'] == obs and type(message['obs']) is dict and type(message['obs']['private']) is dict
child = "import sys,pickle,json; x=pickle.loads(sys.stdin.buffer.read()); assert 'cpu_runtime' not in sys.modules; print(json.dumps(x))"
result = subprocess.run([sys.executable, '-I', '-c', child], input=pickle.dumps(message), capture_output=True, check=True)
assert json.loads(result.stdout) == message
print(json.dumps(dict(status='PASS', referee_import_needed=False, nested_values_preserved=True)))
