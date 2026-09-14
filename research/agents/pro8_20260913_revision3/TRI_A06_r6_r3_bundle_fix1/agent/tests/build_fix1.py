"""Offline build and execution of focused fix1 tests (no competitive games)."""
from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parents[1]
b=json.loads((R/'policy/a06.BUILD.json').read_text())
flags=b['command'][1:b['command'].index('-DA06_EXEC_MODE=2')+1]
compiler='g++'
out=R/'validation_fix1/checks.so'
cmd=[compiler,*flags,str(R/'tests/fix1_checks.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
out.parent.mkdir(exist_ok=True)
print(' '.join(cmd),flush=True)
subprocess.run(cmd,check=True,timeout=90)
subprocess.run([sys.executable,str(R/'tests/run_fix1.py')],check=True,timeout=90)
