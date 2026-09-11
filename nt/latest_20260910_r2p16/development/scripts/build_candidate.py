"""Build only the isolated R2P1 candidate; never replace the released R2."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
POLICY = HERE/'candidate_r2p1/policy'
target = POLICY/'r2p1.so'
assert not target.exists(), 'Do not overwrite a previously built candidate'
sources = {str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in POLICY.rglob('*') if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
command = ['g++-13','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off',
           '-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),
           str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
started = time.perf_counter()
with (HERE/'candidate_r2p1/compile.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(status='BUILT_NOT_VALIDATED' if result.returncode==0 else 'COMPILE_FAILED',
               command=command, seconds=time.perf_counter()-started, sources=sources,
               compiler=subprocess.check_output(['g++-13','--version'],text=True))
if target.exists():
    receipt['binary_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
(HERE/'candidate_r2p1/BUILD.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in {'sources','compiler'}}),flush=True)
raise SystemExit(result.returncode)
