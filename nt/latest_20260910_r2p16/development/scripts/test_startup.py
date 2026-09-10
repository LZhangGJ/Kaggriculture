from pathlib import Path
import subprocess,json,hashlib
HERE=Path(__file__).resolve().parent
out=HERE/'candidate_r2p6';checks=[]
for mode in (0,1,2):
 target=out/f'test_startup_{mode}'
 subprocess.run(['g++-13','-std=c++20','-O2',f'-DR2_STARTUP_MODE={mode}',str(out/'test_startup.cpp'),'-o',str(target)],check=True)
 r=subprocess.run([str(target)],capture_output=True,text=True,check=True)
 checks.append(dict(mode=mode,status='PASS',output=r.stdout.strip()))
off=out/'policy/startup_0.so';old=HERE/'candidate_r2p2/policy/r2p2.so'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(off)==sha(old)
prior=json.loads((HERE/'candidate_r2p2/OFF_CONFORMANCE.json').read_text())
(out/'UNIT_TESTS.json').write_text(json.dumps(dict(status='PASS',checks=checks,off_sha256=sha(off),
    off_identical_to_p2=True,reused_off_conformance='../candidate_r2p2/OFF_CONFORMANCE.json',prior=prior),indent=2))
print(json.dumps(dict(status='PASS',modes=3,off_binary_identical=True)),flush=True)
