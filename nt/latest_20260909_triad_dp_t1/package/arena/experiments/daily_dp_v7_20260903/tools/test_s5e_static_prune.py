"""Isolated mechanism check of exact-order static pruning, not production."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h,rel
out=E/('receipts/s5e_static_prune_'+a.version);out.mkdir(exist_ok=False);sources={}
for rel in ('native/compile_static_prune_prototype.hpp','native/test_compile_static_prune.cpp','native/test_compile_choices.cpp'):
 path=E/rel;(out/path.name).write_bytes(path.read_bytes());sources[rel]=sha(path)
cmd=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(E/'native/test_compile_static_prune.cpp'),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(out/'test')]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if not r.returncode:
 r=subprocess.run([str(out/'test')],capture_output=True,text=True);(out/'test.log').write_text(r.stdout+r.stderr)
receipt=dict(status='PASS_ISOLATED_PROTOTYPE_NOT_PRODUCTION' if not r.returncode else 'FAIL_PRESERVED',returncode=r.returncode,seconds=time.perf_counter()-tic,production_build=build,source_hashes=sources,command=cmd,output=r.stdout,error=r.stderr,policy_changed=False)
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('production_build','source_hashes','command')},indent=2));raise SystemExit(r.returncode)
