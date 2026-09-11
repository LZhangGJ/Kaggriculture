"""Freeze and execute a read-only estimator coverage test; no policy change."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h,rel
out=E/('receipts/s5e_plan_value_coverage_'+a.version);out.mkdir(exist_ok=False)
src=E/'native/test_s5e_plan_value_coverage.cpp';(out/'source.cpp').write_bytes(src.read_bytes())
cmd=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(out/'test')]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if not r.returncode:
 r=subprocess.run([str(out/'test')],capture_output=True,text=True);(out/'test.log').write_text(r.stdout+r.stderr)
receipt=dict(status='PASS_DIAGNOSTIC_NOT_REPAIR' if not r.returncode else 'FAIL_PRESERVED',returncode=r.returncode,seconds=time.perf_counter()-tic,production_build=build,source_sha256=sha(src),command=cmd,output=r.stdout,error=r.stderr,policy_changed=False)
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));raise SystemExit(r.returncode)
