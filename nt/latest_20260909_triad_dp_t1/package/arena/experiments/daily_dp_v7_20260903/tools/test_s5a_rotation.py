"""Freeze and test isolated typed rotation calendar; no production rebuild."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
E=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser();cli.add_argument('--version',required=True);a=cli.parse_args()
out=E/('receipts/s5a_rotation_mechanisms_'+a.version);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
sources=['native/rotation_portfolio_prototype.hpp','native/test_rotation_portfolio.cpp']
hashes={p:sha(E/p) for p in sources}
for p in sources:(out/Path(p).name).write_bytes((E/p).read_bytes())
command=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(E/'native/test_rotation_portfolio.cpp'),str(E/'native/vendor/simulator.cpp'),'-o',str(out/'test')]
tic=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
receipt=dict(source_hashes=hashes,production=build,command=command,compile_seconds=time.perf_counter()-tic,compile_returncode=r.returncode)
if not r.returncode:
 t=time.perf_counter();test=subprocess.run([str(out/'test')],capture_output=True,text=True);receipt.update(test_returncode=test.returncode,test_seconds=time.perf_counter()-t,stdout=test.stdout,stderr=test.stderr)
 (out/'test.log').write_text(test.stdout+test.stderr)
 receipt['status']='PASS_ISOLATED_MECHANISMS' if test.returncode==0 else 'FAILED_PRESERVED'
else:receipt['status']='COMPILE_FAILED_PRESERVED'
for p,h in hashes.items():assert sha(E/p)==h
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('production','command','source_hashes')},ensure_ascii=False),flush=True)
raise SystemExit(0 if receipt['status']=='PASS_ISOLATED_MECHANISMS' else 1)
