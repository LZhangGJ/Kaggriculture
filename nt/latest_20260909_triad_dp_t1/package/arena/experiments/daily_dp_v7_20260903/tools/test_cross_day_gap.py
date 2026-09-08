from pathlib import Path
import hashlib,json,subprocess,time
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4r_cross_day_gap_v1';out.mkdir(exist_ok=False)
b=json.loads((E/'native/build/build_receipt.json').read_text());hashes=b['source_hashes'].copy()
src=E/'native/test_cross_day_gap.cpp';hashes[str(src.relative_to(E))]=hashlib.sha256(src.read_bytes()).hexdigest()
for rel,h in hashes.items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(out/'test')]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('compile failed')
compiled=time.perf_counter()-t;r=subprocess.run([str(out/'test')],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
d=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',returncode=r.returncode,error=r.stderr)
d.update(build=b,source_hashes=hashes,command=cmd,compile_seconds=compiled)
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('build','source_hashes','command')}),flush=True);raise SystemExit(r.returncode)
