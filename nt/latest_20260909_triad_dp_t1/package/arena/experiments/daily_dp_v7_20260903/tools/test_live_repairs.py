from pathlib import Path
import argparse,hashlib,json,subprocess,time
cli=argparse.ArgumentParser();cli.add_argument('--out',default='receipts/s4v_mechanism_v1');args=cli.parse_args()
E=Path(__file__).resolve().parents[1];out=E/args.out;out.mkdir(exist_ok=False)
src=E/'native/test_live_repairs.cpp';binary=out/'test';cmd=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:print(r.stderr[-3000:]);raise SystemExit(r.returncode)
r=subprocess.run([str(binary)],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
d=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',error=r.stderr)
d.update(command=cmd,seconds=time.perf_counter()-tic,source_hashes={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,E/'native/live_execution_repairs.hpp',E/'native/policy.hpp',E/'native/service_recovery.hpp']})
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('command','source_hashes')}),flush=True);raise SystemExit(r.returncode)
