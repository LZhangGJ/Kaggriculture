from pathlib import Path
import argparse,hashlib,json,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
E=Path(__file__).resolve().parents[1];out=E/args.out;out.mkdir(exist_ok=False)
src=E/'native/test_idle_handoff.cpp';binary=out/'test'
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:print(r.stderr[-3000:]);raise SystemExit(r.returncode)
r=subprocess.run([str(binary)],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
d=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',returncode=r.returncode,error=r.stderr)
files=[src,E/'native/policy.hpp',E/'native/idle_task_handoff.hpp',E/'native/observed_day_scenario.hpp']
d.update(source_hashes={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},command=cmd,seconds=time.perf_counter()-tic)
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('source_hashes','command')}),flush=True);raise SystemExit(r.returncode)
