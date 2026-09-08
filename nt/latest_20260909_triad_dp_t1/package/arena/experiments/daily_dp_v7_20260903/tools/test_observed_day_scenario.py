from pathlib import Path
import argparse,hashlib,json,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
E=Path(__file__).resolve().parents[1];out=E/a.out;out.mkdir(exist_ok=False)
sources=[E/'native/test_observed_day_scenario.cpp',E/'native/vendor/simulator.cpp']
inputs=sources+[E/'native/observed_day_scenario.hpp',E/'native/vendor/simulator.hpp',E/'native/policy.hpp',E/'native/resource_exchange.hpp',E/'native/intraday_admission.hpp',E/'native/service_recovery.hpp',E/'native/joint_portfolio.hpp']
inputs += [E/'native/observation_view.hpp',E/'native/day_consequence.hpp',E/'native/crop_delivery.hpp']
before={str(x.relative_to(E)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(E/'native'),*(str(x) for x in sources),'-o',str(out/'test')]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:raise SystemExit(r.returncode)
compiled=time.perf_counter()-t;t=time.perf_counter()
r=subprocess.run([str(out/'test')],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
d=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',error=r.stderr)
d.update(compile_seconds=compiled,run_seconds=time.perf_counter()-t,command=cmd,source_hashes=before,not_policy_strength=True,opponent_scenario='PASS',horizon='current day only, no new random weeds or shop input')
assert before=={str(x.relative_to(E)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('command','source_hashes')}),flush=True);raise SystemExit(r.returncode)
