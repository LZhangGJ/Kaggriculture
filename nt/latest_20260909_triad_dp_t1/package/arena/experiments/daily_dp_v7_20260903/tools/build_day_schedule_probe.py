from pathlib import Path
import argparse,hashlib,json,subprocess,sysconfig,time
import pybind11
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
E=Path(__file__).resolve().parents[1];out=E/a.out;out.mkdir(exist_ok=False)
src=[E/'native/day_schedule_probe.cpp',E/'native/vendor/simulator.cpp']
files=src+[E/'native/observed_day_scenario.hpp',E/'native/vendor/simulator.hpp',E/'native/policy.hpp',E/'native/resource_exchange.hpp',E/'native/intraday_admission.hpp',E/'native/service_recovery.hpp',E/'native/joint_portfolio.hpp']
files += [E/'native/observation_view.hpp',E/'native/day_consequence.hpp',E/'native/crop_delivery.hpp']
hashes={str(x.relative_to(E)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
binary=out/('_dp7_dayprobe'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-fopenmp','-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(E/'native'),*(str(x) for x in src),'-o',str(binary)]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
d=dict(status='PASS_BUILD' if not r.returncode else 'FAIL',seconds=time.perf_counter()-t,command=cmd,source_hashes=hashes,binary=str(binary))
if not r.returncode:d['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
assert hashes=={str(x.relative_to(E)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('command','source_hashes')}),flush=True);raise SystemExit(r.returncode)
