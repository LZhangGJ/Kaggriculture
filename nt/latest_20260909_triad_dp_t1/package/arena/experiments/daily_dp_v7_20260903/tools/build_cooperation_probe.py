from pathlib import Path
import hashlib,json,subprocess,sysconfig,time
import pybind11
E=Path(__file__).resolve().parents[1];out=E/'native/s4s_cooperation_probe_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text());files=build['source_hashes'].copy()
src=E/'native/cooperation_probe.cpp';files[str(src.relative_to(E))]=hashlib.sha256(src.read_bytes()).hexdigest()
for rel,h in files.items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
binary=out/('_dp7_coopprobe'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-fopenmp','-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
d=dict(status='PASS_BUILD' if not r.returncode else 'FAIL',seconds=time.perf_counter()-t,command=cmd,source_hashes=files,binary=str(binary))
if not r.returncode:d['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ('source_hashes','command')}),flush=True)
if r.returncode:print(r.stderr[-3000:])
raise SystemExit(r.returncode)
