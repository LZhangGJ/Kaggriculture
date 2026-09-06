from pathlib import Path
import subprocess,sys,json,hashlib,re,time,sysconfig
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent;B=P/'build';B.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(P/'prepare.py')],check=True)
original=(P/'stage/arena/module.cpp').read_text()
text=original.replace('PYBIND11_MODULE(_dp7_native,m){using namespace bridge;', '#include "runner.hpp"\nPYBIND11_MODULE(_economic_rl_native,m){using namespace bridge;econrl_bind(m);')
assert text!=original
text=re.sub(r'py::class_<([^>]+)>\(m,"([^"]+)"\)',r'py::class_<\1>(m,"\2",py::module_local())',text)
(P/'runner_module.cpp').write_text(text)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
common=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic','-ffp-contract=off','-pthread']
jobs=[]
for name in ('c3auto','c3j7'):
 jobs.append((name,common+(['-DRL_C3_AUTO'] if name=='c3auto' else [])+[str(P/'c3_policy.cpp'),str(P/f'stage/{name}/executor/vendor/simulator.cpp'),'-o',str(B/f'{name}.so')]))
boost=P.parents[1]/'gpt_review/gpt_code/gpt-6-dp/Kaggriculture_CPP_Daily_DP_20260905_v1_improve_c++/Kaggriculture_Daily_DP_CPP_20260905/vendor'
jobs.append(('f3',common+['-I'+str(boost),str(P/'f3_policy.cpp'),'-o',str(B/'f3.so')]))
includes=subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True).split()
arena=P/'stage/arena';ext=sysconfig.get_config_var('EXT_SUFFIX')
objects=[P.parents[1]/'experiments/daily_dp_v7_20260903/native/build'/f for f in ('simulator.o','boatlee_v29.o','kaito_v58.o','lynn_v5.o','fieldbook_adapter.o','three_day_adapter.o','ecobot_v7_core.o','ecobot_v7.o')]
object_hashes={str(f):sha(f) for f in objects}
jobs.append(('runner',common+['-fopenmp',*includes,'-I'+str(arena),'-I'+str(P),str(P/'runner_module.cpp'),*[str(f) for f in objects],'-ldl','-o',str(B/('_economic_rl_native'+ext))]))
def build(job):
 name,cmd=job;start=time.perf_counter();p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (B/f'{name}.log').write_text(p.stdout);r=dict(name=name,command=cmd,seconds=time.perf_counter()-start,returncode=p.returncode)
 print(name,p.returncode,round(r['seconds'],1),p.stdout[-2400:],flush=True);return r
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(build,jobs))
assert all(sha(Path(f))==h for f,h in object_hashes.items())
out=dict(results=rows,python=sys.version,object_hashes=object_hashes,compiler=subprocess.check_output(['g++','--version'],text=True),hashes={str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file() and f.suffix in ('.cpp','.hpp','.inc','.json','.so') and 'logs' not in f.parts})
(P/'BUILD_RECEIPT.json').write_text(json.dumps(out,indent=2))
if any(r['returncode'] for r in rows):raise SystemExit(1)
