"""Mechanical isolated runner generation: per-game mode and policy RNG seed."""
from pathlib import Path
import sys,sysconfig,subprocess,time,json,hashlib
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent;OLD=P.parent/'economic_rl_three_arm_20260906';FIX=P.parent/'economic_rl_f3_ledger_fix_20260906';ROOT=P.parents[1]
B=P/'build';B.mkdir(exist_ok=False)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
frozen=json.loads((FIX/'ACCEPTANCE.json').read_text());hashes=dict(frozen['fixed_hashes']);hashes.update(frozen['preserved_original'])
hashes.update(json.loads((OLD/'SOURCE_FREEZE.json').read_text()))
hashes.update({str(OLD/f):sha(OLD/f)for f in ['runner_module.cpp','runner.hpp','rl_common.hpp']})
for f,h in hashes.items():assert sha(f)==h,f
header=(OLD/'runner.hpp').read_text()
old='int mode,std::vector<uint64_t>seeds,std::vector<int>seats,std::vector<int>opponents,uint64_t sample_seed,int threads,bool trace'
new='std::vector<int>modes,std::vector<uint64_t>seeds,std::vector<int>seats,std::vector<int>opponents,std::vector<uint64_t>sample_seeds,int threads,bool trace'
assert header.count(old)==1;header=header.replace(old,new)
header=header.replace('if(seeds.size()!=seats.size()', 'if(seeds.size()!=modes.size()||seeds.size()!=sample_seeds.size()||seeds.size()!=seats.size()')
old='play(lib,weights,mode,seeds[n],seats[n],opponents[n],sample_seed+0x9e3779b97f4a7c15ULL*(n+1),trace)'
assert header.count(old)==1
header=header.replace(old,'play(lib,weights,modes[n],seeds[n],seats[n],opponents[n],sample_seeds[n],trace)')
(B/'runner.hpp').write_text(header)
module=(OLD/'runner_module.cpp').read_text()
assert module.count('PYBIND11_MODULE(_economic_rl_native,')==1
(B/'runner_module.cpp').write_text(module.replace('PYBIND11_MODULE(_economic_rl_native,','PYBIND11_MODULE(_economic_candidate_native,'))
flags=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic','-ffp-contract=off','-pthread']
boost=ROOT/'gpt_review/gpt_code/gpt-6-dp/Kaggriculture_CPP_Daily_DP_20260905_v1_improve_c++/Kaggriculture_Daily_DP_CPP_20260905/vendor'
includes=subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True).split()
objects=[ROOT/'experiments/daily_dp_v7_20260903/native/build'/f for f in ('simulator.o','boatlee_v29.o','kaito_v58.o','lynn_v5.o','fieldbook_adapter.o','three_day_adapter.o','ecobot_v7_core.o','ecobot_v7.o')]
hashes.update({str(f):sha(f)for f in objects})
jobs=[('policy',flags+['-I'+str(boost),str(P/'intervention.cpp'),'-o',str(B/'audit.so')]),
      ('runner',flags+['-fopenmp',*includes,'-I'+str(OLD/'stage/arena'),'-I'+str(OLD),str(B/'runner_module.cpp'),*[str(f)for f in objects],'-ldl','-o',str(B/('_economic_candidate_native'+sysconfig.get_config_var('EXT_SUFFIX')))])]
def run(job):
    name,cmd=job;t=time.perf_counter();r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (B/f'{name}.log').write_text(r.stdout);print(name,r.returncode,r.stdout[-2000:],flush=True)
    return dict(name=name,command=cmd,seconds=time.perf_counter()-t,exit_code=r.returncode)
with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(run,jobs))
for f,h in hashes.items():assert sha(f)==h,f
(P/'BUILD_RECEIPT.json').write_text(json.dumps(dict(status='PASS'if all(r['exit_code']==0 for r in results)else'FAIL',results=results,preserved=hashes,built={str(f):sha(f)for f in B.glob('*.so')}),indent=2))
if any(r['exit_code'] for r in results):raise SystemExit(1)
