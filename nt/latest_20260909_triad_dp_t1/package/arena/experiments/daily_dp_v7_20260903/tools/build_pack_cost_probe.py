"""Isolated profiling tree: production source and binary remain untouched."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sysconfig,time
import pybind11
E=Path(__file__).resolve().parents[1]
out=E/'native/s4u_cost_probe_v1';out.mkdir(exist_ok=False)
b=json.loads((E/'native/build/build_receipt.json').read_text());hashes={}
for rel,h in b['source_hashes'].items():
    source=E/rel;assert hashlib.sha256(source.read_bytes()).hexdigest()==h
    if rel.startswith('native/'):
        dest=out/rel.removeprefix('native/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        hashes[rel]=h
for name in ('pack_cost_profile.hpp','pack_cost_probe.cpp'):
    source=E/'native'/name;shutil.copy2(source,out/name);hashes['native/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
patches={
 'policy.hpp':[
   ('#include "observation_view.hpp"','#include "observation_view.hpp"\n#include "pack_cost_profile.hpp"'),
   ('bool hire_estimate=false)const{\n  std::stable_sort','bool hire_estimate=false)const{\n  packprofile::Call profile(js,starts,budget,ret,insertion,p.regret_schedule||(hire_estimate?p.regret_hire_estimate:p.regret_compile));\n  std::stable_sort'),
   ('int estimate(const std::vector<Job>&js,int base)const{','int estimate(const std::vector<Job>&js,int base)const{packprofile::Context profiling_context("estimate");')],
 'joint_portfolio.hpp':[
   ('inline double future_wages(const Controller&c,const View&o,const std::vector<std::pair<int,int>>&new_targets){','inline double future_wages(const Controller&c,const View&o,const std::vector<std::pair<int,int>>&new_targets){\n packprofile::Context profiling_context("future_wages");'),
   ('inline void Controller::compare_portfolios(const View&planning,const View&actual,int released){','inline void Controller::compare_portfolios(const View&planning,const View&actual,int released){\n packprofile::Context profiling_context("portfolio_other");')],
 'day_consequence.hpp':[
   ('inline double wages_on(const Controller&c,const std::vector<Work>&works,int day){','inline double wages_on(const Controller&c,const std::vector<Work>&works,int day){\n packprofile::Context profiling_context("residual_wages");'),
   ('inline Value schedule_value(const Controller&source,const View&o,const std::vector<Plan>&plans){','inline Value schedule_value(const Controller&source,const View&o,const std::vector<Plan>&plans){\n packprofile::Context profiling_context("schedule_value_other");')]}
for name,replacements in patches.items():
    path=out/name;text=path.read_text()
    for before,after in replacements:
        assert text.count(before)==1,(name,before);text=text.replace(before,after)
    path.write_text(text)
binary=out/('_dp7_packcost'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-march=native','-DNDEBUG','-fPIC','-fvisibility=hidden','-shared','-fopenmp',
    '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(out),
    str(out/'pack_cost_probe.cpp'),str(out/'vendor/simulator.cpp'),'-o',str(binary)]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
receipt=dict(status='PASS_BUILD' if not r.returncode else 'FAIL',command=cmd,seconds=time.perf_counter()-t,
    base_build=b,original_source_hashes=hashes,mechanical_instrumentation=patches,
    copied_header_hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*.hpp')},
    production_source_changed=False,production_binary_changed=False)
if not r.returncode:receipt['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ('status','seconds')}),flush=True)
if r.returncode:print(r.stderr[-2500:])
raise SystemExit(r.returncode)
