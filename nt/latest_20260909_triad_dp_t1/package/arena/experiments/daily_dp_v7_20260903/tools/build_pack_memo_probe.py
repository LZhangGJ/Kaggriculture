"""Build a mechanically edited, isolated policy copy for exact memo testing."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sysconfig,time
import pybind11
cli=argparse.ArgumentParser();cli.add_argument('--version',choices=['v1','v2','v3'],default='v1');args=cli.parse_args()
E=Path(__file__).resolve().parents[1];out=E/f'native/s4u_memo_probe_{args.version}';out.mkdir(exist_ok=False)
b=json.loads((E/'native/build/build_receipt.json').read_text());hashes={}
for rel,h in b['source_hashes'].items():
    source=E/rel;assert hashlib.sha256(source.read_bytes()).hexdigest()==h
    if rel.startswith('native/'):
        dest=out/rel.removeprefix('native/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[rel]=h
for name in ('pack_memo.hpp','pack_memo_probe.cpp'):
    source=E/'native'/name;shutil.copy2(source,out/name);hashes['native/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
path=out/'policy.hpp';text=path.read_text()
if args.version in ('v2','v3'):
    first=text.index('std::pair<std::vector<Route>,int>regret_pack(')
    last=text.index('std::pair<std::vector<Route>,int>pack(',first)
    optimized=E/'native/regret_pack_incremental.inc'
    hashes['native/regret_pack_incremental.inc']=hashlib.sha256(optimized.read_bytes()).hexdigest()
    shutil.copy2(optimized,out/optimized.name)
    text=text[:first]+optimized.read_text()+'\n '+text[last:]
marker='struct Order{Action a;int priority=9;};'
assert text.count(marker)==1
text=text.replace(marker,'}\n#include "pack_memo.hpp"\nnamespace dp7 {\n'+marker)
signature='std::pair<std::vector<Route>,int>pack(std::vector<Job>js,const std::vector<int>&starts,int budget,bool ret,bool insertion=false,bool hire_estimate=false)const{'
assert text.count(signature)==1
wrapper=signature+'''
  auto*cache=packmemo::active;
  if(!cache)return pack_uncached(std::move(js),starts,budget,ret,insertion,hire_estimate);
  cache->calls++;
  const bool regret=p.regret_schedule||(hire_estimate?p.regret_hire_estimate:p.regret_compile);
  auto key=packmemo::key(js,starts,budget,ret,insertion,regret);
  auto it=cache->entries.find(key);
  if(it!=cache->entries.end()){
   cache->hits++;regret_trials+=it->second.trials;regret_improvements+=it->second.improvements;
   return it->second.result;
  }
  cache->misses++;int trials=regret_trials,improvements=regret_improvements;
  auto result=pack_uncached(std::move(js),starts,budget,ret,insertion,hire_estimate);
  cache->remember(std::move(key),result,regret_trials-trials,regret_improvements-improvements);
  return result;
 }
 '''+signature.replace('>pack(', '>pack_uncached(')
text=text.replace(signature,wrapper);path.write_text(text)
if args.version=='v3':
    # Same exact memo keys and schedules, but caller owns the cache for ONE
    # game. No Controller layout changes or shared cache between games.
    proxy=out/'pack_memo_probe.cpp';code=proxy.read_text()
    edits=[
      (' m.def("act",[](Controller&c,const Simulator&env,int seat,bool enabled){',
       ' py::class_<packmemo::Cache>(m,"Cache").def(py::init<>());\n m.def("act",[](Controller&c,const Simulator&env,int seat,bool enabled,packmemo::Cache*storage){'),
      ('packmemo::Cache cache;PlayerAction result;',
       'packmemo::Cache local;auto&cache=storage?*storage:local;auto old_calls=cache.calls,old_hits=cache.hits,old_misses=cache.misses,old_not_stored=cache.not_stored;PlayerAction result;'),
      ('out["calls"]=cache.calls;out["hits"]=cache.hits;out["misses"]=cache.misses;',
       'out["calls"]=cache.calls-old_calls;out["hits"]=cache.hits-old_hits;out["misses"]=cache.misses-old_misses;'),
      ('out["not_stored"]=cache.not_stored;', 'out["not_stored"]=cache.not_stored-old_not_stored;'),
      ('py::arg("enabled")=true);', 'py::arg("enabled")=true,py::arg("storage")=nullptr);')]
    for before,after in edits:
        assert code.count(before)==1,before;code=code.replace(before,after)
    proxy.write_text(code)
binary=out/('_dp7_packmemo'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-march=native','-DNDEBUG','-fPIC','-fvisibility=hidden','-shared','-fopenmp',
    '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(out),
    str(out/'pack_memo_probe.cpp'),str(out/'vendor/simulator.cpp'),'-o',str(binary)]
t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
receipt=dict(status='PASS_BUILD' if not r.returncode else 'FAIL',command=cmd,seconds=time.perf_counter()-t,base_build=b,
    original_source_hashes=hashes,generated_policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    generated_proxy_sha256=hashlib.sha256((out/'pack_memo_probe.cpp').read_bytes()).hexdigest(),
    production_source_changed=False,production_binary_changed=False)
if not r.returncode:receipt['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ('status','seconds')}),flush=True)
if r.returncode:print(r.stderr[-2500:])
raise SystemExit(r.returncode)
