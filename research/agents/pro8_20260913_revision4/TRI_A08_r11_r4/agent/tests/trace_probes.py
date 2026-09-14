"""Frozen own-observation entry probes, explicitly ZERO new games.
Any divergent saved future is only a probe input, never a candidate continuation.
"""
import argparse,ctypes,gzip,hashlib,importlib.util,json,resource,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--agent',type=Path,default=ROOT/'main.py');p.add_argument('--library',type=Path);p.add_argument('--mode',choices=['root','factory'],default='root');p.add_argument('--trace',type=Path);p.add_argument('--limit',type=int,default=719);p.add_argument('--require-exact',action='store_true');p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
 if a.mode=='root'and a.library:p.error('Root mode uses its actual default library; specify factory for a library override')
 s=importlib.util.spec_from_file_location('probed_main_'+str(time.time_ns()),a.agent);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);a.out_dir.mkdir(parents=True,exist_ok=True);results=[]
 paths=[a.trace]if a.trace else sorted((ROOT/'evidence/round4_feedback/own_traces').glob('*.gz'))
 for path in paths:
  data=json.loads(gzip.decompress(path.read_bytes()));limit=min(a.limit,719,len(data['own_actions']));actions=[];times=[];exact=0;diff=[];last=None
  if a.mode=='root':m.reset();call=m.agent;instance=None
  else:instance=m.create_agent(binary_path=a.library);call=instance
  try:
   for step,obs in enumerate(data['observations'][:limit]):
    t=time.perf_counter();action=call(obs,data['configuration']);times.append(time.perf_counter()-t);actions.append(action);ok=action==data['own_actions'][step];exact+=ok
    if not ok:diff.append(step)
   if a.mode=='root':instance=m._seats[data['observations'][0]['player']]
   default=instance.lib._name
   if hasattr(instance.lib,'td_completion_json'):
    f=instance.lib.td_completion_json;f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_char_p;last=json.loads(f(instance.handle))
   row={'case':path.name,'mode':a.mode,'calls':limit,'exact':exact,'first_difference':diff[0]if diff else None,'difference_steps':diff,'native_path':str(default),'native_sha256':hashlib.sha256(Path(default).read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actions_sha256':hashlib.sha256(json.dumps(actions,sort_keys=True).encode()).hexdigest(),'max_call_seconds':max(times),'median_call_seconds':statistics.median(times),'total_call_seconds':sum(times),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'completion':last,'scope':'saved own observations; zero new games; post-difference inputs not a generated future'}
   (a.out_dir/(path.stem+'.actions.json.gz')).write_bytes(gzip.compress(json.dumps(actions).encode()));results.append(row);(a.out_dir/'SUMMARY.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
   if a.require_exact:assert exact==limit,row
  finally:
   if a.mode=='root':m.reset()
   elif instance:instance.close()
 print('COMPLETE '+str(sum(x['calls']for x in results))+' calls; zero games')
if __name__=='__main__':main()
