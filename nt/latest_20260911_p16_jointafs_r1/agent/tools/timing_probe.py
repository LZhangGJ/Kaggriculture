"""Single-process live timing repeat of one archived worst-latency case.
Repeated match is a diagnostic, not an additional independent strength sample.
"""
from pathlib import Path
import sys,time,json,hashlib,copy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import arena

def one(binary):
 name='nagatakengo_v70';seed=2609205001;seat=0
 pub=arena.FreshPublic(name);player=arena.codec.Agent(config=json.loads((R/'policy/config.json').read_text()),binary_path=str(binary));env=arena.LocalGame(seed,arena.load_engine());actions=[];times=[]
 try:
  while not env.done:
   obs=[env.observation(i)for i in(0,1)];b=pub.call(obs[seat],copy.deepcopy(env.configuration));w=time.perf_counter();c=time.process_time();a=player(obs[1-seat]);cpu=time.process_time()-c;wall=time.perf_counter()-w
   times.append({'step':env.t,'wall_seconds':wall,'cpu_seconds':cpu})
   pair=[b,a];actions.append(copy.deepcopy(pair));env.advance(pair)
  cash=[f['money']for f in env.observation(1)['farms']]
  return {'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'opponent':name,'seed':seed,'opponent_seat':seat,'steps':env.t,'cash':cash,'action_hash':arena.digest(actions),'max_wall_seconds':max(x['wall_seconds']for x in times),'max_cpu_seconds':max(x['cpu_seconds']for x in times),'calls_over_1s':sum(x['wall_seconds']>1 for x in times),'slowest':sorted(times,key=lambda x:x['wall_seconds'],reverse=True)[:12]}
 finally:player.close();pub.close()
if __name__=='__main__':
 out=R/'evidence/TIMING_REPEAT.json'
 if out.exists():raise SystemExit('Refusing overwrite')
 results={k:one(R/p)for k,p in [('parent','baselines/merged.so'),('joint','policy/joint.so')]}
 for arm,run in [('parent','base5'),('joint','joint5')]:
  rr=next(r for r in json.loads((R/'evidence/panels'/run/'rows.json').read_text())if r['seed']==2609205001 and r['opponent']=='nagatakengo_v70'and r['opponent_seat']==0)
  results[arm]['matches_archive']=results[arm]['action_hash']==rr['action_hash']
  results[arm]['panel_wall_max_for_match']=rr['latency_max']
 out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
