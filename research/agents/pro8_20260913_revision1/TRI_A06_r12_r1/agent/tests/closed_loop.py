"""One bounded, official-engine real-policy game; no saved-action opponent.
Run both seats and a parent-vs-parent control externally with this CLI.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,json,resource,sys,time
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def module(label,path):
 spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--seat',type=int,choices=(0,1),required=True);p.add_argument('--variant',choices=('candidate','parent'),required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--referee',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);tag=f'{a.variant}_{a.seed}_seat{a.seat}'
 started=datetime.now(timezone.utc).isoformat();start=time.perf_counter();host=module('referee_host',a.referee/'cpu_runtime.py');engine=host.load_engine();game=host.LocalGame(a.seed,engine)
 pm=module('control_root',a.parent/'main.py');cm=module('candidate_root',R/'main.py');agents=[pm.create_agent(),pm.create_agent()]
 if a.variant=='candidate':agents[a.seat].close();agents[a.seat]=cm.create_agent()
 logs=[];days=[];max_call=[0.,0.];call_seconds=[0.,0.];action_hist=[{},{}];native=[hashlib.sha256((a.parent/'policy/a06.so').read_bytes()).hexdigest()]*2
 if a.variant=='candidate':native[a.seat]=hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest()
 assert game.configuration.seed is None
 for step in range(719):
  obs=[game.observation(0),game.observation(1)];actions=[]
  for seat in (0,1):
   assert obs[seat]['player']==seat and 'seed' not in obs[seat] and 'private' in obs[seat]
   t=time.perf_counter();act=agents[seat](obs[seat]);dt=time.perf_counter()-t;max_call[seat]=max(max_call[seat],dt);call_seconds[seat]+=dt
   assert isinstance(act,dict) and set(act)=={'farmer','hands','market'}
   assert len(act['hands'])==len(obs[seat]['farms'][seat]['hands'])
   assert len(act['market'])<=10
   for atom in [act['farmer'],*act['hands']]:action_hist[seat][atom[0]]=action_hist[seat].get(atom[0],0)+1
   actions.append(act)
  before=[obs[s]['farms'][s]['money'] for s in (0,1)];game.advance(actions)
  after=[game.state[s].observation['farms'][s]['money'] for s in (0,1)]
  logs.append({'step':step,'observation_sha256':[digest(x) for x in obs],'actions':actions,'cash_before':before,'cash_after':after,'state_sha256':digest(game.snapshot())})
  if step%24==23 or step==718:
   for s in (0,1):days.append({'day':step//24,'seat':s,'cash':after[s]})
 assert game.done and game.t==719 and len(days)==60
 assert all(s.status=='DONE' for s in game.state)
 cash=[game.state[s].observation['farms'][s]['money'] for s in (0,1)];margin=cash[a.seat]-cash[1-a.seat]
 final=[{'public_farm':copy.deepcopy(game.state[s].observation['farms'][s]),'own_private':copy.deepcopy(game.state[s].observation['private']),'debug':agents[s].debug()} for s in (0,1)]
 for agent in agents:agent.close()
 data={'scope':'new complete official-engine game; both policies execute against current legal observations','tag':tag,'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'seed_driver_only':a.seed,'tested_seat':a.seat,'variant':a.variant,'opponent':'exact supplied A06_r12 native (not a named public opponent or R2)','native_sha256':native,'engine_sha256':hashlib.sha256((a.referee/'official/kaggriculture.py').read_bytes()).hexdigest(),'steps':game.t,'player_days':len(days),'strict_win':margin>0,'draw':margin==0,'cash':cash,'margin':margin,'statuses':[s.status for s in game.state],'rewards':[s.reward for s in game.state],'seconds':time.perf_counter()-start,'policy_seconds':call_seconds,'max_call_seconds':max_call,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'unit_action_counts':action_hist,'exception_count':0}
 raw={'summary':data,'player_days_cash':days,'steps':logs,'final':final};rawpath=a.out/(tag+'.json.gz');rawpath.write_bytes(gzip.compress(json.dumps(raw).encode(),mtime=0));data['raw_sha256']=hashlib.sha256(rawpath.read_bytes()).hexdigest()
 (a.out/(tag+'.summary.json')).write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
if __name__=='__main__':main()
