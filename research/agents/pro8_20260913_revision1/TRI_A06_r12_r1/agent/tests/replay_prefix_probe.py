"""Rebuild a genuine historical prefix with the official engine.
Compare both natives only on that prefix; stop at FIRST divergence. Permit one
same-tick transition with the known simultaneous opponent action, NOT its suffix.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,json,resource,time,types
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def module(label,path):
 s=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def branch(game):
 out=object.__new__(type(game))
 for key,value in game.__dict__.items():setattr(out,key,value if isinstance(value,types.ModuleType) else copy.deepcopy(value))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 case=next(x for x in json.loads((a.bundle/'SELECTED_CASES.json').read_text()) if x['id']==a.case)
 file=a.bundle/'replays'/(a.case+'.json.gz');replay=json.loads(gzip.decompress(file.read_bytes()));seat=replay['result']['seat'];seed=case['row']['seed'];assert seed==replay['info']['seed']==replay['result']['seed'];host=module('reference_host',a.bundle/'referee/cpu_runtime.py');game=host.LocalGame(seed)
 pm=module('parent_entry',a.parent/'main.py');cm=module('candidate_entry',R/'main.py');parent=pm.create_agent();candidate=cm.create_agent();start=time.perf_counter();hashes=[]
 row={'id':a.case,'scope':'official common-prefix verification and one simultaneous-tick probe; NOT a candidate completed game','native_sha256':sha(R/'policy/a06.so'),'parent_sha256':sha(a.parent/'policy/a06.so'),'historical_replay_sha256':sha(file),'matched':0,'compared':0,'first_difference':None,'completed_candidate_games':0}
 for t in range(719):
  host.compare_frame(game,replay['steps'][t]);obs=game.observation(seat);assert obs['player']==seat and 'seed' not in obs and game.configuration.seed is None
  pa=parent(obs);assert pa==replay['actions'][t][seat];ca=candidate(obs);row['compared']+=1;hashes.append({'step':t,'legal_observation_sha256':digest(obs),'parent_action_sha256':digest(pa),'candidate_action_sha256':digest(ca)})
  if pa!=ca:
   row['first_difference']=t;row['day']=t//24;row['hour']=t%24;counterfactual=branch(game)
   joint=copy.deepcopy(replay['actions'][t]);joint[seat]=ca
   # Other player's same-observation action is known at this tick only.
   counterfactual.advance(joint);game.advance(replay['actions'][t]);host.compare_frame(game,replay['steps'][t+1])
   factual=game.observation(seat);alternate=counterfactual.observation(seat)
   row['one_step']={'own_cash_parent':factual['farms'][seat]['money'],'own_cash_candidate':alternate['farms'][seat]['money'],'parent_seeds':factual['private']['seeds'],'candidate_seeds':alternate['private']['seeds'],'parent_hands':len(factual['farms'][seat]['hands']),'candidate_hands':len(alternate['farms'][seat]['hands']),'unit_actions_identical':pa['farmer']==ca['farmer'] and pa['hands']==ca['hands'],'no_saved_suffix_used':True}
   detail={'summary':row,'hashes':hashes,'legal_observation_before':obs,'parent_action':pa,'candidate_action':ca,'parent_debug':parent.debug(),'candidate_debug':candidate.debug(),'one_step_parent_legal_observation':factual,'one_step_candidate_legal_observation':alternate};break
  row['matched']+=1;game.advance(replay['actions'][t])
 else:detail={'summary':row,'hashes':hashes}
 parent.close();candidate.close();row.update(seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,all_compared_prefix_frames_exact=True)
 raw=a.out/(a.case+'.json.gz');raw.write_bytes(gzip.compress(json.dumps(detail).encode(),mtime=0));row['raw_sha256']=sha(raw);(a.out/(a.case+'.summary.json')).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
if __name__=='__main__':main()
