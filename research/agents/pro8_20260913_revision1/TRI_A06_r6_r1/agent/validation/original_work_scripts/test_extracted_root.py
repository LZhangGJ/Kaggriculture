"""Test main.py:agent from a physically extracted ZIP against two measured traces.
Reproduction of the same revised identity, not extra benchmark games.
"""
import pathlib,importlib.util,sys,json,gzip,hashlib,time,argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();root=a.root.resolve();v=root/'validation'
sys.path.insert(0,str(v/'input/referee'));from cpu_runtime import LocalGame,load_engine,compare_frame
s=importlib.util.spec_from_file_location('physically_extracted_entry',root/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=json.loads((v/'diagnostics/closed_loop_results.json').read_text());engine=load_engine();checked=[];native=hashlib.sha256((root/'policy/a06.so').read_bytes()).hexdigest();assert native==results['candidate_native_sha256']
for row in results['all_rows'][:2]:
 m.reset();seat=row['candidate_seat'];g=LocalGame(row['seed'],engine);ts=time.perf_counter();path=v/row['trace']
 with gzip.open(path,'rt') as f:
  meta=json.loads(next(f));compare_frame(g,meta['initial_state']);count=0
  for line in f:
   x=json.loads(line);obs=g.observation(seat);act=m.agent(obs,g.configuration)
   assert act==x['actions'][seat],('first divergent extracted-root action',row['seed'],seat,g.t,act,x['actions'][seat])
   g.advance(x['actions']);compare_frame(g,x['state']);count+=1
 assert count==719 and g.done
 checked.append({'seed':row['seed'],'seat':seat,'matching_root_agent_calls':count,'verified_transitions':g.t,'wall_seconds':time.perf_counter()-ts});m.reset();assert not m._seats
out={'scope':'Physically extracted ZIP, actual main.py:agent callable in both seats; byte-matched production native; replay reproduction only, not extra wins.','root':str(root),'native_sha256':native,'rows':checked,'all_passed':True};a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
