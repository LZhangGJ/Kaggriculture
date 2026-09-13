"""Bounded real root-entry head-to-head, with source-pinned agents and fresh seeds.
No saved action suffix, opponent private inventory, future seed script, PASS
opponent or stochastic script is passed to either policy.
"""
import sys,importlib.util,pathlib,json,gzip,time,hashlib,datetime,statistics,collections
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import LocalGame,load_engine
PANEL=R/'diagnostics/closed_loop_panel.json';panel=json.loads(PANEL.read_text());engine=load_engine();deadline=time.monotonic()+panel['time_cap_seconds'];results=[]
outdir=R/'diagnostics/closed_loop';outdir.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
mnew=module('tri_revised_game_root',R.parent/'main.py');mold=module('a06_parent_game_root',R/'input/agent/main.py')
assert sha(R.parent/'policy/a06.so')==panel['candidate_native_sha256'];assert sha(R/'input/agent/policy/a06.so')==panel['parent_native_sha256']
# Shape checks complement the local host, which is NOT the Kaggle sandbox.
ops=set(mnew.codec._OPS)
def validate(action,obs):
 assert set(action)=={'farmer','hands','market'}
 assert len(action['hands'])==len(obs['farms'][obs['player']]['hands'])
 for act in [action['farmer'],*action['hands'],*action['market']]:
  assert isinstance(act,list) and act and act[0] in ops,(act,obs['step'])
 assert len(action['market'])<=10

def save_summary():
 done=[r for r in results if r.get('completed')];wins=sum(r['margin']>0 for r in done);loss=sum(r['margin']<0 for r in done);ties=sum(r['margin']==0 for r in done)
 summary={'scope':panel['scope'],'panel_sha256':sha(PANEL),'candidate_native_sha256':panel['candidate_native_sha256'],'parent_native_sha256':panel['parent_native_sha256'],'planned_games':panel['games'],'completed_games':len(done),'wins':wins,'losses':loss,'ties':ties,'strict_win_rate':wins/len(done) if done else None,'mean_margin':statistics.mean(r['margin'] for r in done) if done else None,'all_rows':results,'public_pool_or_R2_win_rate_established':False}
 (R/'diagnostics/closed_loop_results.json').write_text(json.dumps(summary,indent=2));return summary
for seed in panel['seeds']:
 for seat in panel['seats']:
  if time.monotonic()>deadline-45:raise RuntimeError('Bounded panel checkpoint: insufficient remaining test budget for another game')
  name=f'{seed}_candidate_seat{seat}';trace=outdir/f'{name}.jsonl.gz';agent_new=mnew.create_agent();agent_old=mold.create_agent();agents=[None,None];agents[seat]=agent_new;agents[1-seat]=agent_old
  g=LocalGame(seed,engine);ts=time.perf_counter();timings=[[],[]];daily=[];cash_start=[s.observation.farms[i]['money'] for i,s in enumerate(g.state)];actions_hash=hashlib.sha256();money_flows=[collections.Counter(),collections.Counter()];failed=collections.Counter()
  original_commit=engine._commit_unit
  def ledger(op,item,price,farm,private,market,shed_capacity=100):
   sid=0 if farm is g.state[0].observation.farms[0] else 1
   before=farm['money'];ok=original_commit(op,item,price,farm,private,market,shed_capacity)
   if ok:money_flows[sid][op+':'+item]+=farm['money']-before
   else:failed[(sid,op,item)]+=1
   return ok
  engine._commit_unit=ledger
  row={'seed':seed,'candidate_seat':seat,'completed':False,'steps':0,'exceptions':[]}
  try:
   with gzip.open(trace,'wt',compresslevel=5) as f:
    f.write(json.dumps({'kind':'meta','seed':seed,'candidate_seat':seat,'panel_sha256':sha(PANEL),'initial_state':g.snapshot()},separators=(',',':'))+'\n')
    while not g.done:
     assert g.t<719
     if time.perf_counter()-ts>panel['max_game_seconds']:raise TimeoutError('Single-game test wall limit exceeded')
     joint=[]
     for sid in (0,1):
      obs=g.observation(sid);assert obs['player']==sid and 'seed' not in obs and g.configuration.seed is None
      start=time.perf_counter();action=agents[sid](obs,g.configuration);timings[sid].append(time.perf_counter()-start);validate(action,obs);joint.append(action)
     actions_hash.update(json.dumps(joint,sort_keys=True,separators=(',',':')).encode()+b'\n')
     g.advance(joint);row['steps']=g.t
     f.write(json.dumps({'kind':'transition','step':g.t,'actions':joint,'state':g.snapshot()},separators=(',',':'))+'\n')
     if g.t%24==0 or g.done:
      end=[g.state[0].observation.farms[i]['money'] for i in (0,1)]
      daily.extend({'day':(g.t-1)//24,'seat':sid,'cash_start':cash_start[sid],'cash_end':end[sid],'change':end[sid]-cash_start[sid]} for sid in (0,1));cash_start=end
    assert g.t==719 and len(daily)==60 and all(s.status=='DONE' for s in g.state)
    final=[float(s.reward) for s in g.state];assert final==list(map(float,cash_start))
    row.update(completed=True,final_cash=final,margin=final[seat]-final[1-seat],daily_cash=daily,agent_seconds_total=[sum(t) for t in timings],agent_seconds_max=[max(t) for t in timings],actions_sha256=actions_hash.hexdigest(),candidate_debug=agent_new.debug(),money_flows=[dict(x) for x in money_flows],failed_market_attempts=[{'seat':k[0],'op':k[1],'item':k[2],'count':v} for k,v in failed.items()])
    row['candidate_debug'].pop('last_search',None)
  except Exception as exc:
   row['exceptions'].append(type(exc).__name__+': '+str(exc));raise
  finally:
   engine._commit_unit=original_commit;agent_new.close();agent_old.close();row['wall_seconds']=time.perf_counter()-ts;row['trace']=str(trace.relative_to(R));row['trace_sha256']=sha(trace);results.append(row);summary=save_summary();print(json.dumps({k:v for k,v in row.items() if k not in ['daily_cash','candidate_debug','money_flows','failed_market_attempts']}),flush=True)
print(json.dumps({k:v for k,v in save_summary().items() if k!='all_rows'}),flush=True)
