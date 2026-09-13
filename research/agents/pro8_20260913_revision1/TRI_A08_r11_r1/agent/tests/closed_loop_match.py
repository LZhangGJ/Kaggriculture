#!/usr/bin/env python3
"""Full official matches; each policy isolated in its own process.
The seed and referee state never enter the policy workers. Only legal per-seat
observations cross the pipe. An A08-parent opponent is NOT original AFS R2.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, importlib.util, json, multiprocessing as mp
import os, resource, sys, time, traceback

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy_worker(root,binary,connection):
 instance=None
 try:
  spec=importlib.util.spec_from_file_location('isolated_policy',Path(root)/'main.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  instance=module.create_agent(Path(binary))
  connection.send({'ready':True,'pid':os.getpid(),'binary_sha256':sha(binary)})
  while True:
   observation=connection.recv()
   if observation is None:
    connection.send({'closed':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});break
   assert 'seed' not in observation
   assert 'private' in observation and all('private' not in f and 'shed' not in f and 'seeds' not in f for f in observation['farms'])
   start=time.perf_counter();action=instance(observation);seconds=time.perf_counter()-start
   debug=instance.debug();selected={k:v for k,v in debug.items() if k.startswith('sale_') or k.startswith('paid_') or k in ('plan_calls','service_switches','intraday_started','search_calls','search_changes','land_last_selected')}
   sale=selected.get('sale_schedule',{})
   if sale.get('step')!=observation['step']:selected.pop('sale_schedule',None)
   connection.send({'action':action,'seconds':seconds,'debug':selected})
 except BaseException as exc:
  try:connection.send({'error':repr(exc),'traceback':traceback.format_exc()})
  except Exception:pass
 finally:
  if instance is not None:instance.close()
  connection.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--referee',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--candidate-seat',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.referee));from cpu_runtime import LocalGame,load_engine
 ctx=mp.get_context('spawn');workers=[];pipes=[];native=[];day_rows=[];failed=[0,0];invalid_sells=[0,0];events=[];action_hash=[hashlib.sha256(),hashlib.sha256()];worker_rss=[];max_calls=[0.,0.];started=time.perf_counter()
 result={'scope':'new complete closed-loop games versus attached A08_r11 only; NOT original AFS R2/public-pool acceptance','seed_referee_only':a.seed,'candidate_seat':a.candidate_seat,'complete':False,'errors':[],'official_sha256':sha(a.referee/'official/kaggriculture.py')}
 engine=load_engine();game=LocalGame(a.seed,engine)
 try:
  for seat in (0,1):
   root=a.candidate if seat==a.candidate_seat else a.parent
   binary=root/('policy/tri_a08_r11_r1.so' if seat==a.candidate_seat else 'policy/a08_r11.so')
   con,child=ctx.Pipe();proc=ctx.Process(target=policy_worker,args=(str(root),str(binary),child));proc.start();child.close();workers.append(proc);pipes.append(con)
   assert con.poll(20),'Policy initialization timed out';ready=con.recv();assert ready.get('ready'),ready;native.append(ready['binary_sha256'])
  def wrap_money(name):
   original=getattr(engine,name)
   def call(*args,**kwargs):
    farm=args[3] if name=='_commit_unit' else args[0]
    seat=next(s for s in (0,1) if farm is game.state[0].observation.farms[s]);before=farm['money'];ans=original(*args,**kwargs)
    event={'player':seat,'op':args[0] if name=='_commit_unit' else name,'item':args[1] if name=='_commit_unit' else None,'price':args[2] if name=='_commit_unit' else None,'cash_delta':farm['money']-before,'success':bool(ans) if name=='_commit_unit' else farm['money']!=before}
    if name=='_commit_unit' and not ans:failed[seat]+=1
    events.append(event);return ans
   setattr(engine,name,call)
  for name in ('_commit_unit','_do_hire','_do_buy_land'):wrap_money(name)
  original_market=engine._process_market;post_market=[]
  def market(state,env):
   post_market.clear()
   for seat in (0,1):
    shed=copy.deepcopy(state[seat].observation.private['shed']);post_market.append(shed);requested={}
    for order in state[seat].action.get('market',[])[:10]:
     if len(order)>=3 and order[0]=='SELL':requested[order[1]]=requested.get(order[1],0)+int(order[2])
    invalid_sells[seat]+=sum(q>shed.get(item,0) for item,q in requested.items())
   return original_market(state,env)
  engine._process_market=market
  with gzip.open(a.out/'trajectory.jsonl.gz','wt',encoding='utf-8') as trace:
   while not game.done:
    assert game.t<720,'Unexpected episode length'
    observations=[game.observation(s) for s in (0,1)]
    for s in (0,1):pipes[s].send(observations[s])
    replies=[]
    for s in (0,1):
     assert pipes[s].poll(20),f'Action timeout seat={s}, step={game.t}'
     r=pipes[s].recv();assert 'error' not in r,r;assert isinstance(r['action'],dict);replies.append(r);max_calls[s]=max(max_calls[s],r['seconds'])
    actions=[r['action'] for r in replies]
    for s in (0,1):action_hash[s].update(canonical(actions[s]).encode()+b'\n')
    before=[f['money'] for f in game.state[0].observation.farms];events.clear();step=game.t;game.advance(actions)
    after=[f['money'] for f in game.state[0].observation.farms]
    residual=[after[s]-before[s]-sum(e['cash_delta'] for e in events if e['player']==s) for s in (0,1)];assert residual==[0,0],(step,residual)
    assert all(state.status in ('ACTIVE','DONE') for state in game.state)
    row={'step':step,'legal_observations':observations,'actions':actions,'post_units_shed':copy.deepcopy(post_market),'after_cash_both':after,'events':list(events),'cash_identity_residual':residual,'policy_seconds':[r['seconds'] for r in replies],'policy_debug':[r['debug'] for r in replies]}
    trace.write(canonical(row)+'\n')
    if game.t%24==0 or game.done:
     for s in (0,1):day_rows.append({'day':step//24,'player':s,'cash':after[s],'step_after':game.t})
   assert game.t==719 and len(day_rows)==60
  cash=[f['money'] for f in game.state[0].observation.farms];margin=cash[a.candidate_seat]-cash[1-a.candidate_seat]
  result.update(complete=True,steps=game.t,terminal_cash_both=cash,candidate_margin=margin,candidate_strict_win=margin>0,draw=margin==0,day_cash_rows=day_rows,day_cash_rows_count=len(day_rows),cash_identity_residual_all_zero=True,failed_market_units_both=failed,over_requested_post_units_sells_both=invalid_sells,max_policy_call_seconds_both=max_calls,action_sha256_both=[h.hexdigest() for h in action_hash],native_sha256_both=native)
 except BaseException as exc:
  result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc(),'step':game.t});raise
 finally:
  for con,proc in zip(pipes,workers):
   if proc.is_alive():
    try:
     con.send(None)
     if con.poll(5):worker_rss.append(con.recv())
    except Exception:pass
   proc.join(5)
   if proc.is_alive():proc.terminate();proc.join(5)
   con.close()
  result.update(wall_seconds=time.perf_counter()-started,worker_exitcodes=[p.exitcode for p in workers],worker_resource_receipts=worker_rss,driver_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('day_cash_rows',)},separators=(',',':')),flush=True)
if __name__=='__main__':main()
