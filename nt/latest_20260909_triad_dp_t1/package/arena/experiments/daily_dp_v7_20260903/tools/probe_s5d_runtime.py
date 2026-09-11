"""Serial latency and actual-use probe. Stop on a measured slow action."""
from pathlib import Path
import argparse, hashlib, json, sys, time, zlib
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'native/build'));import _dp7_native as n
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');p.add_argument('--configs',default=str(E/'profiles/s5d/configs.json'));a=p.parse_args()
configs=json.loads(Path(a.configs).read_text())
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
out=E/('receipts/s5d_runtime_'+a.version);out.mkdir(exist_ok=False)
rival=n.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())))
# Check the potentially most expensive composition before a large panel.
hard='full_chain_autonomous_timing_first_value_replant'
labels=[hard]+[x for x in configs if x!=hard];rows=[]
for label in labels:
 for seat in (0,1):
  seed=20262701;env=n.Env(seed);c=n.Controller(configs[label]);opstate=n.G001State();times=[];states=[]
  while not env.done:
   step=env.step_count;tic=time.perf_counter()
   try:own=c.act(env,seat)
   except Exception as ex:
    (out/'failure.json').write_text(json.dumps(dict(label=label,seed=seed,seat=seat,step=step,error=str(ex),build=build,debug=c.debug(),observation=env.observation(seat)),indent=2));raise
   seconds=time.perf_counter()-tic;times.append(seconds)
   if step%24==0 or seconds>.25:states.append(dict(step=step,seconds=seconds,stats=n.compile_choice_stats(c)))
   if seconds>1.:
    r=dict(status='STOP_SLOW_ACTION_NOT_COMPLETE_GAME',label=label,seed=seed,seat=seat,step=step,seconds=seconds,build=build,completed_rows=rows,states=states,stats=n.compile_choice_stats(c),debug=c.debug(),observation=env.observation(seat))
    (out/'acceptance.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('build','debug','observation','completed_rows','states')}),flush=True);raise SystemExit(2)
   op=rival.act(env,1-seat,opstate);env.step([own,op] if seat==0 else [op,own])
  row=dict(label=label,seed=seed,seat=seat,steps=env.step_count,max_act_seconds=max(times),total_own_seconds=sum(times),stats=n.compile_choice_stats(c),cash=[f['money'] for f in env.observation(seat)['farms']],states=states)
  rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='states'}),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_NATIVE_PROBE',rows=rows,build=build,final_online_latency=False),indent=2))
