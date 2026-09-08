from pathlib import Path
import argparse,hashlib,json,sys,time,zlib
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'));import _dp7_native as n
cli=argparse.ArgumentParser();cli.add_argument('--version',default='v1');args=cli.parse_args()
cfg=json.loads((E/'profiles/s5b/configs.json').read_text());out=E/('receipts/s5b_runtime_'+args.version);out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
rival=n.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())));rows=[]
for label in ['all_intraday_insert','all_intraday_insert_net','all_intraday_insert_timing','full_chain_autonomous','full_chain_autonomous_net','full_chain_autonomous_timing']:
 for seat in (0,1):
  seed=20262701;env=n.Env(seed);c=n.Controller(cfg[label]);other=n.G001State();times=[];states=[]
  while not env.done:
   step=env.step_count;t=time.perf_counter()
   try:own=c.act(env,seat)
   except Exception as ex:
    (out/'failure.json').write_text(json.dumps(dict(label=label,seed=seed,seat=seat,step=step,error=str(ex),build=build,debug=c.debug(),rotation=n.rotation_stats(c),observation=env.observation(seat)),indent=2));raise
   times.append(time.perf_counter()-t)
   if step%24==0:states.append(dict(step=step,stats=n.rotation_stats(c),seconds=times[-1]))
   op=rival.act(env,1-seat,other);env.step([own,op] if seat==0 else [op,own])
  row=dict(label=label,seed=seed,seat=seat,max_act_seconds=max(times),total_own_seconds=sum(times),stats=n.rotation_stats(c),cash=[f['money'] for f in env.observation(seat)['farms']],states=states)
  rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ('states','stats')}),flush=True)
  # Stop a pathological probe after a verified completed game, not blindly
  # start an expensive panel. Keep the completed evidence and exact state.
  if max(times)>5:
   (out/'acceptance.json').write_text(json.dumps(dict(status='STOP_SLOW_REVIEW',rows=rows,build=build),indent=2));raise SystemExit(2)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_NATIVE_PROBE',rows=rows,build=build,final_online_latency=False),indent=2))
