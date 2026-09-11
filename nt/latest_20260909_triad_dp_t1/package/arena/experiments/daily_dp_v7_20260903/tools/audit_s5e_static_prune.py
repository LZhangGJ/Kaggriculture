"""Live-state equivalence of isolated pruning, not a new strategic candidate."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');a=p.parse_args()
read=lambda p:json.loads((E/p).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
panel=read('receipts/s5d_twelveway_N10_v1/results.json');build=read('native/build/build_receipt.json')
assert read('receipts/s5d_execution_v1/acceptance.json')['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE'
assert panel['build']==build
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h,rel
out=E/('receipts/s5e_static_prune_live_'+a.version);out.mkdir(exist_ok=False)
src=E/'native/compile_static_prune_probe.cpp';sources={}
for path in (src,E/'native/compile_static_prune_prototype.hpp'):
 sources[str(path.relative_to(E))]=sha(path);(out/path.name).write_bytes(path.read_bytes())
inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
binary=out/('_dp7_static_prune_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-march=native','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(binary)]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:
 (out/'failure.json').write_text(json.dumps(dict(status='COMPILE_FAILED_PRESERVED',command=cmd,returncode=r.returncode)));raise SystemExit(r.returncode)
(out/'build.json').write_text(json.dumps(dict(production=build,sources=sources,binary_sha256=sha(binary),seconds=time.perf_counter()-tic,command=cmd),indent=2))
sys.path[:0]=[str(E/'native/build'),str(out)];import _dp7_native as n;import _dp7_static_prune_probe as probe
label='all_intraday_insert_replant';refs={(r['opponent'],r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label}
rows=[];tic=time.perf_counter()
try:
 for opponent,entry in panel['identities'].items():
  if opponent=='pass':continue
  assert sha(E/entry['source'])==entry['source_sha256']
  for seed in range(20262701,20262704):
   for seat in (0,1):
    kind=entry['runtime'];state=None
    if kind in ('searched_route_native','boatlee_v29_native','kaito_v58_native','lynn_v5_native'):
     assert sha(E/entry['asset'])==entry['asset_sha256'];payload=json.loads(zlib.decompress((E/entry['asset']).read_bytes()))
     agent,cls={'searched_route_native':(n.G001,n.G001State),'boatlee_v29_native':(n.BoatleeV29,n.BoatleeState),'kaito_v58_native':(n.KaitoV58,n.KaitoState),'lynn_v5_native':(n.LynnV5,n.LynnState)}[kind]
     rival=agent(payload);state=cls()
    else:rival={'fieldbook_native':n.Fieldbook,'three_day_native':n.ThreeDay,'ecobot_v7_native':n.EcoBotV7}[kind]()
    env=n.Env(seed);c=n.Controller(panel['configurations'][label]);events=[]
    while not env.done:
     if c.debug()['phase']==2:
      before=[c.debug(),n.rotation_stats(c),n.compile_choice_stats(c)]
      event=probe.inspect(c,env,seat)
      assert before==[c.debug(),n.rotation_stats(c),n.compile_choice_stats(c)]
      if event['eligible']:events.append(dict(step=env.step_count,**event))
     own=c.act(env,seat);op=rival.act(env,1-seat,state) if state is not None else rival.act(env,1-seat)
     env.step([own,op] if seat==0 else [op,own])
    cash=[f['money'] for f in env.observation(seat)['farms']];ref=refs[opponent,seed,seat]
    assert (cash[seat],cash[1-seat])==(ref['cash'],ref['opponent_cash'])
    row=dict(opponent=opponent,seed=seed,seat=seat,events=events,cash=cash);rows.append(row)
    (out/'progress.json').write_text(json.dumps([dict(opponent=r['opponent'],seed=r['seed'],seat=r['seat'],cash=r['cash'],checkpoints=len(r['events'])) for r in rows],indent=2));print(json.dumps(dict(opponent=opponent,seed=seed,seat=seat,checkpoints=len(events))),flush=True)
except Exception as exc:
 (out/'failure.json').write_text(json.dumps(dict(status='FAIL_PRESERVED',opponent=opponent,seed=seed,seat=seat,step=env.step_count,error=repr(exc),observation=env.observation(seat),controller=c.debug()),indent=2));raise
with gzip.open(out/'events.json.gz','wt') as f:json.dump(rows,f)
events=[e for r in rows for e in r['events']];assert len(rows)==48 and len(events)>100
receipt=dict(status='PASS_LIVE_STATE_EQUIVALENCE_NOT_PRODUCTION',games=len(rows),checkpoints=len(events),source_build=build,seconds=time.perf_counter()-tic,policy_changed=False,totals={k:sum(e[k] for e in events) for k in ('candidate_count','original_evaluations','pruned_evaluations','original_seconds','pruned_seconds')},caveat='Isolated comparison timing is not full-match throughput; no new strength samples. Original policy continues all real games.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='source_build'}),flush=True)
