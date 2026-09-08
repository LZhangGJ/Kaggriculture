"""Read actual selector counters/wait intents during unchanged native games.

Not another search, not a future-outcome selector. Replays 3 already-used seeds
against all eight live opponents to identify what B actually changed.
"""
from pathlib import Path
import collections,gzip,hashlib,json,sys,time,zlib
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'));import _dp7_native as n
read=lambda p:json.loads((E/p).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
panel=read('receipts/s5d_twelveway_N10_v1/results.json');run=read('receipts/s5d_execution_v1/acceptance.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and run['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE'
build=read('native/build/build_receipt.json');assert build==panel['build']
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
label='all_intraday_insert_replant';out=E/'receipts/s5d_choice_effects_v1';out.mkdir(exist_ok=False)
refs={(r['opponent'],r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label}
rows=[];tic=time.perf_counter()
for opponent,entry in panel['identities'].items():
 if opponent=='pass':continue
 assert sha(E/entry['source'])==entry['source_sha256']
 for seed in range(20262701,20262704):
  for seat in (0,1):
   kind=entry['runtime'];state=None
   if kind in ('searched_route_native','boatlee_v29_native','kaito_v58_native','lynn_v5_native'):
    assert sha(E/entry['asset'])==entry['asset_sha256'];payload=json.loads(zlib.decompress((E/entry['asset']).read_bytes()))
    classes={'searched_route_native':(n.G001,n.G001State),'boatlee_v29_native':(n.BoatleeV29,n.BoatleeState),'kaito_v58_native':(n.KaitoV58,n.KaitoState),'lynn_v5_native':(n.LynnV5,n.LynnState)}
    agent,cls=classes[kind];rival=agent(payload);state=cls()
   else:rival={'fieldbook_native':n.Fieldbook,'three_day_native':n.ThreeDay,'ecobot_v7_native':n.EcoBotV7}[kind]()
   env=n.Env(seed);c=n.Controller(panel['configurations'][label]);changes=0;wait=[0]*100;events=[]
   while not env.done:
    step=env.step_count;own=c.act(env,seat);stats=n.compile_choice_stats(c)
    if stats['changes']!=changes:
     now=n.rotation_stats(c)['plant_not_before'];deferred=[p for p in range(100) if now[p]>wait[p] and now[p]>step//24]
     events.append(dict(step=step,new_deferred_plots=deferred,kind='includes_replant_delay' if deferred else 'schedule_order_or_grouping',stats=stats,action=own));wait=now;changes=stats['changes']
    op=rival.act(env,1-seat,state) if state is not None else rival.act(env,1-seat)
    env.step([own,op] if seat==0 else [op,own])
   cash=[f['money'] for f in env.observation(seat)['farms']];ref=refs[opponent,seed,seat]
   assert (cash[seat],cash[1-seat])==(ref['cash'],ref['opponent_cash'])
   row=dict(label=label,opponent=opponent,seed=seed,seat=seat,cash=cash,unchanged_result=True,counts=dict(collections.Counter(e['kind'] for e in events)),events=events);rows.append(row)
   (out/'progress.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in rows],indent=2));print(json.dumps({k:v for k,v in row.items() if k!='events'}),flush=True)
with gzip.open(out/'events.json.gz','wt') as f:json.dump(rows,f)
counts=dict(collections.Counter(e['kind'] for r in rows for e in r['events']))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_UNCHANGED_LIVE_TRACE',build=build,games=len(rows),counts=counts,seconds=time.perf_counter()-tic,caveat='Actual changes classified from controller counters/wait intents; not an isolated profit attribution or independent evaluation.'),indent=2));print(json.dumps(counts),flush=True)
