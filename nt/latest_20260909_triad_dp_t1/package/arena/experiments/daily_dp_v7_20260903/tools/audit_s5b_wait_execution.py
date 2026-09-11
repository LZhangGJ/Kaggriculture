"""Unchanged live games: connect delay intents to actual plot occupancy."""
from pathlib import Path
import hashlib,json,sys,time,zlib
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'));import _dp7_native as n
panel=json.loads((E/'receipts/s5b_eightway_N10_v1/results.json').read_text())
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
out=E/'receipts/s5b_wait_execution_v1';out.mkdir(exist_ok=False)
rival=n.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())))
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
rows=[];started=time.perf_counter()
for label in ('all_intraday_insert_timing','full_chain_autonomous_timing'):
 for seed in (20262701,20262702,20262703):
  for seat in (0,1):
   env=n.Env(seed);ctl=n.Controller(panel['configurations'][label]);state=n.G001State();days=[]
   while not env.done:
    step=env.step_count;obs=env.observation(seat) if step%24==0 else None
    own=ctl.act(env,seat)
    if obs is not None:
     f=obs['farms'][seat];stats=n.rotation_stats(ctl);target=dict(ctl.debug()['target']);waiting=[]
     for pos,start in enumerate(stats['plant_not_before']):
      if start>step//24:
       tile=f['tiles'][pos//10][pos%10]
       waiting.append(dict(pos=pos,start=start,desired=stats['deferred_kind'][pos],target=target.get(pos),tile=tile))
     days.append(dict(day=step//24,cash=f['money'],waiting=waiting,seeds=obs['private']['seeds'],shed=obs['private']['shed'],rotation=stats))
    other=rival.act(env,1-seat,state);env.step([own,other] if seat==0 else [other,own])
   money=[f['money'] for f in env.observation(seat)['farms']];ref=refs[label,'g003',seed,seat]
   assert money[seat]==ref['cash'] and money[1-seat]==ref['opponent_cash']
   def empty(tile):return tile is None or (isinstance(tile,dict) and tile.get('kind')=='WEED')
   empty_wait=sum(empty(w['tile']) for d in days for w in d['waiting']);longest=0;run={};longest_pos=-1
   for d in days:
    current={w['pos'] for w in d['waiting'] if empty(w['tile'])}
    for pos in set(run)|current:
     run[pos]=run.get(pos,0)+1 if pos in current else 0
     if run[pos]>longest:longest=run[pos];longest_pos=pos
   row=dict(label=label,seed=seed,seat=seat,cash=money[seat],opponent_cash=money[1-seat],empty_wait_plot_days=empty_wait,longest_consecutive_empty_wait=longest,longest_pos=longest_pos,control_equal=True)
   (out/f'{label}_{seed}_{seat}.json').write_text(json.dumps(dict(summary=row,days=days),indent=2));rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_UNCHANGED_WAIT_EXECUTION_AUDIT',build=build,rows=rows,seconds=time.perf_counter()-started,caveat='12 unchanged live games, 3 reused development seeds. Empty waiting is observed behavior, not proof of bad economics or causal loss.'),indent=2))
