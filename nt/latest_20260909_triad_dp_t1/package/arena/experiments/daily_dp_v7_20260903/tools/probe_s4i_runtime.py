"""Profile full live G003 games. No hypothetical continuation or search labels."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
p=argparse.ArgumentParser();p.add_argument('--round',default='s4i1');a=p.parse_args()
assert a.round.replace('_','').isalnum()
cfg=json.loads((EXP/f'profiles/{a.round}/configs.json').read_text())
rival=native.G001(json.loads(zlib.decompress((EXP/'native/g003_frozen.json.zlib').read_bytes())))
out=EXP/f'receipts/{a.round}_runtime_probe_v1';out.mkdir(exist_ok=False);rows=[]
for label in cfg:
    for seed in (20262701,20262702):
        for seat in (0,1):
            env=native.Env(seed);ctl=native.Controller(cfg[label]);rs=native.G001State();times=[];days=[]
            while not env.done:
                step=env.step_count;t=time.perf_counter();action=ctl.act(env,seat);dt=time.perf_counter()-t;times.append(dt)
                if step%24==0:days.append(dict(step=step,seconds=dt,stats=native.portfolio_stats(ctl),target=ctl.debug()['target']))
                opp=rival.act(env,1-seat,rs);actions=[action,opp] if seat==0 else [opp,action];env.step(actions)
            ordered=sorted(times)
            row=dict(label=label,seed=seed,seat=seat,steps=env.step_count,
                     total_own_act_seconds=sum(times),max_act_seconds=max(times),p99_act_seconds=ordered[int(.99*(len(ordered)-1))],
                     day_plans=days,stats=native.portfolio_stats(ctl))
            rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2))
    sample=[r for r in rows if r['label']==label]
    print(json.dumps(dict(label=label,max_act_seconds=max(r['max_act_seconds'] for r in sample),
                         mean_own_seconds=statistics.fmean(r['total_own_act_seconds'] for r in sample),
                         mean_switches=statistics.fmean(r['stats']['switches'] for r in sample))),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_RUNTIME_PROBE_NOT_STRENGTH',build=build,rows=rows,
    games=len(rows),hard_max_action_seconds=max(r['max_act_seconds'] for r in rows),
    caveat='Native CPU act measured from Python wrapper, not the eventual Python submission. N seeds are development; no parameter selected from these wins.'),indent=2))
