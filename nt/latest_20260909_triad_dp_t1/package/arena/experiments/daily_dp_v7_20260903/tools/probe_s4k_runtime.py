"""Native CPU action latency and switch activity during full live games."""
from pathlib import Path
import argparse,hashlib,json,statistics as st,sys,time,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
p=argparse.ArgumentParser();p.add_argument('--round',default='s4k');a=p.parse_args();r=a.round;assert r in ('s4k','s4k2','s4l','s4m1','s4n1','s4q','s4r','s4s','s4t','s4u','s4v')
cfg=json.loads((EXP/f'profiles/{r}/configs.json').read_text())
rival=native.G001(json.loads(zlib.decompress((EXP/'native/g003_frozen.json.zlib').read_bytes())))
out=EXP/f'receipts/{r}_runtime_probe_v1';out.mkdir(exist_ok=False);rows=[]
for label,p in cfg.items():
    for seed in (20262701,20262702):
        for seat in (0,1):
            env=native.Env(seed);ctl=native.Controller(p);rs=native.G001State();times=[]
            while not env.done:
                t=time.perf_counter();action=ctl.act(env,seat);times.append(time.perf_counter()-t)
                opp=rival.act(env,1-seat,rs);env.step([action,opp] if seat==0 else [opp,action])
            rows.append(dict(label=label,seed=seed,seat=seat,steps=env.step_count,max_act_seconds=max(times),total_own_seconds=sum(times),service=native.service_stats(ctl),shared_insertion=native.shared_insertion_stats(ctl),intraday_workforce=native.intraday_workforce_stats(ctl) if r in ('s4n1','s4q','s4r','s4t') else None,day_value=native.day_value_stats(ctl) if r in ('s4q','s4r','s4t') else None,idle_handoff=native.idle_handoff_stats(ctl) if r in ('s4s','s4t') else None))
            if r in ('s4u','s4v'):rows[-1].update(schedule_cache=native.schedule_cache_stats(ctl),portfolio=native.portfolio_stats(ctl),intraday_workforce=native.intraday_workforce_stats(ctl),day_value=native.day_value_stats(ctl),idle_handoff=native.idle_handoff_stats(ctl))
            if r=='s4v':rows[-1]['live_repair']=native.live_repair_stats(ctl)
    subset=[x for x in rows if x['label']==label]
    print(json.dumps(dict(label=label,max_act_seconds=max(x['max_act_seconds'] for x in subset),service={k:st.fmean(x['service'][k] for x in subset) for k in subset[0]['service']})),flush=True)
    (out/'progress.json').write_text(json.dumps(rows,indent=2))
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_RUNTIME_PROBE_NOT_STRENGTH',build=build,rows=rows,games=len(rows),
    hard_max_action_seconds=max(x['max_act_seconds'] for x in rows),
    caveat='Native act through Python wrapper. Not final submitted Python implementation or online timeout acceptance. Development seeds only.'),indent=2))
