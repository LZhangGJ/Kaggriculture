"""Locate slow real action decisions; no policy/source/flag changes."""
from pathlib import Path
import hashlib,json,sys,time,zlib

E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'))
import _dp7_native as native
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
configs=json.loads((E/'profiles/s4t/configs.json').read_text())
out=E/'receipts/s4t_action_cost_trace_v1';out.mkdir(exist_ok=False)
rival=native.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())))
stats={'portfolio':native.portfolio_stats,'day_value':native.day_value_stats,
    'workforce':native.intraday_workforce_stats,'insertions':native.shared_insertion_stats,'handoff':native.idle_handoff_stats}
rows=[];games=[]
for label in ('full_chain','full_chain_autonomous'):
    for seed in (20262701,20262702):
        for seat in (0,1):
            env=native.Env(seed);c=native.Controller(configs[label]);rstate=native.G001State()
            local=[]
            while not env.done:
                step=env.step_count;before={k:fn(c) for k,fn in stats.items()}
                cpu=time.process_time();wall=time.perf_counter();own=c.act(env,seat)
                elapsed=time.perf_counter()-wall;used=time.process_time()-cpu
                after={k:fn(c) for k,fn in stats.items()}
                row=dict(label=label,seed=seed,seat=seat,step=step,day=step//24,hour=step%24,
                    wall_seconds=elapsed,cpu_seconds=used,
                    deltas={k:{f:after[k][f]-before[k][f] for f in after[k]} for k in after})
                local.append(row)
                other=rival.act(env,1-seat,rstate);env.step([own,other] if seat==0 else [other,own])
            assert len(local)==719
            money=[x['money'] for x in env.observation(seat)['farms']]
            game=dict(label=label,seed=seed,seat=seat,money=money,seconds=sum(r['wall_seconds'] for r in local),
                cpu_seconds=sum(r['cpu_seconds'] for r in local),
                slowest=sorted(local,key=lambda r:r['wall_seconds'],reverse=True)[:4])
            games.append(game);rows+=local
            print(json.dumps(game),flush=True)
            (out/'progress.json').write_text(json.dumps(games,indent=2))
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_COST_DIAGNOSTIC_NOT_STRENGTH',build=build,games=games,rows=rows,
    caveat='Single-thread native policy timing, alongside at most one official-check process. CPU time distinguishes waiting. Counters localize expensive decisions but are not a sampled function profiler.'),indent=2))
